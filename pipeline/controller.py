#!/usr/bin/env python3
"""Agent-R pipeline controller.

Runs inside the parent Kubernetes Job. For each iteration it launches one child Job per step,
waits for it, checks its output, and writes a .done marker:

    search -> revise (path_collection) -> build SFT data -> SFT -> eval
for each dataset listed in config `tasks` (e.g. [webshop] to finish one dataset end to end).

Rerunning the parent Job resumes: steps with a .done marker are skipped, child Jobs that already
succeeded are kept, failed ones are recreated. Standard library only (runs in alpine/k8s).
"""
import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent / "templates"
SCIWORLD_TASK_NUMS = 23  # mcts_collection.py task_nums list (maintainers' split)
POLL_SECONDS = 60


def log(*parts):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *parts, flush=True)


def run(args, stdin=None, check=True, retries=3):
    for attempt in range(retries):
        result = subprocess.run(args, input=stdin, text=True, capture_output=True)
        if result.returncode == 0 or not check:
            return result
        if "NotFound" in result.stderr or attempt == retries - 1:
            break
        time.sleep(10 * (attempt + 1))  # transient API errors
    if check:
        raise RuntimeError(f"{' '.join(args)} failed: {result.stderr.strip()}")
    return result


def load_config(path):
    return json.loads(run(["yq", "-o=json", ".", str(path)]).stdout)


def split_evenly(items, parts):
    """Split a list into `parts` contiguous chunks whose sizes differ by at most one."""
    size, extra = divmod(len(items), parts)
    chunks, start = [], 0
    for i in range(parts):
        end = start + size + (1 if i < extra else 0)
        chunks.append(items[start:end])
        start = end
    return [c for c in chunks if c]


class Pipeline:
    def __init__(self, cfg, code_dir, sha):
        self.cfg = cfg
        self.code = Path(code_dir)
        self.sha = sha
        self.ns = cfg["namespace"]
        self.root = Path(cfg["run_root"]) / cfg["run"]
        self.logs = self.root / "logs"
        self.tasks = cfg["tasks"]  # e.g. [webshop]: one dataset end to end

    # ---------- Kubernetes ----------
    def job_state(self, name):
        result = run(["kubectl", "get", "job", name, "-n", self.ns, "-o", "json"], check=False)
        if result.returncode != 0:
            if "NotFound" in result.stderr:
                return None
            raise RuntimeError(result.stderr.strip())
        status = json.loads(result.stdout).get("status", {})
        if status.get("succeeded"):
            return "succeeded"
        for cond in status.get("conditions") or []:
            if cond.get("type") == "Failed" and cond.get("status") == "True":
                return "failed"
        return "running"

    def run_jobs(self, jobs):
        """jobs: list of (name, manifest). Create missing/failed ones, then wait for all to succeed."""
        for name, manifest in jobs:
            state = self.job_state(name)
            if state == "succeeded":
                log(f"  {name}: already succeeded, keeping")
                continue
            if state == "failed":
                log(f"  {name}: previous attempt failed, recreating")
                run(["kubectl", "delete", "job", name, "-n", self.ns, "--wait=true"])
                state = None
            if state is None:
                (self.logs / "manifests").mkdir(parents=True, exist_ok=True)
                (self.logs / "manifests" / f"{name}.yaml").write_text(manifest)
                run(["kubectl", "apply", "-n", self.ns, "-f", "-"], stdin=manifest)
                log(f"  {name}: submitted")
            else:
                log(f"  {name}: already running, waiting")
        pending = {name for name, _ in jobs}
        while pending:
            time.sleep(POLL_SECONDS)
            for name in sorted(pending):
                state = self.job_state(name)
                if state == "succeeded":
                    log(f"  {name}: succeeded")
                    pending.discard(name)
                elif state == "failed":
                    raise RuntimeError(
                        f"child Job {name} failed. Inspect: kubectl logs -n {self.ns} job/{name} -c policy "
                        f"(or -c sft); worker logs under {self.logs}. Resubmit the parent Job to resume."
                    )
                elif state is None:
                    raise RuntimeError(f"child Job {name} disappeared")

    # ---------- templates ----------
    def render(self, template, values):
        text = (TEMPLATES / template).read_text()
        for key, value in values.items():
            text = text.replace("{{" + key + "}}", str(value))
        leftover = re.findall(r"\{\{[A-Z_]+\}\}", text)
        if leftover:
            raise RuntimeError(f"{template}: unfilled placeholders {sorted(set(leftover))}")
        return text

    def job_name(self, iteration, step, shard=None):
        name = f"agr-{self.cfg['run']}-i{iteration}-{step}" + ("" if shard is None else f"-{shard}")
        if len(name) > 63 or not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name):
            raise RuntimeError(f"invalid Job name {name}")
        return name

    def base_values(self, job_name):
        c = self.cfg
        return {
            "JOB_NAME": job_name,
            "RUN": c["run"],
            "PVC": c["pvc"],
            "NODE_KEY": c["node_selector_key"],
            "NODE_VALUE": c["node_selector_value"],
            "IMAGE_VLLM": c["images"]["vllm"],
            "IMAGE_SFT": c["images"]["sft"],
            "IMAGE_ENV_SERVER": c["images"]["env_server"],
            "LOG_DIR": self.logs,
        }

    def inference_env(self, job_name, task, model_dir, workdir, workers, model_type="Raw", temp=None):
        """Env block shared by search, revise and eval containers (12-space YAML indent)."""
        c, inf, s = self.cfg, self.cfg["inference"], self.cfg["search"]
        env = {
            "JOB_NAME": job_name,
            "CODE": self.code,
            "WORKDIR": workdir,
            "LOG_DIR": self.logs,
            "TASK": task,
            "MODEL_NAME": c["model"]["name"],
            "MODEL_DIR": model_dir,
            "MODEL_TYPE": model_type,
            "MAX_DEPTH": s["max_depth"],
            "ITERA": s["itera"],
            "N_GEN": s["n_gen"],
            "TEMP": inf["temp"] if temp is None else temp,
            "MAX_TOKEN_LENGTH": inf["max_token_length"],
            "MAX_NEW_TOKENS": inf["max_new_tokens"],
            "ENABLE_THINKING": inf["enable_thinking"],
            "STOP_TOKENS": "",
            "VLLM_DTYPE": inf["vllm_dtype"],
            "VLLM_API_BASE": "http://127.0.0.1:8000/v1",
            "WORKERS": workers,
            "PYTHONPATH": f"{self.code}:{c['environments'][task]['agentenv']}:/data/envs/policy-site",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/data/models/huggingface",
            "HF_HUB_OFFLINE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
        return "\n".join(f"            - {{name: {k}, value: {json.dumps(str(v))}}}" for k, v in env.items())

    START_VLLM = "\n".join([
        '              python3 -m vllm.entrypoints.openai.api_server --model "$MODEL_DIR" --dtype "$VLLM_DTYPE" \\',
        '                --gpu-memory-utilization 0.90 --port 8000 \\',
        '                > "$LOG_DIR/$JOB_NAME.vllm.log" 2>&1 &',
        '              VLLM_PID=$!',
        '              for i in $(seq 1 180); do',
        '                python3 -c "import urllib.request; urllib.request.urlopen(\'http://127.0.0.1:8000/health\', timeout=3)" 2>/dev/null && break',
        '                kill -0 "$VLLM_PID" 2>/dev/null || { echo "VLLM_FAILED"; tail -40 "$LOG_DIR/$JOB_NAME.vllm.log"; exit 1; }',
        '                [ "$i" -eq 180 ] && { echo "VLLM_TIMEOUT"; exit 1; }',
        '                sleep 10',
        '              done',
        '              echo "vLLM server ready"',
    ])

    def sidecar(self, task):
        return self.render(f"sidecar-{task}.yaml", {"IMAGE_ENV_SERVER": self.cfg["images"]["env_server"]})

    def done(self, step_dir):
        return (step_dir / ".done").exists()

    def mark_done(self, step_dir, info):
        info = dict(info, finished=time.strftime("%Y-%m-%dT%H:%M:%S"), git_sha=self.sha)
        (step_dir / ".done").write_text(json.dumps(info, indent=2, default=str) + "\n")

    # ---------- task shards (released code: same tasks every iteration) ----------
    def webshop_shards(self):
        train = json.loads((self.code / "mcts_utils/webshop/webshop_train_clean.json").read_text())
        test = json.loads((self.code / "mcts_utils/webshop/webshop_test.json").read_text())
        test_ids = {t["item_id"].replace("webshop_", "") for t in test}
        usable = [i for i in range(1000) if str(i) in train and str(i) not in test_ids]  # mcts_collection.py:101
        usable = usable[: self.cfg["search"]["webshop_tasks"]]
        ranges = []
        for chunk in split_evenly(usable, self.cfg["gpus"]):
            ranges.append((chunk[0], chunk[-1] + 1))
        return ranges, len(usable)

    def sciworld_shards(self):
        n = min(self.cfg["search"].get("sciworld_tasks", SCIWORLD_TASK_NUMS), SCIWORLD_TASK_NUMS)
        return [(c[0], c[-1] + 1) for c in split_evenly(list(range(n)), self.cfg["gpus"])]

    # ---------- steps ----------
    def trees_dir(self, iteration, task):
        return self.root / f"iter{iteration}" / f"search-{task}" / "mcts_result" / task / self.cfg["model"]["name"]

    def step_search(self, iteration, task, model_dir):
        step_dir = self.root / f"iter{iteration}" / f"search-{task}"
        if self.done(step_dir):
            log(f"iter{iteration} search {task}: done, skipping")
            return
        trees = self.trees_dir(iteration, task)
        trees.mkdir(parents=True, exist_ok=True)
        seed = (self.cfg["search"].get("seed_trees") or {}).get(f"iter{iteration}", {}).get(task)
        if seed and Path(seed).is_dir():
            copied = 0
            for src in Path(seed).glob("search_results_*.json"):
                if not (trees / src.name).exists():
                    shutil.copy2(src, trees / src.name)
                    copied += 1
            log(f"iter{iteration} search {task}: seeded {copied} trees from {seed}")
        if task == "webshop":
            shards, expected = self.webshop_shards()
            extra = ""
            have = len(list(trees.glob("search_results_*.json")))
            if have >= expected:
                self.mark_done(step_dir, {"trees": have, "model_dir": model_dir, "seeded_from": seed})
                log(f"iter{iteration} search {task}: all {have} trees already present, no jobs needed")
                return
        else:
            shards, expected = self.sciworld_shards(), None
            extra = f"--task_iteration {self.cfg['search']['sciworld_task_iteration']}"
        log(f"iter{iteration} search {task}: {len(shards)} shards {shards}, model {model_dir}")
        jobs = []
        for k, (lo, hi) in enumerate(shards):
            name = self.job_name(iteration, f"search-{task[:2]}", k)
            values = self.base_values(name) | {
                "SIDECAR": self.sidecar(task),
                "COMMON_ENV": self.inference_env(name, task, model_dir, step_dir, workers=hi - lo),
                "START_VLLM": self.START_VLLM,
                "SHARD_MIN": lo,
                "SHARD_MAX": hi,
                "EXTRA_ARGS": extra,
            }
            jobs.append((name, self.render("search.yaml", values)))
        self.run_jobs(jobs)
        count = len(list(trees.glob("search_results_*.json")))
        if expected is not None and count < expected:
            raise RuntimeError(f"search {task}: {count} trees, expected {expected}")
        if count == 0:
            raise RuntimeError(f"search {task}: no trees written")
        self.mark_done(step_dir, {"trees": count, "model_dir": model_dir})
        log(f"iter{iteration} search {task}: {count} trees")

    def step_revise(self, iteration, task, model_dir):
        step_dir = self.root / f"iter{iteration}" / f"revise-{task}"
        if self.done(step_dir):
            log(f"iter{iteration} revise {task}: done, skipping")
            return
        files = sorted(self.trees_dir(iteration, task).glob("search_results_*.json"),
                       key=lambda p: int(p.stem.split("_")[-1]))
        alpha = self.cfg["revise"]["alpha"][iteration - 1]
        beta = self.cfg["revise"]["beta"]
        jobs = []
        for k, chunk in enumerate(split_evenly(files, self.cfg["gpus"])):
            name = self.job_name(iteration, f"revise-{task[:2]}", k)
            input_dir = step_dir / "input" / f"shard{k}"
            if self.job_state(name) in (None, "failed"):  # never touch the inputs of a running job
                shutil.rmtree(input_dir, ignore_errors=True)
                for f in chunk:  # one folder per tree, so one path_collection process per tree
                    (input_dir / f.stem).mkdir(parents=True)
                    (input_dir / f.stem / f.name).symlink_to(f)
            values = self.base_values(name) | {
                "COMMON_ENV": self.inference_env(name, task, model_dir, step_dir / f"shard{k}", workers=len(chunk)),
                "START_VLLM": self.START_VLLM,
                "ALPHA": alpha,
                "BETA": beta,
                "INPUT_DIR": input_dir,
            }
            jobs.append((name, self.render("revise.yaml", values)))
        log(f"iter{iteration} revise {task}: {len(files)} trees over {len(jobs)} shards, alpha={alpha} beta={beta}")
        self.run_jobs(jobs)
        rows = sum(sum(1 for _ in open(p)) for p in step_dir.glob("shard*/out/*/*_centric.jsonl"))
        self.mark_done(step_dir, {"rows": rows, "alpha": alpha, "beta": beta, "model_dir": model_dir})
        log(f"iter{iteration} revise {task}: {rows} rows")

    @staticmethod
    def clean_conversation(messages):
        """Agent-R log -> ms-swift messages. The released code's rewrite() drops the 'loss' flags;
        so do we. A trailing user turn has no response to learn and is dropped, as rewrite() does."""
        out = [{"role": m["role"], "content": m["content"]} for m in messages]
        while out and out[-1]["role"] != "assistant":
            out.pop()
        body = out[1:] if out and out[0]["role"] == "system" else out
        if not body or any(m["role"] != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(body)):
            return None
        return out

    def step_sft_data(self, iteration):
        step_dir = self.root / f"iter{iteration}" / "sft-data"
        data_path = step_dir / "train.jsonl"
        if self.done(step_dir):
            log(f"iter{iteration} sft-data: done, skipping")
            return data_path
        step_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.cfg["sft_data"]
        rng = random.Random(self.cfg["seed"] * 1000 + iteration)
        agent, stats = [], {}
        for task in self.tasks:
            revise, good, seen, invalid = [], [], set(), 0
            for path in sorted((self.root / f"iter{iteration}" / f"revise-{task}").glob("shard*/out/*/*_centric.jsonl")):
                for line in open(path):
                    row = json.loads(line)
                    conv = self.clean_conversation(row["revise_log"])
                    if conv:
                        revise.append(conv)
                    else:
                        invalid += 1
                    key = json.dumps(row["high_log"], sort_keys=True)
                    if key in seen:
                        continue  # one good path is paired with many bad ones; keep it once
                    seen.add(key)
                    high = self.clean_conversation(row["high_log"])
                    if high:
                        good.append(high)
                    else:
                        invalid += 1
            caps = cfg["caps"][task]
            stats[task] = {"revise_available": len(revise), "good_available": len(good), "invalid_dropped": invalid}
            if len(revise) > caps["revise"]:
                revise = rng.sample(revise, caps["revise"])
            if len(good) > caps["good"]:
                good = rng.sample(good, caps["good"])
            stats[task] |= {"revise_used": len(revise), "good_used": len(good)}
            agent += revise + good
        n_general = round(len(agent) * cfg["general_fraction"] / (1 - cfg["general_fraction"]))
        general = []
        roles = {"human": "user", "gpt": "assistant"}
        for conv in json.load(open(cfg["general_path"])):
            turns = conv.get("conversations") or []
            if not turns or any(t.get("from") not in roles or not str(t.get("value", "")).strip() for t in turns):
                continue
            msgs = self.clean_conversation([{"role": roles[t["from"]], "content": t["value"]} for t in turns])
            if msgs:
                general.append(msgs)
        stats["general_available"] = len(general)
        general = rng.sample(general, min(n_general, len(general)))
        stats["general_used"] = len(general)
        rows = [{"messages": m} for m in agent + general]
        rng.shuffle(rows)
        with open(data_path, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats["total"] = len(rows)
        (step_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        if not agent:
            raise RuntimeError(f"sft-data: no agent trajectories for iteration {iteration}: {stats}")
        self.mark_done(step_dir, stats)
        log(f"iter{iteration} sft-data: {stats}")
        return data_path

    def step_sft(self, iteration, model_dir, data_path):
        step_dir = self.root / f"iter{iteration}" / "sft"
        if self.done(step_dir):
            ckpt = json.loads((step_dir / ".done").read_text())["checkpoint"]
            log(f"iter{iteration} sft: done, checkpoint {ckpt}")
            return ckpt
        s, gpus = self.cfg["sft"], self.cfg["gpus"]
        if not s.get("max_length"):
            raise RuntimeError("sft.max_length is not set: revision rows measured 4,297-19,651 tokens "
                               "(median 9,896), so 2048 would cut off every revision. Choose a value, "
                               "commit, and rerun deploy.sh; finished steps are kept.")
        if s["global_batch"] % (gpus * s["per_device_batch"]):
            raise RuntimeError("sft.global_batch must be divisible by gpus x per_device_batch")
        grad_accum = s["global_batch"] // (gpus * s["per_device_batch"])
        name = self.job_name(iteration, "sft")
        out = step_dir / "output"
        values = self.base_values(name) | {
            "GPUS": gpus, "MODEL_DIR": model_dir, "DATA_PATH": data_path, "OUTPUT_DIR": out,
            "EPOCHS": s["epochs"][iteration - 1], "DTYPE": s["dtype"], "MAX_LENGTH": s["max_length"],
            "TRUNCATION": s["truncation_strategy"], "PACKING": str(s["packing"]).lower(),
            "LOSS_SCALE": s["loss_scale"], "PER_DEVICE_BATCH": s["per_device_batch"], "GRAD_ACCUM": grad_accum,
            "LR": s["learning_rate"], "WARMUP": s["warmup_ratio"], "WEIGHT_DECAY": s["weight_decay"],
            "MAX_GRAD_NORM": s["max_grad_norm"], "ADAM_BETA1": s["adam_beta1"], "ADAM_BETA2": s["adam_beta2"],
            "DEEPSPEED": s["deepspeed"], "SEED": self.cfg["seed"],
        }
        log(f"iter{iteration} sft: from {model_dir}, {s['epochs'][iteration - 1]} epoch(s), "
            f"batch {s['per_device_batch']} x {grad_accum} x {gpus} GPUs")
        self.run_jobs([(name, self.render("sft.yaml", values))])
        ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if not ckpts:
            raise RuntimeError(f"sft: no checkpoint under {out}")
        ckpt = ckpts[-1]
        if not (ckpt / "config.json").exists() or not list(ckpt.glob("*.safetensors")):
            raise RuntimeError(f"sft: incomplete checkpoint {ckpt}")
        self.mark_done(step_dir, {"checkpoint": str(ckpt), "base": model_dir, "data": str(data_path)})
        return str(ckpt)

    # Non-weight files the vLLM image reads. ms-swift (transformers 5.16) rewrites them in a format the
    # vLLM image's transformers 4.57 cannot parse (processor_config.json -> KeyError 'qwen3_5'; seen Sep 15).
    # Full fine-tuning changes weights, never shapes, so the base model's files describe the checkpoint exactly.
    SERVING_FILES = ["config.json", "preprocessor_config.json", "video_preprocessor_config.json",
                     "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"]
    SWIFT_ONLY_FILES = ["processor_config.json", "generation_config.json"]  # absent from the base model

    def prepare_checkpoint_for_vllm(self, ckpt):
        """Give a fine-tuned checkpoint the base model's non-weight files; weights are not touched.
        ms-swift's own versions are kept in swift_saved_configs/. Idempotent."""
        ckpt, base = Path(ckpt), Path(self.cfg["model"]["base_dir"])
        if ckpt == base or (ckpt / "SERVING_FILES_FROM_BASE.txt").exists():
            return
        saved = ckpt / "swift_saved_configs"
        saved.mkdir(exist_ok=True)
        for name in self.SERVING_FILES + self.SWIFT_ONLY_FILES:
            if (ckpt / name).exists() and not (saved / name).exists():
                shutil.move(str(ckpt / name), saved / name)
        for name in self.SERVING_FILES:
            if (base / name).exists():
                shutil.copy2(base / name, ckpt / name)
        (ckpt / "SERVING_FILES_FROM_BASE.txt").write_text(
            f"Non-weight files copied from {base} for serving with the vLLM image; ms-swift's originals are in "
            f"swift_saved_configs/. Weights (model-*.safetensors, index) are the fine-tuned ones.\n")
        log(f"prepared {ckpt} for vLLM: base model config/tokenizer files, fine-tuned weights")

    def step_eval(self, iteration, model_dir):
        step_dir = self.root / f"iter{iteration}" / "eval"
        if self.done(step_dir):
            log(f"iter{iteration} eval: done, {json.loads((step_dir / '.done').read_text())}")
            return
        e = self.cfg["eval"]
        model_type = f"agentr-iter{iteration}"
        jobs = []
        for task in self.tasks:
            name = self.job_name(iteration, f"eval-{task[:2]}")
            limit = f'            - {{name: TASK_LIMIT, value: "{e["task_limit"]}"}}' if e["task_limit"] else ""
            values = self.base_values(name) | {
                "SIDECAR": self.sidecar(task),
                "COMMON_ENV": self.inference_env(name, task, model_dir, step_dir / task, workers=1,
                                                 model_type=model_type, temp=e["temp"]),
                "START_VLLM": self.START_VLLM,
                "MAX_STEPS": e["max_steps"],
                "TASK_LIMIT_ENV": limit,
            }
            jobs.append((name, self.render("eval.yaml", values)))
        log(f"iter{iteration} eval: {model_dir}")
        self.run_jobs(jobs)
        summary = {"model_dir": model_dir}
        for task in self.tasks:
            results = list((step_dir / task / "test_result" / task / f"{self.cfg['model']['name']}_{model_type}")
                           .glob("search_results_*.json"))
            scores = [json.load(open(p))["env_score"] for p in results]
            mean = sum(scores) / len(scores) if scores else float("nan")
            summary[task] = {"items": len(scores), "mean_env_score": mean,
                             "reported": mean * 100 if task == "webshop" else mean}
        self.mark_done(step_dir, summary)
        log(f"iter{iteration} eval: {summary}")

    # ---------- main ----------
    def main(self):
        self.logs.mkdir(parents=True, exist_ok=True)
        # A run may be resumed after a fix is committed: every launch is recorded here, and every
        # step's .done names the commit that produced it.
        meta = self.root / "run.json"
        record = json.loads(meta.read_text()) if meta.exists() else {"launches": []}
        record["launches"].append({"git_sha": self.sha, "config": self.cfg,
                                   "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
        meta.write_text(json.dumps(record, indent=2) + "\n")
        model_dir = self.cfg["model"]["base_dir"]
        for it in range(1, self.cfg["iterations"] + 1):
            log(f"===== iteration {it}: model {model_dir}")
            for task in self.tasks:
                self.step_search(it, task, model_dir)
            for task in self.tasks:
                self.step_revise(it, task, model_dir)
            data = self.step_sft_data(it)
            start = model_dir if self.cfg["sft"]["continue_from_previous"] else self.cfg["model"]["base_dir"]
            model_dir = self.step_sft(it, start, data)
            self.prepare_checkpoint_for_vllm(model_dir)
            self.step_eval(it, model_dir)
        log("PIPELINE_DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args()
    try:
        Pipeline(load_config(args.config), args.code_dir, args.sha).main()
    except Exception as exc:  # the Job's log should end with the reason
        log(f"PIPELINE_FAILED: {exc}")
        sys.exit(1)
