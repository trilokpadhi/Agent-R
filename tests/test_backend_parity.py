"""The Slurm runner and the Kubernetes controller must run the SAME experiment.

    python3 tests/test_backend_parity.py

Kubernetes produced every number we report. scripts/run_agentr.sh is a second implementation of the
same five steps for machines without Kubernetes, and a second implementation is a second chance to
drift. Both read the same pipeline/config*.yaml, so this compares what each one would actually do
with a given config:

  * the flags passed to mcts_collection.py, path_collection.py, eval.py and swift sft
  * the VALUES of every setting that affects the result, derived from the same config
  * the shard boundaries, so both cover the same tasks
  * the effective batch size, which must survive a change in GPU count

It does not check placement (how many processes per GPU, which node) - that is allowed to differ
and does not change results.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import controller as C  # noqa: E402

CONFIGS = [
    ("pipeline/config.yaml", "WebShop, 10-step eval"),
    ("pipeline/config-eval100.yaml", "WebShop, 100-step eval"),
    ("pipeline/config-sciworld.yaml", "SciWorld, 100 rounds"),
    ("pipeline/config-intercode.yaml", "InterCode-SQL, 10-step eval"),
    ("pipeline/config-intercode-eval100.yaml", "InterCode-SQL, 100-step eval"),
    ("pipeline/config-gemma-webshop.yaml", "Gemma WebShop, 10-step"),
    ("pipeline/config-gemma-webshop-eval100.yaml", "Gemma WebShop, 100-step"),
    ("pipeline/config-gemma-sciworld.yaml", "Gemma SciWorld Unseen, 100 rounds"),
    ("pipeline/config-gemma-sciworld-seen.yaml", "Gemma SciWorld Seen, 100 rounds"),
]

# Settings that change the experiment. Placement (WORKERS, ENV_SERVERS, CODE, WORKDIR, JOB_NAME,
# LOG_DIR, VLLM_API_BASE) is deliberately excluded.
METHOD_ENV = [
    "TASK", "MODEL_NAME", "MODEL_TYPE", "TEMP", "MAX_DEPTH", "ITERA", "N_GEN",
    "MAX_TOKEN_LENGTH", "MAX_NEW_TOKENS", "ENABLE_THINKING", "STOP_TOKENS", "VLLM_DTYPE",
    "WEBSHOP_PROTOCOL", "SCIWORLD_PROTOCOL", "INTERCODE_PROTOCOL", "MCTS_BATCH_GEN",
    "MCTS_PROFILE", "STEP_BUDGET_MODE", "SCIWORLD_SPLIT",
]


def command_block(text, needle):
    """The whole command starting at `needle`: continuation lines end with a backslash."""
    index = text.index(needle)
    start = text.rfind("\n", 0, index) + 1
    lines = []
    while True:
        end = text.index("\n", start)
        line = text[start:end]
        lines.append(line.strip())
        if not line.rstrip().endswith("\\"):
            break
        start = end + 1
    return " ".join(lines).replace("\\", " ")


def flags(command):
    return sorted(set(re.findall(r"--[a-zA-Z0-9_]+", command)))


def shell_vars(path):
    """Variables the shell script exports, including several per `export` line."""
    exported = set()
    for line in Path(path).read_text().splitlines():
        if "export " not in line:
            continue
        for token in line.split("export ", 1)[1].split():
            name = token.split("=", 1)[0]
            if re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
                exported.add(name)
    return exported


def sft_values(config_path):
    """Flag -> value for `swift sft`, on BOTH backends, with values actually resolved.

    Comparing flag NAMES is not enough: run_agentr.sh once hardcoded `--attn_impl flash_attn`
    while the controller read it from the config, so the same config trained differently on the
    two machines and the name-only check passed. Gemma needs `eager` and DeltaAI has no
    flash-attn installed at all, so that difference is fatal rather than cosmetic.
    """
    cfg = C.load_config(ROOT / config_path)
    sft, gpus = cfg["sft"], cfg["gpus"]
    accum = sft["global_batch"] // (gpus * sft["per_device_batch"])

    # Kubernetes: render the template with the values the controller would pass.
    values = {
        "GPUS": gpus, "MODEL_DIR": "/model", "DATA_PATH": "/data.jsonl", "OUTPUT_DIR": "/out",
        "EPOCHS": sft["epochs"][0], "DTYPE": sft["dtype"], "MAX_LENGTH": sft["max_length"],
        "TRUNCATION": sft["truncation_strategy"], "PACKING": str(sft["packing"]).lower(),
        "LOSS_SCALE": sft["loss_scale"], "PER_DEVICE_BATCH": sft["per_device_batch"],
        "GRAD_ACCUM": accum, "LR": sft["learning_rate"], "WARMUP": sft["warmup_ratio"],
        "WEIGHT_DECAY": sft["weight_decay"], "MAX_GRAD_NORM": sft["max_grad_norm"],
        "ADAM_BETA1": sft["adam_beta1"], "ADAM_BETA2": sft["adam_beta2"],
        "DEEPSPEED": sft["deepspeed"], "SEED": cfg["seed"],
        "ATTN_IMPL": sft.get("attn_impl", "flash_attn"),
    }
    rendered = (ROOT / "pipeline/templates/sft.yaml").read_text()
    for k, v in values.items():
        rendered = rendered.replace("{{" + k + "}}", str(v))
    k_cmd = command_block(rendered, "swift sft")

    # Slurm: take its command and substitute the shell variables config_env.py provides.
    env = subprocess.run([sys.executable, str(ROOT / "scripts/config_env.py"), str(ROOT / config_path)],
                         capture_output=True, text=True).stdout
    shell = dict(re.findall(r"^([A-Z_][A-Z0-9_]*)='?([^'\n]*)'?$", env, re.M))
    shell["epochs"] = str(sft["epochs"][0])          # a loop variable, not from config_env
    s_cmd = command_block((ROOT / "scripts/run_agentr.sh").read_text(), '"$AGENTR_SWIFT" sft')
    for name, val in sorted(shell.items(), key=lambda kv: -len(kv[0])):
        s_cmd = s_cmd.replace(f'"${{{name}}}"', val).replace(f"${{{name}}}", val)
        s_cmd = s_cmd.replace(f'"${name}"', val).replace(f"${name}", val)

    def pairs(cmd):
        return dict(re.findall(r"--([a-zA-Z0-9_]+)\s+(\S+)", cmd))
    return pairs(k_cmd), pairs(s_cmd)


def check_sft_values(config_path, label):
    kv, sv = sft_values(config_path)
    # Only compare flags whose value comes from the config; paths and dirs legitimately differ.
    skip = {"model", "dataset", "output_dir"}
    bad = [(f, kv[f], sv.get(f)) for f in kv
           if f not in skip and not sv.get(f, "").startswith("/") and kv[f] != sv.get(f)]
    if bad:
        print(f"  FAIL {label}")
        for f, a, b in bad:
            print(f"         --{f}: kubernetes={a!r} slurm={b!r}")
        return 1
    print(f"  ok   {label:30s} {len(kv)} sft values match (attn_impl={kv.get('attn_impl')})")
    return 0


def check_commands():
    k8s = {p.name: p.read_text() for p in (ROOT / "pipeline/templates").glob("*.yaml")}
    slurm = (ROOT / "scripts/run_agentr.sh").read_text()
    pairs = [
        ("mcts_collection.py", k8s["search.yaml"], '"$CODE/mcts_collection.py"', '"$REPO/mcts_collection.py"'),
        ("path_collection.py", k8s["revise.yaml"], '"$CODE/path_collection.py"', '"$REPO/path_collection.py"'),
        ("eval.py", k8s["eval.yaml"], '"$CODE/eval.py"', '"$REPO/eval.py"'),
        ("swift sft", k8s["sft.yaml"], "swift sft", '"$AGENTR_SWIFT" sft'),
    ]
    failures = 0
    for name, ktext, kneedle, sneedle in pairs:
        a, b = flags(command_block(ktext, kneedle)), flags(command_block(slurm, sneedle))
        if a != b:
            failures += 1
            print(f"  FAIL {name}: only k8s {sorted(set(a)-set(b))} | only slurm {sorted(set(b)-set(a))}")
        else:
            print(f"  ok   {name:20s} {len(a)} flags identical")
    return failures


def check_env(config_path, label):
    """Both backends must carry the same values for every setting that affects results."""
    cfg = C.load_config(ROOT / config_path)
    pipeline = C.Pipeline(cfg, "/code", "sha")
    task = cfg["tasks"][0]
    e = cfg["eval"]

    rendered = pipeline.inference_env(
        "job", task, "/model", "/work", workers=1, model_type="agentr-iter1", temp=e["temp"],
        sciworld_split=e.get("sciworld_split", "test"),
        step_budget_mode=e.get("step_budget_mode", "fixed"))
    k_env = dict(re.findall(r'\{name: ([A-Z_]+), value: "([^"]*)"\}', rendered))

    out = subprocess.run([sys.executable, str(ROOT / "scripts/config_env.py"), str(ROOT / config_path)],
                         capture_output=True, text=True)
    if out.returncode:
        print(f"  FAIL {label}: config_env.py failed: {out.stderr.strip()[:120]}")
        return 1
    s_env = dict(re.findall(r"^([A-Z_]+)='?([^'\n]*)'?$", out.stdout, re.M))

    exported = shell_vars(ROOT / "scripts/run_agentr.sh")
    problems = []
    for key in METHOD_ENV:
        if key not in k_env:
            continue
        if key not in exported and f"export {key}" not in (ROOT / "scripts/run_agentr.sh").read_text():
            problems.append(f"{key} never exported by run_agentr.sh")

    # Values the shell gets from config_env must equal what the controller renders.
    for key, s_name in (("MAX_DEPTH", "MAX_DEPTH"), ("ITERA", "ITERA"), ("N_GEN", "N_GEN"),
                        ("MAX_TOKEN_LENGTH", "MAX_TOKEN_LENGTH"), ("MAX_NEW_TOKENS", "MAX_NEW_TOKENS"),
                        ("VLLM_DTYPE", "VLLM_DTYPE"), ("MODEL_NAME", "MODEL_NAME")):
        if key in k_env and s_name in s_env and k_env[key] != s_env[s_name]:
            problems.append(f"{key}: k8s={k_env[key]!r} slurm={s_env[s_name]!r}")

    # The eval budget and tag, which is what "for 100 steps" depends on.
    if s_env.get("EVAL_MAX_STEPS") != str(e["max_steps"]):
        problems.append(f"eval max_steps: config={e['max_steps']} slurm={s_env.get('EVAL_MAX_STEPS')}")
    if s_env.get("EVAL_TAG", "") != (e.get("tag") or "").strip():
        problems.append(f"eval tag: config={e.get('tag')!r} slurm={s_env.get('EVAL_TAG')!r}")
    if k_env.get("STEP_BUDGET_MODE") != s_env.get("EVAL_STEP_BUDGET_MODE"):
        problems.append(f"step_budget_mode: k8s={k_env.get('STEP_BUDGET_MODE')} "
                        f"slurm={s_env.get('EVAL_STEP_BUDGET_MODE')}")

    # Shard boundaries: both must cover the same tasks.
    k_shards = (pipeline.webshop_shards()[0] if task == "webshop" else
                pipeline.intercode_sql_shards()[0] if task == "intercode_sql" else
                pipeline.sciworld_shards())
    k_str = " ".join(f"{lo}:{hi}" for lo, hi in k_shards)
    if s_env.get(f"SHARDS_{task.upper()}") != k_str:
        problems.append(f"shards differ:\n      k8s  ={k_str}\n      slurm={s_env.get(f'SHARDS_{task.upper()}')}")

    # Effective batch must survive a different GPU count.
    sft = cfg["sft"]
    if sft["global_batch"] % (cfg["gpus"] * sft["per_device_batch"]):
        problems.append("global_batch not divisible by gpus x per_device_batch")
    else:
        accum = sft["global_batch"] // (cfg["gpus"] * sft["per_device_batch"])
        if s_env.get("SFT_GRAD_ACCUM") != str(accum):
            problems.append(f"grad_accum: controller={accum} slurm={s_env.get('SFT_GRAD_ACCUM')}")

    if problems:
        print(f"  FAIL {label}")
        for p in problems:
            print(f"         {p}")
        return 1
    print(f"  ok   {label:30s} eval={e['max_steps']} steps, tag={e.get('tag') or 'none'!r}, "
          f"accum={s_env.get('SFT_GRAD_ACCUM')}")
    return 0


if __name__ == "__main__":
    print("command flags (kubernetes vs slurm):")
    failures = check_commands()
    print("\nswift sft VALUES (not just flag names):")
    for path, label in CONFIGS:
        if (ROOT / path).exists():
            failures += check_sft_values(path, label)

    print("\nsettings and shards, per config:")
    for path, label in CONFIGS:
        if (ROOT / path).exists():
            failures += check_env(path, label)
    print("\nPASS" if not failures else f"\nFAIL ({failures})")
    sys.exit(1 if failures else 0)
