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
    ("pipeline/config-gemma-webshop.yaml", "Gemma WebShop"),
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
    print("\nsettings and shards, per config:")
    for path, label in CONFIGS:
        if (ROOT / path).exists():
            failures += check_env(path, label)
    print("\nPASS" if not failures else f"\nFAIL ({failures})")
    sys.exit(1 if failures else 0)
