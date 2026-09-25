#!/usr/bin/env python3
"""Print the pipeline config as shell assignments, for scripts/run_agentr.sh.

The Kubernetes controller and the portable runner read the SAME pipeline/config*.yaml, so a run on
Slurm and a run here are configured identically and differ only in how the work is placed on GPUs.
This just flattens the keys the shell needs; nothing is defaulted here that the config can express.

    eval "$(python3 scripts/config_env.py pipeline/config.yaml)"
"""
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from controller import Pipeline, load_config  # noqa: E402


def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")


def main(path):
    c = load_config(path)
    inf, s, sft, ev = c["inference"], c["search"], c["sft"], c["eval"]

    emit("RUN", c["run"])
    emit("TASKS", " ".join(c["tasks"]))
    emit("GPUS", c["gpus"])
    emit("ITERATIONS", c["iterations"])
    emit("SEED", c["seed"])
    emit("RUN_ROOT", c["run_root"])
    emit("MODEL_NAME", c["model"]["name"])
    emit("BASE_MODEL_DIR", c["model"]["base_dir"])
    emit("WEBSHOP_PROTOCOL", c["webshop_protocol"])
    emit("SCIWORLD_PROTOCOL", c.get("sciworld_protocol", "agentgym"))
    emit("INTERCODE_PROTOCOL", c.get("intercode_protocol", "eto"))

    for task, spec in c["environments"].items():
        emit(f"AGENTENV_{task.upper()}", spec["agentenv"])

    emit("TEMP", inf["temp"])
    emit("MAX_TOKEN_LENGTH", inf["max_token_length"])
    emit("MAX_NEW_TOKENS", inf["max_new_tokens"])
    # Emitted only when set: see the note in controller.inference_env - a model whose chat
    # template has no enable_thinking variable must not be sent one.
    if inf.get("enable_thinking") not in (None, ""):
        emit("ENABLE_THINKING", inf["enable_thinking"])
    emit("VLLM_DTYPE", inf["vllm_dtype"])
    emit("ENV_SERVERS", inf.get("env_servers", 1))
    emit("MCTS_BATCH_GEN", "1" if inf.get("mcts_batch_gen") else "0")
    emit("MCTS_PROFILE", "1" if inf.get("mcts_profile") else "0")

    emit("MAX_DEPTH", s["max_depth"])
    emit("ITERA", s["itera"])
    emit("N_GEN", s["n_gen"])
    emit("WEBSHOP_TASKS", s["webshop_tasks"])
    emit("SCIWORLD_TASKS", s.get("sciworld_tasks", 200))
    emit("SCIWORLD_TASK_ITERATION", s["sciworld_task_iteration"])

    # alpha is per iteration; expose the list so the shell indexes it the way the controller does.
    emit("ALPHAS", " ".join(str(a) for a in c["revise"]["alpha"]))
    emit("BETA", c["revise"]["beta"])
    emit("PAIR_SHARDS", c["revise"].get("pair_shards", 1))
    emit("CONCURRENCY", c["revise"].get("concurrency", 1))

    emit("SFT_EPOCHS", " ".join(str(e) for e in sft["epochs"]))
    emit("SFT_CONTINUE", "1" if sft["continue_from_previous"] else "0")
    for key in ("max_length", "truncation_strategy", "learning_rate", "warmup_ratio", "weight_decay",
                "max_grad_norm", "adam_beta1", "adam_beta2", "dtype", "global_batch",
                "per_device_batch", "deepspeed", "loss_scale"):
        emit(f"SFT_{key.upper()}", sft[key])
    emit("SFT_PACKING", str(sft["packing"]).lower())
    emit("SFT_ATTN_IMPL", sft.get("attn_impl", "flash_attn"))

    if sft["global_batch"] % (c["gpus"] * sft["per_device_batch"]):
        sys.exit("config error: sft.global_batch must be divisible by gpus x per_device_batch")
    emit("SFT_GRAD_ACCUM", sft["global_batch"] // (c["gpus"] * sft["per_device_batch"]))

    emit("EVAL_TEMP", ev["temp"])
    emit("EVAL_MAX_STEPS", ev["max_steps"])
    emit("EVAL_TAG", (ev.get("tag") or "").strip())
    emit("EVAL_TASK_LIMIT", ev.get("task_limit", 0))
    emit("EVAL_SCIWORLD_SPLIT", ev.get("sciworld_split", "test"))
    emit("EVAL_STEP_BUDGET_MODE", ev.get("step_budget_mode", "fixed"))

    # Shard boundaries come from the controller's OWN methods, not a second implementation: under
    # webshop_protocol: agentgym they are ranges over usable ids (non-contiguous, read from
    # webshop_train_clean.json), not positions, and only the controller knows that.
    repo = Path(__file__).resolve().parent.parent
    pipeline = Pipeline(c, repo, "local")
    for task in c["tasks"]:
        if task == "webshop":
            shards, expected = pipeline.webshop_shards()
        elif task == "intercode_sql":
            # Without this branch the else fell through to sciworld_shards(), so an InterCode run
            # under Slurm would have collected 200 tasks instead of 300 - a different experiment
            # from the Kubernetes one, silently.
            shards, expected = pipeline.intercode_sql_shards()
        else:
            shards = pipeline.sciworld_shards()
            expected = s.get("sciworld_tasks", 200) if c.get("sciworld_protocol") == "eto" else 0
        emit(f"SHARDS_{task.upper()}", " ".join(f"{lo}:{hi}" for lo, hi in shards))
        emit(f"EXPECTED_{task.upper()}", expected)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__))
