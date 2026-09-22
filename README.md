# Agent-R on the ETO / Co-Evolving environments

[Agent-R](https://arxiv.org/abs/2501.11425) (ByteDance Seed), rebuilt to run on the **ETO /
Co-Evolving** versions of WebShop and ScienceWorld instead of AgentGym.

Full instructions: **[`scripts/README.md`](scripts/README.md)**

## Run it

```bash
cp scripts/site.env.example scripts/site.env   # your paths
$EDITOR pipeline/config.yaml                   # model, tasks, hyperparameters

sbatch --export=ALL,CONFIG=pipeline/config.yaml scripts/slurm/agentr.sbatch   # Slurm
scripts/run_agentr.sh pipeline/config.yaml                                    # a node you hold
pipeline/deploy.sh   pipeline/config.yaml                                     # Kubernetes
```

All three read the same config and resume if you rerun them.

One iteration = **search** (MCTS trees) → **revise** (revision trajectories) → **sft-data** →
**sft** (full fine-tune) → **eval**. Three iterations by default.

---

## What we changed

### 1. The environments (the main work)

Upstream targets AgentGym. **The ETO WebShop test set overlaps AgentGym's by 3 items out of 200**,
so numbers from the two cannot go in the same table. We wrote new environment servers and clients:

**`webshop_eto/`** — full 1,181,430-product catalogue (upstream uses a 1,000-product subset),
ETO's train/test id lists, and their few-shot prompt (instruction + "OK" + one worked example;
upstream is zero-shot).

**`sciworld_eto/`** — `simplificationStr="easy"`, reward = `raw_score` **0–1** kept as the episode
maximum (upstream: 0–100 read at the end), tasks addressed as `(task_name, variation_idx)` from
ETO's split files, and **per-task step budgets** (10–120) instead of one global number. Scoring
imports Co-Evolving's own monkey patch rather than reimplementing it.

The 0–1 reward scale is not cosmetic: `path_collection.py` compares raw node values against
`alpha`, so on a 0–100 scale the paper's `alpha` admits every path and the filter does nothing.

Both protocols still work — set `webshop_protocol` / `sciworld_protocol` to `eto` or `agentgym`.

### 2. Bugs fixed in the upstream code

- `path_collection.py` compared floats against the **strings** `ALPHA`/`BETA` → `TypeError`.
- `eval.py` read `test_id/`, which is not in the repo; rebuilt a vLLM engine per task (OOM on the
  second task); and its resume check looked in a different directory than it wrote to.
- SciWorld sharding walked a 23-entry task list, so only the first shard did any work and the
  rest exited successfully having collected **nothing**.
- `alpha: 1.0` selects **zero** trajectories in iteration 3 — the comparison is `value <= ALPHA`
  and WebShop reward is capped at 1.0. We use `0.999`, which selects exactly the reward-1.0 paths.

### 3. Made it runnable end to end

**`pipeline/`** (Kubernetes) and **`scripts/`** (Slurm / bare node) run all five steps from one
config file, with resume: finished steps are skipped, and finished trees and eval items are
skipped within a step.

### 4. Speed

`revise.pair_shards` splits one tree's pairs across processes with the revision sentences
pre-drawn in serial order — **verified to produce identical rows**, and cut that step from 6h13m to
2h20m. `mcts_batch_gen` is the opposite: it looked like a free speedup but measured **worse**
(49.43 vs 51.66), so it is off by default.

Every deviation from the paper, with its reason, is listed in
[`scripts/README.md`](scripts/README.md).

---

## Before you start

**Three Python environments** — their dependencies conflict, do not merge them:

| | Contents |
|---|---|
| policy | vLLM, `fschat`, `tiktoken` |
| env | **Python 3.8** + the environment server (WebShop needs Java 11; ScienceWorld runs a JVM per session) |
| swift | `ms-swift >= 4.5.3`, `transformers >= 5.2` |

**Environment data.** For WebShop, the full catalogue **and the prebuilt Lucene index** — download
the index, don't rebuild it (links in `k8s/webshop-eto-build.yaml`). For ScienceWorld, the split
files and `max_steps.json` under `eval_agent/data/sciworld`. `k8s/setup/` has the exact build jobs.

Budget a few hours for this; it takes longer than the pipeline.

## Using another base model

1. **`inference.enable_thinking` is Qwen-only.** It is passed to the chat template. For Gemma or
   Llama leave it **empty** and it won't be sent.
2. **Keep `bfloat16`** in `sft.dtype` and `inference.vllm_dtype` — Gemma is unstable in fp16.
3. **Check the trainer loads the model before a long run.** Qwen3.5 needs `transformers >= 5.2`;
   hybrid linear-attention models also need `flash-linear-attention` + `causal-conv1d` and
   `sft.packing: false`.
4. Re-tune `sft.max_length` and `sft.deepspeed` for the model size (OOM notes in
   `scripts/README.md`).

## Status

WebShop: done for Qwen3.5-9B, 3 iterations, evaluated at 10-step and 100-step budgets.
ScienceWorld: running. Results and methodology in `pipeline/REPRODUCTION.md`.

The pipeline internals produced those numbers. `scripts/run_agentr.sh` re-hosts the same steps off
Kubernetes and is newer — if something breaks there, it is the launcher, not the method.
