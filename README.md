# Agent-R on the ETO / Co-Evolving environments

[Agent-R](https://arxiv.org/abs/2501.11425) rebuilt to run on the **ETO / Co-Evolving** versions of
WebShop and ScienceWorld instead of AgentGym.

One run = 3 iterations of: **search** (MCTS trees) → **revise** (revision trajectories) →
**sft-data** → **sft** (full fine-tune) → **eval**. Both launchers below run all of it from one
config file and **resume** if you rerun them after a crash or a time limit.

---

## Run on Kubernetes (what we actually use)

```bash
pipeline/deploy.sh pipeline/config.yaml            # WebShop
pipeline/deploy.sh pipeline/config-sciworld.yaml   # ScienceWorld
```

That is the whole thing. It uploads the current commit to the PVC and starts **one** controller
Job; the controller launches every other job itself (search shards → revise → SFT → eval) and
waits between them. Rerun the same command to resume.

Watch it and find the output:

```bash
kubectl logs -n ii400r87 -f job/agr-q35sci-controller   # progress
ls /data/runs/q35sci/iter1                              # outputs, per step
cat /data/runs/q35sci/iter1/eval/.done                  # the score
```

Requires: `kubectl` access to the namespace, the `tpadhi1` PVC mounted at `/data`, and
`access-pod` running. `deploy.sh` refuses to launch with uncommitted code, so a run always matches
a commit.

---

## Run on Slurm (sample — NOT TESTED)

Same pipeline without Kubernetes, one node with N GPUs. The steps inside are the ones that
produced our numbers, but this launcher has not had a full end-to-end run yet — expect to debug
the launcher, not the method.

```bash
cp scripts/site.env.example scripts/site.env   # paths to your envs + model
$EDITOR pipeline/config.yaml                   # set gpus: to match your allocation
```

```bash
#!/bin/bash
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=48:00:00

scripts/run_agentr.sh pipeline/config.yaml
```

`scripts/slurm/agentr.sbatch` is that same script with a check that `gpus:` matches the
allocation. Details in [`scripts/README.md`](scripts/README.md).

---

## Environments

Four, because their dependencies conflict — WebShop pins Python 3.8 / torch 1.11, Qwen3.5 needs
transformers ≥ 5.2.

| Environment | What | Runs |
|---|---|---|
| `/data/envs/agentenv-webshop` | conda: **Python 3.8** + Java 11 | WebShop server |
| `/data/envs/agentenv-sciworld` | conda: **Python 3.8** + Java 11 | ScienceWorld server (a JVM per session) |
| `vllm/vllm-openai:qwen3_5` | container + `/data/envs/policy-site` (`fschat`, `mmengine`) | MCTS, revision, eval |
| `modelscope … swift4.5.3` | container: ms-swift 4.5.3, transformers 5.16 | the fine-tune |

Built once by `k8s/setup/webshop-env-build.yaml`, `k8s/setup/sciworld-env-build.yaml`,
`k8s/setup/policy-deps-build.yaml`. Without containers, make the last two conda envs: vLLM +
`fschat` + `tiktoken`, and `ms-swift >= 4.5.3` with `transformers >= 5.2`.

**Data.** WebShop needs the full 1.18M-product catalogue **and the prebuilt Lucene index** —
download the index, don't rebuild it (`k8s/webshop-eto-build.yaml` has the links). ScienceWorld
needs the split files and `max_steps.json` from a Co-Evolving checkout. This takes longer than the
pipeline; budget a few hours.

---

## What we changed from upstream Agent-R

**New environments.** Upstream targets AgentGym, and **the ETO WebShop test set overlaps
AgentGym's by 3 items out of 200** — numbers from the two cannot share a table.

- `webshop_eto/` — full 1,181,430-product catalogue (upstream: 1,000), ETO's id lists, their
  few-shot prompt (upstream is zero-shot).
- `sciworld_eto/` — `simplificationStr="easy"`, reward `raw_score` **0–1** kept as the episode max
  (upstream: 0–100 at the end), tasks as `(task_name, variation_idx)`, **per-task** step budgets
  10–120 instead of one number.

The 0–1 scale matters for the method, not just comparability: `path_collection.py` compares raw
values against `alpha`, so on a 0–100 scale the paper's `alpha` admits everything and the
good-trajectory filter does nothing.

Set `webshop_protocol` / `sciworld_protocol` to `eto` or `agentgym`; both work.

**Bugs fixed in the released code**

- `path_collection.py` compared floats to the **strings** `ALPHA`/`BETA` → `TypeError`.
- `eval.py` read a `test_id/` directory not in the repo, rebuilt the vLLM engine per task (OOM on
  task 2), and resumed from a different path than it wrote to.
- SciWorld sharding walked a 23-entry list, so only the first shard did work and the rest exited
  **successfully having collected nothing**.
- `alpha: 1.0` selects **zero** trajectories (`value <= ALPHA`, WebShop reward capped at 1.0). We
  use `0.999`.

**Speed.** `revise.pair_shards` splits a tree's pairs across processes — verified identical rows,
6h13m → 2h20m. `mcts_batch_gen` looked free but measured **worse** (49.43 vs 51.66), so it's off.

---

## Another base model

- **`inference.enable_thinking` is Qwen-only** (it goes to the chat template) — leave it **empty**
  for Gemma or Llama.
- **Keep `bfloat16`** in `sft.dtype` and `inference.vllm_dtype`; Gemma is unstable in fp16.
- Qwen3.5 needs `transformers >= 5.2`; hybrid linear-attention models also need
  `flash-linear-attention` + `causal-conv1d` and `sft.packing: false`.
- Re-tune `sft.max_length` / `sft.deepspeed` for the model size.

Deviations from the paper, with reasons, and the failure modes worth knowing:
[`scripts/README.md`](scripts/README.md).

## Status

WebShop: done for Qwen3.5-9B, 3 iterations, at 10- and 100-step budgets. ScienceWorld: running.
Results in `pipeline/REPRODUCTION.md`.
