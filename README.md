# Agent-R on the ETO / Co-Evolving environments

[Agent-R](https://arxiv.org/abs/2501.11425) rebuilt to run on the **ETO / Co-Evolving** versions of
WebShop and ScienceWorld instead of AgentGym.

## The scripts

One command runs everything — search → revise → build SFT data → fine-tune → eval, three
iterations, for whichever task the config names:

```bash
scripts/run_agentr.sh pipeline/config.yaml            # WebShop
scripts/run_agentr.sh pipeline/config-sciworld.yaml   # ScienceWorld
```

Put that in your own `sbatch` script. It needs one node with N GPUs (`gpus:` in the config), it
prints progress, and **it resumes** — rerun the same command after a crash or a time limit and it
skips what finished.

Everything you change is in `pipeline/config.yaml` (model, tasks, hyperparameters) and
`scripts/site.env` (where things are installed on your machine —
`cp scripts/site.env.example scripts/site.env`).

On our cluster we run the same pipeline through Kubernetes instead, with
`pipeline/deploy.sh pipeline/config-sciworld.yaml`. Same config, same steps.

## The environments we use

Four, because their dependencies conflict. Two are conda envs on shared disk, two are containers:

| What | How we made it | Used for |
|---|---|---|
| `/data/envs/agentenv-webshop` | conda, **python 3.8** + `openjdk 11` — `k8s/setup/webshop-env-build.yaml` | WebShop server |
| `/data/envs/agentenv-sciworld` | conda, **python 3.8** + `openjdk 11` — `k8s/setup/sciworld-env-build.yaml` | ScienceWorld server (a JVM per session) |
| `vllm/vllm-openai:qwen3_5` | container, plus `/data/envs/policy-site` for `fschat` + `mmengine` (`k8s/setup/policy-deps-build.yaml`) | MCTS, revision, eval |
| `modelscope … swift4.5.3` | container: ms-swift 4.5.3, transformers 5.16 | the fine-tune |

Without containers, build the last two as conda envs: vLLM + `fschat` + `tiktoken`, and
`ms-swift >= 4.5.3` with `transformers >= 5.2`.

**Environment data.** WebShop needs the full 1.18M-product catalogue **and the prebuilt Lucene
index** — download the index, don't rebuild it (`k8s/webshop-eto-build.yaml` has the links).
ScienceWorld needs the split files and `max_steps.json` from a Co-Evolving checkout.

This setup takes longer than the pipeline does. Budget a few hours.

## What we changed from upstream Agent-R

**New environments.** Upstream targets AgentGym, and **the ETO WebShop test set overlaps
AgentGym's by 3 items out of 200** — numbers from the two cannot go in the same table.

- `webshop_eto/` — full 1,181,430-product catalogue (upstream: a 1,000-product subset), ETO's id
  lists, their few-shot prompt (upstream is zero-shot).
- `sciworld_eto/` — `simplificationStr="easy"`, reward `raw_score` **0–1** kept as the episode max
  (upstream: 0–100 at the end), tasks as `(task_name, variation_idx)`, and **per-task** step
  budgets of 10–120 instead of one number.

The 0–1 scale matters for the method: `path_collection.py` compares raw values against `alpha`, so
on a 0–100 scale the paper's `alpha` admits everything and the filter does nothing.

Both still work — `webshop_protocol` / `sciworld_protocol` = `eto` or `agentgym`.

**Bugs fixed in the released code.**

- `path_collection.py` compared floats to the **strings** `ALPHA`/`BETA` → `TypeError`.
- `eval.py` read a `test_id/` directory that isn't in the repo, rebuilt the vLLM engine per task
  (OOM on task 2), and resumed from a different path than it wrote to.
- SciWorld sharding walked a 23-entry list, so only the first shard worked and the others exited
  **successfully having collected nothing**.
- `alpha: 1.0` selects **zero** trajectories (`value <= ALPHA`, WebShop reward capped at 1.0). We
  use `0.999`.

**Speed.** `revise.pair_shards` splits a tree's pairs across processes — verified to produce
identical rows, 6h13m → 2h20m. `mcts_batch_gen` looked like a free speedup but measured **worse**
(49.43 vs 51.66), so it's off.

Deviations from the paper, with reasons, are in [`scripts/README.md`](scripts/README.md).

## Another base model

- **`inference.enable_thinking` is Qwen-only** — it goes to the chat template. Leave it **empty**
  for Gemma or Llama.
- **Keep `bfloat16`** (`sft.dtype`, `inference.vllm_dtype`); Gemma is unstable in fp16.
- Qwen3.5 needs `transformers >= 5.2`; hybrid linear-attention models also need
  `flash-linear-attention` + `causal-conv1d` and `sft.packing: false`.
- Re-tune `sft.max_length` / `sft.deepspeed` for the model size.

## Status

WebShop: done for Qwen3.5-9B, 3 iterations, at 10- and 100-step budgets. ScienceWorld: running.
Results in `pipeline/REPRODUCTION.md`.

`scripts/run_agentr.sh` is newer than the Kubernetes path — the steps inside are the ones that
produced our numbers, so breakage there is the launcher, not the method.
