# Running the Agent-R pipeline

This is a reproduction of Agent-R ([arXiv 2501.11425](https://arxiv.org/abs/2501.11425)) on the
WebShop and ScienceWorld environments, set up so it can be run with a different base model or on a
different scheduler than the one it was developed on.

There is one entry point per backend. Both read the **same** config file, so a run on Slurm and a
run on Kubernetes differ only in how work is placed on GPUs — not in anything that affects results.

| Backend | Command |
|---|---|
| Slurm / bare multi-GPU node | `scripts/run_agentr.sh pipeline/config.yaml` |
| Kubernetes | `pipeline/deploy.sh pipeline/config.yaml` |

Both **resume**. Rerun the same command after a crash or a time limit: finished steps are skipped,
and within a step, trees and eval items that already have output files are skipped. You do not need
to clean anything up first.

---

## What the pipeline does

Agent-R trains a model to *recover from its own mistakes*. Each iteration:

1. **search** — Run MCTS from the base model on the training tasks, producing one tree per task.
   Nodes are scored by the environment reward.
   (`mcts_collection.py`, one GPU per shard)
2. **revise** — In each tree, pair a good path with a bad one, find where they diverge, and stitch
   together a *revision trajectory*: the bad prefix, a transition sentence
   ("I need to reconsider…"), then the good continuation.
   (`path_collection.py`, one GPU per shard)
3. **sft-data** — Turn those into a training set: revision trajectories + the good trajectories on
   their own + general instruction data mixed in 8:2.
   (`controller.py --step sft-data`, CPU only)
4. **sft** — Full fine-tune of the model on that set.
   (`swift sft`, all GPUs)
5. **eval** — Score the resulting checkpoint on the held-out test split.
   (`eval.py`, one GPU per shard)

Then repeat, with the new checkpoint as the base. Three iterations by default.

Outputs land under `<run_root>/<run>/iter<N>/<step>/`, and per-step logs under
`<run_root>/<run>/logs/`.

---

## Prerequisites

### Three Python environments

Their dependencies genuinely conflict; do not try to merge them.

| Name | Contents | Used by |
|---|---|---|
| policy | vLLM, `fschat`, `tiktoken`, `mmengine` | `mcts_collection.py`, `path_collection.py`, `eval.py` |
| env | Python **3.8**, the environment server | WebShop / ScienceWorld servers |
| swift | `ms-swift >= 4.5.3`, `transformers >= 5.2` | `swift sft` |

Two constraints that cost us time:

- **WebShop's server needs Python 3.8** and pins `torch 1.11`, `spaCy 3.3`, `pyserini 0.17`. It also
  needs Java 11 for the Lucene index. ScienceWorld needs Java too (it runs a JVM per session).
- **Qwen3.5 does not exist in `transformers` 4.x.** Loading it raises `KeyError: 'qwen3_5'`. It
  needs `transformers >= 5.2`. If your model is a Qwen3.5 / Qwen3-Next-style hybrid, also install
  `flash-linear-attention >= 0.4.2` and `causal-conv1d`, or the Gated DeltaNet layers silently fall
  back to slow, memory-hungry PyTorch ops. Set `packing: false` — linear attention does not support
  variable-length packing.

### Environments and data

- **WebShop** and/or **ScienceWorld**, in whichever protocol you intend to report (see below).
- The base model, in a local HuggingFace cache.
- General instruction data for the 8:2 mixture — we used
  `ShareGPT_V3_unfiltered_cleaned_split.json`. Point `sft_data.general_path` at it.

### Which environment protocol

This matters more than it sounds, and is the single easiest way to produce a number that cannot be
compared with anything.

| | `eto` | `agentgym` |
|---|---|---|
| WebShop catalogue | full 1,181,430 products | 1,000-product subset |
| WebShop test set | ETO's 200 ids | AgentGym's 200 ids |
| ScienceWorld | `simplificationStr="easy"`, reward `raw_score` 0–1 kept as episode max | no simplification, reward 0–100 |
| ScienceWorld tasks | `(task_name, variation_idx)` from split files, per-task step budgets (10–120) | flat index over all 4,639 pairs |

**The two WebShop test sets overlap by 3 items out of 200.** A score collected under one cannot be
compared with a score collected under the other. Pick the one your baselines used and set
`webshop_protocol` / `sciworld_protocol` accordingly.

The reward scale also silently breaks the method: `path_collection.py` compares raw values against
`alpha`, so on ScienceWorld's 0–100 scale, `alpha` of 0.5/0.7/1.0 admits everything and the
good-trajectory filter does nothing at all.

---

## Running it

### 1. Configure

```bash
cp scripts/site.env.example scripts/site.env
$EDITOR scripts/site.env          # where things are installed — never affects results
$EDITOR pipeline/config.yaml      # the method — everything that does
```

Every key in `config.yaml` carries a comment naming its source (paper section, maintainer issue, or
a measurement). The ones you will most likely change:

```yaml
run: my-run-id           # names the output directory and the job names
tasks: [webshop]         # or [sciworld], or both for a multi-task model
gpus: 7
model:
  name: Qwen3.5-9B       # a label, used in output paths
  base_dir: /path/to/model
```

### 2. Launch

```bash
# Slurm
sbatch --export=ALL,CONFIG=pipeline/config.yaml scripts/slurm/agentr.sbatch

# or directly on a node you already hold
scripts/run_agentr.sh pipeline/config.yaml
```

### 3. Read the result

Each eval writes a `.done` containing the summary. Or ask for it directly:

```bash
python3 pipeline/controller.py --config pipeline/config.yaml --code-dir . --sha local \
  --step score --iteration 3
```

```json
{"iteration": 3, "webshop": {"items": 200, "mean_env_score": 0.5255, "reported": 52.55}}
```

`reported` is the number for a table: WebShop's 0–1 reward as a percentage.

---

## Using a different model

1. Point `model.base_dir` at it and give `model.name` a label.
2. Check `swift sft` supports its architecture, and that `transformers` in the swift env is new
   enough. Confirm before a long run: `swift sft --model <path> --max_steps 1 ...` on a tiny dataset.
3. **Check the chat template renders the prompt you expect.** If the model has a thinking mode,
   collection and training must agree about it. We set `enable_thinking: 0`, because Agent-R's own
   base model has no thinking mode — with it on, the model emitted a long reasoning block and hit
   the token cap before ever writing `Action:`.
4. Re-tune `sft.max_length` and `sft.deepspeed` for the model's size. See the OOM notes below.

## Using a different dataset

Adding a third environment means providing, for the new task:

- an environment client with `reset`, `step`, `observe`, `conversation_start`
- a branch in `initialize_environment` in `mcts_collection.py` and `eval.py`
- an `mcts_utils/<task>/` module defining the MCTS node/prompt handling
- caps in `sft_data.caps`

The two existing clients (`webshop_eto/client.py`, `sciworld_eto/client.py`) are the templates to
copy; each is ~110 lines and documents what Agent-R's code touches.

---

## Deviations from the paper, and why

Report these if you publish the numbers. Each is a deliberate decision, not an accident.

| Setting | Paper | Here | Reason |
|---|---|---|---|
| `max_new_tokens` | 500 | 4096 | 500 cut off >10% of replies before `Action:`. Measured over 1,875 real replies: median 34 tokens, p99 692, max 2,332 — so 4096 truncates ~0% and still stops a runaway (one degenerate loop generated 20,000+ tokens and hung a job for 2.5 h). |
| `sft.max_length` | 8,196 | 12,288 | 16,384 OOMed on 7×H100 during backward. Revision rows measured median 2,879 / p90 9,991 tokens. |
| `truncation_strategy` | — | `delete` | Cutting a row's tail removes the recovery and the purchase — the exact defect being trained against. Dropping the ~5% of over-long rows keeps every surviving example complete. |
| `alpha` iteration 3 | 1.0 | 0.999 | `path_collection.py` skips a path when `value <= ALPHA`, and WebShop reward is capped at 1.0, so `alpha=1.0` excludes even perfect trajectories and produced **zero** rows. Paper Eq. 4 requires `alpha < r <= 1`, which is empty at 1.0. 0.999 selects exactly the reward-1.0 paths, the stated intent. |
| loss mask | Eq. 6 masks the bad prefix | no mask | The released code does not implement it: `llm_server.rewrite()` drops the per-turn `loss` flags, so the authors' own pipeline trains on every assistant turn. We follow the code. |
| GPUs | 8×A100 | 7×H100 | Availability. `global_batch` is kept at 112. |

Two settings are **speed only** and provably do not change results:

- `revise.pair_shards` — splits one tree's pairs across processes, with the revision sentences
  pre-drawn in serial order. Verified to produce byte-identical row sets (107 vs 107 rows, identical
  sentence multiset). Cut the revise step from 6 h 13 m to 2 h 20 m. `1` = released behaviour.
- `inference.env_servers` / worker counts — request placement only.

One setting is **measured harmful**, left in the code but off:

- `inference.mcts_batch_gen` — generating the 4 MCTS candidates as one `n=4` request shares the
  prefill, so the candidates correlate and the tree explores less. Saved 9% of search time and cost
  2.2 points (49.43 vs 51.66). Leave it `false`.

---

## Things that will bite you

- **A step that succeeds while doing nothing.** We lost a run to shards that exited 0 having
  collected zero trees. Both backends now assert the expected output count after `search`. If you
  add a task, add its expected count too.
- **OOM during SFT backward, not forward.** It appears at the first *long* row, which may be step
  20. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` recovered 12.7 GiB lost to fragmentation.
  ZeRO-3 fixes OOM but ran 3.4× slower than ZeRO-2 here (169 s/step vs 49 s/step); prefer ZeRO-2 with
  a lower `max_length`.
- **NCCL `NVLS` failure on a partial-node allocation.** If you get fewer GPUs than the node has,
  NVLink SHARP fails with CUDA error 401. `NCCL_NVLS_ENABLE=0` is set for you; it changes the
  collective algorithm, not the math.
- **WebShop's prices are randomly generated at server start** with an unseeded RNG, so two servers
  invent different "under $X" goal caps. Both backends seed it before construction. If you write
  your own launcher, do the same or your environments will silently disagree.
- **Evaluation step budget.** ETO's WebShop config uses 10 steps; the Agent-R paper evaluates at 100.
  This is not a detail — Agent-R's whole point is recovering from a mistake, which takes extra steps.
  Our iteration-2 checkpoint scored 45.42 at 10 steps and 55.24 at 100. Report which you used.
- **Run-to-run variance is real.** Differences of ±3 points between iterations were not reproducible
  from n=1. If an iteration-to-iteration trend matters to your claim, repeat a run.

---

## Measured timings

7×H100, Qwen3.5-9B, WebShop, 300 tasks per iteration:

| Step | Time |
|---|---|
| search | ~2 h 20 m |
| revise | ~2 h 20 m (was 6 h 13 m before `pair_shards`) |
| sft-data | < 5 min |
| sft | ~3–4 h (3 epochs, iteration 1) |
| eval, 200 items | ~40 min at 10 steps, ~2 h at 100 |

A full three-iteration run is roughly **40 hours**. ScienceWorld's search is slower per task, because
its step budgets are per-task (10–120) rather than WebShop's flat 10.
