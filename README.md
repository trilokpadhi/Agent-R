# Agent-R on the ETO / Co-Evolving environments

A reproduction of [Agent-R](https://arxiv.org/abs/2501.11425) (ByteDance Seed) on **WebShop** and
**ScienceWorld**, rebuilt so its numbers are comparable with the ETO / Co-Evolving line of work.

**Start here: [`scripts/README.md`](scripts/README.md)** — how to run the pipeline, what each step
does, every deviation from the paper, and the failures worth knowing about before you hit them.

```bash
cp scripts/site.env.example scripts/site.env   # where things live on your machine
$EDITOR pipeline/config.yaml                   # the method: model, tasks, hyperparameters

sbatch --export=ALL,CONFIG=pipeline/config.yaml scripts/slurm/agentr.sbatch   # Slurm
scripts/run_agentr.sh pipeline/config.yaml                                    # or a node you hold
pipeline/deploy.sh   pipeline/config.yaml                                     # or Kubernetes
```

All three read the same config and resume on rerun.

---

## Why this is not upstream Agent-R

Upstream runs against **AgentGym**. The numbers we need to compare with were produced on the
**ETO / Co-Evolving** environments, and the two are not interchangeable:

| | ETO / Co-Evolving | AgentGym (upstream Agent-R) |
|---|---|---|
| WebShop catalogue | full 1,181,430 products | 1,000-product subset |
| WebShop test ids | ETO's `test_indices.json` | AgentGym's own 200 |
| WebShop prompt | instruction + "OK" + 1 worked example | zero-shot, one hardcoded turn |
| ScienceWorld | `simplificationStr="easy"`, reward `raw_score` 0–1 kept as episode max | no simplification, reward 0–100 |
| ScienceWorld tasks | `(task_name, variation_idx)` split files, per-task budgets 10–120 | flat index over 4,639 pairs |

**The two WebShop test sets overlap by 3 items out of 200.** A score from one cannot be placed in a
table beside a score from the other. That is the reason this fork exists.

The ScienceWorld reward scale matters for the *method*, not just comparability:
`path_collection.py` compares raw node values against `alpha`, so on a 0–100 scale the paper's
`alpha` of 0.5/0.7/1.0 admits every path and the good-trajectory filter silently does nothing.

### What was added

| Path | What |
|---|---|
| `webshop_eto/` | WebShop server + client on ETO's protocol (full catalogue, their splits, their `step()`) |
| `sciworld_eto/` | ScienceWorld server + client on ETO's protocol; imports Co-Evolving's own scoring monkey patch rather than reimplementing it |
| `pipeline/` | One-command orchestration of the 5 steps, with resume, for Kubernetes |
| `scripts/` | The same pipeline without Kubernetes, for Slurm or a bare node |

Upstream's own files (`mcts_collection.py`, `path_collection.py`, `eval.py`, `mcts_utils/`) are
kept, with the protocol handled by a branch rather than a rewrite, so the released code path still
runs under `*_protocol: agentgym`.

---

## Environment setup

This is the part that takes the longest; budget a few hours the first time.

You need **three Python environments** — their dependencies genuinely conflict, so do not merge them:

- **policy**: vLLM + `fschat` + `tiktoken` (+ `mmengine`). Runs MCTS, revision, and eval.
- **env**: **Python 3.8** + the environment server. WebShop pins `torch 1.11` / `spaCy 3.3` /
  `pyserini 0.17` and needs **Java 11**; ScienceWorld runs a **JVM per session**.
- **swift**: `ms-swift >= 4.5.3` with `transformers >= 5.2`.

And the environment data:

- **WebShop (ETO)** — a Co-Evolving checkout plus the full catalogue and the **prebuilt Lucene
  index**, both from the ETO Google Drive links used in `k8s/webshop-eto-build.yaml`. Download the
  index; do not rebuild it.
- **ScienceWorld (ETO)** — a Co-Evolving checkout; the split files and `max_steps.json` under
  `eval_agent/data/sciworld` are all that is needed beyond the `scienceworld` package.

`k8s/setup/` and `k8s/webshop-eto-build.yaml` are the exact jobs that built these, usable as a
recipe on any machine.

---

## Running a different base model

`scripts/README.md` has the full checklist. The three that catch people:

1. **Check the trainer can load it before a long run.** Qwen3.5 does not exist in `transformers`
   4.x at all (`KeyError: 'qwen3_5'`); it needs `>= 5.2`. Hybrid linear-attention models
   (Qwen3.5, Qwen3-Next) also want `flash-linear-attention` and `causal-conv1d`, or the Gated
   DeltaNet layers fall back to slow, memory-hungry PyTorch ops — and they require
   `sft.packing: false`.
2. **`inference.enable_thinking` is a Qwen concept.** It is passed to the chat template as
   `chat_template_kwargs`. For a model whose template has no such variable (Gemma, Llama), leave
   it **empty** in the config and it will not be sent at all.
3. **Use bfloat16.** `sft.dtype` and `inference.vllm_dtype` are both `bfloat16`; Gemma in
   particular is numerically unstable in fp16.

Then re-tune `sft.max_length` and `sft.deepspeed` for the model's size — see the OOM notes in
`scripts/README.md`.

---

## Status

WebShop is complete for Qwen3.5-9B (3 iterations, evaluated at both a 10-step and a 100-step
budget). ScienceWorld is running. `pipeline/REPRODUCTION.md` carries the methodology and results.

The pipeline internals — MCTS, revision, data construction, training, scoring — are what produced
those numbers. `scripts/run_agentr.sh` re-hosts those same steps off Kubernetes and is **newer and
less exercised**: expect to debug the launcher rather than the method, and please report anything
you hit.
