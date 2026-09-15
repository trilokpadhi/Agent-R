# Agent-R reproduction pipeline (Qwen3.5-9B, WebShop + SciWorld, 3 iterations)

One command launches one parent Job. The parent runs `controller.py`, which starts a child Job for
each step, waits for it, checks its output, and moves on:

```text
for iteration in 1..3:
    search WebShop   (7 x 1-GPU jobs)   mcts_collection.py
    search SciWorld  (7 x 1-GPU jobs)   mcts_collection.py
    revise WebShop   (7 x 1-GPU jobs)   path_collection.py --data_type centric --revise 1
    revise SciWorld  (7 x 1-GPU jobs)   path_collection.py --data_type centric --revise 1
    build SFT data   (in the parent)    revise_log + high_log, capped; ShareGPT at 8:2
    SFT              (1 x 7-GPU job)    ms-swift full fine-tune
    eval             (2 x 1-GPU jobs)   eval.py on the 200-item WebShop and SciWorld test sets
    the new checkpoint becomes the model for the next iteration
```

## Run

```bash
pipeline/deploy.sh pipeline/config-smoke.yaml   # tiny end-to-end check first
pipeline/deploy.sh pipeline/config.yaml         # the real run
kubectl logs -n ii400r87 -f job/agr-q35v1-controller
```

`deploy.sh` refuses to run with uncommitted changes, copies `git archive HEAD` to
`/data/src/agentr-pipeline/<sha>`, and applies `rbac.yaml` plus the controller Job.

## Reproducibility

- Every setting is in the config file, with its source (paper, maintainer issue, or user decision).
- All images are pinned by digest. Nothing is installed at runtime.
- `/data/runs/<run>/run.json` records every launch (commit + config), and each step's `.done` names
  the commit that produced it, so a fix can be committed and the run resumed without redoing work.
- Each finished step writes `.done` with its counts, inputs and commit. Rerunning `deploy.sh`
  resumes: finished steps are skipped and failed child Jobs are recreated.
- Rendered child manifests are saved to `/data/runs/<run>/logs/manifests/`.

## Outputs

```text
/data/runs/<run>/
  run.json
  logs/                         controller.log, per-job logs, rendered manifests
  iter<N>/search-<env>/         MCTS trees (mcts_result/<env>/Qwen3.5-9B/search_results_<id>.json)
  iter<N>/revise-<env>/         path_collection output per shard
  iter<N>/sft-data/             train.jsonl, stats.json
  iter<N>/sft/output/           checkpoint used by iteration N+1
  iter<N>/eval/.done            mean final reward per environment
```

## Settings and decisions (recorded Sep 14, 2026)

| Topic | Choice | Why |
|---|---|---|
| Iterations 2-3 start from | previous checkpoint | user; epochs drop to 1 after iteration 1 |
| Loss on bad-prefix turns | included | user: follow the released code (`rewrite()` drops the loss flags) |
| alpha/beta on SciWorld | raw 0-100 scores | released code does not rescale |
| Tasks per iteration | the same tasks every iteration | released code (`range(1000)`, variations `1..task_iteration`) |
| Dataset order | WebShop end to end first (`tasks: [webshop]`), then SciWorld | user |
| Processes per GPU | one per task id (search) and one per tree (revision), sharing one vLLM server | set by the data; affects speed only |
| SFT max length | 8,196 | paper appendix C.1 |
| Precision | bf16 | user; Qwen3.5 is released in bf16 |
| Eval temperature | 0 | paper C.1 (AgentGym setting); search stays at 1 (paper) |
| Output cap per model reply | none (authors: 500) | user; the 500 cap cut off >10% of Qwen3.5 replies before `Action:`, training taught the model those cut-off replies, and iteration-1 eval looped (run q35ws: 14.6, 80 of 104 tasks hit 100 steps) |
| Training setup | single-task (one dataset per run) | user; paper Table 6 "Single" row, WebShop iteration 3 = 60.66 |
| vLLM max model length | not set (model default) | caps one request only; path_collection.py truncates by words, so prompts can exceed 8,192 tokens |
| Global batch | 112 = 1 per GPU x 16 accumulation x 7 GPUs | paper C.1 (paper used 8 A100-80GB) |
| Trainer | ms-swift 4.5.3 | XTuner has no Qwen3.5 support |
| Packing | off | Qwen3.5 linear attention does not support it |
