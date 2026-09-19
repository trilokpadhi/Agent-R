# Runbook: launching the Agent-R pipeline on ARCTIC

Step-by-step, from a cold laptop to a running three-iteration job. Everything below has been used
for the `q35eto` run (Qwen3.5-9B, WebShop, ETO environment) — it is what was actually done, not a
sketch.

The short version:

```bash
# 1. terminal A (yours, once per work session) - open the tunnel and sign in
ssh -o ControlMaster=auto -o ControlPath='~/.ssh/cm-%r@%h:%p' -o ControlPersist=8h \
    -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
    -L 8000:127.0.0.1:8000 tpadhi1@arclogin02.rs.gsu.edu
kubectl get pods -n ii400r87        # prints a URL -> open http://127.0.0.1:8000/ on the Mac

# 2. terminal B - commit, then launch
cd ~/Projects/Agent-R
git add -A && git commit -m "..."
pipeline/deploy.sh pipeline/config.yaml
```

That is the whole launch. Everything else in this file is detail, failure handling, and why.

---

## 1. Connect (once per work session)

You must be on the GSU network or VPN — `arclogin02.rs.gsu.edu` will not even resolve otherwise
(`host arclogin02.rs.gsu.edu` returning NXDOMAIN means VPN is down, not that the cluster is down).

```bash
ssh -o ControlMaster=auto -o ControlPath='~/.ssh/cm-%r@%h:%p' -o ControlPersist=8h \
    -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
    -L 8000:127.0.0.1:8000 tpadhi1@arclogin02.rs.gsu.edu
```

`ControlMaster`/`ControlPersist` create a shared connection that later commands reuse, so Duo is
answered once. `-L 8000` forwards the port the Kubernetes sign-in uses.

Then, in that same terminal:

```bash
kubectl get pods -n ii400r87
```

It prints a `http://localhost:8000/...` URL and waits. Open **http://127.0.0.1:8000/** on the Mac
to finish. When the pod list appears you are authenticated. The token lasts a few hours; repeat
this step when it expires.

**If `kubectl` hangs and prints no URL**, the callback port is already taken on the login node:

```bash
pkill -u "$USER" -x kubectl-oidc_lo; pkill -u "$USER" -x kubectl
ss -ltn | awk '$4 ~ /:(8000|18000)$/'      # should print nothing
```

then retry. Port 8000 is one port per machine, shared by everyone logged into that node, so never
leave stray sign-in helpers running. Cancel a stuck login with Ctrl+C, never Ctrl+Z.

## 2. Check the cluster is free

```bash
kubectl get pods -n ii400r87 --field-selector=status.phase=Running
```

Only `access-pod` should be running before a new launch. The namespace has 7 H100s; a pipeline run
uses all of them.

## 3. Edit the config

`pipeline/config.yaml` is the single source of truth — every setting the run uses, each annotated
with its source (paper, maintainer issue, or a measurement). The ones you are most likely to touch:

| Key | Meaning |
|---|---|
| `run` | short id. **Change it to start a fresh run**; keeping it resumes the existing one |
| `tasks` | `[webshop]` or `[sciworld]` |
| `webshop_protocol` | `eto` (full catalogue, ETO split, few-shot) or `agentgym` (released Agent-R) |
| `iterations` | 3 in the paper |
| `gpus` | 7 |
| `sft.max_length` | 12288; rows above it are dropped, not truncated |
| `eval.max_steps` | 10 for the ETO protocol, 100 for the Agent-R paper protocol |

## 4. Commit — this is enforced

`deploy.sh` refuses to launch if anything under `pipeline/`, `webshop_eto/`, `mcts_utils/`,
`mcts_collection.py`, `path_collection.py` or `eval.py` is uncommitted. The cluster runs
`git archive HEAD`, so an uncommitted edit would silently not be deployed.

```bash
git add -A && git commit -m "what changed and why"
```

## 5. Launch

```bash
pipeline/deploy.sh pipeline/config.yaml
```

Which does, in order:

1. refuses if pipeline code is uncommitted
2. uploads `git archive HEAD` to `/data/src/agentr-pipeline/<sha>` on the PVC (skipped if that sha
   is already there)
3. applies `pipeline/rbac.yaml` — the ServiceAccount the controller uses to create child Jobs
4. applies **one** Job, `agr-<run>-controller`

Expected output ends with:

```text
run=q35eto commit=8b9aeef6da35 config=pipeline/config.yaml
== code on the PVC: /data/src/agentr-pipeline/8b9aeef6da35
uploaded
== controller Job agr-q35eto-controller
job.batch/agr-q35eto-controller created
```

## 6. Watch

```bash
kubectl logs -n ii400r87 -f job/agr-q35eto-controller
kubectl get pods -n ii400r87 -l pipeline-run=q35eto
```

The controller logs one line per child Job submitted/succeeded, plus the summary of each step.
Outputs live under `/data/runs/<run>/`; per-worker logs under `/data/runs/<run>/logs/`.

Reading a child's progress, e.g. training:

```bash
kubectl logs -n ii400r87 job/agr-q35eto-i1-sft --tail=40 | tr '\r' '\n' | grep '^{'
```

(`tr '\r' '\n'` is needed because tqdm writes carriage returns.)

## 7. Resume after a failure

Every completed step writes a `.done` marker, so **re-running deploy.sh resumes** — it never
recomputes finished work. The usual loop is: fix the config or code, commit, relaunch.

If the controller is still running you must stop it first, otherwise `deploy.sh` sees it as healthy
and exits:

```bash
kubectl delete job -n ii400r87 agr-q35eto-controller agr-q35eto-i1-sft --wait=true
pipeline/deploy.sh pipeline/config.yaml
```

Deleting the controller does **not** stop child Jobs already running; the new controller reattaches
to them and waits ("already running, waiting").

To force a step to re-run, delete its marker:

```bash
kubectl exec -n ii400r87 access-pod -- rm /data/runs/q35eto/iter1/sft/.done
```

## 8. Results

```bash
kubectl exec -n ii400r87 access-pod -- cat /data/runs/q35eto/iter1/eval/.done
```

```json
{"webshop": {"items": 200, "mean_env_score": 0.5166, "reported": 51.66}}
```

---

## Concurrency: what actually runs where

A common misreading is that each GPU runs one process. It does not.

```text
one search shard = one Pod = one GPU
  |- sidecar container : WebShop environment server
  |- policy container
       |- ONE vLLM OpenAI server on 127.0.0.1:8000  (holds the model)
       |- 43 x mcts_collection.py, one per task id, all sharing that server
```

`WORKERS` is set to the shard's task count (`workers=hi - lo` in `controller.py`), so with 300 tasks
over 7 shards that is **43 concurrent processes per GPU**, not 1 and not 2. Revise is the same
shape: `xargs -P 43`, one `path_collection.py` per tree.

This is why raising it further does not help. Measured during the pilot: GPU KV-cache occupancy at
this concurrency is **1.5–2.3%**, and the cards draw 190–360 W of a 700 W cap. The GPU is not the
constraint — the environment server and MCTS's replay-from-root behaviour are.

## Known slow spots

| Step | Iteration 1 actual | Note |
|---|---|---|
| search | 2h23m | 300 trees, 7 GPUs |
| revise | 6h13m | the tail is one tree with 1,526 pairs; the other 299 finished in ~2h |
| sft-data | 2 min | |
| SFT | 2h17m | 3 epochs, 219 steps, ~38 s/step |
| eval | 10 min | 200 tasks over 7 shards |

**Pair sharding (from commit `f6031b1`) removes the revise tail without changing the method.** Set in
`pipeline/config.yaml`:

```yaml
revise:
  pair_shards: 8      # processes per tree, each taking every 8th pair; 1 = released serial behaviour
  concurrency: 96     # (tree, shard) units in flight per GPU
```

Each tree's pair list is identical in every process (seeded shuffle), shards are disjoint, and the
revision sentences are pre-drawn in serial order — verified to give the identical set of rows as the
serial run (see REPRODUCTION.md §8). The Job orders units biggest tree first, resumes per shard
(`done/<tree>/p<i>/`), and assembles `out/<tree>/…_centric.jsonl` in the layout downstream reads.
Revise drops from 6–32 h per iteration to roughly 3–6 h.
