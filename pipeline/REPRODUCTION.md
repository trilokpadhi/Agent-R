# Reproducing Agent-R with Qwen3.5-9B on WebShop

What was done, why, and every place it departs from the paper. Written so the numbers can be
defended or re-derived by someone who was not in the room.

Paper: *Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training*
(arXiv 2501.11425). Code: `github.com/ByteDance-Seed/Agent-R`.

---

## 1. Goal

Fill the `Qwen3.5-9B + Agent-R` row of Table 3, alongside SFT (54.41), RFT (55.76), ETO (71.16) and
Ours (72.08) on WebShop. That requires the Agent-R method run on **the same benchmark those rows
were measured on**, which turned out to be the crux of the whole exercise (§4).

## 2. What Agent-R does

Two phases, repeated for three iterations.

**Phase I — build revision trajectories.** For each training task, run MCTS (8 rollouts, depth 20,
4 candidate actions per expansion, c_uct = 0.25, temperature 1) to get a tree of trajectories. Pair
a high-scoring leaf path with a lower-scoring one whose reward differs by more than β = 0.2. Walk
the bad path and ask the model itself to judge each action good/bad/uncertain; the first "bad" is
the **transition point** t'. Splice: bad prefix up to t' + a revision thought (one of ten fixed
sentences) + the good path's continuation. The good path must also clear α, which rises 0.5 → 0.7 →
1.0 across iterations.

**Phase II — SFT on those trajectories**, mixed with self-generated good trajectories and general
chat data (ShareGPT) at 8:2. The new checkpoint becomes the actor for the next iteration. No expert
demonstrations are used at any point — the paper: *"this process relies entirely on self-play."*

## 3. Infrastructure

ARCTIC Kubernetes, namespace `ii400r87`, 7 × H100-80GB, 250 GiB CephFS PVC at `/data`.

One parent Job runs `pipeline/controller.py`, which launches a child Job per step, waits, validates,
writes a `.done` marker, and continues. Re-launching resumes rather than recomputes. Every run
records its git SHA; the cluster executes `git archive HEAD`, so an uncommitted edit cannot run.

Per search/revise shard: one Pod, one GPU, one vLLM OpenAI server, **43 concurrent worker processes**
sharing it (one per task or per tree), plus an environment server as a native sidecar.

Images are pinned by digest: vLLM `qwen3_5`, ms-swift 4.5.3 for training (XTuner, which the authors
used, has no Qwen3.5 support), `condaforge/miniforge3` for environment servers.

## 4. The decision that mattered: which WebShop?

Agent-R ships AgentGym's WebShop. The Table 3 rows were produced on the ETO/Co-Evolving WebShop.
**They are different benchmarks:**

| | AgentGym (Agent-R default) | ETO (the table) |
|---|---|---|
| Products | 1,000-product subset | **1,181,430** |
| Goals | ~6,910 | **12,087** human-written |
| Test set | its own 200 ids | ETO's 200 ids |
| **Test-set overlap between the two** | colspan | **3 of 200** |
| Prompt | zero-shot | one in-context example |
| Step budget | 100 | **10** |
| Invalid format | silently ignored | explicit error observation |

A number collected on AgentGym cannot be compared with that table. We therefore serve the ETO
environment, in `webshop_eto/`:

- `server.py` — one shared `SimServer` across all sessions (their `SharedWebShopFactory` pattern),
  reproducing their `step()` semantics exactly: `Action:` parsed with `findall(...)[0]`, explicit
  format error when the marker is absent, silent no-op on an invalid click, terminal reward only at
  `click[Buy Now]`.
- `client.py` — the four methods Agent-R touches, plus their few-shot prompt rebuilt from
  `prompt_with_icl(icl_num=1)`.

`webshop_protocol: agentgym` in the config restores the released behaviour byte-for-byte.

`/data/src/AgentGym` is never modified. The packages cannot even collide — theirs is
`webshop.web_agent_site`, AgentGym's is `web_agent_site`.

## 5. Deviations from the paper, and why

Each of these is a deliberate choice, not an accident.

| # | Deviation | Reason |
|---|---|---|
| 1 | **ETO environment** instead of AgentGym | §4. Required for comparability with the table. |
| 2 | **10-step eval budget** instead of the paper's 100 | The table's protocol. Handicaps Agent-R (see §7). |
| 3 | **Few-shot eval prompt**; SFT target stays zero-shot | Matches ETO exactly — verified against their `webshop_sft.json`: 2 identical leading turns across 1,824 rows, no in-context example. |
| 4 | **No per-reply token cap** (authors use 500) | 500 fits Llama-3.1's short thoughts but cut >10% of Qwen3.5 replies before `Action:`, producing 10,194 truncated turns and a verifier that failed to parse 44% of its own judgements. |
| 5 | **`max_length` 12288**, over-long rows **dropped not truncated** | The paper's 8,196 truncates 14.5% of our rows; truncation removes the recovery and the purchase. We sample 5,500 rows from 141,548, so dropping the longest ~5% costs nothing. |
| 6 | **ms-swift** instead of XTuner | XTuner has no Qwen3.5 support. |
| 7 | **No loss mask** on the bad prefix | Eq. 6 masks it, but the released code's `rewrite()` drops the flags and trains on every assistant turn. We follow the code. |
| 8 | **RNG seeded before the environment builds** | Upstream WebShop draws product prices from an unseeded global RNG in `load_products()` and only calls `random.seed(233)` afterwards, so every server start invents different "under $X" caps. Without this, the same task id means different things in different pods. |

Everything else follows the paper: 8 rollouts, depth 20, 4 candidates, c_uct 0.25, temperature 1,
α 0.5/0.7/1.0, β 0.2, 300 WebShop simulations per iteration, 5,500 revision + 600 good rows,
ShareGPT at 8:2, lr 2e-5, 3% warmup, cosine, AdamW wd 0, grad clip 1, batch 1 × accumulation 16,
3 epochs then 1, eval temperature 0.

## 6. Data quality checks (all measured, none assumed)

Run before trusting any number:

```
trees collected                       300
trees containing a scoring path       294 / 297  (99%)
trees containing a perfect path       160 / 297  (54%)
verifier replies parsed               ~100%   (was 56% before fix #4)
revision rows produced                141,548
revision rows ending in a purchase    100%
rows with a truncated assistant turn  0
token length of training rows         median 2,879, mean 4,507, p90 9,991, p99 20,776
```

## 7. Results

**Iteration 1, WebShop, ETO protocol, 200 test tasks, 10 steps, temperature 0:**

| | |
|---|---|
| **Score** | **51.66** |
| Completed a purchase | 154 / 200 (77%) |
| Perfect (1.0) | 49 (24.5%) |
| Zero | 49 (24.5%) |
| Hit the 10-step cap | 52 (26%) |
| Median steps | 5 |

Reference points: Agent-R paper iteration 1, Llama-3.1-8B, single-task WebShop = **49.80** — at a
100-step budget. Our earlier broken run (AgentGym, 500-token cap) = 17.08.

**A caveat worth stating in the paper.** Our revision trajectories place the transition point far
earlier than the authors': 91.7% of revision rows flag the *first* action as wrong, giving a
"revision length" of ~1.3 against their 5.7 (Figure 5). Combined with a median 7-turn good
continuation, the learned recovery pattern consumes ≈ 10 steps — exactly the eval budget. This is
the most likely explanation for the 26% cap rate, and it means the 10-step number understates the
method.

## 8. Cost, measured

| Step | Iteration 1 | Note |
|---|---|---|
| search | 2h23m | 300 trees, 7 GPUs |
| revise | 6h13m | 299 trees in ~2h; one tree with 1,526 pairs took the rest |
| sft-data | 2 min | |
| SFT | 2h17m | 3 epochs, 219 steps, ~38 s/step |
| eval | 10 min | 200 tasks over 7 shards |

**The dominant cost was a long tail in revise.** The released `path_collection.py` walks a tree's
pairs one at a time, and pair count is quadratic in the number of scoring leaf paths, so the tail
grows as the model improves: iteration 2's worst tree had 5,228 pairs (~17 h), iteration 3's had
5,480 (~30 h), during which the other six GPUs sat idle. Measured revise times: 6h13m, 17h, 32h.

**Fixed by pair sharding (commit `f6031b1`), which does not change the method.** Each tree now runs
as 8 processes, shard *i* taking pairs *i, i+8, i+16, …* of the identical seeded list. The per-pair
work is untouched. The only quantity that could have differed — which of the ten revision sentences
a pair draws, since the released loop calls `random.randint` once per α-passing pair in serial
order — is pre-drawn in that same order from the same RNG state and passed in, so every shard
reproduces the serial assignment exactly. Verified two ways: an offline simulation of the released
loop (N=1/4/8, trees of 45/531/2,045 pairs: identical index→sentence sets, no duplicates) and an
end-to-end run of the real script on a real tree with `--revise 0` (serial vs 8 shards: 107 vs 107
rows, identical row sets, identical sentence multiset). `pair_shards: 1` is the released behaviour.
Expected revise time per iteration: ~3–6 h instead of 6–32 h; a full 3-iteration run ~28 h
instead of ~68 h. Applies from the first run launched after `f6031b1`; iteration 3 of `q35eto` ran
on the serial code.

## 9. Open items

- SciWorld: their test split addresses tasks by `(task_name, variation_idx)` with per-task step
  budgets (30–120, weighted mean 39.7); Agent-R uses flat `data_idx` integers. Their test set has
  **zero unseen task types**, so the Seen/Unseen columns must come from a source not in the repo.
- Product prices are seeded but not frozen to Co-Evolving's manifest, so per-goal price caps may
  differ from theirs. One reward term out of ~5; unbiased.
- A supplementary eval at 100 steps would separate "could not solve" from "ran out of budget".

## 10. Reproducing

See [RUNBOOK.md](RUNBOOK.md) for the exact commands. Short version: open the SSH tunnel, sign in,
commit, then `pipeline/deploy.sh pipeline/config.yaml`. All settings live in `pipeline/config.yaml`,
each annotated with its source.
