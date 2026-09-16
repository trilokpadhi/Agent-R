#!/usr/bin/env python3
"""Read-only diagnosis of run q35ws, iteration 1 (WebShop): why the trained model scores ~14.

Checks, each on the data the pipeline actually produced:
  A. search trees      - cut-off replies (500-token cap, no 'Action:'), where they occur, what WebShop did
  B. revision rows     - where cut-off turns sit (bad prefix / revision / good continuation), clean rows
  C. SFT training file - cut-off turns, example lengths vs max_length 8196, what truncation removes
  D. SFT log           - the prompt/label rendering ms-swift actually trained on
  E. eval results      - failure categories, reply lengths, loops
  F. inference prompt  - the prompt vLLM renders at eval, to compare with D

Run inside a pod with python3 + transformers + /data (e.g. the vLLM image):
  python3 diagnose_iter1.py --run /data/runs/q35ws > report.txt
"""
import argparse
import collections
import glob
import json
import os
import random
import statistics
import time

CAP = 500          # max_tokens used by the run
CUT_MIN = CAP - 5  # tokenizer counts can differ from vLLM's by a token or two


def pct(a, b):
    return f"{a} / {b} ({100 * a / b:.1f}%)" if b else f"{a} / 0"


def quantiles(xs):
    if not xs:
        return "n/a"
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return f"min {xs[0]}, median {q(.5)}, p90 {q(.9)}, p99 {q(.99)}, max {xs[-1]}"


class Tok:
    def __init__(self, model_dir):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(model_dir)

    def lens(self, texts):
        if not texts:
            return []
        return [len(x) for x in self.t(texts, add_special_tokens=False)["input_ids"]]

    def chat_len(self, messages):
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        return len(self.t.apply_chat_template(msgs, tokenize=True, enable_thinking=False))

    def render(self, messages, gen=True):
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        return self.t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False)


def is_cut(text, n):
    return n >= CUT_MIN and "Action:" not in text


# ---------------------------------------------------------------- A. search trees
def section_a(run, tok):
    print("\n" + "=" * 100 + "\nA. SEARCH TREES (iteration 1, base Qwen3.5-9B, 500-token cap)\n" + "=" * 100)
    files = sorted(glob.glob(f"{run}/iter1/search-webshop/mcts_result/webshop/*/search_results_*.json"))
    tot = cut = noact = same_obs_cut = 0
    by_depth = collections.defaultdict(lambda: [0, 0])
    lens_sample, cut_actions = [], collections.Counter()
    tasks_success = tasks_clean_success = 0
    leaves = leaves_success = leaves_success_clean = leaves_with_cut = 0
    for fi, f in enumerate(files):
        root = json.load(open(f))
        nodes = []  # (node, parent, path_has_cut_before)
        stack = [(c, root) for c in (root.get("children") or [])]
        while stack:
            n, p = stack.pop()
            nodes.append((n, p))
            for c in n.get("children") or []:
                stack.append((c, n))
        texts = [n.get("llm_response", "") for n, _ in nodes]
        lens = tok.lens(texts)
        cutflag = {}
        for (n, p), txt, L in zip(nodes, texts, lens):
            tot += 1
            d = n.get("depth", 0)
            by_depth[d][1] += 1
            if "Action:" not in txt:
                noact += 1
            c = is_cut(txt, L)
            cutflag[id(n)] = c
            if c:
                cut += 1
                by_depth[d][0] += 1
                cut_actions[(n.get("action") or "")[:40]] += 1
                if n.get("obs", "") == p.get("obs", ""):
                    same_obs_cut += 1
            if random.random() < 0.05:
                lens_sample.append(L)
        # leaf paths
        any_s = any_clean_s = False
        def dfs(n, has_cut):
            nonlocal leaves, leaves_success, leaves_success_clean, leaves_with_cut, any_s, any_clean_s
            hc = has_cut or cutflag.get(id(n), False)
            kids = n.get("children") or []
            if not kids:
                leaves += 1
                leaves_with_cut += hc
                if n.get("env_score", 0) > 0:
                    leaves_success += 1
                    any_s = True
                    if not hc:
                        leaves_success_clean += 1
                        any_clean_s = True
                return
            for k in kids:
                dfs(k, hc)
        for k in root.get("children") or []:
            dfs(k, False)
        tasks_success += any_s
        tasks_clean_success += any_clean_s
        if fi % 50 == 0:
            print(f"  ... {fi + 1}/{len(files)} trees", flush=True)
    print(f"trees: {len(files)}   nodes (model replies): {tot}")
    print(f"replies without 'Action:':            {pct(noact, tot)}")
    print(f"replies cut at the cap (>= {CUT_MIN} tokens, no 'Action:'): {pct(cut, tot)}")
    print(f"reply length (5% sample): {quantiles(lens_sample)}")
    print(f"cut replies whose observation equals the parent's (page did not change): {pct(same_obs_cut, cut)}")
    print(f"action recorded for cut replies (top 5): {cut_actions.most_common(5)}")
    print("cut share by depth:", {d: f"{c}/{t}={100 * c / t:.0f}%" for d, (c, t) in sorted(by_depth.items()) if t})
    print(f"leaf paths: {leaves}; containing >= 1 cut reply: {pct(leaves_with_cut, leaves)}")
    print(f"successful leaf paths (env_score > 0): {leaves_success}; of those with no cut reply: {pct(leaves_success_clean, leaves_success)}")
    print(f"tasks with a successful path: {pct(tasks_success, len(files))}; with a successful path free of cut replies: {pct(tasks_clean_success, len(files))}")


# ---------------------------------------------------------------- B. revision rows
REVISION_MARK = "Action: wait"


def classify_log(log, lens):
    """Return counts of cut turns in bad prefix / revision / good part of a revise_log."""
    out = collections.Counter()
    seen_rev = False
    for m, L in zip(log, lens):
        if m["role"] != "assistant" or m.get("loss") is None:
            continue
        txt = m["content"]
        if m.get("loss") is True and txt.rstrip().endswith(REVISION_MARK) and not seen_rev:
            seen_rev = True
            out["revision_turns"] += 1
            continue
        part = "bad" if m.get("loss") is False else "good"
        out[f"{part}_turns"] += 1
        if is_cut(txt, L):
            out[f"{part}_cut"] += 1
    return out


def section_b(run, tok, sample_rows=3000):
    print("\n" + "=" * 100 + "\nB. REVISION ROWS (path_collection.py output)\n" + "=" * 100)
    files = sorted(glob.glob(f"{run}/iter1/revise-webshop/shard*/out/*/*_centric.jsonl"))
    counts = [sum(1 for _ in open(f)) for f in files]
    total = sum(counts)
    step = max(1, total // sample_rows)
    agg, rows_clean, rows_goodclean, n = collections.Counter(), 0, 0, 0
    high_cut_rows = 0
    rev_pos, row_len = [], []
    i = 0
    for f in files:
        for line in open(f):
            i += 1
            if i % step:
                continue
            r = json.loads(line)
            log = r["revise_log"]
            lens = tok.lens([m["content"] for m in log])
            c = classify_log(log, lens)
            agg.update(c)
            n += 1
            rows_clean += (c["bad_cut"] + c["good_cut"]) == 0
            rows_goodclean += c["good_cut"] == 0
            hl = r["high_log"]
            hlens = tok.lens([m["content"] for m in hl])
            high_cut_rows += any(m["role"] == "assistant" and is_cut(m["content"], L) for m, L in zip(hl[3:], hlens[3:]))
            if n <= 400:  # token position of the revision turn in the rendered example
                idx = next((k for k, m in enumerate(log) if m["role"] == "assistant" and m.get("loss") is True
                            and m["content"].rstrip().endswith(REVISION_MARK)), None)
                row_len.append(tok.chat_len(log))
                if idx is not None:
                    rev_pos.append(tok.chat_len(log[:idx + 1]))
    print(f"revision rows: {total} total; analysed a systematic sample of {n}")
    print(f"bad-prefix turns (loss False in the paper, trained in released code): cut {pct(agg['bad_cut'], agg['bad_turns'])}")
    print(f"good-continuation turns (loss True): cut {pct(agg['good_cut'], agg['good_turns'])}")
    print(f"rows with no cut turn anywhere: {pct(rows_clean, n)}")
    print(f"rows whose good continuation has no cut turn: {pct(rows_goodclean, n)}")
    print(f"rows whose high_log (used as 'good' data) contains a cut turn: {pct(high_cut_rows, n)}")
    print(f"rendered example length (first {len(row_len)} sampled): {quantiles(row_len)}")
    print(f"token position where the revision thought ENDS: {quantiles(rev_pos)}; beyond 8196 (cut by SFT max_length): {pct(sum(p > 8196 for p in rev_pos), len(rev_pos))}")


# ---------------------------------------------------------------- C. SFT file
def section_c(run, tok):
    print("\n" + "=" * 100 + "\nC. SFT TRAINING FILE (what ms-swift trained on)\n" + "=" * 100)
    rows = [json.loads(l)["messages"] for l in open(f"{run}/iter1/sft-data/train.jsonl")]
    agent = [m for m in rows if m[0]["role"] == "system"]
    lens = [tok.chat_len(m) for m in agent]
    over = sum(L > 8196 for L in lens)
    print(f"rows: {len(rows)} ({len(agent)} Agent-R, {len(rows) - len(agent)} ShareGPT)")
    print(f"Agent-R rendered length: {quantiles(lens)}")
    print(f"Agent-R rows longer than max_length 8196 (right-truncated): {pct(over, len(agent))}")
    lost_purchase = 0
    rev_rows = rev_beyond = 0
    for m, L in zip(agent, lens):
        if L <= 8196:
            continue
        lost_purchase += 1  # the tail (final steps incl. purchase) is beyond the cut
    for m in agent[:600]:
        idx = next((k for k, x in enumerate(m) if x["role"] == "assistant" and x["content"].rstrip().endswith(REVISION_MARK)), None)
        if idx is not None:
            rev_rows += 1
            rev_beyond += tok.chat_len(m[:idx + 1]) > 8196
    print(f"rows whose final steps (incl. the purchase) are removed by truncation: {pct(lost_purchase, len(agent))}")
    print(f"revision rows (first 600 rows) whose revision thought lies beyond 8196 tokens: {pct(rev_beyond, rev_rows)}")


# ---------------------------------------------------------------- D. SFT log
def section_d(run):
    print("\n" + "=" * 100 + "\nD. WHAT MS-SWIFT RENDERED (first training example printed in the SFT log)\n" + "=" * 100)
    path = f"{run}/logs/agr-q35ws-i1-sft.log" if os.path.exists(f"{run}/logs/agr-q35ws-i1-sft.log") else None
    if not path:
        print("SFT log not found"); return
    text = open(path, errors="replace").read()
    for key in ["[INPUT]", "[LABELS]"]:
        i = text.find(key)
        if i < 0:
            print(f"{key}: not printed in the log"); continue
        chunk = text[i:i + 2500]
        print(f"{key} (first 2500 chars):\n{chunk}\n")
    for key in ["add_non_thinking_prefix", "loss_scale", "template=", "system="]:
        j = text.find(key)
        print(f"{key}: {text[j:j + 80].splitlines()[0] if j >= 0 else 'not found'}")


# ---------------------------------------------------------------- E/F. eval
def section_e(run, tok):
    print("\n" + "=" * 100 + "\nE. EVAL RESULTS (iteration-1 checkpoint, temperature 0, 500-token cap)\n" + "=" * 100)
    files = glob.glob(f"{run}/iter1/eval/webshop/test_result/webshop/*/search_results_*.json")
    cats = collections.Counter()
    reply_lens, first_cut_step, scores = [], [], []
    example_prompt = None
    for f in files:
        r = json.load(open(f))
        st = r["state"]
        if example_prompt is None:
            example_prompt = st
        asst = [m["content"] for m in st[3:] if m["role"] == "assistant"]
        L = tok.lens(asst)
        reply_lens += L
        cuts = [is_cut(t, n) for t, n in zip(asst, L)]
        scores.append(r["env_score"])
        if any(cuts):
            first_cut_step.append(cuts.index(True) + 1)
        last = asst[-5:]
        looping = len(last) == 5 and len(set(last)) == 1
        if r["env_score"] > 0:
            cats["scored > 0"] += 1
        elif r["step_num"] >= 100 and looping and cuts and cuts[-1]:
            cats["0: loop of identical CUT-OFF replies"] += 1
        elif r["step_num"] >= 100 and looping:
            cats["0: loop of identical replies WITH an action"] += 1
        elif r["step_num"] >= 100:
            cats["0: hit 100 steps, varied replies"] += 1
        else:
            cats["0: finished early (e.g. bought the wrong item)"] += 1
    n = len(files)
    print(f"tasks evaluated: {n}; mean score x100 = {100 * statistics.mean(scores):.1f}" if n else "no results")
    for k, v in cats.most_common():
        print(f"  {k}: {pct(v, n)}")
    print(f"reply lengths: {quantiles(reply_lens)}; cut replies: {pct(sum(1 for x in reply_lens if x >= CUT_MIN), len(reply_lens))}")
    print(f"step at which a task's first cut-off happens: {quantiles(first_cut_step)}")
    if example_prompt:
        print("\n" + "=" * 100 + "\nF. INFERENCE PROMPT vLLM RECEIVES (chat template, enable_thinking=False), first 2 steps\n" + "=" * 100)
        rendered = tok.render(example_prompt[:6], gen=True)
        print(rendered[:3000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/data/runs/q35ws")
    ap.add_argument("--model", default="/data/models/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    ap.add_argument("--sections", default="DECBA")
    a = ap.parse_args()
    random.seed(0)
    tok = Tok(a.model)
    t0 = time.time()
    for s in a.sections:
        {"A": lambda: section_a(a.run, tok), "B": lambda: section_b(a.run, tok), "C": lambda: section_c(a.run, tok),
         "D": lambda: section_d(a.run), "E": lambda: section_e(a.run, tok)}[s]()
        print(f"[section {s} done at {time.time() - t0:.0f}s]", flush=True)
