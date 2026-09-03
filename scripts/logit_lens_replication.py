#!/usr/bin/env python
"""Does our reconstructed axis reproduce the published logit-lens token list?

Jiang et al. report the 30 tokens most promoted by their value axis under a logit lens, built on
Qwen3-8B. Our axis is an independent reconstruction on Qwen3-32B. The Qwen3 family shares a
tokenizer (vocab 151936), so the two token lists are directly comparable by id even though the
models differ in size and hidden dimension -- which makes this a cross-model replication test
rather than a sanity check.

The statistic is RANK, not score. Score magnitudes are not comparable across models with
different hidden sizes and unembedding norms, and an earlier version of this analysis went wrong
by reaching for the score standard deviation: that measures alignment with the whole vocabulary,
which no meaningful direction has, and it reported "indistinguishable from random" for a
direction whose top-ranked token is the published rank 3.

The null is the same 30 tokens scored under random unit directions of matched dimensionality,
which is what distinguishes a real replication from tokens that simply sit at the top of any
direction because of unembedding geometry.

Writes data/logit_lens_replication.json.

Reproduce: python scripts/logit_lens_replication.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prognosis.analysis.logit_lens import stream_scores

OUT_DIR = ROOT / "data"

# Jiang et al., "Logit Lens: Top Promoted Tokens", in published rank order. Entries their table
# marks as tokenizer artifacts are kept -- excluding them would be choosing which of their
# results to be graded against after seeing ours.
JIANG_TOP30 = [
    "其它问题", "遗漏", "这才是", "采矿等", "不会再", "Again", "wording", "杜绝", "IfNeeded",
    "進一步", "不影响", "ebenfalls", "again", "瑧", "想办法", "加分", "脱颖", "届时", "下次",
    "不排除", "后再", "网友评论", "范冰", "阿富", "这种情况", "AndUpdate", "找回", "不会有",
    "进一步", "追问",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=Path, default=None)
    ap.add_argument("--axis", type=Path,
                    default=ROOT / "data" / "value_axis.npy")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-32B")
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(20, 65, 4)))
    ap.add_argument("--n-random", type=int, default=20)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    shard = args.shard or Path(sorted(glob.glob(
        "D:/hf-cache/**/*.safetensors", recursive=True))[0])
    from transformers import AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    ids, dropped = {}, []
    for t in JIANG_TOP30:
        for cand in (t, " " + t):
            e = tok.encode(cand, add_special_tokens=False)
            if len(e) == 1:
                ids[t] = e[0]
                break
        else:
            dropped.append(t)
    print(f"{len(ids)}/30 of the published tokens are single tokens in this tokenizer"
          + (f"; dropped {dropped}" if dropped else ""))

    axis = np.load(args.axis).astype(np.float32)
    rng = np.random.default_rng(0)
    dirs = {f"L{L}": axis[L] / max(np.linalg.norm(axis[L]), 1e-12) for L in args.layers}
    for i in range(args.n_random):
        r = rng.normal(size=axis.shape[1]).astype(np.float32)
        dirs[f"rand_{i}"] = r / np.linalg.norm(r)
    scores, _ = stream_scores(shard, "lm_head.weight", dirs)
    vocab = len(next(iter(scores.values())))
    idx = np.array(list(ids.values()))

    def ranks(v):
        # 1-based: the highest-scoring token in the vocabulary is rank 1, not rank 0. These
        # numbers are quoted directly in the paper and compared against Jiang et al.'s published
        # 1-based ranks, so a 0-based convention here would silently shift every one of them.
        order = np.empty(vocab, dtype=np.int64)
        order[np.argsort(v)[::-1]] = np.arange(1, vocab + 1)
        return order[idx]

    null = np.vstack([ranks(scores[f"rand_{i}"]) for i in range(args.n_random)])
    null_median = float(np.median(null))
    print(f"\nnull: median rank of these tokens under {args.n_random} random directions "
          f"= {null_median:,.0f} of {vocab:,}\n")
    print(f"{'layer':>6s} {'median rank':>12s} {'in top 100':>11s} {'in top 1000':>12s} "
          f"{'perm p':>8s}")

    per_layer, best = {}, None
    for L in args.layers:
        r = ranks(scores[f"L{L}"])
        med = float(np.median(r))
        # permutation p: how often does a random direction rank these 30 at least as well?
        p = (np.median(null, axis=1) <= med).mean()
        rec = {"median_rank": med, "n_top100": int((r <= 100).sum()),
               "n_top1000": int((r <= 1000).sum()),
               "perm_p_vs_random": float((p * args.n_random + 1) / (args.n_random + 1)),
               "ranks": {t: int(rr) for t, rr in zip(ids, r)}}
        per_layer[L] = rec
        print(f"{L:>6d} {med:>12,.0f} {rec['n_top100']:>11d} {rec['n_top1000']:>12d} "
              f"{rec['perm_p_vs_random']:>8.3f}")
        if best is None or med < per_layer[best]["median_rank"]:
            best = L

    b = per_layer[best]
    print(f"\nbest layer {best}: median rank {b['median_rank']:,.0f} "
          f"(null {null_median:,.0f}), {b['n_top100']}/{len(ids)} inside the top 100")
    hits = sorted(b["ranks"].items(), key=lambda kv: kv[1])[:10]
    print("  closest matches: " + ", ".join(f"{t}#{r}" for t, r in hits))

    out = {"n_tokens_matched": len(ids), "dropped": dropped, "vocab": int(vocab),
           "null_median_rank": null_median, "n_random": args.n_random,
           "best_layer": best, "per_layer": {str(k): v for k, v in per_layer.items()}}
    path = OUT_DIR / "logit_lens_replication.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
