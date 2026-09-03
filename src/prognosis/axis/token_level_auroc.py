"""Held-out token-level AUROC for the value axis (Fig 2a).

The released construction code scores a held-out reward function from a single
before/after mean projection pair, which saturates near ceiling and is not the
statistic the original figure reports. This scores individual classified tokens
instead, before=0 and after=1, over repeated random splits of the reward functions.

Input is a table of per-token dot products against every function's before/after
mean, at every layer, written by token_projection_extraction.py. Since the axis for
a split is a linear combination of those means, every split is scored from the table
on CPU with no further GPU passes.

Usage: python -m prognosis.axis.token_level_auroc --projections ... --meta ... --out ...
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_SPLITS = 20
N_TRAIN = 24
SEED = 0


def load_projections(npz_path, meta_path):
    """Per-token dot products, their reward function, and their before/after label.

    Columns run [before_f0, after_f0, before_f1, after_f1, ...] in fn_names order.
    """
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    z = np.load(npz_path)
    return z["dots"], z["reward_fns"], z["labels"], list(meta["fn_names"])


def function_differences(dots):
    """(n_tokens, n_layers, n_functions) per-function after-minus-before projection."""
    return dots[:, :, 1::2] - dots[:, :, 0::2]


def heldout_auroc(dots, reward_fns, labels, fn_names,
                  n_splits=N_SPLITS, n_train=N_TRAIN, seed=SEED):
    """Per-layer AUROC over random splits, scoring tokens of held-out functions.

    Each split builds the axis from n_train functions and ranks the tokens of the
    rest. Returns (mean_auc, std_auc, n_splits_used).
    """
    diffs = function_differences(dots)
    is_after = (labels == "after").astype(int)
    fn_index = {name: i for i, name in enumerate(fn_names)}
    token_fn = np.array([fn_index[f] for f in reward_fns])
    n_layers, n_functions = dots.shape[1], len(fn_names)

    rng = np.random.default_rng(seed)
    aucs = np.full((n_splits, n_layers), np.nan)

    for split in range(n_splits):
        perm = rng.permutation(n_functions)
        train, heldout = perm[:n_train], perm[n_train:]
        mask = np.isin(token_fn, heldout)
        y = is_after[mask]
        if y.min() == y.max():
            continue
        scores = diffs[mask][:, :, train].mean(axis=2)
        for layer in range(n_layers):
            aucs[split, layer] = roc_auc_score(y, scores[:, layer])

    return (np.nanmean(aucs, axis=0), np.nanstd(aucs, axis=0),
            int(np.isfinite(aucs[:, 0]).sum()))


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Held-out token-level AUROC by layer.")
    parser.add_argument("--projections", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    dots, reward_fns, labels, fn_names = load_projections(args.projections, args.meta)
    print(f"{dots.shape[0]} tokens, {len(fn_names)} reward functions, {dots.shape[1]} layers")

    mean_auc, std_auc, n_used = heldout_auroc(
        dots, reward_fns, labels, fn_names, args.n_splits, args.n_train, args.seed)
    print(f"Split: {args.n_train} train / {len(fn_names) - args.n_train} held out, "
          f"{n_used}/{args.n_splits} splits used")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"layer": range(dots.shape[1]), "n_splits_used": n_used,
                  "mean_auc": mean_auc, "std_auc": std_auc}).to_csv(args.out, index=False)

    best = int(np.argmax(mean_auc))
    print(f"Best layer: {best} (AUC={mean_auc[best]:.3f})")
    print(f"Saved per-layer AUROC to {args.out}")


if __name__ == "__main__":
    main()
