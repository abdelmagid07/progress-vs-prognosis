#!/usr/bin/env python
"""Can ANY linear direction separate outcomes within a task? (paired probe ceiling)

The frozen axis fails the within-task contrast. That leaves two very different readings:
the outcome information is present in the activations but not aligned with this particular
direction, or it is not linearly decodable from them at all. The full-benchmark fitted-probe
grid cannot settle it, because those probes were fit across 499 single-trajectory tasks and
so can reach high AUROC by reading task difficulty off the prompt -- the same confound the
axis fell to (a probe scores 0.83 on the FIRST fifth of a trajectory, before the agent has
meaningfully acted).

This script asks the question with difficulty held fixed. For each layer:

  leave-one-PAIR-out: fit an L2 logistic probe on the 17 remaining pairs (34 trajectories),
  score the held-out pair's two trajectories, and record whether the resolved arm scores
  above the unresolved one.

That yields one held-out binary outcome per pair per layer -> exact sign test. Because the evaluation
is a within-pair comparison, task difficulty cancels in the test itself, so a probe cannot
score by recognising the task.

POWER AND DIMENSIONALITY. 34 training points in 5120 dimensions are always linearly
separable, so the training fit is meaningless and only the held-out comparison counts.
Regularisation is swept (C) rather than tuned, since tuning on the same pairs used for
evaluation would leak. With 50 pairs an exact sign test needs 33 to reach p<0.05, so this is
a coarse instrument: it can detect a strong decodable signal, not a weak one. A null here
means "no strongly decodable within-task direction", not "no information".

Writes data/paired_probe_ceiling.csv/.json.

Reproduce: python scripts/paired_probe_ceiling.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from prognosis.analysis.pairs import leave_one_pair_out, trajectory_features

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data"


def label_flip_control(X: np.ndarray, meta: pd.DataFrame, C: float, n_trials: int,
                       seed: int = 0) -> dict:
    """Negative control: flip the resolved/unresolved labels within a random subset of pairs.

    A perfect score at every layer -- including near-embedding layers that cannot plausibly
    encode task outcome -- is exactly what a leaky pipeline produces, so it must be ruled out
    rather than assumed away. With labels scrambled, an honest pipeline scores near chance
    (half the pairs). If it still scores high, the procedure is reading the labels' own structure and
    the headline number is worthless.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_trials):
        mm = meta.copy()
        for t in mm["task_id"].unique():
            if rng.random() < 0.5:
                idx = mm["task_id"] == t
                mm.loc[idx, "outcome"] = 1 - mm.loc[idx, "outcome"]
        scores.append(leave_one_pair_out(X, mm, C)["n_correct"])
    return {"n_trials": n_trials, "scores": scores, "mean": float(np.mean(scores)),
            "n_pairs": int(meta["task_id"].nunique())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", type=Path, default=ROOT / "data" / "pairs50_meanpool.npz")
    ap.add_argument("--C", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0])
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Captured-block indices (0..63). Default: all.")
    ap.add_argument("--control-trials", type=int, default=5,
                    help="Label-flip negative-control trials at the best cell.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, meta = trajectory_features(args.activations)
    layers = args.layers if args.layers is not None else sorted(meta["layer"].unique())
    print(f"{X.shape[0]} trajectory-layer feature rows, hidden={X.shape[1]}, "
          f"{meta['task_id'].nunique()} tasks", flush=True)

    rows = []
    for C in args.C:
        for L in layers:
            m = meta["layer"] == L
            r = leave_one_pair_out(X[m.to_numpy()], meta.loc[m].reset_index(drop=True), C)
            r.update({"layer_captured_block": int(L), "layer_new_axis": int(L) + 1, "C": C})
            rows.append(r)
        best = max((x for x in rows if x["C"] == C), key=lambda x: x["frac_correct"])
        print(f"  C={C:<6} best layer {best['layer_new_axis']:>2} (new-axis): "
              f"{best['n_correct']}/{best['n_pairs']} pairs correct, sign p={best['sign_p']:.4f}",
              flush=True)

    df = pd.DataFrame(rows)[
        ["C", "layer_captured_block", "layer_new_axis", "n_pairs", "n_correct",
         "frac_correct", "sign_p"]]
    csv = OUT_DIR / "paired_probe_ceiling.csv"
    df.to_csv(csv, index=False)

    # Negative control at the best cell.
    best_row = df.loc[df["frac_correct"].idxmax()]
    bm = (meta["layer"] == int(best_row["layer_captured_block"])).to_numpy()
    control = label_flip_control(X[bm], meta.loc[bm].reset_index(drop=True),
                                 float(best_row["C"]), args.control_trials)
    # Chance and the pass threshold scale with the number of pairs, so both are derived rather
    # than hardcoded. Each pair is flipped with probability one half, so an honest pipeline
    # lands at n/2 with binomial SD sqrt(n)/2, and two SD above that is the threshold.
    n_ctrl = control["n_pairs"]
    chance = n_ctrl / 2
    leak_at = chance + 2 * (n_ctrl ** 0.5) / 2
    print(f"\nlabel-flip control at the best cell: {control['scores']} "
          f"(mean {control['mean']:.1f}/{n_ctrl}, chance {chance:.0f}, "
          f"leak threshold {leak_at:.1f}) -> "
          f"{'PASS, no leak' if control['mean'] < leak_at else 'FAIL, pipeline leaks'}",
          flush=True)

    n_pairs = int(df["n_pairs"].max())
    need = next((k for k in range(n_pairs // 2, n_pairs + 1)
                 if binomtest(k, n_pairs, 0.5).pvalue < 0.05), None)
    any_sig = df[df["sign_p"] < 0.05]
    summary = {
        "n_pairs": n_pairs,
        "pairs_needed_for_p05": need,
        "n_layer_C_cells": int(len(df)),
        "n_cells_p_lt_05": int(len(any_sig)),
        "expected_false_positives_at_5pct": round(0.05 * len(df), 1),
        "best_cell": df.loc[df["frac_correct"].idxmax()].to_dict(),
        "label_flip_control": control,
        "interpretation": (
            "A probe fit on other pairs and evaluated by a within-pair ranking cannot exploit "
            "task difficulty, because difficulty is identical across the two arms it compares. "
            "If no cell beats chance, outcome information is not strongly linearly decodable "
            "from these activations once task identity is controlled -- a stronger negative "
            "than the frozen axis failing on its own, and it means the full-benchmark probe "
            "ceiling (max 0.913) is itself largely a task-difficulty read."
        ),
    }
    (OUT_DIR / "paired_probe_ceiling.json").write_text(json.dumps(summary, indent=2, default=str),
                                                       encoding="utf-8")
    print(f"\n{n_pairs} pairs; sign test needs {need} correct for p<0.05")
    print(f"cells with p<0.05: {len(any_sig)}/{len(df)} "
          f"(expected ~{0.05*len(df):.0f} by chance)")
    b = summary["best_cell"]
    print(f"best cell: layer {b['layer_new_axis']} (new-axis), C={b['C']}, "
          f"{b['n_correct']}/{b['n_pairs']} correct, p={b['sign_p']:.4f}")
    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
