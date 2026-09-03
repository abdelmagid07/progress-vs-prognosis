#!/usr/bin/env python
"""Does the within-task probe still work once response length is removed?

The opening-move analysis (paired_opening_move_features.py) showed that a probe built from
seven surface text features -- dominated by how long the agent reasons before acting -- matches
the activation probe's 16/18 at k=1. That is consistent with two very different readings:

  (a) the activations carry outcome information, and length is merely one visible correlate
      of it, or
  (b) the activation probe is reading length and nothing else.

This script separates them by removing length from the activations and re-running the identical
leave-one-pair-out probe. Three arms are compared on the same folds:

  raw          the probe as reported in the paper
  residualised the same probe on activations with the length covariate regressed out
  length-only  a probe on the length covariate alone, as a floor

If the residualised probe holds up, (a) is favoured and the paper's claim survives. If it
collapses to the length-only floor, (b) is favoured and the honest statement is that the early
signal is verbosity.

LEAKAGE. The length->activation regression is fit INSIDE each fold, on the 34 training
trajectories only, and then applied to the held-out pair. Residualising globally before the
split would let the held-out pair's own length inform its features, which is the same class of
error as tuning C on the evaluation set.

Writes data/paired_length_controlled_probe.json.

Reproduce: python scripts/paired_length_controlled_probe.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
from paired_probe_ceiling import trajectory_features  # noqa: E402
from paired_direction_control import load, traj_vectors  # noqa: E402

OUT_DIR = ROOT / "results" / "numbers" / "new_axis"


def residualise(X_tr, L_tr, X_all, L_all):
    """Regress the covariate out of every activation dimension, fitting on train only."""
    Lc = L_tr - L_tr.mean()
    denom = float(Lc @ Lc)
    if denom < 1e-12:
        return X_all
    b = (Lc @ (X_tr - X_tr.mean(axis=0))) / denom          # slope per dimension
    a = X_tr.mean(axis=0) - b * L_tr.mean()
    return X_all - (a + np.outer(L_all, b))


def loo_rank(X, meta, L, C, mode):
    """Leave-one-pair-out; score = does the resolved arm outrank the unresolved one?"""
    tasks = sorted(meta["task_id"].unique())
    correct = usable = 0
    for held in tasks:
        te = (meta["task_id"] == held).to_numpy()
        tr = ~te
        if meta.loc[te, "outcome"].nunique() < 2:
            continue
        if mode == "residualised":
            Z = residualise(X[tr], L[tr], X, L)
        elif mode == "length_only":
            Z = L.reshape(-1, 1)
        else:
            Z = X
        sc = StandardScaler().fit(Z[tr])
        clf = LogisticRegression(C=C, max_iter=2000).fit(
            sc.transform(Z[tr]), meta.loc[tr, "outcome"].to_numpy())
        s = clf.decision_function(sc.transform(Z[te]))
        sub = meta.loc[te]
        usable += 1
        correct += int(s[(sub["outcome"] == 1).to_numpy()].mean()
                       > s[(sub["outcome"] == 0).to_numpy()].mean())
    p = binomtest(correct, usable, 0.5).pvalue if usable else float("nan")
    return {"n_correct": correct, "n_pairs": usable, "sign_p": float(p)}


def covariates(meta: pd.DataFrame,
               projections: Path | None = None) -> dict[str, np.ndarray]:
    """Length covariates aligned to meta's row order, one value per trajectory.

    ``projections`` must describe the SAME trajectories as the activations. It was hardcoded to
    the 18-pair file, which made the 50-pair run fail on a missing-covariate assertion rather
    than silently mismatch, but the fix is to pass the matching projections explicitly.
    """
    feats = pd.read_csv(OUT_DIR / "paired_opening_move_features.csv")
    proj = pd.read_parquet(projections or ROOT / "projections_paired_new_axis.parquet")
    tok = (proj[proj.layer == proj.layer.min()]
           .groupby("trajectory_id", as_index=False).n_gen_tokens.sum())
    key = meta["trajectory_id"]
    return {
        "step0_reasoning_words": key.map(
            feats.set_index("trajectory_id").reasoning_len_words).to_numpy(float),
        "total_generated_tokens": key.map(
            tok.set_index("trajectory_id").n_gen_tokens).to_numpy(float),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", type=Path, default=ROOT / "paired_meanpool_32b.npz")
    ap.add_argument("--layer", type=int, default=53, help="new-axis layer for the headline cell")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--sweep", action="store_true", help="also sweep all 64 layers")
    ap.add_argument("--projections", type=Path, default=None,
                    help="projections parquet matching --activations; defaults to the 18-pair file")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    block = args.layer - 1

    # ---- readout 1: all steps (the headline within-task probe) --------------
    Xa, meta_a = trajectory_features(args.activations)
    m = (meta_a["layer"] == block).to_numpy()
    Xall, meta_all = Xa[m], meta_a.loc[m].reset_index(drop=True)
    cov_all = covariates(meta_all, args.projections)

    # ---- readout 2: first step only (where the verbosity claim lives) -------
    acts, mm = load(args.activations, block)
    Xk1, meta_k1 = traj_vectors(acts, mm, max_steps=1)
    cov_k1 = covariates(meta_k1, args.projections)

    results = {}
    for name, X, meta, cov, covname in (
        ("all_steps", Xall, meta_all, cov_all, "total_generated_tokens"),
        ("first_step", Xk1, meta_k1, cov_k1, "step0_reasoning_words"),
    ):
        L = cov[covname]
        assert not np.isnan(L).any(), f"missing covariate for {name}"
        arms = {mode: loo_rank(X, meta, L, args.C, mode)
                for mode in ("raw", "residualised", "length_only")}
        results[name] = {"covariate": covname, **arms}
        print(f"\n{name}  (layer {args.layer}, C={args.C}, covariate = {covname})")
        for mode, r in arms.items():
            print(f"  {mode:<13s} {r['n_correct']:2d}/{r['n_pairs']}  p={r['sign_p']:.4f}")

    # ---- optional: is the headline cell representative across layers? -------
    sweep = None
    if args.sweep:
        rows = []
        for L_ax in range(1, 65):
            mm2 = (meta_a["layer"] == L_ax - 1).to_numpy()
            Xl, ml = Xa[mm2], meta_a.loc[mm2].reset_index(drop=True)
            Lv = covariates(ml, args.projections)["total_generated_tokens"]
            rows.append({"layer_new_axis": L_ax,
                         "raw": loo_rank(Xl, ml, Lv, args.C, "raw")["n_correct"],
                         "residualised": loo_rank(Xl, ml, Lv, args.C,
                                                  "residualised")["n_correct"]})
        sweep = rows
        raw = np.array([r["raw"] for r in rows])
        res = np.array([r["residualised"] for r in rows])
        print(f"\nlayer sweep (C={args.C}, all steps): "
              f"raw median {np.median(raw):.0f}, residualised median {np.median(res):.0f}; "
              f"layers >=14 correct: raw {int((raw >= 14).sum())}/64, "
              f"residualised {int((res >= 14).sum())}/64")

    out = {
        "layer_new_axis": args.layer,
        "C": args.C,
        "readouts": results,
        "layer_sweep": sweep,
        "interpretation": (
            "If the residualised arm stays near the raw arm, the activations carry "
            "outcome information beyond response length and the within-task probe result is "
            "not merely a verbosity effect. If it falls to the length-only arm, the honest "
            "reading is that the probe is reading length. The length-only arm is the floor, "
            "not a competitor: it uses one scalar per trajectory."
        ),
    }
    path = OUT_DIR / "paired_length_controlled_probe.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
