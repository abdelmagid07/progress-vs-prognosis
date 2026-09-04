#!/usr/bin/env python
"""Does the within-task probe survive removing EVERY observable feature we can measure?

paired_length_controlled_probe.py answers this for one covariate at a time -- generated token
count, or step-0 reasoning length. That is the weaker form of the question. A reader can always
name another surface property the probe might be reading instead, and answering them one at a
time never closes the list.

This script closes it. It regresses a whole block of observable trajectory features out of the
activations at once and refits the identical leave-one-pair-out probe. If the probe holds up
after every measurable surface property is removed simultaneously, "it is really reading
verbosity/hedging/command style" stops being available as an explanation.

WHY THIS MATTERS. Within these pairs trajectory length is not balanced: the resolved arm is
shorter in 31 of 50 pairs and length reaches AUROC 0.682, close to the 0.780 it reaches on
the full run. Length is therefore a live confound in this set rather
than a hypothetical one, and a control that removes it -- along with everything else observable --
is doing real work.

THREE ARMS on the same folds:

  raw               the probe as reported in the paper
  residualised      activations with the observable block regressed out
  observable_only   a probe on the observable block alone, as the floor

LEAKAGE. The observable -> activation regression is fitted INSIDE each fold, on the training
trajectories only, then applied to the held-out pair. Residualising globally before the split
would let the held-out pair's own features inform its activations, which is the same class of
error as tuning C on the evaluation set. The covariates are also standardised using training
statistics only, for conditioning rather than for correctness.

Writes data/paired_observable_controlled_probe.json.

Reproduce: python scripts/paired_observable_controlled_probe.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prognosis.analysis.pairs import trajectory_features
from prognosis.analysis.pairs import load, traj_vectors

OUT_DIR = ROOT / "data"

# The observable block. Step-0 text features come from paired_opening_move_features.csv; the two
# trajectory-scale features are derived from the projections. Together these are every surface
# property the repo measures: how much the agent writes, how long it runs, how it hedges, how
# complex its first command is, and whether that command modifies the repo or only reads it.
STEP0_FEATURES = [
    "resp_len_words",
    "reasoning_len_words",
    "thought_len_words",
    "command_len_chars",
    "n_chained_cmds",
    "hedge_count",
    "is_modifying",
]
TRAJ_FEATURES = ["total_generated_tokens", "n_steps"]


def observables(meta: pd.DataFrame, projections: Path,
                step0_only: bool = False) -> tuple[np.ndarray, list[str]]:
    """Observable covariates aligned to meta's row order, one row per trajectory.

    ``step0_only`` drops the two trajectory-scale covariates. It exists for the first-step
    readout, where the full block is the WRONG control: total generated tokens and step count
    are properties of the finished run, so residualising a step-0 probe against them removes
    information the probe could not have had, and the arm no longer answers the question it
    looks like it answers. The seven step-0 features are all measurable from the agent's first
    response alone.
    """
    feats = pd.read_csv(OUT_DIR / "paired_opening_move_features.csv").set_index("trajectory_id")
    proj = pd.read_parquet(projections, columns=["trajectory_id", "layer", "step_index",
                                                 "n_gen_tokens"])
    proj = proj[proj.layer == proj.layer.min()]
    per_traj = proj.groupby("trajectory_id").agg(
        total_generated_tokens=("n_gen_tokens", "sum"),
        n_steps=("step_index", "size"),
    )

    key = meta["trajectory_id"]
    cols = {name: key.map(feats[name]).to_numpy(float) for name in STEP0_FEATURES}
    cols.update({name: key.map(per_traj[name]).to_numpy(float) for name in TRAJ_FEATURES})

    names = STEP0_FEATURES if step0_only else STEP0_FEATURES + TRAJ_FEATURES
    Z = np.column_stack([cols[n] for n in names])
    missing = np.isnan(Z).any(axis=0)
    if missing.any():
        raise SystemExit(f"missing covariates: {[n for n, m in zip(names, missing) if m]}")
    return Z, names


def residualise(X_tr, Z_tr, X_all, Z_all):
    """Regress the whole covariate block out of every activation dimension, fitting on train only.

    Per-dimension OLS with an intercept, solved once for all 5120 dimensions as a single
    least-squares problem. Rank deficiency is expected and harmless -- resp_len_words is nearly
    the sum of reasoning and thought lengths -- and lstsq's SVD handles it by taking the minimum
    norm solution rather than failing.
    """
    mu, sd = Z_tr.mean(axis=0), Z_tr.std(axis=0)
    sd[sd < 1e-12] = 1.0                       # a constant covariate carries no information
    A_tr = np.column_stack([np.ones(len(Z_tr)), (Z_tr - mu) / sd])
    A_all = np.column_stack([np.ones(len(Z_all)), (Z_all - mu) / sd])
    B, *_ = np.linalg.lstsq(A_tr, X_tr, rcond=None)
    return X_all - A_all @ B


def loo_rank(X, meta, Z, C, mode):
    """Leave-one-pair-out; score = does the resolved arm outrank the unresolved one?"""
    tasks = sorted(meta["task_id"].unique())
    correct = usable = 0
    for held in tasks:
        te = (meta["task_id"] == held).to_numpy()
        tr = ~te
        if meta.loc[te, "outcome"].nunique() < 2:
            continue
        if mode == "residualised":
            V = residualise(X[tr], Z[tr], X, Z)
        elif mode == "observable_only":
            V = Z
        else:
            V = X
        sc = StandardScaler().fit(V[tr])
        clf = LogisticRegression(C=C, max_iter=2000).fit(
            sc.transform(V[tr]), meta.loc[tr, "outcome"].to_numpy())
        s = clf.decision_function(sc.transform(V[te]))
        sub = meta.loc[te]
        usable += 1
        correct += int(s[(sub["outcome"] == 1).to_numpy()].mean()
                       > s[(sub["outcome"] == 0).to_numpy()].mean())
    p = binomtest(correct, usable, 0.5).pvalue if usable else float("nan")
    return {"n_correct": correct, "n_pairs": usable, "sign_p": float(p)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # The activations and the projections table must describe the same trajectories: a
    # mismatch produces a complete, plausible, silently wrong result rather than an
    # error.
    ap.add_argument("--activations", type=Path, default=ROOT / "data" / "pairs50_meanpool.npz")
    ap.add_argument("--projections", type=Path,
                    default=ROOT / "data" / "projections_pairs50.parquet")
    ap.add_argument("--layer", type=int, default=53, help="new-axis layer for the headline cell")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--step0-covariates", action=argparse.BooleanOptionalAction, default=True,
                    help="restrict the first-step readout to the seven step-0 features")
    ap.add_argument("--drop-one", action="store_true",
                    help="also report each covariate removed on its own, to see which matters")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    block = args.layer - 1

    # ---- readout 1: all steps (the headline within-task probe) --------------
    Xa, meta_a = trajectory_features(args.activations)
    m = (meta_a["layer"] == block).to_numpy()
    Xall, meta_all = Xa[m].astype(np.float64), meta_a.loc[m].reset_index(drop=True)
    Z_all, names = observables(meta_all, args.projections)

    # ---- readout 2: first step only (where a style explanation is most likely) ---
    acts, mm = load(args.activations, block)
    Xk1, meta_k1 = traj_vectors(acts, mm, max_steps=1)
    Z_k1, names_k1 = observables(meta_k1, args.projections,
                                 step0_only=args.step0_covariates)

    results = {}
    for name, X, meta, Z in (("all_steps", Xall, meta_all, Z_all),
                             ("first_step", Xk1.astype(np.float64), meta_k1, Z_k1)):
        arms = {mode: loo_rank(X, meta, Z, args.C, mode)
                for mode in ("raw", "residualised", "observable_only")}
        results[name] = arms
        k = len(names_k1) if name == "first_step" else len(names)
        print(f"\n{name}  (layer {args.layer}, C={args.C}, {k} covariates)")
        for mode, r in arms.items():
            print(f"  {mode:<16s} {r['n_correct']:2d}/{r['n_pairs']}  p={r['sign_p']:.4g}")

    # ---- optional: which single covariate carries the most? -----------------
    drop_one = None
    if args.drop_one:
        drop_one = []
        for i, nm in enumerate(names):
            r = loo_rank(Xall, meta_all, Z_all[:, [i]], args.C, "residualised")
            o = loo_rank(Xall, meta_all, Z_all[:, [i]], args.C, "observable_only")
            drop_one.append({"covariate": nm, "residualised": r, "observable_only": o})
            print(f"  only {nm:<24s} residualised {r['n_correct']}/{r['n_pairs']}, "
                  f"alone {o['n_correct']}/{o['n_pairs']}")

    out = {
        "axis_layer": args.layer,
        "C": args.C,
        "covariates": names,
        "n_covariates": len(names),
        "first_step_covariates": names_k1,
        "readouts": results,
        "drop_one": drop_one,
        "interpretation": (
            "The residualised arm removes every observable trajectory feature the repo measures "
            "from the activations at once, refitting the regression inside each fold. If it "
            "stays at the raw arm, no surface property in this block explains the within-task "
            "probe, and the one-covariate-at-a-time controls are subsumed. If it falls toward "
            "the observable-only floor, the probe is reading surface form. The floor is not a "
            "competitor: it uses nine scalars per trajectory against 5120 dimensions."
        ),
    }
    path = OUT_DIR / "paired_observable_controlled_probe.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
