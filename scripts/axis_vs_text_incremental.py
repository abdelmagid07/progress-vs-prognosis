#!/usr/bin/env python
"""Does the frozen axis carry anything the trajectory text does not?

text_only_baseline.py showed that simple counts over the generated text reach the frozen axis's
pooled AUROC. Matching accuracy is not the same as being the same signal: two predictors can
agree in aggregate while carrying different information. This asks the sharper question --
whether the axis has any INCREMENTAL predictive value once the text features are already in the
model.

Three comparisons, all under the same task-level 5-fold cross-validation:

  text only          the eight countable features
  axis only          the frozen-axis projection, as reported in the paper
  text + axis        both

If text+axis is indistinguishable from text alone, the axis contributes nothing beyond what can
be counted off the transcript, and the pooled transfer result is not evidence about internal
representations. If it adds, the axis carries something surface features miss -- which would be
a point in the axis's favour and should be reported as such.

We also regress the axis projection itself on the text features. The R^2 says how much of the
axis's variation across trajectories is a function of countable surface properties, independent
of outcome.

Writes data/axis_vs_text_incremental.json.

Reproduce: python scripts/axis_vs_text_incremental.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_agentic import fast_auroc, require_axis_row_layers  # noqa: E402
from text_only_baseline import GROUPS, PAPER_TRAJECTORY_FEATURES, featurise  # noqa: E402

OUT_DIR = ROOT / "data"
TEXT = sum(GROUPS.values(), [])


def cv_auroc(X, y, seed=0, n_rep=5):
    """Out-of-fold AUROC, averaged over repeats so small differences are not fold noise."""
    scores = []
    for r in range(n_rep):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(X, y):
            mdl = make_pipeline(StandardScaler(),
                                LogisticRegression(C=1.0, max_iter=2000,
                                                   class_weight="balanced"))
            mdl.fit(X[tr], y[tr])
            oof[te] = mdl.decision_function(X[te])
        scores.append(fast_auroc(y, oof))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectories", type=Path,
                    default=ROOT / "data" / "normalized")
    ap.add_argument("--projections", type=Path, default=ROOT / "data" / "projections_full.parquet")
    ap.add_argument("--layer", type=int, default=53)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [f for f in (featurise(p) for p in sorted(args.trajectories.glob("*.json"))) if f]
    txt = pd.DataFrame(rows)

    proj = require_axis_row_layers(pd.read_parquet(args.projections), args.projections)
    L = proj[proj.layer == args.layer]
    fin = (L.sort_values("step_index").groupby("trajectory_id", as_index=False).tail(1)
           [["trajectory_id", "proj_mean"]].rename(columns={"proj_mean": "axis_final"}))
    mean = (L.groupby("trajectory_id", as_index=False).proj_mean.mean()
            .rename(columns={"proj_mean": "axis_mean"}))

    df = txt.merge(fin, on="trajectory_id").merge(mean, on="trajectory_id").reset_index(drop=True)
    y = df.outcome.to_numpy()
    print(f"{len(df)} trajectories, {y.sum()} resolved\n")

    # ---- what does the axis track? ----------------------------------------
    corr = {c: {"spearman_rho": float(spearmanr(df[c], df.axis_final)[0]),
                "p": float(spearmanr(df[c], df.axis_final)[1])} for c in TEXT}
    print("axis final-step projection vs each text feature (Spearman):")
    for c, v in sorted(corr.items(), key=lambda kv: -abs(kv[1]["spearman_rho"])):
        print(f"  {c:<20s} rho {v['spearman_rho']:+.3f}  (p={v['p']:.2g})")

    Xt = df[TEXT].to_numpy(float)
    r2 = make_pipeline(StandardScaler(), LinearRegression()).fit(Xt, df.axis_final).score(
        Xt, df.axis_final)
    print(f"\nR^2 of axis projection regressed on the eight text features: {r2:.3f}")

    # ---- incremental value -------------------------------------------------
    # The four features the paper's method section names, and those four plus the axis. These are the arms the paper
    # reports, and they have to be measured under ONE protocol: text_only_baseline.py carries its
    # own cv_auroc, so quoting its number next to a combined number computed here would compare
    # two different cross-validations and manufacture a difference out of the harness.
    Xm = df[PAPER_TRAJECTORY_FEATURES].to_numpy(float)

    sets = {
        "text only":            Xt,
        "method four":          Xm,
        "method four + axis":   np.column_stack([Xm, df.axis_final.to_numpy(float)]),
        "axis only":            df[["axis_final"]].to_numpy(float),
        "axis only (mean)":     df[["axis_mean"]].to_numpy(float),
        "text + axis":          np.column_stack([Xt, df.axis_final.to_numpy(float)]),
        "text + axis (both readouts)":
            np.column_stack([Xt, df[["axis_final", "axis_mean"]].to_numpy(float)]),
    }
    fitted = {}
    print("\nheld-out AUROC (task-level 5-fold CV, mean +- sd over 5 repeats):")
    for name, X in sets.items():
        m, s = cv_auroc(X, y)
        fitted[name] = {"auroc_mean": m, "auroc_sd": s}
        print(f"  {name:<30s} {m:.3f} +- {s:.3f}")
    delta = fitted["text + axis"]["auroc_mean"] - fitted["text only"]["auroc_mean"]
    delta_method = (fitted["method four + axis"]["auroc_mean"]
                    - fitted["method four"]["auroc_mean"])
    print(f"\n  incremental value of the axis over text: {delta:+.4f} AUROC")

    out = {
        "axis_layer": args.layer, "n_trajectories": int(len(df)),
        "axis_vs_text_correlations": corr,
        "axis_explained_by_text_r2": float(r2),
        "fitted_models": fitted,
        "incremental_auroc_axis_over_text": float(delta),
        "incremental_auroc_axis_over_method_four": float(delta_method),
        "interpretation": (
            "If text+axis is within fold noise of text alone, the frozen axis carries no "
            "predictive information beyond countable surface properties of the transcript, and "
            "the pooled transfer result cannot be read as evidence about internal "
            "representations. The R^2 is reported separately from outcome prediction: it says "
            "how much of the axis's variation across trajectories is itself a function of "
            "surface text, whether or not that variation predicts anything."
        ),
    }
    path = OUT_DIR / "axis_vs_text_incremental.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
