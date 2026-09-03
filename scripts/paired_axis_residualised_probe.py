#!/usr/bin/env python
"""Does within-task outcome decoding survive removing the frozen value axis from the activations?

paired_direction_geometry.py answers this geometrically: the fitted outcome direction and the
frozen axis are at 89.9 degrees at layer 53, |cos| = 0.0016. That implies removing the axis should
cost the probe nothing, but it states the result as an angle between two vectors, which is a claim
about directions rather than about decoding. This script makes the same point directly: delete the
axis component from every activation and refit the identical leave-one-pair-out probe.

Three arms on the same folds:

  raw                the probe as reported in the paper
  axis-residualised  activations with their component along v^(l) projected out
  axis only          the frozen axis score alone, as the floor -- this is the paper's 4/18

NO LEAKAGE CONCERN, unlike the length control. The length residualisation in
paired_length_controlled_probe.py must be fitted inside each fold because the length->activation
slope is ESTIMATED from data. The value axis is not estimated here at all: it is a fixed vector
built from single-turn ICRL conversations before any agentic data existed, so projecting it out is
a deterministic transform of each row independently and cannot carry information between folds.

WHAT A NULL RESULT WOULD MEAN. If decoding collapsed after removing one direction out of 5120, the
probe and the axis would be reading the same thing and the paper's central contrast would be an
artifact of how each is scored. That is the failure mode being tested for.

Writes data/paired_axis_residualised_probe.json.

Reproduce: python scripts/paired_axis_residualised_probe.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prognosis.analysis.pairs import trajectory_features

OUT_DIR = ROOT / "data"


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n else v


def deflate(X, v):
    """Remove the component along unit vector v from every row: rank-1 deflation."""
    return X - np.outer(X @ v, v)


def loo_rank(X, meta, C):
    """Leave-one-pair-out; does the resolved arm outrank the unresolved one?"""
    correct = usable = 0
    for held in sorted(meta["task_id"].unique()):
        te = (meta["task_id"] == held).to_numpy()
        tr = ~te
        if meta.loc[te, "outcome"].nunique() < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000).fit(
            sc.transform(X[tr]), meta.loc[tr, "outcome"].to_numpy())
        s = clf.decision_function(sc.transform(X[te]))
        sub = meta.loc[te]
        usable += 1
        correct += int(s[(sub["outcome"] == 1).to_numpy()].mean()
                       > s[(sub["outcome"] == 0).to_numpy()].mean())
    return {"n_correct": correct, "n_pairs": usable,
            "sign_p": float(binomtest(correct, usable, 0.5).pvalue) if usable else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", type=Path, default=ROOT / "data" / "paired_meanpool.npz")
    ap.add_argument("--axis", type=Path,
                    default=ROOT / "data" / "value_axis.npy")
    ap.add_argument("--layer", type=int, default=53)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    axis_all = np.load(args.axis).astype(np.float64)
    Xa, meta_a = trajectory_features(args.activations)

    def arms_at(layer):
        m = (meta_a["layer"] == layer - 1).to_numpy()
        X, meta = Xa[m].astype(np.float64), meta_a.loc[m].reset_index(drop=True)
        v = unit(axis_all[layer])
        proj = (X @ v).reshape(-1, 1)
        return {
            "raw": loo_rank(X, meta, args.C),
            "axis_residualised": loo_rank(deflate(X, v), meta, args.C),
            "axis_only": loo_rank(proj, meta, args.C),
            "residual_cosine_with_axis": float(np.abs(deflate(X, v) @ v).max()),
        }

    res = arms_at(args.layer)
    out = {"layer_new_axis": args.layer, "C": args.C, "n_dims": int(Xa.shape[1]),
           "headline": res}

    print(f"layer {args.layer}, C={args.C}, {Xa.shape[1]} dimensions")
    for k in ("raw", "axis_residualised", "axis_only"):
        r = res[k]
        print(f"  {k:<20s} {r['n_correct']:2d}/{r['n_pairs']}  p={r['sign_p']:.4f}")
    print(f"  (max |residual . axis| after deflation: {res['residual_cosine_with_axis']:.2e})")

    if args.sweep:
        rows = []
        for L in range(1, 65):
            a = arms_at(L)
            rows.append({"layer_new_axis": L,
                         "raw": a["raw"]["n_correct"],
                         "axis_residualised": a["axis_residualised"]["n_correct"]})
        out["layer_sweep"] = rows
        raw = np.array([r["raw"] for r in rows])
        dfl = np.array([r["axis_residualised"] for r in rows])
        out["sweep_summary"] = {"raw_median": float(np.median(raw)),
                                "residualised_median": float(np.median(dfl)),
                                "layers_where_residualised_lower": int((dfl < raw).sum())}
        print(f"\nsweep: raw median {np.median(raw):.0f}/18, "
              f"residualised median {np.median(dfl):.0f}/18, "
              f"lower in {(dfl < raw).sum()} of 64 layers")

    path = OUT_DIR / "paired_axis_residualised_probe.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
