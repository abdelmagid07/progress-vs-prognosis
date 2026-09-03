#!/usr/bin/env python
"""What IS the within-task outcome direction, and how does it sit relative to the value axis?

The paper establishes that outcome information is linearly decodable within a task while the
frozen axis does not capture it. That leaves three things unmeasured, all of which a reviewer
can reasonably ask about given 34 training points in 5120 dimensions:

  1. STABILITY. Is there one shared outcome direction, or are 18 pairs separable for 18
     idiosyncratic reasons? Tested by splitting the pairs into DISJOINT halves, fitting a
     direction on each, and measuring the angle between them. Disjoint is essential: the 18
     leave-one-out directions share 16 of 17 training pairs, so their mutual agreement is
     guaranteed by data overlap and says nothing.

  2. ALIGNMENT. "A failure of alignment, not of information" is currently a rhetorical claim
     supported only indirectly, through the random-direction control. Here we measure the angle
     between the fitted direction and the frozen axis directly, against the null of a random
     direction of matched dimensionality.

  3. WHAT A POOLED PROBE WOULD LEARN. A probe fit across tasks can score by recognising the
     task, so it learns directions that live in the BETWEEN-TASK subspace. If the within-task
     outcome direction is largely orthogonal to that subspace, the two are different signals and
     the confound is a property of pooled training rather than of this particular axis.

     (The direct version of 3 -- fitting a probe on the 499 pooled trajectories -- needs the
     full-run activation cache, which is not in the repository; only the 36 paired trajectories
     are stored. This subspace decomposition asks the same question from the data we have.)

GEOMETRY NOTE. The probe is fit on standardised features, so its weight vector lives in
standardised space. The direction in the original activation space is w / sigma, and that is
what is compared against the axis; using the raw weights would compare vectors in two different
coordinate systems.

Writes data/paired_direction_geometry.json.

Reproduce: python scripts/paired_direction_geometry.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prognosis.analysis.pairs import trajectory_features

OUT_DIR = ROOT / "data"


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def fit_direction(X, y, C):
    """Return the decision direction expressed in ORIGINAL feature space."""
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    return unit(clf.coef_.ravel() / sc.scale_)


def rank_pairs(X, meta, w):
    """Fraction of pairs whose resolved arm projects higher on w."""
    s = X @ w
    t = meta.assign(s=s).pivot_table(index="task_id", columns="outcome", values="s")
    d = (t[1] - t[0]).dropna()
    return int((d > 0).sum()), int(len(d))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", type=Path, default=ROOT / "data" / "paired_meanpool.npz")
    ap.add_argument("--axis", type=Path,
                    default=ROOT / "data" / "value_axis.npy")
    ap.add_argument("--layer", type=int, default=53)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--n-random", type=int, default=2000)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_all, meta_all = trajectory_features(args.activations)
    m = (meta_all["layer"] == args.layer - 1).to_numpy()
    X, meta = X_all[m], meta_all.loc[m].reset_index(drop=True)
    y = meta["outcome"].to_numpy()
    tasks = np.array(sorted(meta["task_id"].unique()))
    d = X.shape[1]
    rng = np.random.default_rng(0)
    print(f"{X.shape[0]} trajectories, {len(tasks)} tasks, {d} dims, layer {args.layer}")

    # ---- 1. stability: disjoint-half agreement ----------------------------
    cos_half, cross_acc = [], []
    for _ in range(args.n_splits):
        perm = rng.permutation(tasks)
        A, B = set(perm[: len(tasks) // 2]), set(perm[len(tasks) // 2:])
        ia = meta["task_id"].isin(A).to_numpy()
        ib = meta["task_id"].isin(B).to_numpy()
        wa = fit_direction(X[ia], y[ia], args.C)
        wb = fit_direction(X[ib], y[ib], args.C)
        cos_half.append(float(wa @ wb))
        # each half's direction scored on the OTHER half's pairs
        for w, idx in ((wa, ib), (wb, ia)):
            k, n = rank_pairs(X[idx], meta.loc[idx].reset_index(drop=True), w)
            cross_acc.append(k / n)
    cos_half = np.array(cos_half)
    cross_acc = np.array(cross_acc)

    # null: two independent random unit directions in d dimensions
    r1 = rng.normal(size=(args.n_splits, d)); r2 = rng.normal(size=(args.n_splits, d))
    cos_null = np.abs(np.sum(r1 / np.linalg.norm(r1, axis=1, keepdims=True)
                             * r2 / np.linalg.norm(r2, axis=1, keepdims=True), axis=1))

    # ---- 2. alignment with the frozen value axis --------------------------
    w_full = fit_direction(X, y, args.C)
    axis = unit(np.load(args.axis)[args.layer].astype(np.float64))
    cos_axis = float(abs(w_full @ axis))
    rr = rng.normal(size=(args.n_random, d))
    rr /= np.linalg.norm(rr, axis=1, keepdims=True)
    cos_axis_null = np.abs(rr @ axis)

    # ---- 3. how much of the direction lies in the between-task subspace ----
    task_mean = np.vstack([X[(meta["task_id"] == t).to_numpy()].mean(axis=0) for t in tasks])
    Bc = task_mean - task_mean.mean(axis=0)
    U, S, _ = np.linalg.svd(Bc, full_matrices=False)
    # right singular vectors span the between-task subspace
    _, _, Vt = np.linalg.svd(Bc, full_matrices=False)
    energy = {}
    for k in (1, 3, 5, 10, 17):
        k = min(k, Vt.shape[0])
        proj = Vt[:k] @ w_full
        energy[k] = {"fitted_direction": float(proj @ proj),
                     "random_expected": k / d,
                     "random_observed": float(np.mean(np.sum((rr @ Vt[:k].T) ** 2, axis=1)))}

    out = {
        "layer_new_axis": args.layer, "C": args.C, "n_dims": int(d),
        "stability_disjoint_halves": {
            "n_splits": args.n_splits,
            "cosine_mean": float(cos_half.mean()), "cosine_median": float(np.median(cos_half)),
            "cosine_min": float(cos_half.min()), "cosine_max": float(cos_half.max()),
            "null_cosine_mean_abs": float(cos_null.mean()),
            "null_cosine_p95_abs": float(np.percentile(cos_null, 95)),
            "cross_half_pair_accuracy_mean": float(cross_acc.mean()),
            "cross_half_pair_accuracy_min": float(cross_acc.min()),
            # Raw draws, so the figure can show the two DISTRIBUTIONS rather than two means.
            # Summary statistics alone cannot answer the question the panel exists to settle --
            # whether the observed agreement and the null overlap at all -- and recomputing them
            # in a plotting script would put the same analysis in two places.
            "cosine_samples": [float(v) for v in cos_half],
            "null_cosine_samples": [float(v) for v in cos_null],
            "cross_half_pair_accuracy_samples": [float(v) for v in cross_acc],
        },
        "alignment_with_value_axis": {
            "abs_cosine": cos_axis,
            "angle_degrees": float(np.degrees(np.arccos(min(cos_axis, 1.0)))),
            "null_abs_cosine_mean": float(cos_axis_null.mean()),
            "null_abs_cosine_p95": float(np.percentile(cos_axis_null, 95)),
            "percentile_vs_null": float((cos_axis_null < cos_axis).mean()),
        },
        "between_task_subspace_energy": energy,
        "interpretation": (
            "Disjoint halves share no training pairs, so their agreement is evidence for a "
            "single shared direction rather than an artifact of overlapping folds. Alignment "
            "with the value axis is reported against the null of a random direction of matched "
            "dimensionality. Subspace energy is the fraction of the fitted direction lying in "
            "the span of the top-k between-task principal directions; a random direction places "
            "k/d of its energy there, so a value near that baseline means the within-task "
            "direction is not what a pooled, task-recognising probe would learn."
        ),
    }
    # sweep_layers was defined but never called: the key it produces was present in a stored
    # JSON from an older revision and silently disappeared the next time this script ran, taking
    # the per-layer alignment data with it. Wired in here so the artifact is reproducible from
    # the script that claims to write it.
    out["alignment_layer_sweep"] = sweep_layers(args.activations, args.axis, args.C)

    path = OUT_DIR / "paired_direction_geometry.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    s = out["stability_disjoint_halves"]
    a = out["alignment_with_value_axis"]
    print(f"\n1. STABILITY (disjoint 9+9 halves, {args.n_splits} splits)")
    print(f"   cosine between half-directions: mean {s['cosine_mean']:+.3f} "
          f"(range {s['cosine_min']:+.3f}..{s['cosine_max']:+.3f})")
    print(f"   null (random pairs of directions): mean |cos| {s['null_cosine_mean_abs']:.4f}, "
          f"95th pct {s['null_cosine_p95_abs']:.4f}")
    print(f"   half-direction scored on the OTHER half's pairs: "
          f"mean {s['cross_half_pair_accuracy_mean']:.3f} of pairs correct")
    print(f"\n2. ALIGNMENT with frozen value axis")
    print(f"   |cos| = {a['abs_cosine']:.4f}  ({a['angle_degrees']:.1f} degrees)")
    print(f"   null |cos|: mean {a['null_abs_cosine_mean']:.4f}, "
          f"95th pct {a['null_abs_cosine_p95']:.4f} -> axis at "
          f"{a['percentile_vs_null']*100:.1f}th percentile of random")
    print(f"\n3. ENERGY IN BETWEEN-TASK SUBSPACE (what a pooled probe would learn)")
    for k, v in energy.items():
        print(f"   top-{k:<2} PCs: fitted {v['fitted_direction']*100:5.1f}%   "
              f"random {v['random_observed']*100:5.2f}%")
    print(f"\nwrote {path}")




def sweep_layers(activations: Path, axis_path: Path, C: float, n_random: int = 500) -> list:
    """Alignment between the fitted within-task direction and the frozen axis, at every layer.

    Reported alongside a per-layer null so the comparison is calibrated: |cos| between two random
    directions in 5120 dimensions is small by construction, and the question is whether the axis
    is any closer than that.
    """
    X_all, meta_all = trajectory_features(activations)
    axis_all = np.load(axis_path).astype(np.float64)
    rng = np.random.default_rng(1)
    rows = []
    for layer in range(1, 65):
        m = (meta_all["layer"] == layer - 1).to_numpy()
        X, meta = X_all[m], meta_all.loc[m].reset_index(drop=True)
        w = fit_direction(X, meta["outcome"].to_numpy(), C)
        ax = unit(axis_all[layer])
        r = rng.normal(size=(n_random, X.shape[1]))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        null = np.abs(r @ ax)
        k, n = rank_pairs(X, meta, w)
        rows.append({"layer_new_axis": layer,
                     "abs_cosine": float(abs(w @ ax)),
                     "angle_degrees": float(np.degrees(np.arccos(min(abs(w @ ax), 1.0)))),
                     # SIGNED cosine as well as absolute. The sign is the finding at layers
                     # 8-20, where the outcome direction points AGAINST the axis -- the same
                     # reversal the within-task contrast reports, visible geometrically. An
                     # absolute value hides it.
                     "signed_cosine": float(w @ ax),
                     "null_abs_cosine_mean": float(null.mean()),
                     "null_abs_cosine_p95": float(np.percentile(null, 95)),
                     "percentile_vs_null": float((null < abs(w @ ax)).mean()),
                     "insample_pairs_correct": k, "n_pairs": n})
    return rows

if __name__ == "__main__":
    main()
