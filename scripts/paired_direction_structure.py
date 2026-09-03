#!/usr/bin/env python
"""Is the within-task outcome direction one representation, and how does it relate to length?

Two questions that follow from paired_direction_geometry.py, which showed the direction is
stable across disjoint halves of the pairs and orthogonal to the frozen axis at every layer.

  1. ACROSS DEPTH. Qwen3-32B's residual stream is a single 5120-dimensional space that every
     block writes into additively, so directions read at different layers live in the same
     coordinate system and their angles are meaningful. If the outcome direction at layer 20
     and at layer 53 are near-parallel, one representation is being read at many depths; if the
     layer-wise directions are mutually orthogonal, the probe is finding a different signal at
     each depth and "the information is linearly decodable" is a weaker statement than it
     sounds.

  2. VERSUS LENGTH. paired_length_controlled_probe.py showed the probe survives regressing the
     length covariate out. That is a statement about predictive power; this is the geometric
     counterpart. We build the direction along which activations vary most with generated-token
     count and measure its angle to the outcome direction. Near-orthogonal means the two
     channels are genuinely distinct rather than one being a rescaling of the other.

The length direction is taken as the activation-length covariance vector rather than a fitted
regression: with 36 samples in 5120 dimensions any fitted coefficient vector is arbitrary in the
null space of the data, whereas the covariance vector is the well-determined part.

Writes data/paired_direction_structure.json.

Reproduce: python scripts/paired_direction_structure.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prognosis.analysis.pairs import trajectory_features
from paired_direction_geometry import fit_direction, unit, rank_pairs  # noqa: E402

OUT_DIR = ROOT / "data"


def length_covariate(meta: pd.DataFrame) -> np.ndarray:
    proj = pd.read_parquet(ROOT / "data" / "projections_paired.parquet")
    tok = (proj[proj.layer == proj.layer.min()]
           .groupby("trajectory_id", as_index=False).n_gen_tokens.sum())
    return meta["trajectory_id"].map(
        tok.set_index("trajectory_id").n_gen_tokens).to_numpy(float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", type=Path, default=ROOT / "data" / "paired_meanpool.npz")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--ref-layer", type=int, default=53)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_all, meta_all = trajectory_features(args.activations)
    d = X_all.shape[1]
    rng = np.random.default_rng(0)

    W, Lw, layers = [], [], list(range(1, 65))
    for layer in layers:
        m = (meta_all["layer"] == layer - 1).to_numpy()
        X, meta = X_all[m], meta_all.loc[m].reset_index(drop=True)
        W.append(fit_direction(X, meta["outcome"].to_numpy(), args.C))
        L = length_covariate(meta)
        Lw.append(unit((L - L.mean()) @ (X - X.mean(axis=0))))
    W = np.vstack(W)
    Lw = np.vstack(Lw)

    # null: |cos| between two independent random unit directions in d dimensions
    r1 = rng.normal(size=(4000, d)); r1 /= np.linalg.norm(r1, axis=1, keepdims=True)
    r2 = rng.normal(size=(4000, d)); r2 /= np.linalg.norm(r2, axis=1, keepdims=True)
    null = np.abs(np.sum(r1 * r2, axis=1))
    null_mean, null_p95 = float(null.mean()), float(np.percentile(null, 95))

    # ---- 1. across depth ---------------------------------------------------
    Cm = W @ W.T
    off = ~np.eye(64, dtype=bool)
    adj = np.array([Cm[i, i + 1] for i in range(63)])
    far = np.array([Cm[i, j] for i in range(64) for j in range(64) if abs(i - j) >= 20])
    ref = Cm[args.ref_layer - 1]

    # ---- 2. outcome direction versus the length direction ------------------
    cos_len = np.array([abs(float(W[i] @ Lw[i])) for i in range(64)])

    out = {
        "C": args.C, "ref_layer": args.ref_layer, "n_dims": int(d),
        "null_abs_cosine": {"mean": null_mean, "p95": null_p95},
        "across_depth": {
            "adjacent_layer_cosine_mean": float(adj.mean()),
            "adjacent_layer_cosine_min": float(adj.min()),
            "distant_layer_cosine_mean": float(far.mean()),
            "all_offdiagonal_cosine_mean": float(Cm[off].mean()),
            "frac_offdiagonal_above_null_p95": float((np.abs(Cm[off]) > null_p95).mean()),
            "ref_layer_cosine_mean": float(np.delete(ref, args.ref_layer - 1).mean()),
            "ref_layer_cosine_max_other": float(np.delete(ref, args.ref_layer - 1).max()),
        },
        "versus_length_direction": {
            "abs_cosine_mean": float(cos_len.mean()),
            "abs_cosine_median": float(np.median(cos_len)),
            "abs_cosine_max": float(cos_len.max()),
            "angle_degrees_median": float(np.degrees(np.arccos(min(np.median(cos_len), 1.0)))),
            "n_layers_above_null_p95": int((cos_len > null_p95).sum()),
        },
        "interpretation": (
            "Directions read at different layers share the residual stream's coordinate system, "
            "so cross-layer angles are interpretable. High off-diagonal cosine means one "
            "representation is being read at many depths rather than a distinct signal per "
            "layer. The length comparison is the geometric counterpart to the residualised "
            "probe: near-orthogonality means the outcome direction is not a rescaling of the "
            "verbosity channel."
        ),
    }
    path = OUT_DIR / "paired_direction_structure.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    a, v = out["across_depth"], out["versus_length_direction"]
    print(f"null |cos| between random directions: mean {null_mean:.4f}, 95th pct {null_p95:.4f}\n")
    print("1. OUTCOME DIRECTION ACROSS DEPTH (cosine between layer-wise directions)")
    print(f"   adjacent layers     mean {a['adjacent_layer_cosine_mean']:+.3f} "
          f"(min {a['adjacent_layer_cosine_min']:+.3f})")
    print(f"   >=20 layers apart   mean {a['distant_layer_cosine_mean']:+.3f}")
    print(f"   all off-diagonal    mean {a['all_offdiagonal_cosine_mean']:+.3f}; "
          f"{a['frac_offdiagonal_above_null_p95']*100:.0f}% exceed the null 95th pct")
    print(f"   layer {args.ref_layer} vs others   mean {a['ref_layer_cosine_mean']:+.3f} "
          f"(max {a['ref_layer_cosine_max_other']:+.3f})")
    print("\n2. OUTCOME DIRECTION VS LENGTH DIRECTION")
    print(f"   |cos| median {v['abs_cosine_median']:.3f} "
          f"({v['angle_degrees_median']:.1f} deg), max {v['abs_cosine_max']:.3f}")
    print(f"   layers above the random null 95th pct: {v['n_layers_above_null_p95']}/64")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
