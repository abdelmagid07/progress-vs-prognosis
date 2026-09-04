#!/usr/bin/env python
"""Put a paired projection table into the layer numbering the paired analyses expect.

Two conventions exist and they differ by one, silently rather than loudly.

``project_steps.py`` writes ``layer`` in captured-block numbering, 0..63. The paired analyses
are written against axis-row numbering, 1..64, with a ``layer_captured_block`` column holding
0..63 alongside: they filter ``df["layer"] == 53`` meaning axis row 53, that is captured block
52. The npz path reaches the same place from the other side, filtering
``meta["layer"] == args.layer - 1`` on block numbering, so the npz needs no change.

Left unconverted, ``--layer 53`` reads block 53, which is axis row 54: an off-by-one that
produces plausible numbers at the wrong depth.

Idempotent: re-running on an already-converted file changes nothing.

Reproduce: python scripts/normalize_pair_layers.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path, default=ROOT / "data" / "projections_pairs50.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.path)
    lo, hi = int(df.layer.min()), int(df.layer.max())

    if "layer_captured_block" in df.columns and (lo, hi) == (1, 64):
        print(f"already normalized: layer {lo}..{hi} with layer_captured_block present")
        return
    if (lo, hi) != (0, 63):
        raise SystemExit(f"unexpected layer range {lo}..{hi}; refusing to guess the convention")

    df.insert(df.columns.get_loc("layer"), "layer_captured_block", df["layer"].astype(int))
    df["layer"] = df["layer_captured_block"] + 1
    print(f"shifted layer 0..63 -> {df.layer.min()}..{df.layer.max()}, "
          f"kept blocks in layer_captured_block")

    df.to_parquet(args.path, index=False)
    print(f"wrote {args.path}")


if __name__ == "__main__":
    main()
