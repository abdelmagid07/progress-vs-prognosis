#!/usr/bin/env python
"""Compare internal value-axis projection vs post-hoc verbalized P(success).

The paper Stage 4: both signals are scored against the same ground-truth
outcomes at each relative position. Uses the same task-level BCa AUROC
machinery as the transfer analyses.

Reproduce: python -m prognosis.analysis.internal_vs_elicited
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prognosis.analysis.by_position import auroc_by_position
from prognosis.analysis.stats import auroc_with_ci
from prognosis.common.config import load_defaults
from prognosis.common.paths import data_file


def _proj_col(df: pd.DataFrame) -> str:
    return "proj_mean" if "proj_mean" in df.columns else "projection"


def compare(
    projections: pd.DataFrame,
    confidence: pd.DataFrame,
    *,
    n_bins: int = 5,
    primary_layer: int | None = None,
    n_boot: int = 1000,
) -> dict:
    proj = projections.copy()
    if primary_layer is not None and "layer" in proj.columns and proj["layer"].nunique() > 1:
        proj = proj[proj["layer"] == primary_layer]

    conf = confidence.dropna(subset=["p_success"]).copy()
    # Join on trajectory + step so both signals share the same units.
    merged = proj.merge(
        conf[["trajectory_id", "step_index", "p_success"]],
        on=["trajectory_id", "step_index"],
        how="inner",
    )
    if merged.empty:
        raise SystemExit(
            "No overlapping (trajectory_id, step_index) rows between projections "
            "and confidence.parquet — regenerate elicitation on the same traj dir."
        )

    col = _proj_col(merged)
    # Final-step headline for both.
    finals = merged.sort_values("step_index").groupby("trajectory_id").tail(1)
    internal_final = auroc_with_ci(finals, col, n_boot=n_boot)
    elicited_final = auroc_with_ci(finals, "p_success", n_boot=n_boot)

    internal_pos = auroc_by_position(merged, n_bins=n_bins, score_col=col, n_boot=n_boot)
    elicited_pos = auroc_by_position(
        merged, n_bins=n_bins, score_col="p_success", n_boot=n_boot
    )

    return {
        "n_joined_rows": int(len(merged)),
        "n_trajectories": int(merged["trajectory_id"].nunique()),
        "internal_final_step": internal_final,
        "elicited_final_step": elicited_final,
        "internal_by_position": internal_pos.to_dict(orient="records"),
        "elicited_by_position": elicited_pos.to_dict(orient="records"),
    }


def run(
    projections_path: Path,
    confidence_path: Path,
    *,
    output_dir: Path,
    n_bins: int = 5,
    primary_layer: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = compare(
        pd.read_parquet(projections_path),
        pd.read_parquet(confidence_path),
        n_bins=n_bins,
        primary_layer=primary_layer,
    )
    path = output_dir / "internal_vs_elicited.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Report -> {path}", flush=True)
    return report


def main() -> None:
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projections", type=Path, default=data_file("projections.parquet"))
    ap.add_argument("--confidence", type=Path, default=data_file("confidence.parquet"))
    ap.add_argument("--output-dir", type=Path, default=data_file("").parent)
    ap.add_argument("--n-bins", type=int, default=defaults["n_bins"])
    ap.add_argument(
        "--primary-layer",
        type=int,
        default=defaults.get("primary_layer", defaults.get("layer")),
    )
    args = ap.parse_args()
    if not args.projections.exists():
        raise SystemExit(f"Projections not found: {args.projections}")
    if not args.confidence.exists():
        raise SystemExit(f"Confidence not found: {args.confidence}")
    run(
        args.projections,
        args.confidence,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
        primary_layer=args.primary_layer,
    )


if __name__ == "__main__":
    main()
