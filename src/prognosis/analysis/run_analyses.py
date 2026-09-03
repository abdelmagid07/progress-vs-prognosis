#!/usr/bin/env python
"""Run trajectory readability analyses and write report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prognosis.analysis.by_position import auroc_by_position
from prognosis.analysis.final_step import final_step_plot
from prognosis.analysis.noise_by_token_type import noise_by_token_type
from prognosis.analysis.signal_to_noise import headline_late_bin_snr, signal_to_noise_by_position
from prognosis.common.config import load_defaults
from prognosis.common.paths import data_file


def interpret_report(
    *,
    final: dict,
    late_snr: float | None,
    mean_std: float | None,
    final_std: float | None,
) -> str:
    lines = []

    auroc = final.get("auroc")
    if auroc is not None and not pd.isna(auroc):
        verdict = "signal" if final.get("signal") else "no signal (CI includes 0.5)"
        lines.append(
            f"Final-step AUROC {auroc:.3f} "
            f"[{final.get('auroc_ci_low'):.3f}, {final.get('auroc_ci_high'):.3f}] "
            f"vs majority baseline {final.get('majority_baseline'):.3f}: {verdict}; "
            f"permutation p={final.get('permutation_p')}."
        )
    else:
        lines.append("Final-step AUROC undefined (single outcome class).")

    if late_snr is not None and late_snr >= 1.0:
        lines.append("Late-bin separation-to-noise is at or above 1 (secondary).")
    elif late_snr is not None:
        lines.append("Late-bin separation-to-noise is below 1 (secondary).")

    if mean_std is not None and final_std is not None and mean_std < final_std:
        lines.append(
            f"Mean-over-G_t readout std ({mean_std:.3f}) < final-token std ({final_std:.3f})."
        )

    return " ".join(lines)


def run_analyses(
    projections_path: Path,
    *,
    n_bins: int = 5,
    output_dir: Path | None = None,
    primary_layer: int | None = None,
) -> dict:
    output_dir = output_dir or data_file("").parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(projections_path)

    # The parquet carries every swept layer; the headline analyses use the single
    # pre-registered primary layer so AUROC isn't mixed across layers.
    if primary_layer is not None and "layer" in df.columns and df["layer"].nunique() > 1:
        df = df[df["layer"] == primary_layer]

    # Primary: AUROC per position bin with task-level BCa CI (The paper).
    pos_df = auroc_by_position(df, n_bins=n_bins)
    pos_csv = output_dir / "auroc_by_position.csv"
    pos_df.to_csv(pos_csv, index=False)

    # Secondary diagnostic: SNR by position.
    snr_df = signal_to_noise_by_position(df, n_bins=n_bins)
    snr_csv = output_dir / "snr_by_position.csv"
    snr_df.to_csv(snr_csv, index=False)

    final_results = final_step_plot(df, output_dir / "final_step_separation.png")
    token_summary = noise_by_token_type(df, output_dir / "noise_by_token_type.png")

    late_snr = headline_late_bin_snr(snr_df)
    label_col = "readout" if "readout" in token_summary.columns else "token_type"
    std_map = {row[label_col]: float(row["std"]) for _, row in token_summary.iterrows()}
    # New schema: mean-over-G_t vs final-token readouts. Legacy fallback keeps the
    # old reasoning/tool_output keys working.
    mean_std = std_map.get("proj_mean", std_map.get("reasoning"))
    final_std = std_map.get("proj_final", std_map.get("tool_output"))
    std_ratio = (
        mean_std / final_std
        if mean_std is not None and final_std and final_std > 0
        else None
    )

    n_traj = df["trajectory_id"].nunique()
    n_success = int(df.groupby("trajectory_id")["outcome"].first().eq(1).sum())
    n_failure = int(n_traj - n_success)

    report = {
        "n_trajectories": int(n_traj),
        "n_success": n_success,
        "n_failure": n_failure,
        "n_projection_rows": len(df),
        # Headline: final-step AUROC + task-level BCa CI + majority baseline +
        # permutation, plus the full positional AUROC sweep.
        "final_step_auroc": final_results["auroc"],
        "final_step_auroc_ci": [
            final_results.get("auroc_ci_low"),
            final_results.get("auroc_ci_high"),
        ],
        "final_step_signal": final_results.get("signal"),
        "majority_baseline": final_results.get("majority_baseline"),
        "permutation_p": final_results.get("permutation_p"),
        "final_step_auroc_proj_final": final_results.get("auroc_proj_final"),
        "final_step_separability": final_results["separability"],
        "within_task_n_mixed": final_results.get("within_task_n_mixed"),
        "within_task_mean_gap": final_results.get("within_task_mean_gap"),
        "auroc_by_position": pos_df.to_dict(orient="records"),
        "auroc_by_position_csv": str(pos_csv),
        # Secondary diagnostics.
        "late_bin_separation_to_noise": late_snr,
        "proj_mean_readout_std": mean_std,
        "proj_final_readout_std": final_std,
        "mean_to_final_std_ratio": std_ratio,
        "snr_by_position_csv": str(snr_csv),
        "final_step_plot": final_results["plot_path"],
        "noise_by_token_type_plot": str(output_dir / "noise_by_token_type.png"),
        "interpretation": interpret_report(
            final=final_results,
            late_snr=late_snr,
            mean_std=mean_std,
            final_std=final_std,
        ),
    }

    report_path = output_dir / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2), flush=True)
    print(f"Report -> {report_path}", flush=True)
    return report


def main():
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projections", type=Path, default=data_file("projections.parquet"))
    ap.add_argument("--n-bins", type=int, default=defaults["n_bins"])
    ap.add_argument("--output-dir", type=Path, default=data_file("").parent)
    ap.add_argument(
        "--primary-layer",
        type=int,
        default=defaults.get("primary_layer", defaults.get("layer")),
        help="Layer used for the headline analyses (default: config primary layer).",
    )
    args = ap.parse_args()

    if not args.projections.exists():
        raise SystemExit(f"Projections file not found: {args.projections}")

    run_analyses(
        args.projections,
        n_bins=args.n_bins,
        output_dir=args.output_dir,
        primary_layer=args.primary_layer,
    )


if __name__ == "__main__":
    main()
