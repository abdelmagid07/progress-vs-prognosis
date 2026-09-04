"""Analysis 2: final-step readout distributions, AUROC + CI, permutation.

The paper's most-favorable test: does the mean-cosine readout (Eq. 1) at the
*final* step separate eventually-resolved from eventually-unresolved
trajectories? Reported as AUROC at the primary layer with a task-level BCa CI,
judged against the majority-class baseline (signal iff CI excludes 0.5), a block
permutation p-value, and the single-final-token read (``proj_final``) as a
robustness column. The unit is the trajectory (one final step each).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prognosis.analysis.stats import (
    auroc_with_ci,
    permutation_test_auroc,
    within_task_contrast,
)


def _proj_col(df: pd.DataFrame) -> str:
    return "proj_mean" if "proj_mean" in df.columns else "projection"


def final_step_rows(df: pd.DataFrame, *, token_type: str = "reasoning") -> pd.DataFrame:
    """One row per trajectory: its last step (tolerant of the older token_type schema)."""
    work = df.copy()
    if "token_type" in work.columns:
        work = work[work["token_type"] == token_type]
    return work.sort_values("step_index").groupby("trajectory_id").tail(1)


def final_step_plot(
    df: pd.DataFrame,
    out_path: Path,
    *,
    token_type: str = "reasoning",
    n_boot: int = 1000,
    n_perm: int = 2000,
) -> dict:
    work = df.copy()
    col = _proj_col(work)
    finals = final_step_rows(work, token_type=token_type)
    finals = finals.dropna(subset=[col])

    succ = finals[finals["outcome"] == 1][col]
    fail = finals[finals["outcome"] == 0][col]

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(finals) > 0:
        bins = np.histogram_bin_edges(finals[col], bins=min(12, max(3, len(finals))))
        ax.hist(fail, bins=bins, alpha=0.6, label=f"Failed (n={len(fail)})")
        ax.hist(succ, bins=bins, alpha=0.6, label=f"Succeeded (n={len(succ)})")
        if len(fail) > 0:
            ax.axvline(fail.mean(), linestyle="--", linewidth=1, color="C0")
        if len(succ) > 0:
            ax.axvline(succ.mean(), linestyle="--", linewidth=1, color="C1")

    ax.set_xlabel("Value-axis projection at final step")
    ax.set_ylabel("Count")
    ax.set_title("Final-step projection: success vs. failure")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Primary: mean-cosine readout AUROC with task-level BCa CI + majority
    # baseline + signal flag, plus a block-permutation p-value.
    ci = auroc_with_ci(finals, col, n_boot=n_boot)
    perm = permutation_test_auroc(finals, col, n_perm=n_perm)
    contrast = within_task_contrast(finals, col)

    # Robustness read: the single final-token cosine, when available.
    auroc_final = float("nan")
    if "proj_final" in finals.columns:
        ff = finals.dropna(subset=["proj_final"])
        auroc_final = auroc_with_ci(ff, "proj_final", n_boot=n_boot)["auroc"]

    return {
        "n_final_steps": len(finals),
        "n_success": int(len(succ)),
        "n_failure": int(len(fail)),
        "auroc": ci["auroc"],
        "auroc_ci_low": ci["ci_low"],
        "auroc_ci_high": ci["ci_high"],
        "majority_baseline": ci["majority_baseline"],
        "signal": ci["signal"],
        "permutation_p": perm["p_value"],
        "auroc_proj_final": auroc_final,
        "separability": ci["separability"],
        "within_task_n_mixed": contrast["n_mixed_tasks"],
        "within_task_mean_gap": contrast["mean_within_task_gap"],
        "plot_path": str(out_path),
    }
