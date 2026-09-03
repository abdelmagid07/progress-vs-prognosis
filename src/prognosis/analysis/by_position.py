"""AUROC of the value-axis readout by relative-position bin (with task-level CI).

The paper reports the separation "at every relative position", not just the
final step. This replaces the old SNR-by-position *headline* with AUROC per
equal-width ``rel_pos`` bin, each with a task-level BCa confidence interval and
the pre-registered signal flag (CI excludes 0.5). SNR-by-position is retained as
a secondary diagnostic in :mod:`prognosis.analysis.signal_to_noise`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prognosis.analysis.stats import auroc_with_ci


def _proj_col(df: pd.DataFrame) -> str:
    return "proj_mean" if "proj_mean" in df.columns else "projection"


def auroc_by_position(
    df: pd.DataFrame,
    *,
    n_bins: int = 5,
    score_col: str | None = None,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """AUROC + task-level BCa CI per equal-width relative-position bin.

    Each step contributes one row; within a bin the AUROC contrasts the readout
    against the trajectory outcome, resampling tasks for the CI.
    """
    work = df.copy()
    col = score_col or _proj_col(work)
    rng = rng or np.random.default_rng(0)
    # Fixed equal-width bins over [0, 1] so bins are comparable across runs.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    work["bin"] = np.clip(
        np.digitize(work["rel_pos"], edges[1:-1], right=False), 0, n_bins - 1
    )

    rows = []
    for b in range(n_bins):
        sub = work[work["bin"] == b]
        if sub.empty:
            continue
        res = auroc_with_ci(sub, col, n_boot=n_boot, rng=rng)
        rows.append(
            {
                "bin": b,
                "rel_pos_mid": (b + 0.5) / n_bins,
                "n": res["n"],
                "auroc": res["auroc"],
                "ci_low": res["ci_low"],
                "ci_high": res["ci_high"],
                "signal": res["signal"],
                "majority_baseline": res["majority_baseline"],
            }
        )
    return pd.DataFrame(rows)
