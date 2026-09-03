"""Permutation test for the matched-pair single-turn design.

``prognosis.analysis.stats.permutation_test_auroc`` reassigns whole *blocks* of
outcomes between tasks (matched by block length) — the right null when
different tasks carry different outcome patterns, which is true for agentic
rollouts (a task's resolved/unresolved mix across seeds varies). It is
**degenerate here**: every DebugBench problem contributes the same fixed
pair by construction (one ``outcome=1`` original row, one ``outcome=0``
corrupted row per variant), so swapping identical blocks between problems
never changes anything — the block-swap p-value is pinned at 1.0 regardless
of signal strength (confirmed empirically: AUROC ~1.0 on synthetic separable
data still returned p=1.0 with the block-swap version).

The appropriate null for a matched-pair/small-group design is a **within-task
label permutation**: for each task, shuffle which of *that task's own* rows
carry which outcome label (preserving the task's own count of 1s/0s), so the
score-to-label pairing is actually perturbed. For the 1-vs-1 per-variant
frame this is exactly a 50/50 paired sign-flip; for the pooled frame (one
original vs. several corrupted rows per task) it permutes which row within
the task is treated as "original".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prognosis.analysis.stats import auroc


def within_task_permutation_test(
    df: pd.DataFrame,
    score_col: str,
    *,
    task_col: str = "slug",
    outcome_col: str = "outcome",
    n_perm: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Two-sided permutation p-value on ``|AUROC - 0.5|``, task-local label shuffle."""
    rng = rng or np.random.default_rng(0)
    observed = auroc(df[outcome_col].to_numpy(), df[score_col].to_numpy())
    if np.isnan(observed):
        return {"estimate": observed, "p_value": float("nan"), "n_perm": 0}

    order = df.reset_index(drop=True)
    scores = order[score_col].to_numpy()
    outcomes = order[outcome_col].to_numpy()
    task_ids = order[task_col].to_numpy()

    positions = [np.where(task_ids == t)[0] for t in pd.unique(task_ids)]

    obs_stat = abs(observed - 0.5)
    ge = 0
    for _ in range(n_perm):
        permuted = outcomes.copy()
        for idx in positions:
            if len(idx) > 1:
                permuted[idx] = rng.permutation(outcomes[idx])
        stat = abs(auroc(permuted, scores) - 0.5)
        if not np.isnan(stat) and stat >= obs_stat:
            ge += 1
    p = (1 + ge) / (n_perm + 1)
    return {"estimate": observed, "p_value": float(p), "n_perm": int(n_perm)}
