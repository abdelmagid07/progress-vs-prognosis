"""Regression test for the within-task permutation null.

This exists specifically because ``prognosis.analysis.stats.permutation_test_auroc``
turned out to be degenerate for this design (see ``stats_local.py``'s
docstring): it silently returned p=1.0 even for a perfectly separable signal,
because it swaps identical fixed [1, 0] blocks between tasks. These tests
would have caught that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prognosis.single_turn.stats_local import within_task_permutation_test


def _paired_df(n_pairs: int, gap: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pairs):
        slug = f"p{i}"
        rows.append({"slug": slug, "outcome": 1, "score": rng.normal(gap, 0.2)})
        rows.append({"slug": slug, "outcome": 0, "score": rng.normal(0.0, 0.2)})
    return pd.DataFrame(rows)


def test_significant_for_a_strongly_separable_paired_signal():
    df = _paired_df(n_pairs=40, gap=3.0, seed=0)
    res = within_task_permutation_test(df, "score", n_perm=500,
                                        rng=np.random.default_rng(1))
    assert res["p_value"] < 0.05, res


def test_not_significant_under_the_null():
    df = _paired_df(n_pairs=40, gap=0.0, seed=0)
    res = within_task_permutation_test(df, "score", n_perm=500,
                                        rng=np.random.default_rng(1))
    assert res["p_value"] > 0.05, res


def test_single_row_tasks_are_left_unpermuted_without_crashing():
    # Pooled-style frame where some tasks have >1 row and this exercises the
    # len(idx) > 1 guard for degenerate single-row groups.
    df = pd.DataFrame({
        "slug": ["a", "a", "b"],
        "outcome": [1, 0, 1],
        "score": [1.0, -1.0, 0.5],
    })
    res = within_task_permutation_test(df, "score", n_perm=50)
    assert res["n_perm"] == 50
    assert 0.0 <= res["p_value"] <= 1.0
