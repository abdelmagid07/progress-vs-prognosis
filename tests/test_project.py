"""Offline tests for the diff-window and aggregation logic (no model)."""

from __future__ import annotations

import numpy as np
import pytest

from prognosis.single_turn.project import aggregate_stats, code_token_span, diff_window_mask


def test_diff_window_mask_identical_sequences_is_all_false():
    ids = [1, 2, 3, 4, 5]
    mask_a, mask_b = diff_window_mask(ids, ids, radius=10)
    assert not mask_a.any()
    assert not mask_b.any()


def test_diff_window_mask_substitution():
    ids_a = list(range(10))
    ids_b = ids_a.copy()
    ids_b[3] = 99  # difflib reports this as a 'replace' op: i1=3,i2=4,j1=3,j2=4
    mask_a, mask_b = diff_window_mask(ids_a, ids_b, radius=2)

    expected = np.zeros(10, dtype=bool)
    expected[3:6] = True  # [3, 4+2)
    np.testing.assert_array_equal(mask_a, expected)
    np.testing.assert_array_equal(mask_b, expected)


def test_diff_window_mask_insertion_zero_width_change_point():
    ids_a = [1, 2, 3, 4, 5]
    ids_b = [1, 2, 3, 99, 4, 5]  # difflib: insert i1=3,i2=3 (zero-width) / j1=3,j2=4
    mask_a, mask_b = diff_window_mask(ids_a, ids_b, radius=2)

    expected_a = np.zeros(5, dtype=bool)
    expected_a[3:5] = True  # [3, 3+2) -- zero-width point still opens a window
    expected_b = np.zeros(6, dtype=bool)
    expected_b[3:6] = True  # [3, 4+2)
    np.testing.assert_array_equal(mask_a, expected_a)
    np.testing.assert_array_equal(mask_b, expected_b)


def test_diff_window_mask_radius_clips_at_array_end():
    ids_a = list(range(5))
    ids_b = ids_a.copy()
    ids_b[4] = 99  # change right at the last index
    mask_a, mask_b = diff_window_mask(ids_a, ids_b, radius=10)
    # Window would extend to 5+10=15, but must clip to len(ids)=5, not raise.
    expected = np.zeros(5, dtype=bool)
    expected[4:5] = True
    np.testing.assert_array_equal(mask_a, expected)
    np.testing.assert_array_equal(mask_b, expected)


def test_diff_window_mask_multiple_regions_union():
    ids_a = list(range(20))
    ids_b = ids_a.copy()
    ids_b[2] = 99
    ids_b[15] = 88
    mask_a, mask_b = diff_window_mask(ids_a, ids_b, radius=1)
    assert mask_a[2] and mask_a[3] and not mask_a[4]
    assert mask_a[15] and mask_a[16]
    assert not mask_a[0] and not mask_a[10]


def test_aggregate_stats_whole_array():
    proj = np.array([0.1, 0.2, 0.3, 0.4])
    stats = aggregate_stats(proj)
    assert stats["mean"] == pytest.approx(0.25)
    assert stats["final"] == pytest.approx(0.4)
    assert stats["n_tokens"] == 4


def test_aggregate_stats_with_mask():
    proj = np.array([0.1, 0.2, 0.3, 0.4])
    mask = np.array([False, True, True, False])
    stats = aggregate_stats(proj, mask)
    assert stats["mean"] == pytest.approx(0.25)  # mean(0.2, 0.3)
    assert stats["final"] == pytest.approx(0.3)  # last True position
    assert stats["n_tokens"] == 2


def test_aggregate_stats_empty_mask_is_nan_not_crash():
    proj = np.array([0.1, 0.2, 0.3])
    mask = np.zeros(3, dtype=bool)
    stats = aggregate_stats(proj, mask)
    assert np.isnan(stats["mean"])
    assert np.isnan(stats["final"])
    assert stats["n_tokens"] == 0


def test_code_token_span_still_works():
    # Regression: this function moved but its behavior must be unchanged.
    offsets = [(0, 1), (1, 2), (2, 3), (3, 4)]
    ts, te = code_token_span(offsets, 1, 3)
    assert (ts, te) == (1, 3)
