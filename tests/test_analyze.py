"""Offline tests for analyze.py against synthetic projections (no model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from prognosis.single_turn.analyze import headline_report, layer_sweep
from tests.fixtures.mock_projections import VARIANTS, mock_projections

PRIMARY_LAYER = 5
OTHER_LAYER = 2


@pytest.fixture()
def separable_df(tmp_path: Path):
    out = tmp_path / "projections.parquet"
    return mock_projections(
        out, n_problems=30, n_layers=8, primary_layer=PRIMARY_LAYER,
        signal_at_primary=0.6, seed=0,
    )


def test_headline_report_separates_at_primary_layer(separable_df):
    report = headline_report(
        separable_df, variants=VARIANTS, primary_layer=PRIMARY_LAYER,
        score_col="proj_mean", n_boot=200, n_perm=200,
    )
    assert report["primary_layer"] == PRIMARY_LAYER
    for variant in VARIANTS:
        res = report["by_variant"][variant]
        assert res["auroc"] > 0.8, (variant, res["auroc"])
        assert res["ci_low"] > 0.5, (variant, res)
        assert res["permutation_p"] < 0.05, (variant, res)
    pooled = report["pooled"]
    assert pooled["auroc"] > 0.8
    # Every variant contributes its own (original, corrupted) pair now (the
    # diff window is pair-specific) -> 2 rows per problem per variant.
    assert pooled["n"] == 30 * len(VARIANTS) * 2
    assert pooled["permutation_p"] < 0.05, pooled


def test_headline_report_near_chance_at_non_primary_layer(separable_df):
    report = headline_report(
        separable_df, variants=VARIANTS, primary_layer=OTHER_LAYER,
        score_col="proj_mean", n_boot=200, n_perm=200,
    )
    pooled = report["pooled"]
    assert 0.3 < pooled["auroc"] < 0.7, pooled["auroc"]
    assert pooled["permutation_p"] > 0.05, pooled


def test_layer_sweep_peaks_at_primary_layer(separable_df):
    sweep = layer_sweep(separable_df, variants=VARIANTS, score_col="proj_mean")
    pooled_by_layer = sweep["pooled"]
    best_layer = max(pooled_by_layer, key=lambda k: pooled_by_layer[k])
    assert int(best_layer) == PRIMARY_LAYER


def test_majority_baseline_is_balanced(separable_df):
    report = headline_report(
        separable_df, variants=VARIANTS, primary_layer=PRIMARY_LAYER,
        score_col="proj_mean", n_boot=50, n_perm=50,
    )
    # Every variant pairing contributes exactly one original + one corrupted
    # row -> balanced classes, majority baseline == 0.5.
    assert report["pooled"]["majority_baseline"] == pytest.approx(0.5, abs=1e-6)


def test_diff_window_readout_present_and_directionally_sane(separable_df):
    report = headline_report(
        separable_df, variants=VARIANTS, primary_layer=PRIMARY_LAYER,
        score_col="proj_window_mean", n_boot=200, n_perm=200,
    )
    assert report["score_col"] == "proj_window_mean"
    assert report["pooled"]["auroc"] > 0.7, report["pooled"]
