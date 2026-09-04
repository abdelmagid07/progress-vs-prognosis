#!/usr/bin/env python
"""Agentic transfer analyses against the frozen value axis.

Input: the full-run projections table. Its ``layer`` column is in
embedding-inclusive numbering, covering layers 1-64; layer 0 (the pure embedding
row) is not recoverable from the cached mean-pooled activations.

Produces, into data/:
  * agentic_final_step_auroc_by_layer.csv  -- full 64-layer final-step sweep, with CIs
  * agentic_position_bins_peak_layer.csv   -- position-bin evolution at the peak layer
  * agentic_summary.json                   -- headline numbers

STATISTICS. The repo's authoritative machinery is prognosis/analysis/stats.py
(``auroc_with_ci`` = task-level BCa bootstrap, ``permutation_test_auroc`` = block
permutation). That implementation rebuilds a DataFrame with ``pd.concat`` on every
bootstrap iteration, which costs ~46s per layer here -- ~50 min for a 64-layer sweep.

This script instead uses a vectorized BCa that is valid *because of a property of
this particular dataset*: the run is single-seed (seed = -1, one trajectory per
task, 499/499), and the final-step readout takes exactly one row per trajectory, so
"resample tasks with replacement" collapses to "resample rows with replacement" and
can be done as pure array indexing. ``--validate`` re-runs the authoritative slow
path at the peak layer and asserts the two agree; do not trust the fast path on any
future multi-seed run, where a task spans several rows and the collapse no longer
holds.

PEAK-LAYER SELECTION IS POST-HOC. The peak layer is chosen by scanning all 64 layers
on the same trajectories whose AUROC is then reported, so the peak-layer value is
optimistically biased and is an exploratory descriptive statistic, not a held-out
estimate. The full sweep is reported alongside the
peak layer's position-bin evolution, with the selection explicitly labelled.

Reproduce: python scripts/analyze_agentic.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT_DIR = ROOT / "data"
CHANCE = 0.5


def fast_auroc(outcome: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUROC, tie-aware (matches sklearn's roc_auc_score)."""
    n_pos = int(outcome.sum())
    n_neg = int(len(outcome) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[outcome == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fast_auprc(outcome: np.ndarray, score: np.ndarray) -> float:
    """Average precision (AUPRC). No-skill baseline is the positive prevalence,
    NOT 0.5 -- with ~12% resolved, an uninformative scorer sits near 0.12, so an
    AUPRC must always be read against the prevalence reported alongside it."""
    if len(np.unique(outcome)) < 2:
        return float("nan")
    return float(average_precision_score(outcome, score))


def _bca_bounds(boot: np.ndarray, jack: np.ndarray, observed: float, alpha: float) -> tuple[float, float]:
    """BCa interval from bootstrap + jackknife replicates, matching
    prognosis.analysis.stats.bca_bootstrap_ci's logic including its degenerate fallback."""
    boot = boot[~np.isnan(boot)]
    jack = jack[~np.isnan(jack)]
    if len(boot) < 2:
        return float("nan"), float("nan")
    prop = float(np.mean(boot < observed))
    if prop <= 0.0 or prop >= 1.0 or len(jack) < 2:
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lo), float(hi)
    z0 = norm.ppf(prop)
    diffs = jack.mean() - jack
    denom = 6.0 * (np.sum(diffs**2) ** 1.5)
    a = float(np.sum(diffs**3) / denom) if denom != 0 else 0.0

    def _adj(z_alpha: float) -> float:
        return float(norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))))

    lo, hi = np.percentile(boot, [100 * _adj(norm.ppf(alpha / 2)), 100 * _adj(norm.ppf(1 - alpha / 2))])
    return float(lo), float(hi)


def fast_auroc_bca(
    outcome: np.ndarray, score: np.ndarray, *, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """BCa CIs for AUROC *and* AUPRC, resampling rows (== tasks here; see module
    docstring). Both metrics are computed on the SAME resamples, so the two
    intervals are mutually consistent and the bootstrap is only paid for once.

    Keys: auroc/ci_low/ci_high (AUROC, kept under the original names so existing
    callers are unaffected) and auprc/auprc_ci_low/auprc_ci_high/prevalence.
    """
    rng = np.random.default_rng(seed)
    n = len(outcome)
    obs_roc, obs_prc = fast_auroc(outcome, score), fast_auprc(outcome, score)
    prevalence = float(np.mean(outcome)) if n else float("nan")
    out = {
        "auroc": obs_roc, "ci_low": float("nan"), "ci_high": float("nan"),
        "auprc": obs_prc, "auprc_ci_low": float("nan"), "auprc_ci_high": float("nan"),
        "prevalence": prevalence, "n_boot": 0,
    }
    if np.isnan(obs_roc) or n < 2:
        return out

    idx = rng.integers(0, n, size=(n_boot, n))
    b_roc = np.empty(n_boot, dtype=float)
    b_prc = np.empty(n_boot, dtype=float)
    for k, i in enumerate(idx):
        o_i, s_i = outcome[i], score[i]
        b_roc[k] = fast_auroc(o_i, s_i)
        b_prc[k] = fast_auprc(o_i, s_i)
    out["n_boot"] = int(np.sum(~np.isnan(b_roc)))

    j_roc = np.empty(n, dtype=float)
    j_prc = np.empty(n, dtype=float)
    for i in range(n):
        o_j, s_j = np.delete(outcome, i), np.delete(score, i)
        j_roc[i] = fast_auroc(o_j, s_j)
        j_prc[i] = fast_auprc(o_j, s_j)

    out["ci_low"], out["ci_high"] = _bca_bounds(b_roc, j_roc, obs_roc, alpha)
    if not np.isnan(obs_prc):
        out["auprc_ci_low"], out["auprc_ci_high"] = _bca_bounds(b_prc, j_prc, obs_prc, alpha)
    return out


def fast_perm_p(outcome: np.ndarray, score: np.ndarray, *, n_perm: int = 2000, seed: int = 0) -> float:
    """Permutation p for AUROC. One row per task here, so a block permutation over
    tasks is a plain label shuffle (add-one smoothed, matching stats.py)."""
    rng = np.random.default_rng(seed)
    observed = fast_auroc(outcome, score)
    if np.isnan(observed):
        return float("nan")
    obs_dev = abs(observed - CHANCE)
    null = np.array(
        [abs(fast_auroc(rng.permutation(outcome), score) - CHANCE) for _ in range(n_perm)], dtype=float
    )
    null = null[~np.isnan(null)]
    return float((np.sum(null >= obs_dev) + 1) / (len(null) + 1))


def final_step_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Last step of each trajectory, per layer."""
    return df.sort_values("step_index").groupby(["trajectory_id", "layer"], as_index=False).tail(1)


def auroc_by_step(df: pd.DataFrame, peak_layer: int, *, n_boot: int = 1000) -> pd.DataFrame:
    """Final-outcome AUROC at each ABSOLUTE step index, with BCa intervals.

    Absolute step index rather than rel_pos bins: rel_pos is length-normalized and length
    tracks outcome here, so within-bin averaging smooths the classes unequally and manufactures
    a spurious "signal emerges late" pattern (see the position-bin block below).

    ``n_resolved`` is carried because it, not ``n``, is what limits this curve. The resolved
    class is exhausted long before the unresolved one -- past step ~13 only a handful of
    resolved runs survive, and an AUROC computed against 4 positives is noise regardless of how
    many negatives remain. Consumers should truncate on ``n_resolved``, and the BCa interval
    makes the cost of not doing so visible: it widens to span chance exactly where the point
    estimate starts swinging.
    """
    rows = []
    for st, g in df[df["layer"] == peak_layer].groupby("step_index"):
        if len(g) < 30 or g["outcome"].nunique() < 2:
            continue
        r = fast_auroc_bca(g["outcome"].to_numpy(), g["proj_mean"].to_numpy(),
                           n_boot=n_boot, seed=int(st))
        rows.append({"step_index": int(st), "n": int(len(g)),
                     "n_resolved": int(g["outcome"].sum()), "auroc": r["auroc"],
                     "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projections", type=Path, default=ROOT / "data" / "projections_full.parquet")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--validate", action="store_true",
                    help="Cross-check the fast BCa against prognosis.analysis.stats at the peak layer.")
    ap.add_argument("--steps-only", type=int, default=None, metavar="PEAK_LAYER",
                    help="Regenerate ONLY the by-absolute-step CSV at the given peak layer, "
                         "skipping the 64-layer sweep (~50 min) and everything downstream of "
                         "it. Use when the sweep is unchanged and only the step curve needs "
                         "rebuilding; the layer must match the sweep's peak or the output will "
                         "not correspond to the rest of the analysis.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.projections)
    finals = final_step_rows(df)

    if args.steps_only is not None:
        steps_df = auroc_by_step(df, args.steps_only, n_boot=args.n_boot)
        steps_path = OUT_DIR / "agentic_auroc_by_step_index_peak_layer.csv"
        steps_df.to_csv(steps_path, index=False)
        print(steps_df.to_string(index=False), flush=True)
        print(f"\n  wrote {steps_path}", flush=True)
        return

    n_traj = df["trajectory_id"].nunique()
    n_task = df["task_id"].nunique()
    traj_level = df.drop_duplicates("trajectory_id")
    resolution_rate = float(traj_level["outcome"].mean())
    print(f"{len(df):,} rows | {n_traj} trajectories | {n_task} tasks "
          f"| resolution rate {resolution_rate:.3f}", flush=True)
    if n_traj != n_task:
        raise SystemExit(
            f"Fast path assumes one trajectory per task (got {n_traj} vs {n_task}). "
            "Use prognosis.analysis.stats directly instead -- see module docstring."
        )

    # ---- full final-step layer sweep -------------------------------------
    print(f"\nFinal-step sweep over {finals['layer'].nunique()} layers...", flush=True)
    sweep = []
    for L, g in finals.groupby("layer"):
        o = g["outcome"].to_numpy()
        s = g["proj_mean"].to_numpy()
        r = fast_auroc_bca(o, s, n_boot=args.n_boot, seed=int(L))
        r["layer"] = int(L)
        r["n"] = int(len(g))
        r["signal"] = bool(
            not np.isnan(r["ci_low"]) and (r["ci_low"] > CHANCE or r["ci_high"] < CHANCE)
        )
        sweep.append(r)
    sweep_df = pd.DataFrame(sweep)[
        ["layer", "auroc", "ci_low", "ci_high", "signal",
         "auprc", "auprc_ci_low", "auprc_ci_high", "prevalence", "n", "n_boot"]
    ].sort_values("layer")
    sweep_path = OUT_DIR / "agentic_final_step_auroc_by_layer.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"  wrote {sweep_path}", flush=True)

    peak = sweep_df.loc[sweep_df["auroc"].idxmax()]
    peak_layer = int(peak["layer"])
    print(f"\nPeak layer (POST-HOC): {peak_layer}  AUROC={peak['auroc']:.4f} "
          f"[{peak['ci_low']:.4f}, {peak['ci_high']:.4f}]  signal={bool(peak['signal'])}", flush=True)

    pk = finals[finals["layer"] == peak_layer]
    perm_p = fast_perm_p(pk["outcome"].to_numpy(), pk["proj_mean"].to_numpy(),
                         n_perm=args.n_perm, seed=peak_layer)
    print(f"  permutation p = {perm_p:.4g}", flush=True)

    if args.validate:
        from prognosis.analysis.stats import auroc_with_ci  # noqa: PLC0415
        print("\nValidating fast BCa against prognosis.analysis.stats (slow path)...", flush=True)
        ref = auroc_with_ci(pk, "proj_mean", n_boot=args.n_boot)
        print(f"  slow: AUROC={ref['auroc']:.4f} [{ref['ci_low']:.4f}, {ref['ci_high']:.4f}]")
        print(f"  fast: AUROC={peak['auroc']:.4f} [{peak['ci_low']:.4f}, {peak['ci_high']:.4f}]")
        assert abs(ref["auroc"] - peak["auroc"]) < 1e-9, "point estimates diverge"
        for k in ("ci_low", "ci_high"):
            assert abs(ref[k] - peak[k]) < 0.02, f"{k} diverges beyond bootstrap noise"
        print("  OK: point estimates identical, CI bounds agree within bootstrap noise.", flush=True)

    # ---- position-bin evolution at the peak layer ------------------------
    # TWO readouts per bin, because they disagree and only one of them is sound:
    #
    #   auroc_binmean  -- mean of proj_mean over every step falling in the bin.
    #                     UNSOUND HERE. rel_pos is length-normalized, so a bin spans
    #                     more absolute steps for a long trajectory than a short one,
    #                     and unresolved runs are much longer. In bin 0, resolved runs
    #                     average ~2 steps and unresolved ~5.6, so the two classes get
    #                     systematically different amounts of smoothing and the
    #                     comparison is confounded by length rather than position.
    #                     Retained for comparability with previously reported values.
    #   auroc_midstep  -- the single step nearest the bin's midpoint, one row per
    #                     trajectory per bin. Position-matched with no differential
    #                     averaging, so this is the one to trust.
    #
    # On this data the difference is not cosmetic: bin 0 reads 0.52 under binmean and
    # 0.72 under midstep, i.e. the "no early separation" pattern is an artifact of the
    # averaging, not a property of the axis.
    print(f"\nPosition-bin evolution at layer {peak_layer} ({args.n_bins} bins)...", flush=True)
    at_peak = df[df["layer"] == peak_layer].copy()
    edges = np.linspace(0.0, 1.0, args.n_bins + 1)
    at_peak["bin"] = np.clip(np.digitize(at_peak["rel_pos"], edges[1:-1]), 0, args.n_bins - 1)

    binmean = (
        at_peak.groupby(["trajectory_id", "bin"], as_index=False)
        .agg(outcome=("outcome", "first"), proj_mean=("proj_mean", "mean"))
    )
    at_peak["mid_dist"] = (at_peak["rel_pos"] - at_peak["bin"].map(
        {b: (edges[b] + edges[b + 1]) / 2 for b in range(args.n_bins)})).abs()
    midstep = (
        at_peak.sort_values("mid_dist")
        .groupby(["trajectory_id", "bin"], as_index=False)
        .first()[["trajectory_id", "bin", "outcome", "proj_mean"]]
    )

    bins = []
    for b in range(args.n_bins):
        gm = binmean[binmean["bin"] == b]
        gs = midstep[midstep["bin"] == b]
        rm = fast_auroc_bca(gm["outcome"].to_numpy(), gm["proj_mean"].to_numpy(),
                            n_boot=args.n_boot, seed=100 + b)
        rs = fast_auroc_bca(gs["outcome"].to_numpy(), gs["proj_mean"].to_numpy(),
                            n_boot=args.n_boot, seed=300 + b)
        bins.append({
            "bin": b,
            "rel_pos_lo": float(edges[b]), "rel_pos_hi": float(edges[b + 1]),
            "auroc_binmean": rm["auroc"], "binmean_ci_low": rm["ci_low"], "binmean_ci_high": rm["ci_high"],
            "auroc_midstep": rs["auroc"], "midstep_ci_low": rs["ci_low"], "midstep_ci_high": rs["ci_high"],
            "auprc_midstep": rs["auprc"], "midstep_auprc_ci_low": rs["auprc_ci_low"],
            "midstep_auprc_ci_high": rs["auprc_ci_high"], "prevalence": rs["prevalence"],
            "midstep_signal": bool(not np.isnan(rs["ci_low"]) and (rs["ci_low"] > CHANCE or rs["ci_high"] < CHANCE)),
            "midstep_perm_p": fast_perm_p(gs["outcome"].to_numpy(), gs["proj_mean"].to_numpy(),
                                          n_perm=args.n_perm, seed=400 + b),
            "n_traj": int(len(gs)),
            "mean_steps_in_bin_resolved": float(
                at_peak[(at_peak["bin"] == b) & (at_peak["outcome"] == 1)].groupby("trajectory_id").size().mean()),
            "mean_steps_in_bin_unresolved": float(
                at_peak[(at_peak["bin"] == b) & (at_peak["outcome"] == 0)].groupby("trajectory_id").size().mean()),
        })
    bins_df = pd.DataFrame(bins)
    bins_path = OUT_DIR / "agentic_position_bins_peak_layer.csv"
    bins_df.to_csv(bins_path, index=False)
    print(bins_df[["bin", "rel_pos_lo", "rel_pos_hi", "auroc_binmean", "auroc_midstep",
                   "auprc_midstep", "prevalence", "midstep_signal", "midstep_perm_p",
                   "n_traj"]].to_string(index=False), flush=True)
    print(f"  wrote {bins_path}", flush=True)

    # ---- AUROC by absolute step index (no length normalization at all) ----
    print("\nAUROC by absolute step index at the peak layer (first 12 steps)...", flush=True)
    steps_df = auroc_by_step(df, peak_layer, n_boot=args.n_boot)
    steps_path = OUT_DIR / "agentic_auroc_by_step_index_peak_layer.csv"
    steps_df.to_csv(steps_path, index=False)
    print(steps_df.head(12).to_string(index=False), flush=True)
    print(f"  wrote {steps_path}", flush=True)

    # ---- summary ----------------------------------------------------------
    summary = {
        "source_parquet": str(args.projections.name),
        "n_rows": int(len(df)),
        "n_trajectories": int(n_traj),
        "n_tasks": int(n_task),
        "single_seed_run": bool(traj_level["seed"].nunique() == 1),
        "resolution_rate": resolution_rate,
        "majority_baseline": float(max(resolution_rate, 1 - resolution_rate)),
        "layer_range": [int(df["layer"].min()), int(df["layer"].max())],
        "layer_indexing": "axis-row numbering, embedding-inclusive; layer 0 unavailable from cache",
        "final_step": {
            "peak_layer_post_hoc": peak_layer,
            "auroc": float(peak["auroc"]),
            "ci": [float(peak["ci_low"]), float(peak["ci_high"])],
            "auprc": float(peak["auprc"]),
            "auprc_ci": [float(peak["auprc_ci_low"]), float(peak["auprc_ci_high"])],
            "auprc_no_skill_prevalence": float(peak["prevalence"]),
            "signal": bool(peak["signal"]),
            "permutation_p": perm_p,
            "n": int(peak["n"]),
            "n_layers_with_signal": int(sweep_df["signal"].sum()),
            "auroc_min": float(sweep_df["auroc"].min()),
            "auroc_min_layer": int(sweep_df.loc[sweep_df["auroc"].idxmin(), "layer"]),
        },
        "position_bins": bins_df.to_dict(orient="records"),
        "auroc_by_step_index_head": steps_df.head(12).to_dict(orient="records"),
        "caveats": [
            "Peak layer selected post-hoc from the same trajectories used to report it; "
            "the peak-layer AUROC is optimistically biased and is exploratory, not held-out.",
            "Position bins: use auroc_midstep, not auroc_binmean. rel_pos bins span more "
            "absolute steps for longer (predominantly unresolved) trajectories, so averaging "
            "within a bin smooths the two classes unequally and manufactures a spurious "
            "'no early separation' pattern (bin 0: 0.52 binmean vs 0.72 midstep).",
            "Separation is present from step 0 onward, before the agent has meaningfully "
            "acted, which is consistent with the axis reading static task difficulty off the "
            "prompt rather than tracking trajectory progress. See confound_cascade.json.",
            "proj_mean here is cosine-of-meanvec (from the cached mean-pooled activations), "
            "not the original per-token mean-of-cosines.",
            "proj_final (single-final-token readout) is not recoverable from the cache.",
            "Single-seed run (one trajectory per task): no within-task contrast is possible.",
        ],
    }
    summary_path = OUT_DIR / "agentic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
