#!/usr/bin/env python
"""Within-task paired contrast on the same-task resolved/unresolved trajectory pairs.

Implements the experimental-setup spec for the paired sub-study:

  "For the N same-task pairs, we compare the alignment scores of the resolved and
   unresolved trajectories within each task. We use a Wilcoxon signed-rank test on
   the paired differences and report the number of pairs in which the resolved
   trajectory has a higher score using an exact sign test."

WHY THIS DESIGN ANSWERS SOMETHING THE 500-TASK RUN CANNOT. On the full benchmark the
axis separates outcomes, but three confounds fire (see
data/agentic_confound_checks.json): trajectory length alone
out-predicts the axis, separation is already present at step 0 before the agent has
acted, and the effect largely dissolves inside length strata. All are symptoms of the
same alternative explanation -- the axis may be reading *static task difficulty* off
the prompt rather than tracking how a run is going. Pairing resolved against
unresolved trajectories of the SAME task holds difficulty exactly fixed, so a
within-pair effect cannot be explained by it. Here the two members of a pair differ
only by temperature sampling noise (same task, same prompt, no seed change), which is
the cleanest available contrast.

STATISTICS. Both tests are paired and distribution-free, matching the spec:
  * Wilcoxon signed-rank on the per-pair differences -- primary, since it uses the
    magnitude of each difference, not just its sign.
  * Exact binomial sign test on the count of pairs where resolved > unresolved --
    secondary and deliberately conservative; it throws away magnitude and therefore
    needs 33/50 pairs to reach p < 0.05.
No bootstrap/BCa here: those are for the 500-task benchmark's AUROC/AUPRC. The paired
sub-study is a different question with a different, smaller design.

POWER. With 50 pairs the sign test needs 33/50 for p < 0.05, and a true within-pair
rate equal to the pooled AUROC (0.747) would yield ~0.94 power. A null result at this
size is therefore informative rather than merely underpowered, and this script prints
the achieved counts so that can be stated honestly.

Reproduce: python scripts/paired_within_task_contrast.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data"

READOUTS = {
    "final": "score at the trajectory's final step",
    "mean": "mean score over all of the trajectory's steps",
}


def trajectory_scores(df: pd.DataFrame, layer: int, readout: str) -> pd.DataFrame:
    """One score per trajectory at ``layer`` under the chosen readout."""
    d = df[df["layer"] == layer]
    if readout == "final":
        d = d.sort_values("step_index").groupby("trajectory_id", as_index=False).tail(1)
        out = d[["task_id", "trajectory_id", "outcome", "proj_mean"]].copy()
    elif readout == "mean":
        out = (
            d.groupby(["task_id", "trajectory_id"], as_index=False)
            .agg(outcome=("outcome", "first"), proj_mean=("proj_mean", "mean"))
        )
    else:
        raise ValueError(f"unknown readout {readout!r}")
    return out.rename(columns={"proj_mean": "score"})


def build_pairs(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per task carrying its resolved and unresolved score.

    Tasks that are not genuinely mixed are dropped loudly rather than silently
    averaged, since a non-mixed task contributes nothing to a within-task contrast.
    If a task has several trajectories per outcome, their scores are averaged so each
    task still contributes exactly one paired difference (equal weight per task).
    """
    rows = []
    for task, g in scores.groupby("task_id"):
        res = g[g["outcome"] == 1]["score"]
        unres = g[g["outcome"] == 0]["score"]
        if len(res) == 0 or len(unres) == 0:
            continue
        rows.append({
            "task_id": task,
            "score_resolved": float(res.mean()),
            "score_unresolved": float(unres.mean()),
            "n_resolved": int(len(res)),
            "n_unresolved": int(len(unres)),
        })
    pairs = pd.DataFrame(rows)
    if len(pairs):
        pairs["diff"] = pairs["score_resolved"] - pairs["score_unresolved"]
    return pairs


def paired_tests(diffs: np.ndarray) -> dict:
    """Wilcoxon signed-rank (primary) + exact binomial sign test (secondary)."""
    n = len(diffs)
    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    n_tie = int(np.sum(diffs == 0))

    res: dict = {
        "n_pairs": n,
        "n_resolved_higher": n_pos,
        "n_unresolved_higher": n_neg,
        "n_ties": n_tie,
        "mean_diff": float(np.mean(diffs)) if n else float("nan"),
        "median_diff": float(np.median(diffs)) if n else float("nan"),
    }

    # Wilcoxon needs at least one non-zero difference.
    nz = diffs[diffs != 0]
    if len(nz) >= 1:
        # Exact method when small and untied, which is the case at n=20.
        stat, p = wilcoxon(nz, alternative="two-sided",
                           mode="exact" if len(nz) <= 25 else "auto")
        res["wilcoxon_stat"] = float(stat)
        res["wilcoxon_p"] = float(p)
    else:
        res["wilcoxon_stat"] = float("nan")
        res["wilcoxon_p"] = float("nan")

    # Exact sign test over non-tied pairs.
    if n_pos + n_neg > 0:
        bt = binomtest(n_pos, n_pos + n_neg, 0.5, alternative="two-sided")
        res["sign_test_p"] = float(bt.pvalue)
        res["sign_test_prop_resolved_higher"] = n_pos / (n_pos + n_neg)
    else:
        res["sign_test_p"] = float("nan")
        res["sign_test_prop_resolved_higher"] = float("nan")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projections", type=Path,
                    default=ROOT / "data" / "projections_pairs50.parquet")
    ap.add_argument("--readout", choices=sorted(READOUTS), default="final")
    ap.add_argument("--layer", type=int, default=53,
                    help="Layer to report as the headline. Default: the layer with the "
                         "smallest Wilcoxon p across the sweep, which is POST-HOC and "
                         "labelled as such in the output.")
    ap.add_argument("--out-prefix", default="paired_within_task")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.projections)
    layers = sorted(df["layer"].unique())
    print(f"{len(df):,} rows | {df['trajectory_id'].nunique()} trajectories "
          f"| {df['task_id'].nunique()} tasks | layers {min(layers)}-{max(layers)}", flush=True)
    print(f"readout: {args.readout} ({READOUTS[args.readout]})\n", flush=True)

    # ---- sweep every layer -------------------------------------------------
    sweep = []
    for L in layers:
        pairs = build_pairs(trajectory_scores(df, L, args.readout))
        if not len(pairs):
            continue
        r = paired_tests(pairs["diff"].to_numpy())
        r["layer"] = int(L)
        sweep.append(r)
    sweep_df = pd.DataFrame(sweep)
    cols = ["layer", "n_pairs", "n_resolved_higher", "wilcoxon_p", "sign_test_p",
            "mean_diff", "median_diff"]
    sweep_df = sweep_df[cols + [c for c in sweep_df.columns if c not in cols]]
    sweep_path = OUT_DIR / f"{args.out_prefix}_by_layer_{args.readout}.csv"
    sweep_df.to_csv(sweep_path, index=False)

    n_pairs = int(sweep_df["n_pairs"].iloc[0])
    need = next((k for k in range(n_pairs // 2, n_pairs + 1)
                 if binomtest(k, n_pairs, 0.5, alternative="two-sided").pvalue < 0.05), None)
    print(f"{n_pairs} usable pairs. Sign test needs {need}/{n_pairs} for p<0.05.\n", flush=True)

    print(sweep_df[cols].to_string(index=False), flush=True)
    print(f"\nwrote {sweep_path}", flush=True)

    # ---- headline layer ----------------------------------------------------
    if args.layer is not None:
        head = sweep_df[sweep_df["layer"] == args.layer]
        if not len(head):
            raise SystemExit(f"layer {args.layer} not present")
        head = head.iloc[0]
        selection = "pre-specified"
    else:
        head = sweep_df.loc[sweep_df["wilcoxon_p"].idxmin()]
        selection = "POST-HOC (smallest Wilcoxon p across the sweep)"

    print(f"\n=== headline: layer {int(head['layer'])} [{selection}] ===")
    print(f"  pairs with resolved higher : {int(head['n_resolved_higher'])}/{n_pairs}")
    print(f"  Wilcoxon signed-rank       : W={head['wilcoxon_stat']:.1f}, p={head['wilcoxon_p']:.4g}")
    print(f"  exact sign test            : p={head['sign_test_p']:.4g}")
    print(f"  mean paired difference     : {head['mean_diff']:+.4f}")

    n_sig = int((sweep_df["wilcoxon_p"] < 0.05).sum())
    print(f"\n  layers with Wilcoxon p<0.05: {n_sig}/{len(sweep_df)} "
          f"(uncorrected; the sweep is exploratory)")

    summary = {
        "readout": args.readout,
        "n_pairs": n_pairs,
        "sign_test_pairs_needed_for_p05": need,
        "headline_layer": int(head["layer"]),
        "headline_layer_selection": selection,
        "headline": {k: (float(head[k]) if isinstance(head[k], (int, float, np.floating)) else head[k])
                     for k in ("n_resolved_higher", "wilcoxon_stat", "wilcoxon_p",
                               "sign_test_p", "mean_diff", "median_diff")},
        "n_layers_wilcoxon_p_lt_05": n_sig,
        "n_layers_total": int(len(sweep_df)),
        "caveats": [
            "Pair members differ only by temperature sampling (no seed change), so task "
            "difficulty is held exactly fixed -- this is the confound-free contrast.",
            "Mixed-outcome tasks are a non-random, intermediate-difficulty subset; the "
            "contrast is interpreted within that scope.",
            f"With {n_pairs} pairs the sign test needs {need} to reach p<0.05, and power "
            "is ~0.6 even if the true within-pair rate matched the pooled AUROC. A null "
            "here is inconclusive, not evidence of absence.",
        ],
    }
    if args.layer is None:
        summary["caveats"].append(
            "Headline layer chosen post-hoc by smallest p across the sweep; that p is "
            "optimistically biased and is not corrected for 64 comparisons."
        )
    out = OUT_DIR / f"{args.out_prefix}_summary_{args.readout}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
