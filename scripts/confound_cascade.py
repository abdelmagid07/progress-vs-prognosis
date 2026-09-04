#!/usr/bin/env python
"""The paper's argument as one quantity, measured under progressively tighter conditioning.

Every headline number in the results section is the SAME estimand:

    theta = P(a resolved run scores above an unresolved one)

Pooled AUROC is exactly that probability with no conditioning (AUROC = P(X_pos > X_neg) +
0.5 P(tie)). The within-task result is the same probability conditioned on the two runs sharing
a task, which is what a paired sign count over the 50 pairs estimates. The length-stratified result
conditions on the two runs sharing a length quartile. So all five rows below can share one axis
with chance at 0.5 -- they are not different metrics that happen to be plotted together, and
the figure would be misleading if they were.

Reading down the rows, the apparent transfer is progressively removed by conditioning:

    all pairs, final step          0.747   the headline
    all pairs, at step 0           0.723   present before the agent acts
    same length stratum            ~0.60   most of it was length
    same task, final step          31/50   nothing left
    same task, trajectory mean      22/50  at chance

Intervals: BCa bootstrap over trajectories for the two unconditioned rows and the stratified
row (matching the rest of the paper); Wilson score intervals for the two paired rows, since
those are binomial counts over 50 pairs, not bootstrapped statistics.

Writes data/confound_cascade.json.

Reproduce: python scripts/confound_cascade.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_agentic_new_axis import _bca_bounds, fast_auroc  # noqa: E402

OUT_DIR = ROOT / "data"
ALPHA = 0.05


def stratified_auroc(outcome: np.ndarray, score: np.ndarray, stratum: np.ndarray) -> float:
    """P(resolved > unresolved | same stratum), pooled over strata.

    Weighted by the number of (resolved, unresolved) pairs each stratum contributes, which is
    the natural weighting: it is the probability for a pair drawn uniformly from all
    within-stratum discordant pairs. Strata containing only one class contribute no pairs and
    drop out on their own -- which is what happens to the longest length quartile, where no run
    was resolved.
    """
    num = den = 0.0
    for s in np.unique(stratum):
        m = stratum == s
        o, v = outcome[m], score[m]
        if len(np.unique(o)) < 2:
            continue
        pos, neg = v[o == 1], v[o == 0]
        n_pairs = len(pos) * len(neg)
        # fast_auroc is P(pos>neg)+0.5P(tie); multiply back up to a pair count.
        num += fast_auroc(o, v) * n_pairs
        den += n_pairs
    return float(num / den) if den else float("nan")


def bca(stat_fn, n: int, observed: float, *, n_boot: int, seed: int,
        alpha: float = ALPHA) -> tuple[float, float]:
    """BCa interval for a statistic over n resampled trajectories.

    alpha is a parameter so the same estimator can yield the 95% interval the paper reports and
    the 50% inner interval a two-tier forest plot draws. The seed is fixed, so calling this twice
    with different alphas regenerates identical bootstrap and jackknife draws and the two
    intervals are nested by construction rather than by luck.
    """
    rng = np.random.default_rng(seed)
    boot = np.array([stat_fn(rng.integers(0, n, n)) for _ in range(n_boot)])
    jack = np.array([stat_fn(np.delete(np.arange(n), i)) for i in range(n)])
    lo, hi = _bca_bounds(boot[~np.isnan(boot)], jack[~np.isnan(jack)], observed, alpha)
    return float(lo), float(hi)


def wilson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Wilson score interval -- the right one for small binomial counts."""
    z = norm.ppf(1 - alpha / 2)
    p, d = k / n, 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return float(centre - half), float(centre + half)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projections", type=Path, default=ROOT / "data" / "projections_full.parquet")
    ap.add_argument("--paired", type=Path, default=ROOT / "data" / "projections_pairs50.parquet")
    ap.add_argument("--layer", type=int, default=53)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--confidence-paired", type=Path,
                    default=ROOT / "data" / "confidence_pairs50.parquet")
    ap.add_argument("--probe-csv", type=Path,
                    default=OUT_DIR / "paired_probe_ceiling.csv")
    ap.add_argument("--probe-C", type=float, default=0.1,
                    help="Regularisation for the reference probe row. Default 0.1 matches the "
                         "prefix-probe analysis. NOTE this is a swept parameter, not a "
                         "pre-specified one: at layer 53 the probe scores 44/50 at C=0.01 and "
                         "42/50 at C=1.0, so quoting the best C would be selection.")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    full = pd.read_parquet(args.projections)
    L = full[full.layer == args.layer]

    # ---- row 1: pooled, final step ----------------------------------------
    fin = L.sort_values("step_index").groupby("trajectory_id", as_index=False).tail(1)
    o, s = fin.outcome.to_numpy(), fin.proj_mean.to_numpy()
    a_final = fast_auroc(o, s)
    ci_final = bca(lambda i: fast_auroc(o[i], s[i]), len(o), a_final,
                   n_boot=args.n_boot, seed=1)


    # ---- row 2: pooled, at step 0 -----------------------------------------
    st0 = L[L.step_index == 0]
    o0, s0 = st0.outcome.to_numpy(), st0.proj_mean.to_numpy()
    a_step0 = fast_auroc(o0, s0)
    ci_step0 = bca(lambda i: fast_auroc(o0[i], s0[i]), len(o0), a_step0,
                   n_boot=args.n_boot, seed=2)


    # ---- row 3: same length stratum (final step, length quartiles) --------
    lengths = L.groupby("trajectory_id").step_index.max().add(1)
    fin = fin.assign(n_steps=fin.trajectory_id.map(lengths))
    q = pd.qcut(fin.n_steps, 4, labels=False, duplicates="drop").to_numpy()
    ofs, sfs = fin.outcome.to_numpy(), fin.proj_mean.to_numpy()
    a_strat = stratified_auroc(ofs, sfs, q)
    ci_strat = bca(lambda i: stratified_auroc(ofs[i], sfs[i], q[i]), len(ofs), a_strat,
                   n_boot=args.n_boot, seed=3)

    strata = [{"quartile": int(k), "n": int((q == k).sum()),
               "n_resolved": int(ofs[q == k].sum()),
               "steps_min": int(fin.n_steps.to_numpy()[q == k].min()),
               "steps_max": int(fin.n_steps.to_numpy()[q == k].max()),
               "auroc": (float(fast_auroc(ofs[q == k], sfs[q == k]))
                         if len(np.unique(ofs[q == k])) > 1 else None)}
              for k in sorted(np.unique(q))]

    # ---- rows 4-5: same task (paired) -------------------------------------
    P = pd.read_parquet(args.paired)
    PL = P[P.layer == args.layer]
    pf = PL.sort_values("step_index").groupby(["task_id", "outcome"], as_index=False).tail(1)
    piv_f = pf.pivot_table(index="task_id", columns="outcome", values="proj_mean").dropna()
    k_final, n_pairs = int((piv_f[1] > piv_f[0]).sum()), len(piv_f)

    # TRUNCATED to the shorter arm, matching how the fitted-probe row below is constructed.
    # Without this the two within-task rows are not like for like: the probe's features truncate
    # and this one did not, and the difference is not cosmetic. The pair-difference vectors are
    # dominated by a length component -- their first principal component carries 35% of the
    # variance and correlates 0.79 with the difference in generated tokens -- so averaging over
    # unequal numbers of steps reads length. Untruncated this row is 16/50, a significant
    # reversal; truncated it is 22/50, at chance. The reversal was the length channel.
    n_steps = (PL.groupby(["task_id", "trajectory_id"]).step_index.size()
               .rename("n").reset_index())
    cap = n_steps.groupby("task_id").n.min().rename("cap")
    PT = PL.merge(cap, on="task_id")
    PT = PT[PT.step_index < PT.cap]
    pm = PT.groupby(["task_id", "outcome"], as_index=False).proj_mean.mean()
    piv_m = pm.pivot_table(index="task_id", columns="outcome", values="proj_mean").dropna()
    k_mean = int((piv_m[1] > piv_m[0]).sum())

    # --- reference row: a probe FIT in-domain, same 50 pairs -----------------
    # Not part of the cascade: this row varies the METHOD, not the conditioning, so it is drawn
    # below a divider. It is here because the cascade alone invites "there was never anything to
    # detect" -- this row shows the information is near ceiling exactly where the frozen axis
    # bottoms out. Read from paired_probe_ceiling.csv, which applies the step-count truncation
    # and L2 normalisation the probe result requires; paired_direction_control.py's all-steps
    # row omits the truncation and reads 18/18 here instead of 17/18.
    pc = pd.read_csv(args.probe_csv)
    cell = pc[(pc.layer_new_axis == args.layer) & (pc.C == args.probe_C)]
    if len(cell) != 1:
        raise SystemExit(f"probe cell not unique for layer {args.layer}, C={args.probe_C}")
    k_probe = int(cell.n_correct.iloc[0])
    n_probe = int(cell.n_pairs.iloc[0])

    # LABELS. The first three rows pool UNPAIRED trajectories, so calling them "All pairs" (as an
    # earlier version did) named the wrong design and invited exactly the confusion the figure
    # exists to prevent. "Across tasks" / "Within task" names the design directly, which is the
    # distinction the whole figure exists to draw.
    #
    # SUBLABELS carry one kind of thing only: the evidence base. They previously mixed editorial
    # commentary ("the headline result"), a method detail ("length quartiles") and sign counts
    # (a bare "n/50") in one column, which read as noise. The commentary belongs in the caption,
    # The paired rows also carry their sign count in parentheses. That is the same number as
# theta, but printing "22/50" alongside "0.440" shows what the decimal hides: the estimate is
# a count over fifty pairs, so it moves in steps of 0.02 and cannot be read as a
# continuous quantity measured to three digits.
    rows = [
        {"key": "pooled_final", "label": "Across tasks, final step",
         "sublabel": f"{len(o)} trajectories", "theta": a_final,
         "ci": list(ci_final), "n": int(len(o)), "kind": "auroc"},
        {"key": "pooled_step0", "label": "Across tasks, at step 0",
         "sublabel": f"{len(o0)} trajectories", "theta": a_step0,
         "ci": list(ci_step0), "n": int(len(o0)), "kind": "auroc"},
        {"key": "same_length", "label": "Across tasks, length-matched",
         "sublabel": f"{len(ofs)} trajectories", "theta": a_strat,
         "ci": list(ci_strat), "n": int(len(ofs)), "kind": "auroc_stratified"},
        {"key": "same_task_final", "label": "Within task, final step",
         "sublabel": f"{n_pairs} pairs ({k_final}/{n_pairs})", "theta": k_final / n_pairs,
         "ci": list(wilson(k_final, n_pairs)), "n": n_pairs, "kind": "paired"},
        {"key": "same_task_mean", "label": "Within task, value axis",
         "sublabel": f"{n_pairs} pairs ({k_mean}/{n_pairs})", "theta": k_mean / n_pairs,
         "ci": list(wilson(k_mean, n_pairs)), "n": n_pairs, "kind": "paired"},
        {"key": "probe_reference", "label": "Within task, fitted probe",
         "sublabel": f"{n_probe} pairs ({k_probe}/{n_probe})", "theta": k_probe / n_probe,
         "ci": list(wilson(k_probe, n_probe)), "n": n_probe, "kind": "reference"},
    ]

    # ---- within-pair comparison of the three signals ------------------------
    # Same 50 pairs, same evaluation (does the resolved arm rank above the unresolved one?),
    # three different signals. Reported with the EXACT SIGN TEST for all three rather than the
    # Wilcoxon used elsewhere for the continuous ones: the bar is a count of pairs, so the
    # comparable test is the one on counts. This is the figure that answers "the agent is just
    # clueless, so there was nothing to find" -- the report is at chance where the probe is not.
    conf = pd.read_parquet(args.confidence_paired)
    ct = conf.groupby(["task_id", "outcome"], as_index=False).p_success.mean()
    cpiv = ct.pivot_table(index="task_id", columns="outcome", values="p_success").dropna()
    k_elic = int((cpiv[1] > cpiv[0]).sum())

    comparison = [
        {"key": "axis", "label": "Frozen value axis", "n_correct": k_mean, "n_pairs": n_pairs,
         "sign_p": float(binomtest(k_mean, n_pairs, 0.5).pvalue)},
        {"key": "elicited", "label": "Stated $P$(success)", "n_correct": k_elic,
         "n_pairs": int(len(cpiv)),
         "sign_p": float(binomtest(k_elic, len(cpiv), 0.5).pvalue)},
        {"key": "probe", "label": "Fitted probe", "n_correct": k_probe, "n_pairs": n_probe,
         "sign_p": float(binomtest(k_probe, n_probe, 0.5).pvalue)},
    ]

    out = {
        "layer_new_axis": args.layer,
        "estimand": "P(resolved run scores above unresolved run), under increasing conditioning",
        "chance": 0.5,
        "n_boot": args.n_boot,
        "rows": rows,
        "within_pair_comparison": comparison,
        "length_strata": strata,
        "note": (
            "Pooled AUROC and the paired sign fraction estimate the same probability under "
            "different conditioning, which is what licenses a shared axis. Interval methods "
            "differ by row kind: BCa bootstrap over trajectories for AUROC rows, Wilson score "
            "for the binomial paired rows."
        ),
    }
    path = OUT_DIR / "confound_cascade.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for r in rows:
        print(f"{r['label']:<30s} {r['theta']:.3f}  "
              f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  n={r['n']}  ({r['sublabel']})")
    print("\nlength strata:", json.dumps(strata))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
