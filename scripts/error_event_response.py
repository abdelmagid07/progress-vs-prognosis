#!/usr/bin/env python
"""When the agent hits an explicit error, does anything move?

The paper's claims about progress monitoring are made at the level of whole trajectories: the
axis separates outcomes pooled but not within task, and verbalized confidence is miscalibrated
and at chance within pairs. Neither says what happens at the moment something visibly goes
wrong. This measures the event-locked response to an error appearing in the environment's
output -- a traceback, a non-zero return code, a missing module or file -- for both signals, on
all 499 trajectories and roughly 9,000 steps.

TIMING. Step $t$ is: the agent generates a response (whose activations give the axis projection
$A_t$), then the environment returns $\\text{obs}_t$. So $A_t$ is computed BEFORE $\\text{obs}_t$
and cannot respond to it; $A_{t+1}$ is the first activation that sees it. To avoid any ambiguity
about whether the elicitation prefix at step $t$ includes $\\text{obs}_t$, both signals are
measured the same way, as a forward difference across the event:

    dA(t) = A_{t+1} - A_t          dP(t) = P_{t+1} - P_t

and compared between steps where obs_t contained an error and steps where it did not.

WITHIN TRAJECTORY. Trajectories differ in overall level and in drift, and error-heavy runs are
also longer and more likely to fail, so a pooled comparison of the two groups would recover that
rather than an event response. Each trajectory therefore contributes one paired contrast --- its
mean delta after an error minus its mean delta after a clean step --- and those contrasts are
aggregated across trajectories.

Writes data/error_event_response.json.

Reproduce: python scripts/error_event_response.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_rule import is_error  # noqa: E402

OUT_DIR = ROOT / "data"


def load_errors(traj_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(traj_dir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for s in d.get("steps", []):
            rows.append({"trajectory_id": d["trajectory_id"],
                         "step_index": int(s["step_index"]),
                         "is_error": is_error(s.get("observation"))})
    return pd.DataFrame(rows)


def paired_contrast(df: pd.DataFrame, col: str) -> dict:
    """One contrast per trajectory: mean delta after an error minus after a clean step."""
    per = []
    for tid, g in df.groupby("trajectory_id"):
        a = g.loc[g.is_error, col].dropna()
        b = g.loc[~g.is_error, col].dropna()
        if len(a) >= 2 and len(b) >= 2:
            per.append(a.mean() - b.mean())
    per = np.array(per, dtype=float)
    stat = wilcoxon(per) if len(per) > 5 else None
    return {"n_trajectories": int(len(per)), "mean_contrast": float(per.mean()),
            "median_contrast": float(np.median(per)),
            "frac_negative": float((per < 0).mean()),
            "wilcoxon_p": float(stat.pvalue) if stat is not None else float("nan")}


def shuffle_null(df: pd.DataFrame, col: str, *, n_perm: int, seed: int = 0) -> dict:
    """Null for the event response: relabel which steps are errors, within each trajectory.

    The observed contrast could in principle come from drift plus arbitrary step selection --
    the signal wanders over a run, errors are not uniformly placed, and differencing two
    non-random subsets of steps could produce a gap with no event response at all. Permuting
    the is_error flags WITHIN each trajectory keeps every trajectory's step count, its error
    count and its actual delta values, and destroys only which steps the errors land on.

    Not the same as a random-direction control, which would ask whether any direction responds
    this way and needs per-step projections of all 499 trajectories onto random vectors. Those
    activations are not in the repository; only the 36 paired trajectories are stored.
    """
    rng = np.random.default_rng(seed)
    out = []
    groups = [(g[col].to_numpy(), g["is_error"].to_numpy()) for _, g in df.groupby("trajectory_id")]
    for _ in range(n_perm):
        per = []
        for vals, flags in groups:
            f = rng.permutation(flags)
            a_, b_ = vals[f], vals[~f]
            a_, b_ = a_[~np.isnan(a_)], b_[~np.isnan(b_)]
            if len(a_) >= 2 and len(b_) >= 2:
                per.append(a_.mean() - b_.mean())
        out.append(float(np.mean(per)))
    out = np.array(out)
    return {"n_perm": int(n_perm), "null_mean": float(out.mean()),
            "null_sd": float(out.std()), "null_p2_5": float(np.percentile(out, 2.5)),
            "null_p97_5": float(np.percentile(out, 97.5)), "samples": [float(v) for v in out]}


def per_trajectory_null(df: pd.DataFrame, col: str, *, n_perm: int = 20, seed: int = 1) -> dict:
    """Null on the SAME scale as one trajectory's contrast, so it can be drawn on the histogram.

    shuffle_null() collapses each permutation to the mean across trajectories, which is the right
    null for that mean but is far narrower than a single trajectory's contrast -- overlaying it
    on the per-trajectory distribution would compare two different quantities.

    Here each permuted trajectory contributes its OWN contrast, exactly as the observed points do,
    so the two distributions are directly comparable. Replicates are pooled rather than averaged
    within a trajectory: averaging would shrink the null's width by sqrt(n_perm) and make the
    observed shift look larger than it is. The figure divides the counts by n_perm to put the
    pooled null on the same vertical scale as the 433 observed points.
    """
    rng = np.random.default_rng(seed)
    groups = [(g[col].to_numpy(), g["is_error"].to_numpy()) for _, g in df.groupby("trajectory_id")]
    out = []
    for _ in range(n_perm):
        for vals, flags in groups:
            f = rng.permutation(flags)
            a_, b_ = vals[f], vals[~f]
            a_, b_ = a_[~np.isnan(a_)], b_[~np.isnan(b_)]
            if len(a_) >= 2 and len(b_) >= 2:
                out.append(float(a_.mean() - b_.mean()))
    arr = np.array(out)
    return {"n_perm": int(n_perm), "contrasts": out,
            "frac_negative": float((arr < 0).mean()), "mean": float(arr.mean())}


def peri_event(df: pd.DataFrame, col: str, *, window: int = 3) -> dict:
    """Event-locked time course: the signal at each offset around an error observation.

    The contrast statistic answers "does it move?"; this answers "what does the movement look
    like?", which is the claim the section actually makes. A signal that tracks CURRENT STATE
    should move at the event and then relax back; one that accumulates evidence about the
    eventual outcome should step and stay stepped.

    TIMING follows the module docstring: obs_t is returned after the step-t response, so A_t is
    the last value computed BEFORE the error is visible and A_{t+1} is the first that can react.
    Offsets -window..0 are therefore all pre-event, and each curve is baselined on their MEAN --
    not on offset 0 alone, which would pin one point to zero and hide the approach.

    Clean steps are aligned the same way. This is not a neutral control: a clean observation is
    a command that worked, so it is the opposite event, and the two curves together show whether
    the signal is bidirectional or only drops.

    Averaged per trajectory first, then across trajectories: error-heavy runs are longer, so
    pooling events would weight them more and recover run-level differences instead of the event
    response. The difference curve uses only trajectories contributing BOTH event types, so it is
    paired and cannot be produced by which runs enter which condition.
    """
    offs = list(range(-window, window + 1))
    pre = slice(0, window + 1)                      # offsets -window .. 0, all pre-event
    per_traj = {True: {}, False: {}}
    for tid, g in df.groupby("trajectory_id"):
        g = g.sort_values("step_index")
        v = g[col].to_numpy(dtype=float)
        flags = g["is_error"].to_numpy()
        for want in (True, False):
            curves = []
            for i in np.flatnonzero(flags == want):
                if i - window < 0 or i + window >= len(v):
                    continue
                seg = v[i - window: i + window + 1]
                if np.isnan(seg).any():
                    continue
                curves.append(seg - seg[pre].mean())
            if curves:
                per_traj[want][tid] = np.mean(curves, axis=0)

    out = {"offsets": offs, "window": window}
    for name, want in (("error", True), ("clean", False)):
        A = np.vstack(list(per_traj[want].values()))
        out[name] = {"n_trajectories": int(A.shape[0]),
                     "mean": [float(x) for x in A.mean(0)],
                     "sem": [float(x) for x in A.std(0, ddof=1) / np.sqrt(A.shape[0])]}
    both = sorted(set(per_traj[True]) & set(per_traj[False]))
    D = np.vstack([per_traj[True][t] - per_traj[False][t] for t in both])
    out["difference"] = {"n_trajectories": int(D.shape[0]),
                         "mean": [float(x) for x in D.mean(0)],
                         "sem": [float(x) for x in D.std(0, ddof=1) / np.sqrt(D.shape[0])],
                         "wilcoxon_p": [float(wilcoxon(D[:, j]).pvalue) if np.any(D[:, j]) else 1.0
                                        for j in range(D.shape[1])]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectories", type=Path,
                    default=ROOT / "data" / "normalized")
    ap.add_argument("--projections", type=Path, default=ROOT / "data" / "projections_full.parquet")
    ap.add_argument("--confidence", type=Path,
                    default=ROOT / "data" / "confidence.parquet")
    ap.add_argument("--layer", type=int, default=53)
    ap.add_argument("--n-perm", type=int, default=1000,
                    help="within-trajectory relabelling permutations for the null")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    err = load_errors(args.trajectories)
    proj = pd.read_parquet(args.projections)
    axis = (proj[proj.layer == args.layer][["trajectory_id", "step_index", "proj_mean", "outcome"]]
            .rename(columns={"proj_mean": "axis"}))
    conf = pd.read_parquet(args.confidence)[["trajectory_id", "step_index", "p_success"]]

    df = (axis.merge(conf, on=["trajectory_id", "step_index"], how="inner")
               .merge(err, on=["trajectory_id", "step_index"], how="inner")
               .sort_values(["trajectory_id", "step_index"]).reset_index(drop=True))
    # forward differences across the event, within trajectory
    for src, dst in (("axis", "d_axis"), ("p_success", "d_conf")):
        df[dst] = df.groupby("trajectory_id")[src].shift(-1) - df[src]
    print(f"{len(df)} steps over {df.trajectory_id.nunique()} trajectories; "
          f"{df.is_error.mean()*100:.1f}% of steps end in an error observation\n")

    out = {"layer_new_axis": args.layer, "n_steps": int(len(df)),
           "n_trajectories": int(df.trajectory_id.nunique()),
           "error_step_fraction": float(df.is_error.mean())}

    for label, col in (("frozen axis projection", "d_axis"),
                       ("verbalized P(success)", "d_conf")):
        r = paired_contrast(df, col)
        # Per-trajectory contrasts kept so a figure can show the DISTRIBUTION rather than a
        # mean: "80% of trajectories move down" is a claim about a distribution's mass.
        per = []
        for tid, g in df.groupby("trajectory_id"):
            x = g.loc[g.is_error, col].dropna()
            y = g.loc[~g.is_error, col].dropna()
            if len(x) >= 2 and len(y) >= 2:
                per.append(float(x.mean() - y.mean()))
        r["contrasts"] = per
        r["shuffle_null"] = shuffle_null(df, col, n_perm=args.n_perm)
        r["per_trajectory_null"] = per_trajectory_null(df, col)
        r["peri_event"] = peri_event(df, {"d_axis": "axis", "d_conf": "p_success"}[col])
        out[col] = r
        print(f"{label}: change across the event, error minus clean")
        print(f"   mean {r['mean_contrast']:+.5f}   median {r['median_contrast']:+.5f}   "
              f"lower after an error in {r['frac_negative']*100:.0f}% of "
              f"{r['n_trajectories']} trajectories   Wilcoxon p={r['wilcoxon_p']:.3g}")

    # scale reference: how big is that change relative to each signal's own spread?
    for label, col, base in (("axis", "d_axis", "axis"), ("confidence", "d_conf", "p_success")):
        sd = float(df[base].std())
        out[f"{col}_in_sd_units"] = float(out[col]["mean_contrast"] / sd) if sd else float("nan")
        # Kept so a figure can put both signals on one standardised axis without re-deriving it.
        out[col]["signal_sd"] = sd
        print(f"   ({label}: contrast is {out[f'{col}_in_sd_units']:+.3f} "
              f"of the signal's own SD)")

    path = OUT_DIR / "error_event_response.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
