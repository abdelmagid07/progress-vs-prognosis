#!/usr/bin/env python
"""Does the fitted outcome direction respond to local errors, the way the axis does?

The paper reports two facts about two different directions, each measured on its own
terrain: the frozen value axis responds to error events but does not separate outcomes
within a task, and the fitted outcome direction separates outcomes within a task. That
is suggestive of a dissociation but does not establish one, because neither direction
has been measured on the other's terrain.

This runs the error-response analysis of error_event_response.py on BOTH directions over
the same trajectories and the same steps, so all four cells of

                        local error response     within-task outcome
    value axis                  ?                        ?
    outcome direction           ?                        ?

are measured the same way. Two readings are informative. If the outcome direction is
error-insensitive, the dissociation is double and the two directions are doing genuinely
different jobs. If it is error-sensitive, then "separate directions" is too strong a
claim for what the data show, and the paper should say so.

PROCEDURE, matching the error-event analysis exactly. Step t is: the agent generates a
response (whose activations give the projection S_t), then the environment returns
obs_t. So S_t precedes obs_t and cannot respond to it, and the forward difference

    dS(t) = S_{t+1} - S_t

is the first quantity that sees the event. Each trajectory contributes one paired
contrast, its mean delta after an error minus its mean delta after a clean step, and
those contrasts are aggregated across trajectories. The null permutes the is_error flags
within each trajectory, preserving every trajectory's step count, error count and delta
values, and destroying only where the errors land.

LEAKAGE. The outcome direction is fitted leave-one-pair-out: a pair's steps are always
projected onto a direction fitted without that pair. The value axis is frozen and never
fitted here.

Runs on the paired set, which is the only data with per-step activations stored, so both
directions can be projected from the same vectors.

READOUT. Projections here are the cosine between the axis and each step's mean-pooled
activation. The projections table used elsewhere instead stores the mean of the per-token
cosines, which is a different quantity: on these same trajectories the axis contrast is
-0.52 SD under the per-token readout and -0.71 SD under this one. The comparison below is
between two directions measured the same way, so the choice does not affect it, but the
axis figure here is not directly comparable to the one reported from the projections table.
For reference the script also reports the per-token axis contrast when the paired
projections table is available.

Reproduce: python scripts/error_response_outcome_direction.py
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_rule import is_error  # noqa: E402

OUT_DIR = ROOT / "data"


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def load_steps(npz_path: Path, block: int):
    """Per-step activation vectors at one captured block, with their metadata."""
    d = np.load(npz_path, allow_pickle=True)
    meta = pd.DataFrame({
        "task_id": d["task_id"].astype(str),
        "trajectory_id": d["trajectory_id"].astype(str),
        "outcome": d["outcome"].astype(int),
        "layer": d["layer"].astype(int),
        "step_index": d["step_index"].astype(int),
    })
    meta["row"] = np.arange(len(meta))
    m = meta[meta.layer == block].reset_index(drop=True)
    X = d["activations"][m.row.to_numpy()].astype(np.float32)
    return X, m.drop(columns=["row", "layer"])


def trajectory_vectors(X, meta):
    """One L2-normalised mean vector per trajectory, both arms truncated to the shorter."""
    n = meta.groupby(["task_id", "trajectory_id"]).step_index.size().rename("n").reset_index()
    cap = n.groupby("task_id").n.min().rename("cap")
    m = meta.merge(cap, on="task_id")
    m = m[m.step_index < m.cap]
    rows, V = [], []
    for (t, tr, o), g in m.groupby(["task_id", "trajectory_id", "outcome"]):
        V.append(unit(X[g.index.to_numpy()].mean(axis=0)))
        rows.append({"task_id": t, "trajectory_id": tr, "outcome": o})
    return np.vstack(V), pd.DataFrame(rows)


def fit_direction(X, y, C):
    """Decision direction in original feature space."""
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
    return unit(clf.coef_.ravel() / sc.scale_)


def loo_outcome_projections(X, meta, C):
    """Project every step onto an outcome direction fitted without that step's own pair."""
    V, vmeta = trajectory_vectors(X, meta)
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    proj = np.full(len(meta), np.nan)
    ranked = []
    for task in sorted(meta.task_id.unique()):
        tr = vmeta.task_id != task
        w = fit_direction(V[tr.to_numpy()], vmeta.loc[tr, "outcome"].to_numpy(), C)
        sel = (meta.task_id == task).to_numpy()
        proj[sel] = Xn[sel] @ w
        held = vmeta[~tr]
        if held.outcome.nunique() == 2:
            s = {int(r.outcome): float(V[i] @ w) for i, r in zip(held.index, held.itertuples())}
            ranked.append(s[1] > s[0])
    return proj, ranked


def load_errors(zip_path: Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            d = json.loads(z.read(name))
            for s in d.get("steps", []):
                rows.append({"trajectory_id": d["trajectory_id"],
                             "step_index": int(s["step_index"]),
                             "is_error": is_error(s.get("observation"))})
    return pd.DataFrame(rows)


def paired_contrast(df: pd.DataFrame, col: str) -> dict:
    per = []
    for _tid, g in df.groupby("trajectory_id"):
        a = g.loc[g.is_error, col].dropna()
        b = g.loc[~g.is_error, col].dropna()
        if len(a) >= 2 and len(b) >= 2:
            per.append(a.mean() - b.mean())
    per = np.array(per, dtype=float)
    stat = wilcoxon(per) if len(per) > 5 else None
    return {"n_trajectories": int(len(per)), "mean_contrast": float(per.mean()),
            "median_contrast": float(np.median(per)),
            "frac_negative": float((per < 0).mean()),
            "wilcoxon_p": float(stat.pvalue) if stat is not None else float("nan"),
            "contrasts": [float(v) for v in per]}


def shuffle_null(df: pd.DataFrame, col: str, *, n_perm: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    groups = [(g[col].to_numpy(), g.is_error.to_numpy()) for _, g in df.groupby("trajectory_id")]
    out = []
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
    return {"n_perm": int(n_perm), "null_mean": float(out.mean()), "null_sd": float(out.std()),
            "null_p2_5": float(np.percentile(out, 2.5)),
            "null_p97_5": float(np.percentile(out, 97.5))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activations", type=Path, default=ROOT / "data" / "pairs50_meanpool.npz")
    ap.add_argument("--trajectories", type=Path, default=ROOT / "data" / "normalized_pairs.zip")
    ap.add_argument("--axis", type=Path,
                    default=ROOT / "data" / "value_axis.npy")
    ap.add_argument("--layer", type=int, default=53, help="axis row")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--paired-projections", type=Path,
                    default=ROOT / "data" / "projections_pairs50.parquet",
                    help="optional: reports the axis contrast under the per-token readout too")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    block = args.layer - 1
    X, meta = load_steps(args.activations, block)
    print(f"{len(meta)} steps over {meta.trajectory_id.nunique()} trajectories "
          f"({meta.task_id.nunique()} pairs) at axis row {args.layer}")

    axis = unit(np.load(args.axis)[args.layer].astype(np.float32))
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    meta["axis"] = Xn @ axis
    meta["outcome_dir"], ranked = loo_outcome_projections(X, meta, args.C)
    n_correct = int(sum(ranked))
    n_pairs = len(ranked)
    print(f"outcome direction fitted leave-one-pair-out: ranks the resolved arm higher in "
          f"{n_correct}/{n_pairs} pairs")

    err = load_errors(args.trajectories)
    df = (meta.merge(err, on=["trajectory_id", "step_index"], how="inner")
              .sort_values(["trajectory_id", "step_index"]).reset_index(drop=True))
    for src, dst in (("axis", "d_axis"), ("outcome_dir", "d_outcome")):
        df[dst] = df.groupby("trajectory_id")[src].shift(-1) - df[src]
    print(f"{len(df)} steps matched to observations; "
          f"{df.is_error.mean()*100:.1f}% end in an error\n")

    # Within-task outcome separation for the frozen axis, on the same truncated vectors.
    V, vmeta = trajectory_vectors(X, meta[["task_id", "trajectory_id", "outcome", "step_index"]])
    axis_pairs = []
    for task, g in vmeta.groupby("task_id"):
        if g.outcome.nunique() != 2:
            continue
        s = {int(r.outcome): float(V[i] @ axis) for i, r in zip(g.index, g.itertuples())}
        axis_pairs.append(s[1] > s[0])
    axis_correct = int(sum(axis_pairs))

    out = {"axis_layer": args.layer, "C": args.C,
           "n_steps": int(len(df)), "n_trajectories": int(df.trajectory_id.nunique()),
           "n_pairs": n_pairs, "error_step_fraction": float(df.is_error.mean()),
           "within_task_outcome": {
               "value_axis": {"n_correct": axis_correct, "n_pairs": len(axis_pairs),
                              "sign_p": float(binomtest(axis_correct, len(axis_pairs), 0.5).pvalue)},
               "outcome_direction": {"n_correct": n_correct, "n_pairs": n_pairs,
                                     "sign_p": float(binomtest(n_correct, n_pairs, 0.5).pvalue)}}}

    for label, col, base in (("value axis", "d_axis", "axis"),
                             ("outcome direction", "d_outcome", "outcome_dir")):
        r = paired_contrast(df, col)
        r["shuffle_null"] = shuffle_null(df, col, n_perm=args.n_perm)
        sd = float(df[base].std())
        r["signal_sd"] = sd
        r["mean_contrast_in_sd_units"] = r["mean_contrast"] / sd if sd else float("nan")
        nl = r["shuffle_null"]
        r["null_sds_from_zero"] = ((r["mean_contrast"] - nl["null_mean"]) / nl["null_sd"]
                                   if nl["null_sd"] else float("nan"))
        out["error_response_" + base] = r
        print(f"{label}: change across the event, error minus clean")
        print(f"   mean {r['mean_contrast']:+.5f} ({r['mean_contrast_in_sd_units']:+.2f} SD)   "
              f"lower after an error in {r['frac_negative']*100:.0f}% of "
              f"{r['n_trajectories']} trajectories   Wilcoxon p={r['wilcoxon_p']:.3g}")
        print(f"   {abs(r['null_sds_from_zero']):.1f} null SDs from the shuffled-label null\n")

    print("2x2 (rows: direction, cols: what it tracks)")
    print(f"  {'':<20}{'local error response':>26}{'within-task outcome':>24}")
    a, o = out["error_response_axis"], out["error_response_outcome_dir"]
    w = out["within_task_outcome"]
    print(f"  {'value axis':<20}{a['mean_contrast_in_sd_units']:>+18.2f} SD, p={a['wilcoxon_p']:.1g}"
          f"{w['value_axis']['n_correct']:>15}/{w['value_axis']['n_pairs']}, "
          f"p={w['value_axis']['sign_p']:.2g}")
    print(f"  {'outcome direction':<20}{o['mean_contrast_in_sd_units']:>+18.2f} SD, p={o['wilcoxon_p']:.1g}"
          f"{w['outcome_direction']['n_correct']:>15}/{w['outcome_direction']['n_pairs']}, "
          f"p={w['outcome_direction']['sign_p']:.2g}")

    # Same trajectories, the per-token readout the projections table stores, so the two
    # axis figures can be compared directly rather than looking like a discrepancy.
    if args.paired_projections.exists():
        pp = pd.read_parquet(args.paired_projections)
        pa = (pp[pp.layer == args.layer][["trajectory_id", "step_index", "proj_mean"]]
              .rename(columns={"proj_mean": "axis_pertoken"}))
        pdf = (pa.merge(err, on=["trajectory_id", "step_index"], how="inner")
                 .sort_values(["trajectory_id", "step_index"]).reset_index(drop=True))
        pdf["d"] = pdf.groupby("trajectory_id")["axis_pertoken"].shift(-1) - pdf["axis_pertoken"]
        pr = paired_contrast(pdf, "d")
        psd = float(pdf["axis_pertoken"].std())
        pr["signal_sd"] = psd
        pr["mean_contrast_in_sd_units"] = pr["mean_contrast"] / psd if psd else float("nan")
        pr.pop("contrasts", None)
        out["error_response_axis_per_token_readout"] = pr
        print(f"value axis under the per-token readout: {pr['mean_contrast_in_sd_units']:+.2f} SD, "
              f"lower in {pr['frac_negative']*100:.0f}% of {pr['n_trajectories']} trajectories\n")

    path = OUT_DIR / "error_response_outcome_direction.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
