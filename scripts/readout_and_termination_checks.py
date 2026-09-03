#!/usr/bin/env python
"""Why the final step, and what the final step is made of.

Two questions a reader should ask about the headline number that the other confound scripts
do not answer.

1. WHY THIS READOUT? The final step is the pre-specified gate (fixed in the proposal before
   any agentic data existed), which is what protects it from readout shopping. But
   pre-specified is not the same as conservative, and it turns out to be the single most
   favourable pooled summary of a trajectory available: better than the trajectory mean, the
   min, the max, the last three steps, and step 0. That should be stated rather than left for
   a reader to discover, so this script tabulates every natural readout at the headline layer.

2. WHAT DOES "FINAL STEP" MEAN? It is not the same kind of event in both classes. Every
   resolved run ends by submitting a patch, but 115 of 439 unresolved runs (26%) end in
   ``LimitsExceeded`` -- the harness cutting them off at the 60-step budget -- and no resolved
   run does. For a quarter of the unresolved class the "final step" is therefore an arbitrary
   truncation, not a conclusion the agent reached, and the readout is partly comparing
   submissions against interruptions.

   Restricting to trajectories that terminate the same way in both arms (``Submitted``) tests
   this. It also partially controls length, because budget-exhausted runs are by construction
   the longest ones -- the two cannot be cleanly separated in this design, and the script says
   so rather than claiming a pure termination effect.

Writes data/readout_and_termination.json.

Reproduce: python scripts/readout_and_termination_checks.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_agentic_new_axis import fast_auroc, fast_auroc_bca  # noqa: E402

OUT_DIR = ROOT / "data"


def load_exit_status(traj_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(traj_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"trajectory_id": d["trajectory_id"], "exit_status": d.get("exit_status")})
    return pd.DataFrame(rows)


def readouts(L: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every natural way to reduce a trajectory's per-step projections to one number."""
    keys = ["trajectory_id", "outcome", "exit_status"]
    srt = L.sort_values("step_index")
    return {
        "final_step": srt.groupby("trajectory_id", as_index=False).tail(1),
        "last_3_steps": (srt.groupby("trajectory_id", as_index=False).tail(3)
                         .groupby(keys, as_index=False).proj_mean.mean()),
        "step_0": L[L.step_index == 0],
        "trajectory_mean": L.groupby(keys, as_index=False).proj_mean.mean(),
        "trajectory_min": L.groupby(keys, as_index=False).proj_mean.min(),
        "trajectory_max": L.groupby(keys, as_index=False).proj_mean.max(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projections", type=Path, default=ROOT / "data" / "projections_full.parquet")
    ap.add_argument("--trajectories", type=Path,
                    default=ROOT / "data" / "normalized")
    ap.add_argument("--layer", type=int, default=53)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ex = load_exit_status(args.trajectories)
    proj = pd.read_parquet(args.projections)
    L = proj[proj.layer == args.layer].merge(ex, on="trajectory_id", how="left")
    if L.exit_status.isna().any():
        raise SystemExit("some projected trajectories have no exit_status; check the traj dir")

    traj = L.drop_duplicates("trajectory_id")
    xtab = (pd.crosstab(traj.exit_status, traj.outcome)
            .rename(columns={0: "unresolved", 1: "resolved"}))
    print("termination type by outcome (projected set):")
    print(xtab.to_string(), "\n")

    rs = readouts(L)

    # --- readout comparison, all trajectories -------------------------------
    table = []
    for name, d in rs.items():
        table.append({"readout": name, "n": int(len(d)), "n_resolved": int(d.outcome.sum()),
                      "auroc_all": fast_auroc(d.outcome.to_numpy(), d.proj_mean.to_numpy())})
    table.sort(key=lambda r: -r["auroc_all"])
    print("readout comparison (all trajectories), layer", args.layer)
    for r in table:
        print(f"  {r['readout']:<18s} {r['auroc_all']:.4f}")

    # --- same-termination restriction ---------------------------------------
    # Both arms end the same way, so 'final step' denotes the same kind of event in each.
    print("\nrestricted to Submitted (both arms terminate the same way):")
    restricted = {}
    for name, d in rs.items():
        s = d[d.exit_status == "Submitted"]
        r = fast_auroc_bca(s.outcome.to_numpy(), s.proj_mean.to_numpy(),
                           n_boot=args.n_boot, seed=args.layer)
        restricted[name] = {
            "n": int(len(s)), "n_resolved": int(s.outcome.sum()),
            "auroc": r["auroc"], "ci": [r["ci_low"], r["ci_high"]],
            "auprc": r["auprc"], "prevalence": r["prevalence"],
        }
        print(f"  {name:<18s} {r['auroc']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    for row in table:
        row["auroc_submitted_only"] = restricted[row["readout"]]["auroc"]

    full_final = fast_auroc_bca(rs["final_step"].outcome.to_numpy(),
                                rs["final_step"].proj_mean.to_numpy(),
                                n_boot=args.n_boot, seed=args.layer)
    out = {
        "layer_new_axis": args.layer,
        "termination_by_outcome": xtab.to_dict(),
        "n_limits_exceeded_unresolved": int(xtab.loc["LimitsExceeded", "unresolved"])
        if "LimitsExceeded" in xtab.index else 0,
        "n_limits_exceeded_resolved": int(xtab.loc["LimitsExceeded", "resolved"])
        if "LimitsExceeded" in xtab.index else 0,
        "readout_comparison": table,
        "final_step_all": {"auroc": full_final["auroc"],
                           "ci": [full_final["ci_low"], full_final["ci_high"]],
                           "n": int(len(rs["final_step"]))},
        "same_termination_submitted_only": restricted,
        "interpretation": (
            "The final step is the most favourable pooled readout of the trajectory, and it is "
            "also the one most inflated by termination asymmetry: every resolved run ends by "
            "submitting, while 26% of unresolved runs end by exhausting the step budget, so "
            "the readout partly compares submissions against interruptions. Holding termination "
            "type fixed, the ordering inverts -- the final step becomes the WORST of the three "
            "main readouts rather than the best, while step 0 is essentially unchanged because "
            "it precedes any termination. CAVEAT: budget-exhausted runs are by definition the "
            "longest, so this restriction also partially controls length; termination type and "
            "length cannot be cleanly separated in this design."
        ),
    }
    path = OUT_DIR / "readout_and_termination.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
