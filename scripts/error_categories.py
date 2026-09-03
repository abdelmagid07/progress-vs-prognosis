#!/usr/bin/env python
"""Break failed commands into kinds, and check the axis response is not driven by one of them.

error_rule.py settles WHETHER a step failed (non-zero exit status). This asks WHAT failed, for
two reasons. A reviewer should be able to see that "error" covers a spread of ordinary failure
modes rather than one dominant pattern. And if the axis response came entirely from, say,
tracebacks while test failures did nothing, "responds to local evidence of failure" would be too
broad a description of it.

Categories are assigned within failed steps only, first match wins, so they partition rather than
overlap. The order runs from most specific to least: a traceback that also mentions a missing file
is a traceback. Steps that exit non-zero without matching any pattern land in "other", which is
reported rather than hidden -- a large "other" would mean the taxonomy is not describing the data.

The per-category effect is the same forward difference the headline uses, A_{t+1} - A_t, pooled
over events rather than averaged per trajectory. Pooling is the right choice here because the
question is about categories, not runs, and some categories are rare enough that a per-trajectory
mean would be undefined for most trajectories.

NO TEST-FAILURE CATEGORY. One was tried and removed: a test-runner signature appears in 4 of the
12,464 steps, and an explicit FAILED line in one. This agent almost never executes the test suite,
so failing tests are not an error category here -- "error" in this setting means a command or
script that did not run, not a test that did not pass. The counter is kept in the output so the
claim is checkable rather than asserted.

Writes data/error_categories.json.

Reproduce: python scripts/error_categories.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
from error_rule import is_error  # noqa: E402

OUT_DIR = ROOT / "results" / "numbers" / "new_axis"

# First match wins; most specific first.
CATEGORIES = [
    ("traceback / exception", re.compile(r"Traceback \(most recent call last\)|"
                                         r"\b[A-Za-z_]*(Error|Exception):", re.M)),
    ("missing file or path", re.compile(r"No such file or directory|"
                                        r"cannot find|does not exist", re.I)),
    ("command not found", re.compile(r"command not found|not recognized", re.I)),
    ("usage / bad argument", re.compile(r"usage:|unrecognized arguments|invalid option", re.I)),
]


def categorise(observation: str | None) -> str:
    o = observation or ""
    for name, rx in CATEGORIES:
        if rx.search(o):
            return name
    return "other non-zero exit"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectories", type=Path,
                    default=ROOT / "stage2" / "data" / "normalized_full" / "normalized_full")
    ap.add_argument("--projections", type=Path, default=ROOT / "projections_full_new_axis.parquet")
    ap.add_argument("--layer", type=int, default=53)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(args.trajectories.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for s in d.get("steps", []):
            obs = s.get("observation")
            err = is_error(obs)
            rows.append({"trajectory_id": d["trajectory_id"],
                         "step_index": int(s["step_index"]),
                         "is_error": err,
                         "category": categorise(obs) if err else "clean"})
    lab = pd.DataFrame(rows)

    # Kept as a recorded fact, not a category: see the module docstring.
    TEST_RUNNER = re.compile(r"=+ test session starts|^FAILED |\d+ (passed|failed)|Ran \d+ test",
                             re.M)
    n_test_steps = sum(
        bool(TEST_RUNNER.search(s.get("observation") or ""))
        for p in sorted(args.trajectories.glob("*.json"))
        for s in json.loads(p.read_text(encoding="utf-8")).get("steps", []))

    proj = pd.read_parquet(args.projections)
    axis = (proj[proj.layer == args.layer][["trajectory_id", "step_index", "proj_mean"]]
            .rename(columns={"proj_mean": "axis"}))
    df = (axis.merge(lab, on=["trajectory_id", "step_index"], how="inner")
              .sort_values(["trajectory_id", "step_index"]).reset_index(drop=True))
    df["d_axis"] = df.groupby("trajectory_id")["axis"].shift(-1) - df["axis"]

    sd = float(df["axis"].std())
    clean = df.loc[df.category == "clean", "d_axis"].dropna()
    out = {"layer_new_axis": args.layer, "n_steps": int(len(df)),
           "signal_sd": sd,
           "clean": {"n": int(len(clean)), "mean_delta": float(clean.mean()),
                     "mean_delta_sd_units": float(clean.mean() / sd)},
           "steps_running_tests": int(n_test_steps),
           "categories": []}

    # RAW vs MEASURABLE. Every failed step gets a category (df.category is set on all 12,464
    # rows), but a step's response A_{t+1} - A_t is undefined when t is the last step of its
    # trajectory -- one such step per trajectory, 499 in total, 107 of them failures. Using the
    # RAW category count as the numerator of "share_of_errors" while "n" below is the
    # MEASURABLE (dropna'd) count made the two columns describe different populations: a
    # traceback share of 72.2% (6260/8667 raw) sat next to an n of 6171 (measurable only),
    # silently inconsistent by up to 0.2 points per category. Both are now computed on the
    # measurable population, so the table's own numbers reconstruct each other.
    n_total = int(df.is_error.sum())
    n_measurable = int(sum(len(df.loc[df.category == name, "d_axis"].dropna())
                          for name, _ in CATEGORIES + [("other non-zero exit", None)]))
    print(f"{len(df)} steps; {n_total} failed ({100 * n_total / len(df):.1f}%); "
          f"{n_measurable} with a measurable response "
          f"({n_total - n_measurable} failed steps are a trajectory's last step)\n")
    print(f"{'category':<24}{'n':>7}{'share':>8}{'mean dA':>11}{'vs clean':>11}")
    print(f"{'clean':<24}{len(clean):>7}{'':>8}{clean.mean():>+11.5f}{'':>11}")
    for name, _ in CATEGORIES + [("other non-zero exit", None)]:
        g = df.loc[df.category == name, "d_axis"].dropna()
        if not len(g):
            continue
        rec = {"category": name, "n": int(len(g)),
               "share_of_errors": float(len(g) / max(1, n_measurable)),
               "mean_delta": float(g.mean()),
               "vs_clean": float(g.mean() - clean.mean()),
               "vs_clean_sd_units": float((g.mean() - clean.mean()) / sd)}
        out["categories"].append(rec)
        print(f"{name:<24}{rec['n']:>7}{100 * rec['share_of_errors']:>7.1f}%"
              f"{rec['mean_delta']:>+11.5f}{rec['vs_clean']:>+11.5f}")

    path = OUT_DIR / "error_categories.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
