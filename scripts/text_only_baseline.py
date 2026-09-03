#!/usr/bin/env python
"""How much of the pooled agentic result is recoverable from the trajectory text alone?

The paper's claim is that the frozen axis's pooled AUROC of 0.747 is a task-difficulty read
rather than progress monitoring. The sharpest demonstration available without new compute is to
predict the same outcomes from the generated TEXT, using no activations at all. If counting
words and commands matches an activation-space direction, the pooled number is not evidence
about representations.

This also scales up two findings that currently rest on 18 pairs:

  * the opening-move result (arms that resolve reason longer at step 0), and
  * the qualitative lead from the extremes analysis, that low-scoring steps are re-issues of a
    failing command -- here measured as a command-repetition rate over all 500 trajectories.

FEATURE GROUPS are reported separately, because they are not equally interesting. Length is
already known to beat the axis (separability 0.780) and would otherwise carry the combined
model on its own; the question is what the other groups add once it is present.

  A  length      steps, total generated words
  B  opening     step-0 reasoning/response length, hedging
  C  behaviour   command repetition, error-observation rate, read-vs-write mix

EVALUATION mirrors the fitted-probe grid: task-level 5-fold cross-validation, so no task appears
in both train and test, and AUROC on held-out folds. Single features are scored directly (no
fitting), so their AUROC needs no cross-validation.

Writes data/text_only_baseline.json.

Reproduce: python scripts/text_only_baseline.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
from analyze_agentic_new_axis import fast_auroc  # noqa: E402
from paired_opening_move_features import HEDGE_RE, extract_command, split_content  # noqa: E402

OUT_DIR = ROOT / "results" / "numbers" / "new_axis"

GROUPS = {
    "A_length":    ["n_steps", "total_words"],
    "B_opening":   ["s0_reasoning_words", "s0_response_words", "s0_hedge"],
    "C_behaviour": ["repeat_rate", "error_rate", "write_frac"],
}

# The trajectory-level baseline exactly as METHOD defines it: "four observable properties of the
# run: total generated words, number of trajectory steps, command repetition rate, and error
# observation rate". No such model existed here. Those four are split across A_length and
# C_behaviour, and C_behaviour carries a fifth feature (write_frac) the Method does not name, so
# the paper was quoting an eight-feature number for a four-feature claim. Kept separate from
# GROUPS because GROUPS is summed to build "all text features" and these four are a subset.
METHOD_TRAJECTORY_FEATURES = ["total_words", "n_steps", "repeat_rate", "error_rate"]

from error_rule import is_error  # noqa: E402


def norm_cmd(c: str) -> str:
    """Collapse whitespace so trivially reformatted re-issues count as repeats."""
    return re.sub(r"\s+", " ", c).strip()


def featurise(path: Path) -> dict | None:
    d = json.loads(path.read_text(encoding="utf-8"))
    steps = d.get("steps") or []
    if not steps:
        return None
    cmds, words, errs = [], 0, 0
    writes = 0
    for s in steps:
        reasoning, content = split_content(s)
        words += len((reasoning + " " + content).split())
        c = norm_cmd(extract_command(content))
        if c:
            cmds.append(c)
            if re.search(r">>|(?<![0-9&])>(?!&)(?!>)|\bsed -i\b|\bmv \S|\btouch \S|\btee\b", c):
                writes += 1
        if is_error(s.get("observation")):
            errs += 1
    r0, c0 = split_content(steps[0])
    seen, repeats = set(), 0
    for c in cmds:
        if c in seen:
            repeats += 1
        seen.add(c)
    n = len(steps)
    return {
        "trajectory_id": d["trajectory_id"], "task_id": d.get("task_id", d["trajectory_id"]),
        "outcome": int(d["outcome"]),
        "n_steps": n, "total_words": words,
        "s0_reasoning_words": len(r0.split()),
        "s0_response_words": len((r0 + " " + c0).split()),
        "s0_hedge": len(HEDGE_RE.findall(r0 + " " + c0)),
        "repeat_rate": repeats / max(len(cmds), 1),
        "error_rate": errs / n,
        "write_frac": writes / max(len(cmds), 1),
    }


def cv_auroc(X, y, groups, seed=0):
    """Task-level 5-fold CV; returns out-of-fold AUROC."""
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        mdl = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
        mdl.fit(X[tr], y[tr])
        oof[te] = mdl.decision_function(X[te])
    return fast_auroc(y, oof)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectories", type=Path,
                    default=ROOT / "stage2" / "data" / "normalized_full" / "normalized_full")
    ap.add_argument("--projections", type=Path, default=ROOT / "projections_full_new_axis.parquet")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [f for f in (featurise(p) for p in sorted(args.trajectories.glob("*.json"))) if f]
    df = pd.DataFrame(rows)
    # restrict to the analysed set so the comparison against the axis is like for like
    keep = set(pd.read_parquet(args.projections, columns=["trajectory_id"]).trajectory_id.unique())
    df = df[df.trajectory_id.isin(keep)].reset_index(drop=True)
    y = df.outcome.to_numpy()
    print(f"{len(df)} trajectories, {y.sum()} resolved ({y.mean():.3f})\n")

    single = {}
    for col in sum(GROUPS.values(), []):
        v = df[col].to_numpy(float)
        a = fast_auroc(y, v)
        single[col] = {"auroc": a, "auroc_oriented": max(a, 1 - a),
                       "direction": "higher=resolved" if a > 0.5 else "higher=unresolved"}
    print("single features (AUROC, oriented so >0.5 means predictive):")
    for k, v in sorted(single.items(), key=lambda kv: -kv[1]["auroc_oriented"]):
        print(f"  {k:<20s} {v['auroc_oriented']:.3f}   ({v['direction']})")

    combos = {}
    for name, cols in GROUPS.items():
        combos[name] = cv_auroc(df[cols].to_numpy(float), y, df.task_id)
    combos["B_opening + C_behaviour (no length)"] = cv_auroc(
        df[GROUPS["B_opening"] + GROUPS["C_behaviour"]].to_numpy(float), y, df.task_id)
    combos["four trajectory features (Method)"] = cv_auroc(
        df[METHOD_TRAJECTORY_FEATURES].to_numpy(float), y, df.task_id)
    combos["all text features"] = cv_auroc(
        df[sum(GROUPS.values(), [])].to_numpy(float), y, df.task_id)
    print("\nfitted text-only models (task-level 5-fold CV, held-out AUROC):")
    for k, v in combos.items():
        print(f"  {k:<38s} {v:.3f}")
    print(f"\n  reference: frozen axis final step 0.747, axis at step 0 0.723, "
          f"activation probe grid max 0.913")

    out = {
        "n_trajectories": int(len(df)), "resolution_rate": float(y.mean()),
        "single_features": single, "fitted_models": combos,
        "reference": {"axis_final_step": 0.747, "axis_step0": 0.723,
                      "activation_probe_grid_max": 0.913},
        "interpretation": (
            "These predictors use no activations. If a text-only model reaches the frozen "
            "axis's pooled AUROC, the pooled transfer result is not evidence about "
            "representations: it is recoverable by counting. Feature groups are separated "
            "because trajectory length alone is already known to outpredict the axis, so the "
            "informative comparisons are group B and C on their own, and whether they add "
            "anything once length is present."
        ),
    }
    path = OUT_DIR / "text_only_baseline.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
