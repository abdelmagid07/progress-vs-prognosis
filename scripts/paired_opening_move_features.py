#!/usr/bin/env python
"""What property of the agent's opening move carries the prefix-probe signal?

The prefix-restricted probe (``paired_direction_control.py``) ranks the resolved arm above
the unresolved arm in 16 of 18 pairs using ONLY step 0 -- before the agent has done anything
but read the prompt and issue one command. The single-turn control shows this same axis
family responds strongly to whole-text rewrites (~0.75 AUROC) but barely to small semantic
breaks (~0.53), which suggests a stylistic reading: maybe the probe is picking up surface
features of *how* the agent writes its opening move, not what it understood about the task.

This script tests that reading directly and cheaply, with no new GPU work:

1. Re-fits the SAME leave-one-pair-out probe used for the k=1 prefix result (peak layer 53,
   C=0.1), but keeps each held-out trajectory's decision-function SCORE instead of collapsing
   to a correct/incorrect bit. That gives a continuous per-pair margin
   (resolved score - unresolved score) instead of just its sign.
2. Parses step 0 of all 36 trajectories from ``normalized_mixed_within_task.zip`` into five
   text features: opening-response length, thought length, first-command length, hedging-
   language count, and whether the first command modifies the repo (write) or only reads it.
3. Correlates the per-pair margin against the per-pair feature DIFFERENCE (resolved -
   unresolved), and separately fits a probe on the text features ALONE (same leave-one-pair-
   out procedure) to see how much of the 16/18 result these five surface features can
   reconstruct on their own.

CAVEATS. n=18 pairs: correlations below are exploratory, not confirmatory -- nothing here is
powered to publish on its own. The write-vs-read command classification is a regex heuristic
over bash commands, not a parser; spot-check ``is_modifying`` before trusting it.

Writes data/paired_opening_move_features.csv/.json.

Reproduce: python scripts/paired_opening_move_features.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
from paired_direction_control import load, traj_vectors  # noqa: E402
from paired_probe_ceiling import leave_one_pair_out  # noqa: E402

OUT_DIR = ROOT / "results" / "numbers" / "new_axis"

HEDGE_RE = re.compile(
    r"\b(might|may|could|perhaps|maybe|possibly|probably|not sure|unsure|uncertain|"
    r"i think|i believe|seems?\s+(?:to|like)|appears?\s+to|should be|let(?:'s| us)? check|"
    r"need to (?:check|verify|confirm)|going to (?:check|explore|investigate)|"
    r"i(?:'ll| will) (?:check|look|explore|investigate))\b",
    re.IGNORECASE,
)

# Heuristic: a command counts as "modifying" if it writes to the filesystem. Redirects that
# duplicate a file descriptor (2>&1, &>) or append-compare operators are deliberately excluded
# so read-only commands piping stderr to stdout aren't misclassified.
WRITE_PATTERNS = [
    r"\bsed\s+-i\b",
    r"\bapply_patch\b",
    r"\bgit\s+apply\b",
    r"\bpatch\s+-p\d\b",
    r"\btee\b",
    r"\bmv\s+\S",
    r"\btouch\s+\S",
    r"\brm\s+(?:-\w+\s+)*\S",
    r"(?<![0-9])>>",
    r"(?<![0-9&])>(?!&)(?!>)",
    r"open\([^)]*['\"]w",
]
WRITE_RE = re.compile("|".join(WRITE_PATTERNS))


def split_content(step: dict) -> tuple[str, str]:
    """Return (reasoning_text, post_think_content) for one trajectory step."""
    am = step.get("assistant_message")
    if am and am.get("content"):
        return am.get("reasoning_content") or "", am["content"]
    resp = step.get("assistant_response", "")
    m = re.search(r"<think>(.*?)</think>(.*)", resp, re.DOTALL)
    return (m.group(1), m.group(2)) if m else ("", resp)


def extract_command(content: str) -> str:
    m = re.search(r"```(?:mswea_bash_command|bash)\s*\n(.*?)```", content, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_thought(content: str) -> str:
    idx = content.find("```")
    pre = content[:idx] if idx != -1 else content
    return re.sub(r"^\s*THOUGHT:\s*", "", pre.strip(), flags=re.IGNORECASE).strip()


def step0_features(step: dict) -> dict:
    reasoning, content = split_content(step)
    command = extract_command(content)
    thought = extract_thought(content)
    full_text = f"{reasoning} {content}"
    return {
        "resp_len_words": len(full_text.split()),
        "reasoning_len_words": len(reasoning.split()),
        "thought_len_words": len(thought.split()),
        "command_len_chars": len(command),
        "n_chained_cmds": command.count("&&") + command.count("||") + (1 if command else 0),
        "hedge_count": len(HEDGE_RE.findall(full_text)),
        "is_modifying": int(bool(WRITE_RE.search(command))),
    }


def load_features(zip_path: Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            with z.open(name) as f:
                d = json.load(f)
            # The pair archives carry a protocol_violations.json report alongside the
            # trajectories. Anything without steps is not a trajectory.
            if not d.get("steps"):
                continue
            feats = step0_features(d["steps"][0])
            rows.append({
                "task_id": d["task_id"],
                "trajectory_id": d["trajectory_id"],
                "outcome": int(d["outcome"]),
                **feats,
            })
    return pd.DataFrame(rows)


def leave_one_pair_out_scores(X: np.ndarray, meta: pd.DataFrame, C: float) -> pd.DataFrame:
    """Same procedure as leave_one_pair_out, but keeps the held-out decision score."""
    tasks = sorted(meta["task_id"].unique())
    rows = []
    for held in tasks:
        te = meta["task_id"] == held
        tr = ~te
        if meta.loc[te, "outcome"].nunique() < 2 or meta.loc[tr, "outcome"].nunique() < 2:
            continue
        scaler = StandardScaler().fit(X[tr.to_numpy()])
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(scaler.transform(X[tr.to_numpy()]), meta.loc[tr, "outcome"].to_numpy())
        sub = meta.loc[te].copy()
        sub["score"] = clf.decision_function(scaler.transform(X[te.to_numpy()]))
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


FEATURE_COLS = ["resp_len_words", "reasoning_len_words", "thought_len_words",
                 "command_len_chars", "n_chained_cmds", "hedge_count", "is_modifying"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", type=Path, default=ROOT / "paired_meanpool_32b.npz")
    ap.add_argument("--trajectories", type=Path,
                     default=ROOT / "normalized_mixed_within_task.zip")
    ap.add_argument("--layer-new-axis", type=int, default=53)
    ap.add_argument("--C", type=float, default=0.1)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. held-out margin from the SAME k=1 probe as the paper's 16/18 result -----------
    block = args.layer_new_axis - 1
    acts, m = load(args.activations, block)
    V, info = traj_vectors(acts, m, max_steps=1)
    scored = leave_one_pair_out_scores(V, info, args.C)
    n_correct = int((scored.pivot_table(index="task_id", columns="outcome",
                                        values="score")[1]
                     > scored.pivot_table(index="task_id", columns="outcome",
                                          values="score")[0]).sum())
    print(f"sanity check: k=1 probe margin agrees with sign result on "
          f"{n_correct}/{scored['task_id'].nunique()} pairs (paper reports 16/18)")

    # --- 2. step-0 text features -----------------------------------------------------------
    feats = load_features(args.trajectories)
    merged = scored.merge(feats, on=["task_id", "trajectory_id", "outcome"], how="inner")
    assert len(merged) == len(scored), "feature/score join dropped rows"

    # --- 3a. per-pair margin vs per-pair feature-difference correlation --------------------
    piv_score = merged.pivot_table(index="task_id", columns="outcome", values="score")
    margin = (piv_score[1] - piv_score[0]).rename("margin_diff")
    corr_rows = []
    for col in FEATURE_COLS:
        piv_f = merged.pivot_table(index="task_id", columns="outcome", values=col)
        fdiff = (piv_f[1] - piv_f[0]).reindex(margin.index)
        pr, pp = pearsonr(margin, fdiff)
        sr, sp = spearmanr(margin, fdiff)
        # Sign test excludes ties (fdiff == 0): a tie carries no directional information, and
        # counting it as a trial (as a naive binomtest(n_pos, n_pairs, 0.5) would) spuriously
        # inflates significance when a binary feature like is_modifying ties on most pairs.
        n_pos = int((fdiff > 0).sum())
        n_neg = int((fdiff < 0).sum())
        n_tied = int((fdiff == 0).sum())
        n_decided = n_pos + n_neg
        sign_p = binomtest(n_pos, n_decided, 0.5).pvalue if n_decided else float("nan")
        corr_rows.append({
            "feature": col, "n_pairs_resolved_higher": n_pos, "n_pairs_unresolved_higher": n_neg,
            "n_pairs_tied": n_tied, "n_pairs_decided": n_decided,
            "sign_p_resolved_gt_unresolved": sign_p,
            "pearson_r_vs_margin": pr, "pearson_p_vs_margin": pp,
            "spearman_r_vs_margin": sr, "spearman_p_vs_margin": sp,
        })
    corr_df = pd.DataFrame(corr_rows).sort_values("pearson_p_vs_margin")

    # --- 3b. can text features alone reconstruct the probe's ranking? ----------------------
    Xtext = merged.sort_values(["task_id", "outcome"])[FEATURE_COLS].to_numpy(dtype=float)
    meta_text = merged.sort_values(["task_id", "outcome"])[["task_id", "outcome"]].reset_index(drop=True)
    text_probe = leave_one_pair_out(Xtext, meta_text, args.C)

    print("\nper-pair margin (activation probe, k=1) vs step-0 text feature differences:")
    with pd.option_context("display.width", 120):
        print(corr_df.to_string(index=False))
    print(f"\ntext-features-only probe (same leave-one-pair-out procedure): "
          f"{text_probe['n_correct']}/{text_probe['n_pairs']} pairs, "
          f"sign p={text_probe['sign_p']:.4f}")
    print(f"(activation probe for comparison: {n_correct}/{scored['task_id'].nunique()})")

    out_csv = OUT_DIR / "paired_opening_move_features.csv"
    merged.drop(columns=["score"]).to_csv(out_csv, index=False)
    summary = {
        "layer_new_axis": args.layer_new_axis,
        "C": args.C,
        "activation_probe_k1": {
            "n_correct": n_correct, "n_pairs": int(scored["task_id"].nunique()),
        },
        "text_features_only_probe_k1": text_probe,
        "feature_vs_margin_correlations": corr_df.to_dict(orient="records"),
        "interpretation": (
            "If the text-only probe approaches 16/18 and/or individual features correlate "
            "strongly with the activation-probe margin, the k=1 signal is substantially "
            "stylistic (consistent with the single-turn control's sensitivity to whole-text "
            "rewrites). If it stays near chance despite the activation probe scoring well, "
            "the five surface features tested here do not explain the signal and a semantic "
            "(content-based) reading is favoured. Either way, n=18 pairs means this is "
            "suggestive, not confirmatory."
        ),
    }
    (OUT_DIR / "paired_opening_move_features.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_csv}")
    print(f"wrote {OUT_DIR / 'paired_opening_move_features.json'}")


if __name__ == "__main__":
    main()
