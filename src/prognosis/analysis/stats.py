"""Statistics for every separation reported in the paper.

* Success (resolved) is the *minority* class, so separation is reported as
  **AUROC**, whose no-skill / majority-class value is 0.5 — never accuracy,
  which a majority-class predictor would inflate.
* Uncertainty uses a **bootstrap with the task as the unit of resampling**,
  respecting the within-trajectory correlation of steps, with **95% BCa**
  (bias-corrected and accelerated) confidence intervals.
* Headline separations are additionally tested by a **block permutation**: each
  task's set of trajectory outcomes is reassigned across tasks as a block,
  preserving within-task outcome structure under the null of no association.

These functions are deliberately data-frame oriented (``task_id`` / ``outcome`` /
score columns) so the same code serves the final-step headline, the per-position
sweep, and the within-task contrast.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

# The AUROC of a majority-class (constant) predictor: the line a real signal's
# confidence interval must exclude. This is 0.5, NOT the majority-class accuracy
# reported by majority_baseline(), which is a different quantity.
CHANCE_AUROC = 0.5


def auroc(outcomes, scores) -> float:
    """AUROC of ``scores`` against binary ``outcomes``; NaN if single-class."""
    outcomes = np.asarray(outcomes)
    scores = np.asarray(scores)
    mask = ~(np.isnan(scores))
    outcomes, scores = outcomes[mask], scores[mask]
    if len(np.unique(outcomes)) < 2 or len(outcomes) < 2:
        return float("nan")
    return float(roc_auc_score(outcomes, scores))


def separability(value: float) -> float:
    """Direction-agnostic separation: ``max(auroc, 1 - auroc)``."""
    if np.isnan(value):
        return float("nan")
    return float(max(value, 1.0 - value))


def majority_baseline(outcomes) -> float:
    """Majority-class rate = accuracy of always predicting the majority class.

    Reported for context under class imbalance; the AUROC decision line stays
    :data:`CHANCE_AUROC` (0.5), which is the majority predictor's *AUROC*.
    """
    outcomes = np.asarray(outcomes)
    if len(outcomes) == 0:
        return float("nan")
    p = float(np.mean(outcomes == 1))
    return max(p, 1.0 - p)


def _auroc_stat(df: pd.DataFrame, score_col: str, outcome_col: str) -> float:
    return auroc(df[outcome_col].to_numpy(), df[score_col].to_numpy())


def bca_bootstrap_ci(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    *,
    task_col: str = "task_id",
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """95% BCa CI for ``stat_fn`` with the task as the unit of resampling.

    Resamples whole tasks with replacement (all of a task's rows move together,
    respecting within-trajectory/step correlation), then applies the BCa
    bias-correction (``z0``) and acceleration (jackknife over tasks). Returns the
    observed estimate and the CI bounds; bounds are NaN when the statistic is
    undefined (e.g. single-class) or too few resamples were valid.
    """
    rng = rng or np.random.default_rng(0)
    observed = stat_fn(df)

    tasks = df[task_col].unique().tolist()
    groups = {t: df[df[task_col] == t] for t in tasks}
    n = len(tasks)

    out = {"estimate": observed, "ci_low": float("nan"), "ci_high": float("nan"),
           "n_boot": 0}
    if np.isnan(observed) or n < 2:
        return out

    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, n, size=n)
        sub = pd.concat([groups[tasks[i]] for i in pick], ignore_index=True)
        boot[b] = stat_fn(sub)
    boot = boot[~np.isnan(boot)]
    out["n_boot"] = int(len(boot))
    if len(boot) < 2:
        return out

    # Jackknife over tasks for the acceleration term.
    jack = np.array(
        [
            stat_fn(pd.concat([groups[t] for t in tasks if t != leave], ignore_index=True))
            for leave in tasks
        ],
        dtype=float,
    )
    jack = jack[~np.isnan(jack)]

    prop = float(np.mean(boot < observed))
    # Degenerate bootstrap (all equal) -> fall back to percentile-free point.
    if prop <= 0.0 or prop >= 1.0 or len(jack) < 2:
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out["ci_low"], out["ci_high"] = float(lo), float(hi)
        return out

    z0 = norm.ppf(prop)
    jack_mean = jack.mean()
    diffs = jack_mean - jack
    denom = 6.0 * (np.sum(diffs ** 2) ** 1.5)
    a = float(np.sum(diffs ** 3) / denom) if denom != 0 else 0.0

    def _adj(z_alpha: float) -> float:
        return norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))

    lo_p = _adj(norm.ppf(alpha / 2))
    hi_p = _adj(norm.ppf(1 - alpha / 2))
    lo, hi = np.percentile(boot, [100 * lo_p, 100 * hi_p])
    out["ci_low"], out["ci_high"] = float(lo), float(hi)
    return out


def auroc_with_ci(
    df: pd.DataFrame,
    score_col: str,
    *,
    task_col: str = "task_id",
    outcome_col: str = "outcome",
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """AUROC + task-level BCa CI + majority baseline + signal flag.

    ``signal`` is True iff the CI excludes :data:`CHANCE_AUROC` (0.5), the
    pre-registered decision rule.
    """
    res = bca_bootstrap_ci(
        df,
        lambda d: _auroc_stat(d, score_col, outcome_col),
        task_col=task_col,
        n_boot=n_boot,
        alpha=alpha,
        rng=rng,
    )
    lo, hi = res["ci_low"], res["ci_high"]
    signal = bool(not np.isnan(lo) and not np.isnan(hi) and (lo > CHANCE_AUROC or hi < CHANCE_AUROC))
    return {
        "auroc": res["estimate"],
        "separability": separability(res["estimate"]),
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": res["n_boot"],
        "majority_baseline": majority_baseline(df[outcome_col].to_numpy()),
        "chance_auroc": CHANCE_AUROC,
        "signal": signal,
        "n": int(len(df)),
    }


def permutation_test_auroc(
    df: pd.DataFrame,
    score_col: str,
    *,
    task_col: str = "task_id",
    outcome_col: str = "outcome",
    n_perm: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Block permutation p-value for an AUROC headline (one row per unit).

    Each task's ordered outcome vector is a block; the null reassigns blocks
    across tasks (only among tasks with the same block length, so the shuffle is
    well defined), preserving within-task outcome structure. Two-sided on
    ``|AUROC - 0.5|``. Expects one row per unit (e.g. per trajectory for the
    final-step headline).
    """
    rng = rng or np.random.default_rng(0)
    scores = df[score_col].to_numpy()
    observed = auroc(df[outcome_col].to_numpy(), scores)
    if np.isnan(observed):
        return {"estimate": observed, "p_value": float("nan"), "n_perm": 0}

    # Blocks of outcomes per task, and the score row-slice each block sits over.
    order = df.reset_index(drop=True)
    scores = order[score_col].to_numpy()
    outcomes = order[outcome_col].to_numpy()
    task_ids = order[task_col].to_numpy()

    blocks: dict[object, np.ndarray] = {}
    positions: dict[object, np.ndarray] = {}
    for t in pd.unique(task_ids):
        idx = np.where(task_ids == t)[0]
        positions[t] = idx
        blocks[t] = outcomes[idx]

    # Group tasks by block length so blocks are swapped only between size-matched
    # tasks (unequal rollout counts after exclusions).
    by_len: dict[int, list] = {}
    for t, idx in positions.items():
        by_len.setdefault(len(idx), []).append(t)

    obs_stat = abs(observed - 0.5)
    ge = 0
    for _ in range(n_perm):
        permuted = outcomes.copy()
        for length, tlist in by_len.items():
            if len(tlist) < 2:
                continue  # nothing to swap with; block stays put
            donors = list(tlist)
            rng.shuffle(donors)
            for recv, give in zip(tlist, donors):
                permuted[positions[recv]] = blocks[give]
        stat = abs(auroc(permuted, scores) - 0.5)
        if not np.isnan(stat) and stat >= obs_stat:
            ge += 1
    p = (1 + ge) / (n_perm + 1)
    return {"estimate": observed, "p_value": float(p), "n_perm": int(n_perm)}


def within_task_contrast(
    df: pd.DataFrame,
    score_col: str,
    *,
    task_col: str = "task_id",
    outcome_col: str = "outcome",
) -> dict:
    """Within-task resolved-vs-unresolved score gap on mixed-outcome tasks only.

    Where the same task is resolved on some seeds and not others, compare
    resolved vs unresolved *within* the task, holding task difficulty fixed. Returns the number of mixed tasks and the mean paired gap
    (resolved minus unresolved task means). One row per unit expected.
    """
    diffs = []
    for _t, g in df.groupby(task_col):
        succ = g[g[outcome_col] == 1][score_col]
        fail = g[g[outcome_col] == 0][score_col]
        if len(succ) >= 1 and len(fail) >= 1:
            diffs.append(float(succ.mean() - fail.mean()))
    if not diffs:
        return {"n_mixed_tasks": 0, "mean_within_task_gap": float("nan"),
                "frac_positive": float("nan")}
    diffs = np.array(diffs)
    return {
        "n_mixed_tasks": int(len(diffs)),
        "mean_within_task_gap": float(diffs.mean()),
        "frac_positive": float(np.mean(diffs > 0)),
    }
