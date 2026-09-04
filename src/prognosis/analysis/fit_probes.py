#!/usr/bin/env python
"""Fit logistic probes on mean-pooled agentic activations.

The frozen-axis test asks whether the *original* representation persists. A
frozen axis can fail while outcome information remains decodable in a different
direction; this module is that decodability reference.

Protocol:
  - Features: per-step mean-pooled residual activations dumped by
    ``project_steps --activations-npz`` (one vector per step × layer).
  - Standardization fit on the training fold only.
  - ``l2``-regularized logistic regression.
  - Cross-validation at the *task* level: all trajectories of a task share a
    fold; folds are stratified by task-level outcome mix so each fold contains
    resolved trajectories.
  - Report the full layer × position-bin AUROC grid (never just the max).

Reproduce: python -m prognosis.analysis.fit_probes --activations data/full_meanpool.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from prognosis.common.config import load_defaults
from prognosis.common.paths import data_file


def _task_labels(df: pd.DataFrame) -> pd.Series:
    """One binary label per task: 1 if any of its trajectories resolved."""
    return df.groupby("task_id")["outcome"].max().astype(int)


def _stratified_task_folds(
    task_ids: np.ndarray,
    task_y: pd.Series,
    *,
    n_folds: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_task_mask_over_rows, test_task_mask_over_rows).

    Stratifies by task-level outcome so each fold has resolved tasks when possible.
    """
    unique_tasks = np.array(sorted(task_y.index.tolist()))
    y = task_y.loc[unique_tasks].to_numpy()
    n_folds = min(n_folds, int(np.min(np.bincount(y))) if len(np.unique(y)) > 1 else 2)
    n_folds = max(2, n_folds)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
    folds = []
    for train_idx, test_idx in skf.split(unique_tasks, y):
        train_tasks = set(unique_tasks[train_idx].tolist())
        test_tasks = set(unique_tasks[test_idx].tolist())
        train_mask = np.array([t in train_tasks for t in task_ids])
        test_mask = np.array([t in test_tasks for t in task_ids])
        folds.append((train_mask, test_mask))
    return folds


def fit_grid(
    activations: np.ndarray,
    meta: pd.DataFrame,
    *,
    n_bins: int = 5,
    n_folds: int = 5,
    C: float = 1.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Cross-validated AUROC for every (layer, position-bin) cell."""
    rng = rng or np.random.default_rng(0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(
        np.digitize(meta["rel_pos"].to_numpy(), edges[1:-1], right=False), 0, n_bins - 1
    )
    meta = meta.copy()
    meta["bin"] = bins

    task_y = _task_labels(meta)
    layers = sorted(meta["layer"].unique().tolist())
    rows: list[dict] = []

    for layer in layers:
        for b in range(n_bins):
            mask = (meta["layer"].to_numpy() == layer) & (meta["bin"].to_numpy() == b)
            if mask.sum() < 4:
                rows.append(
                    {
                        "layer": int(layer),
                        "bin": b,
                        "rel_pos_mid": (b + 0.5) / n_bins,
                        "auroc": float("nan"),
                        "n": int(mask.sum()),
                        "n_folds_used": 0,
                    }
                )
                continue

            X = activations[mask]
            y = meta.loc[mask, "outcome"].to_numpy().astype(int)
            task_ids = meta.loc[mask, "task_id"].to_numpy()
            # Task labels restricted to tasks present in this cell.
            present = pd.unique(task_ids)
            cell_task_y = task_y.reindex(present).dropna()
            if len(cell_task_y) < 2 or cell_task_y.nunique() < 2 or len(np.unique(y)) < 2:
                rows.append(
                    {
                        "layer": int(layer),
                        "bin": b,
                        "rel_pos_mid": (b + 0.5) / n_bins,
                        "auroc": float("nan"),
                        "n": int(mask.sum()),
                        "n_folds_used": 0,
                    }
                )
                continue

            folds = _stratified_task_folds(task_ids, cell_task_y, n_folds=n_folds, rng=rng)
            fold_scores: list[float] = []
            for train_m, test_m in folds:
                if train_m.sum() < 2 or test_m.sum() < 1:
                    continue
                if len(np.unique(y[train_m])) < 2 or len(np.unique(y[test_m])) < 2:
                    continue
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_m])
                X_test = scaler.transform(X[test_m])
                clf = LogisticRegression(
                    penalty="l2",
                    C=C,
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                )
                clf.fit(X_train, y[train_m])
                scores = clf.predict_proba(X_test)[:, 1]
                fold_scores.append(float(roc_auc_score(y[test_m], scores)))

            rows.append(
                {
                    "layer": int(layer),
                    "bin": b,
                    "rel_pos_mid": (b + 0.5) / n_bins,
                    "auroc": float(np.mean(fold_scores)) if fold_scores else float("nan"),
                    "n": int(mask.sum()),
                    "n_folds_used": int(len(fold_scores)),
                }
            )
    return pd.DataFrame(rows)


def load_activations_npz(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    data = np.load(path, allow_pickle=True)
    activations = data["activations"].astype(np.float32)
    meta = pd.DataFrame(
        {
            "trajectory_id": data["trajectory_id"],
            "task_id": data["task_id"],
            "seed": data["seed"],
            "outcome": data["outcome"],
            "step_index": data["step_index"],
            "rel_pos": data["rel_pos"],
            "layer": data["layer"],
        }
    )
    if len(activations) != len(meta):
        raise SystemExit(
            f"activations/meta length mismatch: {len(activations)} vs {len(meta)}"
        )
    return activations, meta


def run(
    activations_path: Path,
    *,
    output_dir: Path,
    n_bins: int = 5,
    n_folds: int = 5,
    C: float = 1.0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    activations, meta = load_activations_npz(activations_path)
    grid = fit_grid(activations, meta, n_bins=n_bins, n_folds=n_folds, C=C)
    grid_csv = output_dir / "probe_auroc_grid.csv"
    grid.to_csv(grid_csv, index=False)

    # Never promote the max as a tuned ceiling — report it only as a summary
    # alongside the full grid.
    valid = grid.dropna(subset=["auroc"])
    summary = {
        "n_rows": int(len(meta)),
        "n_tasks": int(meta["task_id"].nunique()),
        "n_layers": int(meta["layer"].nunique()),
        "n_bins": n_bins,
        "n_folds": n_folds,
        "C": C,
        "grid_csv": str(grid_csv),
        "grid_max_auroc": float(valid["auroc"].max()) if len(valid) else float("nan"),
        "grid_mean_auroc": float(valid["auroc"].mean()) if len(valid) else float("nan"),
        "note": (
            "grid_max_auroc is a summary only; selection across the layer×bin "
            "family is subject to overfitting — report the full grid."
        ),
    }
    report_path = output_dir / "probe_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Grid -> {grid_csv}", flush=True)
    return summary


def main() -> None:
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--activations",
        type=Path,
        required=True,
        help="NPZ from project_steps --activations-npz",
    )
    ap.add_argument("--output-dir", type=Path, default=data_file("probe_report"))
    ap.add_argument("--n-bins", type=int, default=defaults["n_bins"])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0, help="Inverse l2 strength")
    args = ap.parse_args()
    if not args.activations.exists():
        raise SystemExit(f"Activations not found: {args.activations}")
    run(
        args.activations,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
        n_folds=args.n_folds,
        C=args.C,
    )


if __name__ == "__main__":
    main()
