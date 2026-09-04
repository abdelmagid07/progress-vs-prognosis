"""Loading and pooling within-task trajectory pairs.

Shared by the paired probes and the leakage controls, so that truncation and
normalisation are decided in exactly one place.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load(npz_path: Path, layer_block: int):
    d = np.load(npz_path, allow_pickle=True)
    meta = pd.DataFrame({
        "task_id": d["task_id"].astype(str),
        "trajectory_id": d["trajectory_id"].astype(str),
        "outcome": d["outcome"].astype(int),
        "layer": d["layer"].astype(int),
        "step_index": d["step_index"].astype(int),
    })
    meta["row"] = np.arange(len(meta))
    return d["activations"].astype(np.float32), meta[meta["layer"] == layer_block]


def traj_vectors(acts, m, max_steps=None):
    """L2-normalised trajectory-mean vector per trajectory (optionally first k steps)."""
    if max_steps is not None:
        m = m[m["step_index"] < max_steps]
    V, rows = [], []
    for (t, tr, o), g in m.groupby(["task_id", "trajectory_id", "outcome"]):
        v = acts[g["row"].to_numpy()].mean(axis=0)
        n = np.linalg.norm(v)
        V.append(v / n if n > 1e-8 else v)
        rows.append({"task_id": t, "trajectory_id": tr, "outcome": o})
    return np.vstack(V), pd.DataFrame(rows)


def trajectory_features(npz_path: Path, *, truncate: bool = True,
                        l2norm: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    """Mean-pool each trajectory's per-step vectors, per layer.

    Both corrections are essential. Without them the probe separates every pair at every
    layer, including layers too early to plausibly encode task outcome.

    ``truncate`` -- the two arms of a pair have very different step counts, since failures
    have a long tail of long runs. Averaging over different numbers of steps shifts the
    mean vector systematically, so the classes become separable by step count rather than
    by content. Both arms are truncated to the shorter arm's length before pooling.

    ``l2norm`` -- even after truncation, magnitude drifts with context length, and the norm
    of the mean vector separates pairs on its own at layer 1. L2 normalising each
    trajectory's mean vector removes that scalar channel and forces the probe to use
    direction.
    """
    d = np.load(npz_path, allow_pickle=True)
    meta = pd.DataFrame({
        "task_id": d["task_id"].astype(str),
        "trajectory_id": d["trajectory_id"].astype(str),
        "outcome": d["outcome"].astype(int),
        "layer": d["layer"].astype(int),
        "step_index": d["step_index"].astype(int),
    })
    acts = d["activations"].astype(np.float32)
    meta["row"] = np.arange(len(meta))

    if truncate:
        # Steps per trajectory (layer-invariant), then per-task cap = shorter arm.
        per_traj = meta[meta["layer"] == meta["layer"].min()].groupby(
            ["task_id", "trajectory_id"]).size().rename("n").reset_index()
        cap = per_traj.groupby("task_id")["n"].min().rename("cap")
        meta = meta.merge(cap, on="task_id", how="left")
        meta = meta[meta["step_index"] < meta["cap"]]

    grouped = meta.groupby(["layer", "task_id", "trajectory_id", "outcome"], sort=True)
    feats, rows = [], []
    for (layer, task, traj, outcome), g in grouped:
        v = acts[g["row"].to_numpy()].mean(axis=0)
        if l2norm:
            n = np.linalg.norm(v)
            if n > 1e-8:
                v = v / n
        feats.append(v)
        rows.append({"layer": layer, "task_id": task, "trajectory_id": traj,
                     "outcome": outcome, "n_steps_used": len(g)})
    return np.vstack(feats), pd.DataFrame(rows)


def leave_one_pair_out(X: np.ndarray, meta: pd.DataFrame, C: float) -> dict:
    """One held-out ranking per pair: does the resolved arm outscore the unresolved one?"""
    tasks = sorted(meta["task_id"].unique())
    correct, usable = 0, 0
    for held in tasks:
        te = meta["task_id"] == held
        tr = ~te
        if meta.loc[te, "outcome"].nunique() < 2 or meta.loc[tr, "outcome"].nunique() < 2:
            continue
        scaler = StandardScaler().fit(X[tr.to_numpy()])
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(scaler.transform(X[tr.to_numpy()]), meta.loc[tr, "outcome"].to_numpy())
        sub = meta.loc[te]
        s = clf.decision_function(scaler.transform(X[te.to_numpy()]))
        res = s[(sub["outcome"] == 1).to_numpy()].mean()
        unres = s[(sub["outcome"] == 0).to_numpy()].mean()
        usable += 1
        correct += int(res > unres)
    p = binomtest(correct, usable, 0.5).pvalue if usable else float("nan")
    return {"n_pairs": usable, "n_correct": correct,
            "frac_correct": correct / usable if usable else float("nan"), "sign_p": p}
