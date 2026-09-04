#!/usr/bin/env python
"""CPU step: projections.parquet -> AUROC/CI/permutation report + layer plot.

Reuses ``prognosis.analysis.stats.auroc_with_ci`` (task-level BCa bootstrap)
verbatim, rather than the reference script's bare "frac(original >
corrupted)", so the control reports separation the same way as every other
result in the paper. "Task" here is the DebugBench problem (``slug``): each problem
contributes one ``outcome=1`` (original) and one ``outcome=0`` (corrupted)
row per variant, and BCa resampling by slug keeps a problem's pair together.

The permutation test is **not** ``prognosis.analysis.stats.permutation_test_auroc``
— that one reassigns whole outcome-pattern *blocks* between tasks, which is
degenerate here because every problem's block is the identical fixed pair
[1, 0] by construction (confirmed empirically while wiring this up: it
returned p=1.0 even on synthetic data with AUROC~1.0). See
``prognosis.single_turn.stats_local`` for the within-task label permutation
used instead.

Reproduce: python -m prognosis.single_turn.analyze
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from prognosis.single_turn.config import load_defaults
from prognosis.single_turn.paths import data_file
from prognosis.single_turn.stats_local import within_task_permutation_test
from prognosis.analysis.stats import auroc, auroc_with_ci


def load_primary_layer(manifest_path: Path, fallback: int) -> int:
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        return int(manifest.get("primary_layer", fallback))
    return fallback


def variant_frame(df: pd.DataFrame, variant: str, layer: int, score_col: str) -> pd.DataFrame:
    """One (original, corrupted) row pair per slug, at a fixed layer, for one variant.

    Every row in the parquet already belongs to exactly one variant pairing
    (``run_control.py``'s ``rows_for_problem`` emits "original" once per
    variant it's compared against, not once per problem — the diff window is
    pair-specific), so this is just a straight filter, no "original" special
    case needed.
    """
    sub = df[(df["layer"] == layer) & (df["variant"] == variant)]
    return sub[["slug", "outcome", score_col]].copy()


def pooled_frame(df: pd.DataFrame, layer: int, score_col: str) -> pd.DataFrame:
    """Every variant pairing at a fixed layer (each contributes one 1/0 pair)."""
    sub = df[df["layer"] == layer]
    return sub[["slug", "outcome", score_col]].copy()


def headline_report(
    df: pd.DataFrame,
    *,
    variants: list[str],
    primary_layer: int,
    score_col: str,
    n_boot: int,
    n_perm: int,
    rng_seed: int = 0,
) -> dict:
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    by_variant = {}
    for variant in variants:
        vf = variant_frame(df, variant, primary_layer, score_col)
        if vf.empty:
            continue
        ci = auroc_with_ci(vf, score_col, task_col="slug", outcome_col="outcome",
                            n_boot=n_boot, rng=rng)
        perm = within_task_permutation_test(vf, score_col, task_col="slug",
                                             outcome_col="outcome", n_perm=n_perm, rng=rng)
        by_variant[variant] = {**ci, "permutation_p": perm["p_value"]}

    pooled_df = pooled_frame(df, primary_layer, score_col)
    pooled_ci = auroc_with_ci(pooled_df, score_col, task_col="slug", outcome_col="outcome",
                               n_boot=n_boot, rng=rng)
    pooled_perm = within_task_permutation_test(pooled_df, score_col, task_col="slug",
                                                outcome_col="outcome", n_perm=n_perm, rng=rng)

    return {
        "primary_layer": primary_layer,
        "score_col": score_col,
        "by_variant": by_variant,
        "pooled": {**pooled_ci, "permutation_p": pooled_perm["p_value"]},
    }


def layer_sweep(df: pd.DataFrame, *, variants: list[str], score_col: str) -> dict:
    """Plain AUROC (no CI) per layer, per variant + pooled — for the localization plot."""
    layers = sorted(df["layer"].unique().tolist())
    sweep: dict[str, dict[str, float]] = {v: {} for v in variants}
    sweep["pooled"] = {}
    for layer in layers:
        for variant in variants:
            vf = variant_frame(df, variant, layer, score_col)
            if vf.empty:
                continue
            sweep[variant][str(layer)] = auroc(vf["outcome"], vf[score_col])
        pf = pooled_frame(df, layer, score_col)
        sweep["pooled"][str(layer)] = auroc(pf["outcome"], pf[score_col])
    return sweep


def plot_layer_sweep(sweep: dict, primary_layer: int, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, by_layer in sweep.items():
        layers = sorted(int(k) for k in by_layer.keys())
        values = [by_layer[str(l)] for l in layers]
        style = {"linewidth": 2.5} if name == "pooled" else {"linewidth": 1, "alpha": 0.7}
        ax.plot(layers, values, marker="o", markersize=2, label=name, **style)
    ax.axvline(x=primary_layer, color="gray", linestyle="--", alpha=0.5,
                label=f"primary layer ({primary_layer})")
    ax.axhline(y=0.5, color="black", linestyle=":", alpha=0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC (original vs. corrupted)")
    ax.set_title("Single-turn coding control: frozen axis by layer")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projections", type=Path, default=data_file("projections.parquet"))
    ap.add_argument("--output-dir", type=Path, default=data_file("report"))
    ap.add_argument("--primary-layer", type=int, default=None,
                     help="Default: axis_manifest_path's primary_layer, else config default.")
    ap.add_argument("--variants", nargs="+", default=defaults["variants"])
    ap.add_argument("--n-boot", type=int, default=defaults["n_boot"])
    ap.add_argument("--n-perm", type=int, default=defaults["n_perm"])
    args = ap.parse_args()

    df = pd.read_parquet(args.projections)
    n_problems = df["slug"].nunique()
    print(f"Loaded {len(df)} rows, {n_problems} problems from {args.projections}", flush=True)

    primary_layer = args.primary_layer
    if primary_layer is None:
        primary_layer = load_primary_layer(defaults["axis_manifest_path"], defaults["primary_layer"])
    print(f"Primary layer: {primary_layer}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "n_problems": int(n_problems),
        "variants": args.variants,
        "primary": headline_report(
            df, variants=args.variants, primary_layer=primary_layer, score_col="proj_mean",
            n_boot=args.n_boot, n_perm=args.n_perm,
        ),
        "final_token_robustness": headline_report(
            df, variants=args.variants, primary_layer=primary_layer, score_col="proj_final",
            n_boot=args.n_boot, n_perm=args.n_perm,
        ),
    }
    if "proj_window_mean" in df.columns:
        # The original paper's actual metric: mean projection on tokens after
        # the original/corrupted code diverges (the original paper "Correlation
        # with code correctness"), not a whole-span mean or bare final token.
        report["diff_window"] = headline_report(
            df, variants=args.variants, primary_layer=primary_layer,
            score_col="proj_window_mean", n_boot=args.n_boot, n_perm=args.n_perm,
        )
    sweep = layer_sweep(df, variants=args.variants, score_col="proj_mean")
    report["layer_sweep_auroc"] = sweep

    report_path = args.output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    plot_path = args.output_dir / "layer_sweep.png"
    plot_layer_sweep(sweep, primary_layer, plot_path)

    def print_headline(label: str, headline: dict) -> None:
        print(f"\n{label} (layer {primary_layer}):", flush=True)
        for variant, res in headline["by_variant"].items():
            print(
                f"  {variant:<14} AUROC {res['auroc']:.3f} "
                f"[{res['ci_low']:.3f}, {res['ci_high']:.3f}]  "
                f"perm p={res['permutation_p']:.3f}  n={res['n']}",
                flush=True,
            )
        pooled = headline["pooled"]
        print(
            f"  {'pooled':<14} AUROC {pooled['auroc']:.3f} "
            f"[{pooled['ci_low']:.3f}, {pooled['ci_high']:.3f}]  "
            f"perm p={pooled['permutation_p']:.3f}  n={pooled['n']}  "
            f"(majority baseline {pooled['majority_baseline']:.3f})",
            flush=True,
        )

    print_headline("Whole-span mean", report["primary"])
    print_headline("Final token", report["final_token_robustness"])
    if "diff_window" in report:
        print_headline("Diff window (anchor-paper metric)", report["diff_window"])
    print(f"\nSaved {report_path} and {plot_path}", flush=True)


if __name__ == "__main__":
    main()
