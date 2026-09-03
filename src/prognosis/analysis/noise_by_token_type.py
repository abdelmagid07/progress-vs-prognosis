"""Analysis 3: readout noise — mean-over-G_t vs single final token.

The paper excludes tool-output tokens from the readout, so the old
reasoning-vs-tool_output comparison retires. This repoints the same diagnostic
at the two readouts we now compute: ``proj_mean`` (Eq. 1, mean cosine over the
generated tokens) vs ``proj_final`` (the single final-token robustness read).
The expectation is that averaging reduces variance, i.e. ``proj_mean`` is the
less noisy readout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def noise_by_token_type(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Compare readout std of ``proj_mean`` vs ``proj_final`` (legacy-tolerant)."""
    if "proj_mean" in df.columns:
        readouts = [c for c in ("proj_mean", "proj_final") if c in df.columns]
        summary = (
            df[readouts]
            .agg(["mean", "std", "count"])
            .T.reset_index()
            .rename(columns={"index": "readout"})
        )
        label_col = "readout"
    else:
        # Legacy parquet with token_type / projection columns.
        summary = (
            df.groupby("token_type")["projection"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        label_col = "token_type"

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(summary[label_col], summary["std"])
    ax.set_ylabel("Projection std")
    ax.set_title("Readout noise: mean-over-G_t vs final token")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return summary
