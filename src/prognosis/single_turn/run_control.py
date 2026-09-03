#!/usr/bin/env python
"""GPU step: problems_manifest.json + frozen axis -> projections.parquet.

Resumable checkpoint loop at the granularity of one DebugBench problem
(slug): all of a slug's rows, the original plus every corrupted variant, are
computed and checkpointed together, so a checkpoint is always slug-atomic.
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prognosis.single_turn.config import load_defaults
from prognosis.single_turn.paths import data_file
from prognosis.single_turn.project import aggregate_stats, diff_window_mask, run_forward
from prognosis.extract.hooks import unit_direction


def load_axis_directions(
    axis_path: Path, layers: list[int], axis_layer_offset: int = 0
) -> dict[int, torch.Tensor]:
    """``layers`` are captured-block indices (0..n_layers-1, matching
    ``LayerActivationCapture``/``code_acts`` indexing in ``project.py`` -- must stay in that
    space, NOT the axis's own row numbering, since the returned dict's keys are reused
    downstream to index ``code_acts`` directly). ``axis_layer_offset`` is the only place the
    axis's own indexing convention enters: 0 for the old axis (no embedding row, captured block i
    IS axis row i), 1 for the new Jiang-faithful axis (embedding-inclusive, captured block i is
    axis row i+1). The output ``layer`` column downstream stays in captured-block numbering
    either way -- see the module docstring / notebook for why relabeling it was intentionally
    avoided.
    """
    import numpy as np

    axis = np.load(axis_path)
    return {
        L: torch.tensor(unit_direction(axis[L + axis_layer_offset].astype("float32")), dtype=torch.float32)
        for L in layers
    }


def _copy_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".mirror_tmp" + dst.suffix)
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)
    if dst.exists():
        dst.unlink()
    shutil.move(str(tmp), str(dst))


def _mirror_checkpoint(output_path: Path, mirror_output: Path | None) -> None:
    if mirror_output is not None and output_path.exists():
        _copy_replace(output_path, mirror_output)
        print(f"    mirrored parquet -> {mirror_output}", flush=True)


def rows_for_problem(
    entry: dict,
    model,
    tokenizer,
    *,
    n_layers: int,
    directions: dict[int, torch.Tensor],
    device,
) -> list[dict]:
    """Two rows per (variant, layer): the diff window is pair-specific (original's
    window against ``shuffled`` differs from its window against ``buggy``), so
    "original" is no longer a single shared row — it's recomputed per variant
    pairing, keeping ``proj_mean``/``proj_final`` identical across pairings but
    letting ``proj_window_mean`` vary correctly.
    """
    rows: list[dict] = []

    original = run_forward(
        model, tokenizer,
        entry["original"]["full_text"],
        entry["original"]["code_char_start"], entry["original"]["code_char_end"],
        n_layers=n_layers, directions=directions, device=device,
    )
    if original is None:
        print(f"    SKIP {entry['slug']}/original: code span not found in tokenization",
              flush=True)
        return rows

    for variant, payload in entry["variants"].items():
        corrupted = run_forward(
            model, tokenizer,
            payload["full_text"], payload["code_char_start"], payload["code_char_end"],
            n_layers=n_layers, directions=directions, device=device,
        )
        if corrupted is None:
            print(f"    SKIP {entry['slug']}/{variant}: code span not found in tokenization",
                  flush=True)
            continue

        mask_orig, mask_corr = diff_window_mask(original.token_ids, corrupted.token_ids)
        if not mask_orig.any() and not mask_corr.any():
            print(f"    SKIP {entry['slug']}/{variant}: no token-level diff found",
                  flush=True)
            continue

        for layer in directions:
            whole_orig = aggregate_stats(original.proj_by_layer[layer])
            whole_corr = aggregate_stats(corrupted.proj_by_layer[layer])
            window_orig = aggregate_stats(original.proj_by_layer[layer], mask_orig)
            window_corr = aggregate_stats(corrupted.proj_by_layer[layer], mask_corr)

            rows.append({
                "slug": entry["slug"], "category": entry.get("category", ""),
                "subtype": entry.get("subtype", ""), "variant": variant, "side": "original",
                "outcome": 1, "layer": layer,
                "proj_mean": whole_orig["mean"], "proj_final": whole_orig["final"],
                "proj_window_mean": window_orig["mean"],
                "n_tokens": whole_orig["n_tokens"], "n_window_tokens": window_orig["n_tokens"],
            })
            rows.append({
                "slug": entry["slug"], "category": entry.get("category", ""),
                "subtype": entry.get("subtype", ""), "variant": variant, "side": "corrupted",
                "outcome": 0, "layer": layer,
                "proj_mean": whole_corr["mean"], "proj_final": whole_corr["final"],
                "proj_window_mean": window_corr["mean"],
                "n_tokens": whole_corr["n_tokens"], "n_window_tokens": window_corr["n_tokens"],
            })
    return rows


def run(
    manifest_path: Path,
    *,
    axis_path: Path,
    layers: list[int],
    model_name: str,
    dtype: str,
    n_layers: int,
    output_path: Path,
    mirror_output: Path | None = None,
    mirror_every: int = 10,
    axis_layer_offset: int = 0,
) -> pd.DataFrame:
    with open(manifest_path) as f:
        problems = json.load(f)
    if not problems:
        raise FileNotFoundError(f"No problems in {manifest_path}")

    directions = load_axis_directions(axis_path, layers, axis_layer_offset=axis_layer_offset)

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True,
    ).eval()
    device = next(model.parameters()).device

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_slugs: set[str] = set()
    all_rows: list[dict] = []
    if output_path.exists():
        prev = pd.read_parquet(output_path)
        if len(prev):
            all_rows = prev.to_dict(orient="records")
            done_slugs = set(prev["slug"].astype(str).unique())
            print(f"Resuming: {len(done_slugs)} problems already in {output_path}", flush=True)

    n_total = len(problems)
    if mirror_output is not None:
        print(f"Drive/durable mirror every {mirror_every} problems (output={mirror_output})",
              flush=True)

    n_skipped = 0
    for entry in problems:
        slug = entry["slug"]
        n_done = len(done_slugs)
        if slug in done_slugs:
            n_skipped += 1
            continue
        if n_skipped:
            print(f"  (skipped {n_skipped} already-done problems)", flush=True)
            n_skipped = 0

        n_variants = len(entry["variants"])
        print(f"  [{n_done}/{n_total} done] {slug}: {n_variants} corrupted variant(s)",
              flush=True)
        rows = rows_for_problem(
            entry, model, tokenizer, n_layers=n_layers, directions=directions, device=device,
        )
        all_rows.extend(rows)
        done_slugs.add(slug)
        n_done = len(done_slugs)

        df = pd.DataFrame(all_rows)
        df.to_parquet(output_path, index=False)
        print(f"    checkpointed {len(df)} rows -> {output_path} [{n_done}/{n_total} done]",
              flush=True)

        if mirror_output is not None and mirror_every > 0 and n_done % mirror_every == 0:
            _mirror_checkpoint(output_path, mirror_output)

    if mirror_output is not None:
        print("Final mirror to durable storage...", flush=True)
        _mirror_checkpoint(output_path, mirror_output)

    return pd.DataFrame(all_rows)


def main():
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problems-manifest", type=Path, default=data_file("problems_manifest.json"))
    ap.add_argument("--output", type=Path, default=data_file("projections.parquet"))
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                     help="Captured-block indices to project (0..n_layers-1, NOT axis row "
                     "numbers -- these index the model's captured activations directly). "
                     "Default: all layers 0..n_layers-1.")
    ap.add_argument("--axis-layer-offset", type=int, default=0,
                     help="Axis row = captured-block index + this offset. 0 for the old axis "
                     "(no embedding row, captured block i IS axis row i). 1 for the new "
                     "Jiang-faithful axis (embedding-inclusive: captured block i is axis row "
                     "i+1). Output 'layer' column stays in captured-block numbering either way.")
    ap.add_argument("--mirror-output", type=Path, default=None,
                     help="Optional durable copy path for the projections parquet (e.g. Drive).")
    ap.add_argument("--mirror-every", type=int, default=10,
                     help="Copy to --mirror-output every N newly completed problems.")
    ap.add_argument("--model", default=defaults["model"])
    ap.add_argument("--axis-path", type=Path, default=defaults["axis_path"])
    ap.add_argument("--dtype", default=defaults["dtype"])
    ap.add_argument("--n-layers", type=int, default=defaults["n_layers"])
    args = ap.parse_args()

    if not args.axis_path.exists():
        raise SystemExit(f"Value axis not found at {args.axis_path}. Run Stage 1 first.")

    layers = args.layers if args.layers is not None else list(range(args.n_layers))

    print(f"Projecting {args.problems_manifest} over {len(layers)} layer(s)...", flush=True)
    df = run(
        args.problems_manifest,
        axis_path=args.axis_path,
        layers=layers,
        model_name=args.model,
        dtype=args.dtype,
        n_layers=args.n_layers,
        output_path=args.output,
        mirror_output=args.mirror_output,
        mirror_every=args.mirror_every,
        axis_layer_offset=args.axis_layer_offset,
    )
    print(f"Done. {len(df)} rows -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
