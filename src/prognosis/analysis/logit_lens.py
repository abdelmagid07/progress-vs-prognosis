"""Streaming reads of the unembedding matrix for logit-lens scoring."""

import json
import struct
from pathlib import Path

import numpy as np


def _st_header(path: Path) -> tuple[dict, int]:
    """safetensors header: 8-byte little-endian length, then JSON. Costs a ~2 KB read."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """bfloat16 is the top 16 bits of float32, so widening is exact and needs no lookup."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def stream_scores(shard: Path, key: str, directions: dict[str, np.ndarray],
                  block_rows: int = 8192) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute W @ d for each direction, and each row's norm, without loading W.

    This machine has ~1.5 GB of free RAM against a 1.56 GB unembedding, so neither
    safetensors.load_file (which loads every tensor in the shard) nor safe_open (which
    memory-maps the whole file) can be used -- both raise a Windows paging-file error. The lens
    only ever needs W @ d, which is a streaming reduction: read a block of vocabulary rows,
    accumulate, discard. Peak memory is one block, about 170 MB at the default size.
    """
    hdr, data_start = _st_header(shard)
    if key not in hdr:
        raise SystemExit(f"{key} not in shard; have {[k for k in hdr if k != '__metadata__'][:4]}")
    entry = hdr[key]
    if entry["dtype"] != "BF16":
        raise SystemExit(f"expected BF16, got {entry['dtype']} -- widen _bf16_to_f32 first")
    vocab, hidden = entry["shape"]
    start = data_start + entry["data_offsets"][0]

    for name, d in directions.items():
        if d.shape != (hidden,):
            raise SystemExit(f"direction {name} has shape {d.shape}, expected ({hidden},)")

    scores = {name: np.empty(vocab, dtype=np.float32) for name in directions}
    rownorm = np.empty(vocab, dtype=np.float32)
    with open(shard, "rb") as f:
        f.seek(start)
        done = 0
        while done < vocab:
            rows = min(block_rows, vocab - done)
            buf = np.frombuffer(f.read(rows * hidden * 2), dtype=np.uint16)
            W = _bf16_to_f32(buf).reshape(rows, hidden)
            rownorm[done:done + rows] = np.linalg.norm(W, axis=1)
            for name, d in directions.items():
                scores[name][done:done + rows] = W @ d
            done += rows
            del W, buf
    return scores, rownorm
