"""One forward pass over a prefilled (prompt + code) text -> per-layer projections.

Reuses ``prognosis.extract.hooks.LayerActivationCapture`` /
``cosine_projection`` instead of the reference script's ad hoc ``per_token_cs``
hook. Layer indexing hooks ``model.model.layers[i]`` directly and reads axis
row ``i``, *not* the reference script's ``hook_idx = probe_layer - 1``.

Three readouts, in increasing faithfulness to the original paper:

- ``mean``/``final`` (whole code span): mean over every code token, or just
  the last one. Ours; the whole-span mean turned out to be diluted by a
  positional ramp unrelated to correctness (confirmed via a random-direction
  control), so this is *not* a faithful reproduction of
  the original paper's headline stat.
- ``window`` (``diff_window_mask`` + masked mean): the original paper's actual
  metric — "the average value-axis projection on the assistant tokens after
  the bug" (the original paper, "Correlation with code correctness"). Locates
  where original and corrupted code diverge at the token level (diff, like
  their reference script's ``after10_mean``) and averages only over
  ``[diff_start, diff_end + radius)`` per diverging region, unioned across
  regions. This is pair-specific — original's window differs depending on
  which variant it's being compared against — so it's computed per
  (original, variant) pair, not once per problem.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

import numpy as np
import torch

from prognosis.extract.hooks import LayerActivationCapture, cosine_projection

# Matches the reference script's R=10: token radius after each diverging
# region (the original paper doesn't specify the exact radius in prose; this value
# is only in the reference implementation, code_correlation.py).
DIFF_WINDOW_RADIUS = 10


def code_token_span(
    offset_mapping: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int | None, int | None]:
    """Token index range covering [char_start, char_end) in the source text."""
    ts = te = None
    for i, (s, _e) in enumerate(offset_mapping):
        if s >= char_start and ts is None:
            ts = i
        if s < char_end:
            te = i + 1
    return ts, te


@dataclass
class CodeSpanActivations:
    """Raw (not yet aggregated) per-token result for one prefilled code span."""

    token_ids: list[int]
    proj_by_layer: dict[int, np.ndarray]  # each array shape (n_code_tokens,)


def run_forward(
    model,
    tokenizer,
    full_text: str,
    code_char_start: int,
    code_char_end: int,
    *,
    n_layers: int,
    directions: dict[int, torch.Tensor],
    device,
) -> CodeSpanActivations | None:
    """Per-layer raw per-token cosine projections over the code span.

    Returns ``None`` if the code span can't be located in the tokenization
    (mirrors the reference script's skip behavior for unmapped spans).
    """
    enc = tokenizer(
        full_text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False
    )
    offset_mapping = [(int(a), int(b)) for a, b in enc["offset_mapping"][0].tolist()]
    input_ids = enc["input_ids"].to(device)

    ts, te = code_token_span(offset_mapping, code_char_start, code_char_end)
    if ts is None or te is None or te <= ts:
        return None

    capture = LayerActivationCapture(model, n_layers=n_layers)
    with torch.no_grad():
        model(input_ids=input_ids)
    layer_acts = capture.all_layers(n_layers)  # (L, seq, H) cpu float32
    capture.remove()

    code_acts = layer_acts[:, ts:te, :]  # (L, n_code_tok, H)
    code_token_ids = enc["input_ids"][0][ts:te].tolist()

    proj_by_layer = {
        layer: cosine_projection(code_acts[layer], direction).numpy()
        for layer, direction in directions.items()
    }
    return CodeSpanActivations(token_ids=code_token_ids, proj_by_layer=proj_by_layer)


def diff_window_mask(
    ids_a: list[int], ids_b: list[int], *, radius: int = DIFF_WINDOW_RADIUS
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks over (ids_a, ids_b) covering the original paper's readout window.

    Diffs the two token-id sequences (``difflib.SequenceMatcher``, matching
    the reference implementation) and marks ``[region_start, region_end +
    radius)`` for every non-equal opcode, unioned across regions, separately
    per side. A zero-width change point (pure insertion/deletion) still
    opens a window starting at that point, matching the reference's
    ``end = e if e > s else s`` handling.
    """
    ops = [
        (i1, i2, j1, j2)
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ids_a, ids_b, autojunk=False).get_opcodes()
        if tag != "equal"
    ]
    mask_a = np.zeros(len(ids_a), dtype=bool)
    mask_b = np.zeros(len(ids_b), dtype=bool)
    for i1, i2, j1, j2 in ops:
        end_a = i2 if i2 > i1 else i1
        mask_a[max(0, i1):min(len(ids_a), end_a + radius)] = True
        end_b = j2 if j2 > j1 else j1
        mask_b[max(0, j1):min(len(ids_b), end_b + radius)] = True
    return mask_a, mask_b


def aggregate_stats(proj: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int]:
    """mean/final/n_tokens over ``proj``, optionally restricted to a boolean ``mask``.

    ``final`` is the last element of the selected subset — for the whole
    span that's the code's actual last token; for a window mask it's the
    last token inside that window, not necessarily the code's last token.
    """
    sel = proj[mask] if mask is not None else proj
    if sel.size == 0:
        return {"mean": float("nan"), "final": float("nan"), "n_tokens": 0}
    return {"mean": float(sel.mean()), "final": float(sel[-1]), "n_tokens": int(sel.size)}
