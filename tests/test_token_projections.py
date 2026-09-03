"""The column contract between the extraction step and the held-out AUROC.

Extraction writes one column per function mean, interleaved before/after; the AUROC
step reads those columns back as per-function (after - before) differences. A silent
disagreement about that order would not raise anywhere, it would just produce the
wrong direction, so the round trip is pinned here.
"""

import numpy as np
import torch

from prognosis.axis.token_level_auroc import function_differences, heldout_auroc
from prognosis.axis.token_projection_extraction import stack_means

N_LAYERS = 3
HIDDEN = 4


def fake_means(names):
    return {
        name: {
            "before_mean": {layer: torch.arange(HIDDEN).float() + i
                            for layer in range(N_LAYERS)},
            "after_mean": {layer: torch.arange(HIDDEN).float() + i + 100
                           for layer in range(N_LAYERS)},
        }
        for i, name in enumerate(names)
    }


def test_stack_means_orders_functions_alphabetically():
    stacked, fn_names = stack_means(fake_means(["charlie", "alpha", "bravo"]), N_LAYERS)
    assert fn_names == ["alpha", "bravo", "charlie"]
    assert stacked.shape == (N_LAYERS, 6, HIDDEN)


def test_columns_interleave_before_and_after():
    means = fake_means(["alpha", "bravo"])
    stacked, fn_names = stack_means(means, N_LAYERS)
    for i, name in enumerate(fn_names):
        assert torch.equal(stacked[0, 2 * i], means[name]["before_mean"][0])
        assert torch.equal(stacked[0, 2 * i + 1], means[name]["after_mean"][0])


def test_dots_match_explicit_projection():
    means = fake_means(["alpha", "bravo"])
    stacked, _ = stack_means(means, N_LAYERS)
    token = torch.tensor([1.0, 2.0, 3.0, 4.0])
    dots = (stacked * token.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    assert dots.shape == (N_LAYERS, 4)
    for layer in range(N_LAYERS):
        for col in range(4):
            assert torch.allclose(dots[layer, col], stacked[layer, col] @ token)


def test_auroc_reads_the_columns_extraction_writes():
    """An after-heavy token must score above a before-heavy one at every layer."""
    means = fake_means(["alpha", "bravo", "charlie", "delta"])
    stacked, fn_names = stack_means(means, N_LAYERS)

    rows, labels = [], []
    rng = np.random.default_rng(0)
    for name in fn_names:
        for label in ("before", "after"):
            base = means[name][f"{label}_mean"][0]
            for _ in range(4):
                token = base + torch.tensor(rng.normal(0, 0.01, HIDDEN)).float()
                rows.append((stacked * token.unsqueeze(0).unsqueeze(0)).sum(dim=-1).numpy())
                labels.append(label)

    dots = np.stack(rows)
    diffs = function_differences(dots)
    assert diffs.shape == (len(rows), N_LAYERS, len(fn_names))

    mean_auc, _, n_used = heldout_auroc(
        dots,
        np.array([n for n in fn_names for _ in range(8)]),
        np.array(labels),
        fn_names,
        n_splits=4,
        n_train=2,
    )
    assert n_used == 4
    assert (mean_auc > 0.9).all()
