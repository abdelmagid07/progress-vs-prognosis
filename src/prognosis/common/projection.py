"""Load frozen value axis and project activations."""

import json

import numpy as np

from prognosis.extract.hooks import cosine_projection, unit_direction
from prognosis.common.config import load_defaults


def load_axis_direction(
    axis_path,
    layer: int | None = None,
) -> np.ndarray:
    """Return unit direction for the given layer from value_axis.npy."""
    axis = np.load(axis_path)
    if layer is None:
        defaults = load_defaults()
        layer = defaults["layer"]
        if defaults.get("axis_manifest_path") and defaults["axis_manifest_path"].exists():
            with open(defaults["axis_manifest_path"]) as f:
                manifest = json.load(f)
            layer = manifest.get("default_layer", layer)
    direction = axis[layer].astype(np.float32)
    return unit_direction(direction)


def project_activation(activation, direction: np.ndarray) -> float:
    """Cosine projection of a single activation vector onto axis direction."""
    import torch

    act = torch.tensor(activation, dtype=torch.float32).unsqueeze(0)
    dir_t = torch.tensor(direction, dtype=torch.float32)
    return float(cosine_projection(act, dir_t)[0].item())
