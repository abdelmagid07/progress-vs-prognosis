"""Load the agentic-transfer config."""

import yaml

from prognosis.common.paths import config_file, resolve_root_path


def load_defaults():
    cfg = yaml.safe_load(config_file("defaults.yaml").read_text(encoding="utf-8"))
    cfg["axis_path"] = resolve_root_path(cfg["axis_path"])
    if "axis_manifest_path" in cfg:
        cfg["axis_manifest_path"] = resolve_root_path(cfg["axis_manifest_path"])
    return cfg
