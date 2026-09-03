"""Load the single-turn control config."""

import yaml

from prognosis.single_turn.paths import config_file, resolve_root_path


def load_defaults():
    cfg = yaml.safe_load(config_file("single_turn.yaml").read_text(encoding="utf-8"))
    cfg["axis_path"] = resolve_root_path(cfg["axis_path"])
    cfg["axis_manifest_path"] = resolve_root_path(cfg["axis_manifest_path"])
    return cfg
