"""Paths for the single-turn control, under the repository-wide data/ folder."""

from prognosis.common.paths import DATA_DIR, config_file, resolve_root_path  # noqa: F401

SINGLE_TURN_DIR = DATA_DIR / "single_turn"


def data_file(name):
    return SINGLE_TURN_DIR / name
