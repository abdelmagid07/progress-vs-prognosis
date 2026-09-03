"""Repository paths. Inputs live in configs/, outputs in data/ and results/."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def config_file(name):
    return CONFIG_DIR / name


def data_file(name):
    return DATA_DIR / name


def resolve_root_path(relative):
    """Resolve a config-declared path relative to the repository root."""
    return (ROOT / relative).resolve()
