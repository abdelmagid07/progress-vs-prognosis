"""Repository paths. Configuration lives in configs/, everything generated in data/."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
TRAJ_DIR = DATA_DIR / "trajectories"
NORMALIZED_DIR = DATA_DIR / "normalized"


def config_file(name):
    return CONFIG_DIR / name


def data_file(name):
    return DATA_DIR / name


def resolve_root_path(relative):
    """Resolve a config-declared path relative to the repository root."""
    return (ROOT / relative).resolve()


def require_axis_path(axis_path):
    if not Path(axis_path).exists():
        raise FileNotFoundError(
            f"Value axis not found at {axis_path}. Reconstruct it first; see docs/REPRODUCE.md."
        )
    return axis_path
