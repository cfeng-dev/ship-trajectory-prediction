"""Locations of Stan model files."""

from pathlib import Path

STAN_ROOT = Path(__file__).resolve().parents[3] / "stan"


def stan_path(relative_path):
    """Return an absolute path resolved from the project Stan directory."""
    return (STAN_ROOT / relative_path).resolve()
