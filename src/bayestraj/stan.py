"""Locations of Stan model files."""

from pathlib import Path

STAN_ROOT = Path(__file__).resolve().parents[1] / "stan"


def stan_path(relative_path):
    """Return an absolute path resolved from ``src/stan``."""
    return (STAN_ROOT / relative_path).resolve()
