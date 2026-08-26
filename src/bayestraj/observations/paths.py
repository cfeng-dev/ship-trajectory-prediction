"""Locations of recorded and simulated trajectory data."""

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


def data_path(relative_path):
    """Return an absolute path resolved from the project data directory."""
    return (DATA_ROOT / relative_path).resolve()
