"""Tests for trajectory data path resolution."""

from ship_trajectory_prediction.observations.paths import DATA_ROOT, data_path


def test_data_path_resolves_relative_to_data_root():
    """Resolve relative paths from the trajectory data directory."""
    assert data_path("raw") == (DATA_ROOT / "raw").resolve()
