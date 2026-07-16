"""Tests for shared project path resolution."""

from ship_trajectory_prediction.paths import PROJECT_ROOT, project_path


def test_project_path_resolves_relative_to_project_root():
    """Resolve relative paths independently of the calling module."""
    assert project_path("data/raw") == (PROJECT_ROOT / "data" / "raw").resolve()
