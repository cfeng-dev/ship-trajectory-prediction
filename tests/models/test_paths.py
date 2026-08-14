"""Tests for Stan model path resolution."""

from ship_trajectory_prediction.models.paths import STAN_ROOT, stan_path


def test_stan_path_resolves_relative_to_stan_root():
    """Resolve relative paths from the Stan model directory."""
    assert stan_path("models") == (STAN_ROOT / "models").resolve()
