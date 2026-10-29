"""Locations of Stan model files."""

from bayestraj.paths import project_path

STAN_ROOT = project_path("src/stan")


def stan_path(relative_path):
    """Return an absolute path resolved from ``src/stan``."""
    return (STAN_ROOT / relative_path).resolve()
