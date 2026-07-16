"""Shared paths for project data and resources."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(relative_path):
    """Return an absolute path resolved from the project root."""
    return (PROJECT_ROOT / relative_path).resolve()
