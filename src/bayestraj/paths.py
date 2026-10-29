"""Resolve project-relative paths independently of the working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(relative_path: str | Path = "") -> Path:
    """Return an absolute path resolved from the project root directory."""
    return (PROJECT_ROOT / relative_path).resolve()
