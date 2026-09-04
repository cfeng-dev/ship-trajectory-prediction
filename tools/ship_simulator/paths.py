"""Default output locations owned by the standalone simulator tool."""

from pathlib import Path


def default_simulation_data_directory() -> Path:
    """Return a useful data directory without importing project infrastructure."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return (candidate / "data" / "simulated").resolve()

    return (Path.cwd() / "data" / "simulated").resolve()
