"""Default output locations owned by the standalone simulator tool."""

from pathlib import Path

from bayestraj.paths import project_path


def default_simulation_data_directory() -> Path:
    """Return the simulation data directory relative to the project root."""
    return project_path("data/simulated")
