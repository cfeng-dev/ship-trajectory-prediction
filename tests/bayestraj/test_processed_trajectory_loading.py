"""Processed local trajectories have a dedicated CSV loading contract."""

import numpy as np
import pandas as pd
import pytest

import bayestraj.observations.io as observations_io


def test_processed_trajectory_reader_projects_elapsed_local_positions(tmp_path):
    """Read time/x/y as sorted elapsed seconds and local metres only."""
    csv_path = tmp_path / "trajectory.csv"
    pd.DataFrame(
        {
            "time": [34.0, 24.0, 44.0],
            "x": [12.0, 1.0, 23.0],
            "y": [-5.0, -2.0, -8.0],
            "source": ["b", "a", "c"],
        }
    ).to_csv(csv_path, index=False)

    reader = getattr(observations_io, "read_processed_trajectory", None)

    assert callable(reader)
    trajectory = reader(csv_path)
    assert trajectory.columns.tolist() == ["time", "x", "y"]
    np.testing.assert_allclose(trajectory["time"], [0.0, 10.0, 20.0])
    np.testing.assert_allclose(trajectory["x"], [1.0, 12.0, 23.0])
    np.testing.assert_allclose(trajectory["y"], [-2.0, -5.0, -8.0])


def test_processed_trajectory_reader_rejects_an_empty_csv(tmp_path):
    """An empty header-only file reports an input error instead of indexing a row."""
    csv_path = tmp_path / "trajectory.csv"
    pd.DataFrame(columns=["time", "x", "y"]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="at least one"):
        observations_io.read_processed_trajectory(csv_path)
