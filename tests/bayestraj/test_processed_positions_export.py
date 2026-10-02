"""Tests for the shared generic position-trajectory export."""

import numpy as np
import pandas as pd

from bayestraj.observations.io import export_processed_positions


def test_export_processed_positions_sorts_and_rounds_elapsed_time(tmp_path):
    """A generic position export writes sorted elapsed seconds and x/y only."""
    output_csv = tmp_path / "processed" / "trajectory.csv"

    output_path = export_processed_positions(
        time=[101.24, 100.0, 100.44],
        x=[3.0, 1.0, 2.0],
        y=[-3.0, -1.0, -2.0],
        output_csv=output_csv,
    )

    exported = pd.read_csv(output_path)

    assert output_path == output_csv
    assert exported.columns.tolist() == ["time", "x", "y"]
    assert exported["time"].tolist() == [0.0, 0.4, 1.2]
    np.testing.assert_allclose(exported["x"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(exported["y"], [-1.0, -2.0, -3.0])
