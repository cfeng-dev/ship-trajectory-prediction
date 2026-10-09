"""Tests for exporting the BR24 target reference position trajectory."""

import numpy as np
import pandas as pd
import pytest

from bayestraj.observations.io import export_br24_target_reference_trajectory


def test_export_br24_target_reference_trajectory_sorts_elapsed_time(tmp_path):
    """The position-only export starts at zero and preserves source intervals."""
    input_csv = tmp_path / "BR24_target_reference_state.csv"
    pd.DataFrame(
        {
            "time": [101.24, 100.0, 100.44],
            "x": [3.0, 1.0, 2.0],
            "y": [-3.0, -1.0, -2.0],
            "yaw": [0.3, 0.1, 0.2],
            "velocity": [3.0, 1.0, 2.0],
        }
    ).to_csv(input_csv, index=False)
    raw_contents = input_csv.read_bytes()

    output_path = export_br24_target_reference_trajectory(
        input_csv=input_csv,
        output_csv=tmp_path / "processed" / "BR24_target_reference_trajectory.csv",
    )

    exported = pd.read_csv(output_path)

    assert output_path == tmp_path / "processed" / "BR24_target_reference_trajectory.csv"
    assert exported.columns.tolist() == ["time", "x", "y"]
    assert exported["time"].tolist() == [0.0, 0.4, 1.2]
    np.testing.assert_allclose(exported["x"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(exported["y"], [-1.0, -2.0, -3.0])
    assert exported.dtypes.tolist() == [np.dtype("float64")] * 3
    assert input_csv.read_bytes() == raw_contents


def test_export_br24_target_reference_trajectory_rejects_one_sample(tmp_path):
    """A single point cannot define a processed trajectory."""
    input_csv = tmp_path / "BR24_target_reference_state.csv"
    pd.DataFrame({"time": [100.0], "x": [1.0], "y": [-1.0]}).to_csv(
        input_csv,
        index=False,
    )

    with pytest.raises(ValueError, match="at least two samples"):
        export_br24_target_reference_trajectory(input_csv, tmp_path / "output.csv")


def test_export_br24_target_reference_trajectory_rejects_repeated_times(tmp_path):
    """Equal times would violate the processed trajectory time contract."""
    input_csv = tmp_path / "BR24_target_reference_state.csv"
    pd.DataFrame({"time": [100.0, 100.0], "x": [1.0, 2.0], "y": [-1.0, -2.0]}).to_csv(
        input_csv, index=False
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        export_br24_target_reference_trajectory(input_csv, tmp_path / "output.csv")


@pytest.mark.parametrize("column", ("x", "y"))
def test_export_br24_target_reference_trajectory_skips_non_finite_positions(
    tmp_path, column
):
    """Missing Cartesian positions are omitted without changing valid time gaps."""
    input_csv = tmp_path / "BR24_target_reference_state.csv"
    source_data = pd.DataFrame(
        {
            "time": [100.0, 110.0, 121.0],
            "x": [1.0, 2.0, 3.0],
            "y": [-1.0, -2.0, -3.0],
        }
    )
    source_data.loc[1, column] = np.nan
    source_data.to_csv(input_csv, index=False)

    exported = pd.read_csv(
        export_br24_target_reference_trajectory(input_csv, tmp_path / "output.csv")
    )

    assert exported["time"].tolist() == [0.0, 21.0]
    np.testing.assert_allclose(exported["x"], [1.0, 3.0])
    np.testing.assert_allclose(exported["y"], [-1.0, -3.0])


def test_export_br24_target_reference_trajectory_rejects_non_finite_time(tmp_path):
    """Elapsed time cannot be reconstructed from a missing source time."""
    input_csv = tmp_path / "BR24_target_reference_state.csv"
    pd.DataFrame({"time": [100.0, np.nan], "x": [1.0, 2.0], "y": [-1.0, -2.0]}).to_csv(
        input_csv, index=False
    )

    with pytest.raises(ValueError, match="time must contain finite numeric values"):
        export_br24_target_reference_trajectory(input_csv, tmp_path / "output.csv")


def test_export_br24_target_reference_trajectory_rejects_missing_positions(tmp_path):
    """The reference state data must supply every generic position column."""
    input_csv = tmp_path / "BR24_target_reference_state.csv"
    pd.DataFrame({"time": [100.0, 110.0], "x": [1.0, 2.0]}).to_csv(
        input_csv,
        index=False,
    )

    with pytest.raises(ValueError, match=r"Missing required columns: \['y'\]"):
        export_br24_target_reference_trajectory(input_csv, tmp_path / "output.csv")
