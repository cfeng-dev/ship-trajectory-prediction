"""Tests for loading and filtering real ship trajectory data."""

import pandas as pd

from ship_trajectory_prediction.trajectory.io import (
    read_ship_data,
    resample_trajectory_data,
)


def test_read_ship_data_filters_run_and_time(tmp_path):
    """The loader should filter rows and parse timestamps as UTC."""
    csv_path = tmp_path / "trajectory.csv"
    data = pd.DataFrame(
        {
            "time": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:10Z",
                "2026-01-01T00:00:20Z",
            ],
            "run_id": [1, 1, 2],
            "gps_latitude": [47.0, 47.1, 47.2],
            "gps_longitude": [8.0, 8.1, 8.2],
            "gps_speed": [0.0, 1.0, 2.0],
            "shaft_speed": [0.0, 10.0, 20.0],
            "thruster_speed": [0.0, 0.0, 0.0],
        }
    )
    data.to_csv(csv_path, index=False)

    result = read_ship_data(
        csv_path,
        run_id=1,
        start_time="2026-01-01T00:00:10Z",
    )

    assert len(result) == 1
    assert result.iloc[0]["gps_speed"] == 1.0
    assert str(result["time"].dt.tz) == "UTC"


def test_read_ship_data_accepts_simulated_trajectory_columns(tmp_path):
    """The loader should not require Shiptech-specific propulsion columns."""
    csv_path = tmp_path / "simulated_trajectory.csv"
    data = pd.DataFrame(
        {
            "time": ["2026-01-01T00:00:00Z"],
            "run_id": [0],
            "gps_latitude": [47.0],
            "gps_longitude": [8.0],
            "gps_speed": [7.2],
            "x_true": [0.0],
            "y_true": [0.0],
        }
    )
    data.to_csv(csv_path, index=False)

    result = read_ship_data(csv_path, run_id=0)

    assert len(result) == 1
    assert result.iloc[0]["gps_speed"] == 7.2
    assert "shaft_speed" not in result.columns
    assert "thruster_speed" not in result.columns


def test_resample_trajectory_data_uses_separate_ten_second_run_windows():
    """Resampling should keep one original row per window and per run."""
    data = pd.DataFrame(
        {
            "time": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00.050Z",
                "2026-01-01T00:00:10Z",
                "2026-01-01T00:00:03Z",
                "2026-01-01T00:00:03.050Z",
                "2026-01-01T00:00:13Z",
            ],
            "run_id": [1, 1, 1, 2, 2, 2],
            "value": [0, 1, 2, 3, 4, 5],
        }
    )

    result = resample_trajectory_data(data, interval_seconds=10)

    assert result["value"].tolist() == [0, 2, 3, 5]
    assert result["run_id"].tolist() == [1, 1, 2, 2]
    assert str(result["time"].dt.tz) == "UTC"


def test_resample_trajectory_data_supports_simulation_data_without_run_id():
    """Simulation data should be resampled even when no run ID is present."""
    data = pd.DataFrame(
        {
            "time": [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00.050000+00:00",
                "2026-01-01T00:00:10+00:00",
            ],
            "t": [0.0, 0.05, 10.0],
        }
    )

    result = resample_trajectory_data(data, interval_seconds=10)

    assert result["t"].tolist() == [0.0, 10.0]
