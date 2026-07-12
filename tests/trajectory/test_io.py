"""Tests for loading and filtering real ship trajectory data."""

import pandas as pd

from ship_trajectory_prediction.trajectory.io import read_ship_data


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
