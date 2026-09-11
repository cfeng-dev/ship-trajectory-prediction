"""Tests for the minimal Ship Simulator CSV export schema."""

from types import SimpleNamespace

import numpy as np
import pytest
from ship_simulator import coordinates
from ship_simulator import io as simulation_io
from ship_simulator.view import format_elapsed_time


@pytest.mark.parametrize(
    ("seconds", "expected"),
    ((20.0, "20.0 s"), (80.0, "1 min 20 s"), (3_735.0, "1 h 02 min 15 s")),
)
def test_status_time_uses_adaptive_elapsed_format(seconds, expected):
    assert format_elapsed_time(seconds) == expected


def test_csv_export_contains_unnoised_gps_and_motion_values_only():
    simulator = SimpleNamespace(
        x_all=[0.0, 10.0],
        y_all=[0.0, -5.0],
        t_all=[0.0, 1.0],
        v_all=[5.0, 6.0],
        theta_all=[0.0, 0.1],
        omega_all=[0.0, 0.2],
        motor_state_all=[False, True],
    )
    reference_longitude = 8.0
    reference_latitude = 47.0

    data = simulation_io.create_simulation_dataframe(
        simulator,
        start_time="2026-01-01T00:00:00Z",
        reference_longitude=reference_longitude,
        reference_latitude=reference_latitude,
    )

    expected_longitude, expected_latitude = coordinates.local_to_gps_coordinates(
        np.asarray(simulator.x_all),
        np.asarray(simulator.y_all),
        reference_longitude=reference_longitude,
        reference_latitude=reference_latitude,
    )
    assert data.columns.tolist() == [
        "time",
        "gps_latitude",
        "gps_longitude",
        "gps_speed",
        "t",
        "theta",
        "omega",
        "v",
        "simulation_running",
    ]
    np.testing.assert_allclose(data["gps_longitude"], expected_longitude)
    np.testing.assert_allclose(data["gps_latitude"], expected_latitude)
    np.testing.assert_allclose(data["gps_speed"], [18.0, 21.6])
