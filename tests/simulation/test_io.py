"""Tests for simulated trajectory DataFrame creation and CSV export."""

import pandas as pd
import pytest

from ship_trajectory_prediction.simulation.core import ShipSimulator
from ship_trajectory_prediction.simulation.io import (
    create_simulation_dataframe,
    save_trajectory_data,
)

REFERENCE_LONGITUDE = 8.3122
REFERENCE_LATITUDE = 47.0515


def create_test_simulator():
    """Create a simulator containing two deterministic trajectory samples."""
    simulator = ShipSimulator(v=2.0, sigma=0.0, dt=0.05)
    simulator.step(omega=0.0, motor_running=True)
    simulator.step(omega=0.0, motor_running=True)
    return simulator


def test_simulation_dataframe_contains_utc_timestamps():
    """Absolute timestamps should start at UTC time zero and follow simulation time."""
    simulator = create_test_simulator()

    data = create_simulation_dataframe(
        simulator,
        start_time="2026-01-09 23:00:10+00:00",
        reference_longitude=REFERENCE_LONGITUDE,
        reference_latitude=REFERENCE_LATITUDE,
    )

    assert list(data.columns[:5]) == [
        "time",
        "gps_latitude",
        "gps_longitude",
        "gps_speed",
        "t",
    ]
    assert str(data["time"].dtype) == "datetime64[ns, UTC]"
    assert data.loc[0, "time"] == pd.Timestamp("2026-01-09 23:00:10+00:00")
    assert (data.loc[1, "time"] - data.loc[0, "time"]).total_seconds() == 0.05
    assert data.loc[0, "gps_latitude"] == pytest.approx(REFERENCE_LATITUDE)
    assert data.loc[0, "gps_longitude"] == pytest.approx(REFERENCE_LONGITUDE)
    assert data.loc[1, "gps_longitude"] > data.loc[0, "gps_longitude"]
    assert data["gps_speed"].tolist() == pytest.approx([7.2, 7.2])


def test_csv_uses_shiptech_style_timestamp(tmp_path):
    """CSV timestamps should use the timezone-aware format found in Shiptech data."""
    simulator = create_test_simulator()
    simulator.step(omega=0.0, motor_running=True)

    data = create_simulation_dataframe(
        simulator,
        start_time="2026-01-09 23:00:10+00:00",
        reference_longitude=REFERENCE_LONGITUDE,
        reference_latitude=REFERENCE_LATITUDE,
    )

    output_path = save_trajectory_data(data, tmp_path / "trajectory.csv")
    csv_lines = output_path.read_text(encoding="utf-8").splitlines()

    assert csv_lines[0].startswith("time,gps_latitude,gps_longitude,gps_speed,t,")
    assert csv_lines[1].startswith(
        "2026-01-09 23:00:10.00+00:00,47.05150000,8.31220000,7.200,0.000,"
    )
    assert csv_lines[2].startswith("2026-01-09 23:00:10.05+00:00,")
    assert csv_lines[3].startswith("2026-01-09 23:00:10.10+00:00,")


def test_simulation_dataframe_exports_speed_history_in_mps_and_kmh():
    """Exported speed columns should preserve changes and use their stated units."""
    simulator = ShipSimulator(v=2.0, sigma=0.0, dt=0.05)
    simulator.step(omega=0.0, motor_running=True)
    simulator.v = 3.0
    simulator.step(omega=0.0, motor_running=True)

    data = create_simulation_dataframe(
        simulator,
        reference_longitude=REFERENCE_LONGITUDE,
        reference_latitude=REFERENCE_LATITUDE,
    )

    assert data["v"].tolist() == pytest.approx([2.0, 3.0])
    assert data["gps_speed"].tolist() == pytest.approx([7.2, 10.8])
