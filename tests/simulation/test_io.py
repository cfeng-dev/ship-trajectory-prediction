"""Tests for simulated trajectory DataFrame creation and CSV export."""

import pandas as pd

from ship_trajectory_prediction.simulation.core import ShipSimulator
from ship_trajectory_prediction.simulation.io import (
    create_simulation_dataframe,
    save_trajectory_data,
)


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
    )

    assert list(data.columns[:2]) == ["time", "t"]
    assert str(data["time"].dtype) == "datetime64[ns, UTC]"
    assert data.loc[0, "time"] == pd.Timestamp("2026-01-09 23:00:10+00:00")
    assert (data.loc[1, "time"] - data.loc[0, "time"]).total_seconds() == 0.05


def test_csv_uses_shiptech_style_timestamp(tmp_path):
    """CSV timestamps should use the timezone-aware format found in Shiptech data."""
    data = create_simulation_dataframe(
        create_test_simulator(),
        start_time="2026-01-09 23:00:10+00:00",
    )

    output_path = save_trajectory_data(data, tmp_path / "trajectory.csv")
    csv_lines = output_path.read_text(encoding="utf-8").splitlines()

    assert csv_lines[0].startswith("time,t,")
    assert csv_lines[1].startswith("2026-01-09 23:00:10+00:00,0.000,")
    assert csv_lines[2].startswith("2026-01-09 23:00:10.050000+00:00,0.050,")
