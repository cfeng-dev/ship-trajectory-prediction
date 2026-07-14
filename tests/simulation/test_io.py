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

    save_result = save_trajectory_data(data, tmp_path / "trajectory.csv")
    csv_lines = save_result.output_path.read_text(encoding="utf-8").splitlines()

    assert save_result.run_id == 0
    assert save_result.appended is False
    assert save_result.continued is False

    assert csv_lines[0].startswith(
        "time,run_id,gps_latitude,gps_longitude,gps_speed,t,"
    )
    assert csv_lines[1].startswith(
        "2026-01-09 23:00:10.00+00:00,0,47.05150000,8.31220000,7.200,0.000,"
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


def test_save_appends_trajectory_with_next_run_id(tmp_path):
    """Saving to the same file should append a new, incremented run."""
    output_path = tmp_path / "trajectory.csv"
    first_run = create_simulation_dataframe(
        create_test_simulator(),
        start_time="2026-01-09 23:00:10+00:00",
        reference_longitude=REFERENCE_LONGITUDE,
        reference_latitude=REFERENCE_LATITUDE,
    )
    second_run = create_simulation_dataframe(
        create_test_simulator(),
        start_time="2026-01-09 23:01:10+00:00",
        reference_longitude=REFERENCE_LONGITUDE,
        reference_latitude=REFERENCE_LATITUDE,
    )

    first_result = save_trajectory_data(first_run, output_path)
    second_result = save_trajectory_data(second_run, output_path)

    saved_data = pd.read_csv(output_path)
    assert first_result.run_id == 0
    assert first_result.appended is False
    assert first_result.continued is False
    assert second_result.run_id == 1
    assert second_result.appended is True
    assert second_result.continued is False
    assert saved_data["run_id"].tolist() == [0, 0, 1, 1]
    assert len(saved_data) == 4


def test_save_continues_existing_run_without_duplicate_samples(tmp_path):
    """New samples from one session should retain the same run ID."""
    output_path = tmp_path / "trajectory.csv"
    trajectory = create_simulation_dataframe(
        create_test_simulator(),
        reference_longitude=REFERENCE_LONGITUDE,
        reference_latitude=REFERENCE_LATITUDE,
    )

    first_result = save_trajectory_data(trajectory.iloc[:1], output_path)
    continued_result = save_trajectory_data(
        trajectory.iloc[1:],
        output_path,
        existing_run_id=first_result.run_id,
    )

    saved_data = pd.read_csv(output_path)
    assert continued_result.run_id == 0
    assert continued_result.appended is True
    assert continued_result.continued is True
    assert saved_data["run_id"].tolist() == [0, 0]
    assert len(saved_data) == 2


def test_save_migrates_compatible_csv_without_run_id(tmp_path):
    """A compatible older export should become run zero before appending."""
    output_path = tmp_path / "trajectory.csv"
    existing_run = create_simulation_dataframe(
        create_test_simulator(),
        start_time="2026-01-09 23:00:10+00:00",
        reference_longitude=REFERENCE_LONGITUDE,
        reference_latitude=REFERENCE_LATITUDE,
    )
    existing_run.to_csv(output_path, index=False)

    new_run = create_simulation_dataframe(
        create_test_simulator(),
        start_time="2026-01-09 23:01:10+00:00",
        reference_longitude=REFERENCE_LONGITUDE,
        reference_latitude=REFERENCE_LATITUDE,
    )
    save_trajectory_data(new_run, output_path)

    saved_data = pd.read_csv(output_path)
    assert saved_data.columns[1] == "run_id"
    assert saved_data["run_id"].tolist() == [0, 0, 1, 1]


def test_save_rejects_incompatible_existing_csv_without_overwriting(tmp_path):
    """An unrelated existing file should remain unchanged."""
    output_path = tmp_path / "trajectory.csv"
    original_content = "unexpected_column\nkeep-me\n"
    output_path.write_text(original_content, encoding="utf-8")
    new_run = create_simulation_dataframe(
        create_test_simulator(),
        reference_longitude=REFERENCE_LONGITUDE,
        reference_latitude=REFERENCE_LATITUDE,
    )

    with pytest.raises(ValueError, match="do not match"):
        save_trajectory_data(new_run, output_path)

    assert output_path.read_text(encoding="utf-8") == original_content
