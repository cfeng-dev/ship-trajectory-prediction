"""Tests for exporting recorded ship runs to the generic position CSV format."""

import numpy as np
import pandas as pd
import pytest

import bayestraj.observations.io as observations_io
from bayestraj.observations.io import export_shiptech_trajectory


def _write_raw_ship_data(csv_path, rows):
    """Write the minimum recorded-ship schema accepted by ``read_ship_data``."""
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _raw_ship_row(
    time,
    *,
    run_id=42,
    gps_latitude=47.0,
    gps_longitude=8.0,
):
    return {
        "time": time,
        "run_id": run_id,
        "gps_latitude": gps_latitude,
        "gps_longitude": gps_longitude,
        "gps_speed": 0.0,
    }


def test_export_shiptech_trajectory_writes_sorted_position_only_data(tmp_path):
    """Exporting one run preserves its irregular timestamp spacing and positions."""
    input_csv = tmp_path / "raw_ship_data.csv"
    _write_raw_ship_data(
        input_csv,
        [
            {
                "time": "2026-01-09 23:00:30+00:00",
                "run_id": 42,
                "gps_latitude": 47.00030,
                "gps_longitude": 8.00030,
                "gps_speed": 0.0,
            },
            {
                "time": "2026-01-09 23:00:00+00:00",
                "run_id": 42,
                "gps_latitude": 47.00000,
                "gps_longitude": 8.00000,
                "gps_speed": 0.0,
            },
            {
                "time": "2026-01-09 23:00:10+00:00",
                "run_id": 42,
                "gps_latitude": 47.00010,
                "gps_longitude": 8.00010,
                "gps_speed": 0.0,
            },
            {
                "time": "2026-01-09 23:00:21+00:00",
                "run_id": 42,
                "gps_latitude": 47.00021,
                "gps_longitude": 8.00021,
                "gps_speed": 0.0,
            },
            {
                "time": "2026-01-09 23:00:05+00:00",
                "run_id": 7,
                "gps_latitude": 47.01000,
                "gps_longitude": 8.01000,
                "gps_speed": 0.0,
            },
        ],
    )
    raw_contents = input_csv.read_bytes()

    output_path = export_shiptech_trajectory(
        input_csv=input_csv,
        run_id=42,
        output_dir=tmp_path / "processed",
    )

    exported = pd.read_csv(output_path)

    assert output_path == tmp_path / "processed" / "ship_trajectory_run_42.csv"
    assert exported.columns.tolist() == ["time", "x", "y"]
    assert exported["time"].tolist() == [0.0, 10.0, 21.0, 30.0]
    np.testing.assert_allclose(exported.loc[0, ["x", "y"]], [0.0, 0.0])
    assert (exported.loc[1:, "x"] > 0.0).all()
    assert (exported.loc[1:, "y"] > 0.0).all()
    assert exported.dtypes.tolist() == [np.dtype("float64")] * 3
    assert input_csv.read_bytes() == raw_contents


def test_export_shiptech_trajectory_rejects_unknown_run_id(tmp_path):
    """A missing run ID must not produce an empty processed CSV."""
    input_csv = tmp_path / "raw_ship_data.csv"
    _write_raw_ship_data(
        input_csv,
        [_raw_ship_row("2026-01-09 23:00:00+00:00", run_id=7)],
    )

    with pytest.raises(ValueError, match="run_id 42 was not found"):
        export_shiptech_trajectory(input_csv, run_id=42, output_dir=tmp_path)


@pytest.mark.parametrize("run_id", (None, (42, 7)))
def test_export_shiptech_trajectory_rejects_multiple_run_selection(tmp_path, run_id):
    """The position-only output represents exactly one selected source run."""
    input_csv = tmp_path / "raw_ship_data.csv"
    _write_raw_ship_data(
        input_csv,
        [
            _raw_ship_row("2026-01-09 23:00:00+00:00", run_id=42),
            _raw_ship_row("2026-01-09 23:00:10+00:00", run_id=7),
        ],
    )

    with pytest.raises(ValueError, match="run_id must select exactly one run"):
        export_shiptech_trajectory(input_csv, run_id=run_id, output_dir=tmp_path)


def test_export_shiptech_trajectory_rejects_run_with_one_sample(tmp_path):
    """One position cannot form a trajectory for the processed format."""
    input_csv = tmp_path / "raw_ship_data.csv"
    _write_raw_ship_data(
        input_csv,
        [_raw_ship_row("2026-01-09 23:00:00+00:00")],
    )

    with pytest.raises(ValueError, match="at least two samples"):
        export_shiptech_trajectory(input_csv, run_id=42, output_dir=tmp_path)


def test_export_shiptech_trajectory_rejects_repeated_timestamps(tmp_path):
    """Duplicate source timestamps would make a non-increasing time column."""
    input_csv = tmp_path / "raw_ship_data.csv"
    _write_raw_ship_data(
        input_csv,
        [
            _raw_ship_row("2026-01-09 23:00:00+00:00"),
            _raw_ship_row(
                "2026-01-09 23:00:00+00:00",
                gps_latitude=47.00001,
                gps_longitude=8.00001,
            ),
        ],
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        export_shiptech_trajectory(input_csv, run_id=42, output_dir=tmp_path)


def test_export_shiptech_trajectory_rejects_invalid_source_timestamp(tmp_path):
    """Missing source timestamps cannot define elapsed trajectory time."""
    input_csv = tmp_path / "raw_ship_data.csv"
    _write_raw_ship_data(
        input_csv,
        [
            _raw_ship_row("2026-01-09 23:00:00+00:00"),
            _raw_ship_row(""),
        ],
    )

    with pytest.raises(ValueError, match="Source timestamps must be valid"):
        export_shiptech_trajectory(input_csv, run_id=42, output_dir=tmp_path)


@pytest.mark.parametrize(
    ("gps_latitude", "gps_longitude"),
    ((np.nan, 8.0), (47.0, np.inf)),
)
def test_export_shiptech_trajectory_rejects_non_finite_gps_coordinates(
    tmp_path,
    gps_latitude,
    gps_longitude,
):
    """Invalid GPS measurements cannot be transformed into local positions."""
    input_csv = tmp_path / "raw_ship_data.csv"
    _write_raw_ship_data(
        input_csv,
        [
            _raw_ship_row("2026-01-09 23:00:00+00:00"),
            _raw_ship_row(
                "2026-01-09 23:00:10+00:00",
                gps_latitude=gps_latitude,
                gps_longitude=gps_longitude,
            ),
        ],
    )

    with pytest.raises(ValueError, match="GPS coordinates must be finite"):
        export_shiptech_trajectory(input_csv, run_id=42, output_dir=tmp_path)


def test_export_shiptech_trajectory_rejects_mismatched_generated_coordinates(
    tmp_path,
    monkeypatch,
):
    """A coordinate conversion must return one east/north pair per timestamp."""
    input_csv = tmp_path / "raw_ship_data.csv"
    _write_raw_ship_data(
        input_csv,
        [
            _raw_ship_row("2026-01-09 23:00:00+00:00"),
            _raw_ship_row("2026-01-09 23:00:10+00:00"),
        ],
    )
    monkeypatch.setattr(
        observations_io.coordinates,
        "gps_to_local_coordinates",
        lambda *args, **kwargs: (np.array([0.0]), np.array([0.0, 1.0])),
    )

    with pytest.raises(ValueError, match="must match the time sample count"):
        export_shiptech_trajectory(input_csv, run_id=42, output_dir=tmp_path)


def test_export_shiptech_trajectory_rejects_non_finite_generated_coordinates(
    tmp_path,
    monkeypatch,
):
    """The exported local position data must remain finite after conversion."""
    input_csv = tmp_path / "raw_ship_data.csv"
    _write_raw_ship_data(
        input_csv,
        [
            _raw_ship_row("2026-01-09 23:00:00+00:00"),
            _raw_ship_row("2026-01-09 23:00:10+00:00"),
        ],
    )
    monkeypatch.setattr(
        observations_io.coordinates,
        "gps_to_local_coordinates",
        lambda *args, **kwargs: (
            np.array([0.0, np.nan]),
            np.array([0.0, 1.0]),
        ),
    )

    with pytest.raises(ValueError, match="Generated x/y coordinates must be finite"):
        export_shiptech_trajectory(input_csv, run_id=42, output_dir=tmp_path)
