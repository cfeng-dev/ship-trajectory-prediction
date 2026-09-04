"""Tests for the standalone trajectory checker."""

import pandas as pd
import pytest
from trajectory_checker import cli
from trajectory_checker.checker import check_trajectory_data


def _trajectory_data(**overrides):
    values = {
        "time": pd.to_datetime(
            [
                "2026-01-01 00:00:00+00:00",
                "2026-01-01 00:00:10+00:00",
                "2026-01-01 00:00:20+00:00",
            ],
            utc=True,
        ),
        "run_id": [4, 4, 4],
        "gps_latitude": [47.0, 47.00001, 47.00002],
        "gps_longitude": [8.0, 8.00001, 8.00002],
        "gps_speed": [0.0, 0.5, 0.5],
    }
    values.update(overrides)
    return pd.DataFrame(values)


def test_check_trajectory_data_accepts_valid_data():
    report = check_trajectory_data(_trajectory_data())

    assert report.is_valid
    assert report.errors == ()
    assert report.warnings == ()
    assert len(report.runs) == 1
    assert report.runs[0].run_id == "4"
    assert report.runs[0].row_count == 3
    assert report.runs[0].duration_seconds == pytest.approx(20.0)
    assert report.runs[0].median_interval_seconds == pytest.approx(10.0)
    assert report.runs[0].distance_meters > 0


def test_check_trajectory_data_reports_missing_columns():
    data = _trajectory_data().drop(columns="gps_longitude")

    report = check_trajectory_data(data)

    assert not report.is_valid
    assert "Missing required columns: ['gps_longitude']" in report.errors
    assert report.runs == ()


def test_check_trajectory_data_reports_invalid_coordinates_and_speeds():
    report = check_trajectory_data(
        _trajectory_data(
            gps_latitude=[47.0, 91.0, 47.1],
            gps_speed=[0.0, -1.0, "invalid"],
        )
    )

    assert not report.is_valid
    assert "Latitude values outside [-90, 90]: 1" in report.errors
    assert "Negative gps_speed values: 1" in report.errors
    assert "Non-numeric or non-finite gps_speed values: 1" in report.errors


def test_check_trajectory_data_reports_duplicate_times_and_large_gaps():
    report = check_trajectory_data(
        _trajectory_data(
            time=pd.to_datetime(
                [
                    "2026-01-01 00:00:00+00:00",
                    "2026-01-01 00:00:00+00:00",
                    "2026-01-01 00:01:00+00:00",
                ],
                utc=True,
            )
        )
    )

    assert not report.is_valid
    assert "Run 4 contains duplicate timestamps: 1" in report.errors
    assert "Run 4 contains 1 time gap(s) larger than 15 seconds." in report.warnings
    assert report.runs[0].large_gap_count == 1


def test_check_trajectory_data_warns_about_source_order():
    data = _trajectory_data().iloc[[1, 0, 2]].reset_index(drop=True)

    report = check_trajectory_data(data)

    assert report.is_valid
    assert report.warnings == ("Run 4 is not ordered by timestamp.",)


def test_cli_returns_zero_for_valid_csv(tmp_path, capsys):
    csv_path = tmp_path / "trajectory.csv"
    _trajectory_data().to_csv(csv_path, index=False)

    exit_code = cli.main([str(csv_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Run 4:" in output
    assert "Result: VALID" in output


def test_print_report_limits_run_summaries_by_default(capsys):
    data = pd.concat(
        [_trajectory_data(run_id=[run_id] * 3) for run_id in range(11)],
        ignore_index=True,
    )
    report = check_trajectory_data(data)

    cli.print_report(report)

    output = capsys.readouterr().out
    assert "Run 9:" in output
    assert "Run 10:" not in output
    assert "1 additional run summaries omitted" in output


def test_cli_returns_one_for_invalid_csv(tmp_path, capsys):
    csv_path = tmp_path / "trajectory.csv"
    _trajectory_data(gps_speed=[0.0, -1.0, 0.5]).to_csv(csv_path, index=False)

    exit_code = cli.main([str(csv_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "Negative gps_speed values: 1" in output
    assert "Result: INVALID" in output


def test_cli_returns_two_for_missing_file(tmp_path, capsys):
    exit_code = cli.main([str(tmp_path / "missing.csv")])

    assert exit_code == 2
    assert "Trajectory CSV file not found" in capsys.readouterr().err
