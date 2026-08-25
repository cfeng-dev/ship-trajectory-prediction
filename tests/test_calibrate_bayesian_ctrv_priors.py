"""Tests for parametric Bayesian CTRV prior exploration."""

import numpy as np
import pandas as pd
import pytest

from experiments.data_exploration.calibrate_bayesian_ctrv_priors import (
    _print_report,
    calibrate_parametric_ctrv_priors,
)
from ship_trajectory_prediction.observations.coordinates import (
    local_to_gps_coordinates,
)


def create_position_data(*, count=20):
    """Create one curved position-only run."""
    time_seconds = np.arange(count, dtype=float) * 5.0
    heading = 0.2 + 0.01 * time_seconds
    x_meters = np.zeros(count)
    y_meters = np.zeros(count)
    for index in range(1, count):
        x_meters[index] = x_meters[index - 1] + 3.0 * 5.0 * np.cos(heading[index - 1])
        y_meters[index] = y_meters[index - 1] + 3.0 * 5.0 * np.sin(heading[index - 1])
    longitude, latitude = local_to_gps_coordinates(
        x_meters,
        y_meters,
        reference_longitude=8.0,
        reference_latitude=54.0,
    )
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=count, freq="5s", tz="UTC"),
            "run_id": 0,
            "gps_latitude": latitude,
            "gps_longitude": longitude,
        }
    )


@pytest.mark.parametrize("history_position_count", [20, 10])
def test_calibration_supports_both_comparison_histories(history_position_count):
    """K=20 and K=10 should both yield finite constant-motion estimates."""
    result = calibrate_parametric_ctrv_priors(
        create_position_data(),
        run_start=0,
        run_stop=1,
        history_position_count=history_position_count,
    )

    assert result.history_position_count == history_position_count
    assert len(result.estimates) == 20 // history_position_count
    assert result.speed_summary.median == pytest.approx(3.0, rel=0.02)
    assert result.turn_rate_summary.median == pytest.approx(0.01, abs=0.002)


def test_calibration_uses_no_external_speed_columns():
    """Adding implausible GPS speed must not alter position-only estimates."""
    data = create_position_data()
    reference = calibrate_parametric_ctrv_priors(
        data,
        run_start=0,
        run_stop=1,
        history_position_count=10,
    )
    changed = data.assign(gps_speed=100_000.0)
    changed_result = calibrate_parametric_ctrv_priors(
        changed,
        run_start=0,
        run_stop=1,
        history_position_count=10,
    )

    pd.testing.assert_frame_equal(changed_result.estimates, reference.estimates)


def test_report_marks_candidates_as_pending_validation(capsys):
    """Reporting must not claim that transferred candidates are final."""
    result = calibrate_parametric_ctrv_priors(
        create_position_data(),
        run_start=0,
        run_stop=1,
        history_position_count=10,
    )

    _print_report(result)

    output = capsys.readouterr().out
    assert "empirical candidates" in output
    assert "held-out validation pending" in output
    assert "process" not in output.lower()
