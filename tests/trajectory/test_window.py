"""Tests for shared trajectory-window preparation."""

import numpy as np
import pandas as pd
import pytest

from ship_trajectory_prediction.trajectory import (
    TrajectoryWindowData,
    prepare_trajectory_window,
)
from ship_trajectory_prediction.trajectory.window import (
    estimate_initial_heading,
    estimate_positive_speed_median,
    estimate_turn_direction,
)


def create_trajectory_data():
    """Create one deliberately unsorted trajectory with mixed speed values."""
    time_offsets = np.array([30, 0, 50, 10, 40, 20])
    speed_by_offset = {
        0: 0.0,
        10: 18.0,
        20: -3.6,
        30: "invalid",
        40: 36.0,
        50: 54.0,
    }
    return pd.DataFrame(
        {
            "time": pd.Timestamp("2026-01-10 08:00:00", tz="UTC")
            + pd.to_timedelta(time_offsets, unit="s"),
            "run_id": 1,
            "gps_latitude": 47.0 + time_offsets * 1e-5,
            "gps_longitude": 8.0 + time_offsets * 2e-5,
            "gps_speed": [speed_by_offset[offset] for offset in time_offsets],
        }
    )


def test_prepare_trajectory_window_sorts_slices_and_converts_values():
    """The selected window should be chronological, local, and expressed in SI."""
    window = prepare_trajectory_window(
        create_trajectory_data(),
        observation_count=3,
        prediction_count=2,
        start_index=1,
    )

    assert isinstance(window, TrajectoryWindowData)
    assert window.timestamps.is_monotonic_increasing
    assert str(window.timestamps.tz) == "UTC"
    assert window.time_seconds.tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert window.x_meters[0] == pytest.approx(0.0)
    assert window.y_meters[0] == pytest.approx(0.0)
    assert window.gps_speed_mps[:2] == pytest.approx([5.0, -1.0])
    assert np.isnan(window.gps_speed_mps[2])
    assert window.gps_speed_mps[3:] == pytest.approx([10.0, 15.0])
    assert window.observation_count == 3
    assert window.prediction_count == 2
    assert window.observed_slice == slice(0, 3)
    assert window.prediction_slice == slice(3, None)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("observation_count", 2),
        ("observation_count", True),
        ("prediction_count", 0),
        ("start_index", -1),
        ("speed_unit", "knots"),
    ],
)
def test_prepare_trajectory_window_rejects_invalid_arguments(argument, value):
    """Shared window arguments should fail before trajectory conversion."""
    arguments = {
        "observation_count": 3,
        "prediction_count": 2,
        "start_index": 0,
        "speed_unit": "km/h",
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=argument):
        prepare_trajectory_window(create_trajectory_data(), **arguments)


def test_prepare_trajectory_window_rejects_missing_columns_and_multiple_runs():
    """A model window requires all shared columns from exactly one run."""
    with pytest.raises(ValueError, match="gps_longitude"):
        prepare_trajectory_window(
            create_trajectory_data().drop(columns="gps_longitude"),
            observation_count=3,
            prediction_count=2,
        )

    multiple_runs = create_trajectory_data()
    multiple_runs.loc[multiple_runs.index[-1], "run_id"] = 2
    with pytest.raises(ValueError, match="exactly one run_id"):
        prepare_trajectory_window(
            multiple_runs,
            observation_count=3,
            prediction_count=2,
        )


def test_prepare_trajectory_window_rejects_short_or_non_chronological_windows():
    """The selected window must exist and contain unique increasing timestamps."""
    with pytest.raises(ValueError, match="Not enough trajectory rows"):
        prepare_trajectory_window(
            create_trajectory_data(),
            observation_count=4,
            prediction_count=3,
        )

    duplicate_time = create_trajectory_data()
    duplicate_time.loc[duplicate_time.index[0], "time"] = duplicate_time.loc[
        duplicate_time.index[4], "time"
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        prepare_trajectory_window(
            duplicate_time,
            observation_count=4,
            prediction_count=2,
        )


def test_shared_estimators_match_existing_model_conventions():
    """Shared estimators should use moving positions and positive speeds only."""
    heading = estimate_initial_heading(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
    )

    assert heading == pytest.approx(np.pi / 4)
    assert estimate_positive_speed_median(
        np.array([np.nan, -1.0, 0.0, 2.0, 4.0])
    ) == pytest.approx(3.0)
    assert (
        estimate_turn_direction(
            np.array([0.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        == 1
    )
    assert (
        estimate_turn_direction(
            np.array([0.0, 1.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
        )
        == -1
    )

    with pytest.raises(ValueError, match="positive finite"):
        estimate_positive_speed_median(np.array([np.nan, -1.0, 0.0]))
