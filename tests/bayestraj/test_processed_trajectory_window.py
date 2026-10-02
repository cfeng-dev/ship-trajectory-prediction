"""Local processed trajectories prepare inference windows without GPS metadata."""

import numpy as np
import pandas as pd
import pytest

import bayestraj.observations.window as observation_window


def test_processed_window_uses_selected_elapsed_positions_without_gps_origin():
    """A time/x/y window preserves local positions and restarts time at zero."""
    data = pd.DataFrame(
        {
            "time": [0.0, 10.0, 20.0, 30.0, 40.0],
            "x": [0.0, 2.0, 5.0, 9.0, 14.0],
            "y": [0.0, -1.0, -1.0, 0.0, 2.0],
        }
    )

    prepare = getattr(observation_window, "prepare_processed_trajectory_window", None)

    assert callable(prepare)
    window = prepare(
        data,
        observation_count=3,
        prediction_count=1,
        start_index=1,
    )
    np.testing.assert_allclose(window.time_seconds, [0.0, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(window.x_meters, [2.0, 5.0, 9.0, 14.0])
    np.testing.assert_allclose(window.y_meters, [-1.0, -1.0, 0.0, 2.0])
    assert window.reference_longitude is None
    assert window.reference_latitude is None
    assert np.isnan(window.gps_speed_mps).all()


def test_processed_window_accepts_a_real_gap_by_default():
    """A position-only trajectory retains its real irregular sample interval."""
    data = pd.DataFrame(
        {
            "time": [0.0, 10.0, 45.6, 55.6],
            "x": [0.0, 2.0, 9.0, 12.0],
            "y": [0.0, -1.0, 3.0, 5.0],
        }
    )

    window = observation_window.prepare_processed_trajectory_window(
        data,
        observation_count=3,
        prediction_count=1,
    )

    np.testing.assert_allclose(window.time_seconds, [0.0, 10.0, 45.6, 55.6])
    np.testing.assert_allclose(window.x_meters, [0.0, 2.0, 9.0, 12.0])
    np.testing.assert_allclose(window.y_meters, [0.0, -1.0, 3.0, 5.0])


def test_processed_window_honors_an_explicit_time_gap_limit():
    """An opt-in limit still protects callers that require continuity."""
    data = pd.DataFrame(
        {
            "time": [0.0, 10.0, 45.6, 55.6],
            "x": [0.0, 2.0, 9.0, 12.0],
            "y": [0.0, -1.0, 3.0, 5.0],
        }
    )

    with pytest.raises(ValueError, match="time gap of 35.6 seconds"):
        observation_window.prepare_processed_trajectory_window(
            data,
            observation_count=3,
            prediction_count=1,
            max_time_gap_seconds=15.0,
        )
