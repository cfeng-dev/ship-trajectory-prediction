"""Tests for the deterministic CTRV single-window prediction workflow."""

import numpy as np
import pandas as pd

from ship_trajectory_prediction.prediction.deterministic_ctrv import (
    _add_position_observation_noise,
)
from ship_trajectory_prediction.trajectory import TrajectoryWindowData


def _window():
    return TrajectoryWindowData(
        timestamps=pd.date_range("2026-01-01", periods=5, freq="s", tz="UTC"),
        time_seconds=np.arange(5, dtype=float),
        x_meters=np.arange(5, dtype=float),
        y_meters=np.arange(5, dtype=float) * 2,
        reference_longitude=10.0,
        reference_latitude=54.0,
        gps_speed_mps=np.ones(5),
        observation_count=3,
    )


def test_position_observation_noise_is_reproducible_and_keeps_targets():
    """Noise should affect observations without changing held-out truth."""
    window = _window()

    first = _add_position_observation_noise(
        window,
        additional_noise_std_m=2.0,
        seed=2026,
    )
    second = _add_position_observation_noise(
        window,
        additional_noise_std_m=2.0,
        seed=2026,
    )

    np.testing.assert_array_equal(first.x_meters, second.x_meters)
    np.testing.assert_array_equal(first.y_meters, second.y_meters)
    assert not np.array_equal(
        first.x_meters[first.observed_slice],
        window.x_meters[window.observed_slice],
    )
    np.testing.assert_array_equal(
        first.x_meters[first.prediction_slice],
        window.x_meters[window.prediction_slice],
    )
    np.testing.assert_array_equal(
        first.y_meters[first.prediction_slice],
        window.y_meters[window.prediction_slice],
    )
