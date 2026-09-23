"""Tests for reproducible position-observation simulation."""

import numpy as np
import pandas as pd
import pytest

import bayestraj.observations.position as position_observations
import bayestraj.observations.window as observation_window


def _trajectory_window(*, observation_count=4, position_count=6):
    time_seconds = np.arange(position_count, dtype=float)
    return observation_window.TrajectoryWindowData(
        timestamps=pd.date_range(
            "2026-01-01",
            periods=position_count,
            freq="s",
            tz="UTC",
        ),
        time_seconds=time_seconds,
        x_meters=10.0 + 2.0 * time_seconds,
        y_meters=-5.0 + 3.0 * time_seconds,
        reference_longitude=8.0,
        reference_latitude=47.0,
        gps_speed_mps=np.full(position_count, np.nan),
        observation_count=observation_count,
    )


def test_zero_noise_returns_reference_positions_exactly():
    window = _trajectory_window()

    observations = position_observations.simulate_position_observations(
        window,
        position_noise_std_m=0.0,
        seed=17,
    )

    observed = window.observed_slice
    assert np.array_equal(observations.x_meters, window.x_meters[observed])
    assert np.array_equal(observations.y_meters, window.y_meters[observed])


def test_nonzero_noise_samples_each_coordinate_around_its_reference(monkeypatch):
    window = _trajectory_window()

    class CenteredNormalGenerator:
        def __init__(self):
            self.coordinate_index = 0

        def normal(self, loc, scale, size=None):
            if size is not None:
                raise AssertionError(
                    "Reference positions must be supplied through normal(loc=...)."
                )
            self.coordinate_index += 1
            return np.asarray(loc, dtype=float) + self.coordinate_index * scale

    generator = CenteredNormalGenerator()
    monkeypatch.setattr(
        position_observations.np.random,
        "default_rng",
        lambda seed: generator,
    )

    observations = position_observations.simulate_position_observations(
        window,
        position_noise_std_m=2.0,
        seed=17,
    )

    observed = window.observed_slice
    assert np.array_equal(observations.x_meters, window.x_meters[observed] + 2.0)
    assert np.array_equal(observations.y_meters, window.y_meters[observed] + 4.0)


def test_nonzero_noise_is_finite_shaped_and_reproducible():
    window = _trajectory_window()

    first = position_observations.simulate_position_observations(
        window,
        position_noise_std_m=3.0,
        seed=2026,
    )
    second = position_observations.simulate_position_observations(
        window,
        position_noise_std_m=3.0,
        seed=2026,
    )

    expected_shape = (window.observation_count,)
    assert first.x_meters.shape == expected_shape
    assert first.y_meters.shape == expected_shape
    assert np.all(np.isfinite(first.x_meters))
    assert np.all(np.isfinite(first.y_meters))
    assert np.array_equal(first.x_meters, second.x_meters)
    assert np.array_equal(first.y_meters, second.y_meters)


def test_configured_noise_is_per_coordinate_standard_deviation():
    window = _trajectory_window(observation_count=20_000, position_count=20_001)
    standard_deviation_m = 3.0

    observations = position_observations.simulate_position_observations(
        window,
        position_noise_std_m=standard_deviation_m,
        seed=2026,
    )

    observed = window.observed_slice
    x_noise = observations.x_meters - window.x_meters[observed]
    y_noise = observations.y_meters - window.y_meters[observed]
    assert np.std(x_noise) == pytest.approx(standard_deviation_m, rel=0.02)
    assert np.std(y_noise) == pytest.approx(standard_deviation_m, rel=0.02)
    assert abs(np.corrcoef(x_noise, y_noise)[0, 1]) < 0.03
