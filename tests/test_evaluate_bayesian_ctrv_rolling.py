"""Tests for the Bayesian CTRV rolling-evaluation experiment."""

import numpy as np

from experiments.trajectory_prediction.evaluate_bayesian_ctrv_rolling import (
    EXPERIMENT,
    _build_window_position_observations,
    _simulate_route_position_noise,
)
from ship_trajectory_prediction.models.ctrv import CTRVState
from ship_trajectory_prediction.simulation.synthetic_ctrv import (
    SyntheticCTRVNoise,
    simulate_synthetic_ctrv_data,
)
from ship_trajectory_prediction.trajectory import prepare_trajectory_window


def _create_route():
    """Return one deterministic route for overlapping rolling windows."""
    return simulate_synthetic_ctrv_data(
        count=10,
        dt_seconds=5.0,
        initial_state=CTRVState(
            x=0.0,
            y=0.0,
            speed=3.0,
            heading=0.2,
            turn_rate=0.01,
        ),
        noise=SyntheticCTRVNoise(
            sigma_position_gps=0.0,
            sigma_speed_gps=0.0,
            sigma_position_process=0.0,
            sigma_speed_process=0.0,
            sigma_turn_rate_process=0.0,
        ),
        seed=7,
    )


def test_rolling_experiment_adds_reproducible_two_meter_position_noise():
    """The rolling default should match the single-fit 2 m noise scenario."""
    assert EXPERIMENT.additional_position_noise_std_m == 2.0
    assert EXPERIMENT.position_noise_seed == 2026

    first_x, first_y = _simulate_route_position_noise(
        10,
        additional_noise_std_m=EXPERIMENT.additional_position_noise_std_m,
        seed=EXPERIMENT.position_noise_seed,
    )
    second_x, second_y = _simulate_route_position_noise(
        10,
        additional_noise_std_m=EXPERIMENT.additional_position_noise_std_m,
        seed=EXPERIMENT.position_noise_seed,
    )

    np.testing.assert_array_equal(first_x, second_x)
    np.testing.assert_array_equal(first_y, second_y)
    assert np.any(first_x != 0.0)
    assert np.any(first_y != 0.0)


def test_overlapping_windows_reuse_the_same_route_position_noise():
    """One physical observation must retain its perturbation across windows."""
    route = _create_route()
    route_noise_x, route_noise_y = _simulate_route_position_noise(
        len(route),
        additional_noise_std_m=2.0,
        seed=2026,
    )
    first_window = prepare_trajectory_window(
        route,
        observation_count=5,
        prediction_count=2,
        start_index=0,
    )
    second_window = prepare_trajectory_window(
        route,
        observation_count=5,
        prediction_count=2,
        start_index=2,
    )

    first_observations = _build_window_position_observations(
        first_window,
        route_start_index=0,
        route_noise_x=route_noise_x,
        route_noise_y=route_noise_y,
        additional_noise_std_m=2.0,
        noise_seed=2026,
    )
    second_observations = _build_window_position_observations(
        second_window,
        route_start_index=2,
        route_noise_x=route_noise_x,
        route_noise_y=route_noise_y,
        additional_noise_std_m=2.0,
        noise_seed=2026,
    )

    first_x_noise = (
        first_observations.x_meters - first_window.x_meters[first_window.observed_slice]
    )
    first_y_noise = (
        first_observations.y_meters - first_window.y_meters[first_window.observed_slice]
    )
    second_x_noise = (
        second_observations.x_meters
        - second_window.x_meters[second_window.observed_slice]
    )
    second_y_noise = (
        second_observations.y_meters
        - second_window.y_meters[second_window.observed_slice]
    )

    np.testing.assert_allclose(first_x_noise[2:], second_x_noise[:3])
    np.testing.assert_allclose(first_y_noise[2:], second_y_noise[:3])
    np.testing.assert_allclose(first_x_noise, route_noise_x[:5])
    np.testing.assert_allclose(second_y_noise, route_noise_y[2:7])
    assert first_observations.additional_noise_std_m == 2.0
    assert first_observations.noise_seed == 2026


def test_zero_position_noise_keeps_the_recorded_route_unchanged():
    """Disabling the option should preserve the original observations exactly."""
    route = _create_route()
    window = prepare_trajectory_window(
        route,
        observation_count=5,
        prediction_count=2,
    )
    route_noise_x, route_noise_y = _simulate_route_position_noise(
        len(route),
        additional_noise_std_m=0.0,
        seed=2026,
    )
    observations = _build_window_position_observations(
        window,
        route_start_index=0,
        route_noise_x=route_noise_x,
        route_noise_y=route_noise_y,
        additional_noise_std_m=0.0,
        noise_seed=2026,
    )

    np.testing.assert_array_equal(
        observations.x_meters,
        window.x_meters[window.observed_slice],
    )
    np.testing.assert_array_equal(
        observations.y_meters,
        window.y_meters[window.observed_slice],
    )
