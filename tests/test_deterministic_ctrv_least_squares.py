"""Focused tests for deterministic nonlinear CTRV trajectory fitting."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import bayestraj.forecasting.deterministic_ctrv as forecasting
import bayestraj.models.deterministic_ctrv as deterministic_model
import bayestraj.observations.window as observation_window


def _positions(initial_state, time_seconds):
    positions = []
    for elapsed_time in time_seconds:
        state = (
            initial_state
            if elapsed_time == 0.0
            else deterministic_model.ctrv_step(initial_state, float(elapsed_time))
        )
        positions.append((state.x, state.y))
    return np.asarray(positions)


def _window(initial_state, observed_times, future_times=(30.0, 35.0)):
    time_seconds = np.concatenate(
        (np.asarray(observed_times, dtype=float), np.asarray(future_times, dtype=float))
    )
    positions = _positions(initial_state, time_seconds)
    return observation_window.TrajectoryWindowData(
        timestamps=pd.DatetimeIndex(
            pd.Timestamp("2026-01-01", tz="UTC")
            + pd.to_timedelta(time_seconds, unit="s")
        ),
        time_seconds=time_seconds,
        x_meters=positions[:, 0],
        y_meters=positions[:, 1],
        reference_longitude=8.0,
        reference_latitude=47.0,
        gps_speed_mps=np.full(time_seconds.size, np.nan),
        observation_count=len(observed_times),
    )


def _angle_error(actual, expected):
    return np.arctan2(np.sin(actual - expected), np.cos(actual - expected))


def _state_parameters(state):
    return np.asarray(
        [state.x, state.y, state.speed, state.heading, state.turn_rate],
        dtype=float,
    )


def _position_rmse(parameters, elapsed_times, x_observed, y_observed):
    residuals = forecasting._ctrv_position_residuals(
        parameters,
        elapsed_times,
        x_observed,
        y_observed,
    ).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))


def test_exact_straight_fit_recovers_state_and_future_position():
    true_state = deterministic_model.CTRVState(
        x=12.0,
        y=-7.0,
        speed=4.2,
        heading=0.6,
        turn_rate=0.0,
    )
    observed_times = np.asarray([0.0, 2.0, 5.0, 9.0, 14.0])
    window = _window(true_state, observed_times, future_times=(19.0, 24.0))

    fitted_initial = forecasting.fit_ctrv_least_squares(
        observed_times,
        window.x_meters[: len(observed_times)],
        window.y_meters[: len(observed_times)],
    )
    forecast_origin = forecasting.estimate_ctrv_state(window)
    predicted_future = deterministic_model.ctrv_step(forecast_origin, 5.0)
    expected_future = deterministic_model.ctrv_step(true_state, 19.0)

    assert fitted_initial.speed == pytest.approx(true_state.speed, abs=1e-7)
    assert fitted_initial.turn_rate == pytest.approx(0.0, abs=1e-8)
    assert _angle_error(fitted_initial.heading, true_state.heading) == pytest.approx(
        0.0, abs=1e-8
    )
    assert predicted_future.x == pytest.approx(expected_future.x, abs=1e-6)
    assert predicted_future.y == pytest.approx(expected_future.y, abs=1e-6)


def test_exact_curved_fit_handles_irregular_times_and_heading_wrap():
    true_state = deterministic_model.CTRVState(
        x=100.0,
        y=-50.0,
        speed=5.0,
        heading=3.0,
        turn_rate=0.035,
    )
    observed_times = np.asarray([0.0, 1.5, 4.0, 8.5, 14.0, 21.0])
    positions = _positions(true_state, observed_times)

    fitted = forecasting.fit_ctrv_least_squares(
        observed_times,
        positions[:, 0],
        positions[:, 1],
    )
    future_elapsed = 29.0
    fitted_future = deterministic_model.ctrv_step(fitted, future_elapsed)
    expected_future = deterministic_model.ctrv_step(true_state, future_elapsed)

    assert fitted.speed == pytest.approx(true_state.speed, abs=1e-6)
    assert fitted.turn_rate == pytest.approx(true_state.turn_rate, abs=1e-8)
    assert _angle_error(fitted.heading, true_state.heading) == pytest.approx(
        0.0, abs=1e-8
    )
    assert fitted_future.x == pytest.approx(expected_future.x, abs=1e-5)
    assert fitted_future.y == pytest.approx(expected_future.y, abs=1e-5)


def test_noisy_curved_fit_returns_finite_physical_parameters():
    true_state = deterministic_model.CTRVState(10.0, 20.0, 5.5, -2.8, -0.025)
    times = np.asarray([0.0, 3.0, 7.0, 12.0, 18.0, 25.0, 33.0])
    positions = _positions(true_state, times)
    positions += np.random.default_rng(2026).normal(0.0, 1.5, positions.shape)

    fitted = forecasting.fit_ctrv_least_squares(
        times,
        positions[:, 0],
        positions[:, 1],
    )

    assert np.all(np.isfinite(_state_parameters(fitted)))
    assert fitted.speed >= 0.0


def test_near_zero_turn_rate_fit_is_numerically_stable():
    true_state = deterministic_model.CTRVState(2.0, -3.0, 3.0, -0.7, 1e-8)
    times = np.asarray([0.0, 2.0, 6.0, 11.0, 17.0, 24.0])
    positions = _positions(true_state, times)

    fitted = forecasting.fit_ctrv_least_squares(
        times,
        positions[:, 0],
        positions[:, 1],
    )

    assert np.all(np.isfinite(_state_parameters(fitted)))
    assert fitted.speed == pytest.approx(true_state.speed, abs=1e-6)
    assert fitted.turn_rate == pytest.approx(true_state.turn_rate, abs=1e-7)


def test_stationary_fit_enforces_non_negative_speed():
    times = np.asarray([0.0, 2.0, 5.0, 9.0])
    x_observed = np.full(times.size, 7.0)
    y_observed = np.full(times.size, -4.0)

    fitted = forecasting.fit_ctrv_least_squares(
        times,
        x_observed,
        y_observed,
    )

    assert fitted.speed >= 0.0
    assert fitted.speed == pytest.approx(0.0, abs=1e-6)


def test_gps_speed_does_not_influence_position_only_fit():
    true_state = deterministic_model.CTRVState(0.0, 0.0, 4.0, 0.4, 0.02)
    window = _window(true_state, [0.0, 5.0, 10.0, 15.0, 20.0])
    alternate_window = dataclasses.replace(
        window,
        gps_speed_mps=np.linspace(100.0, 500.0, len(window.time_seconds)),
    )

    fitted = forecasting.estimate_ctrv_state(window)
    alternate_fitted = forecasting.estimate_ctrv_state(alternate_window)

    np.testing.assert_allclose(
        _state_parameters(fitted),
        _state_parameters(alternate_fitted),
    )


def test_forecast_origin_is_propagated_fitted_position_not_last_observation():
    true_state = deterministic_model.CTRVState(5.0, -2.0, 4.5, 0.8, 0.018)
    observed_times = np.asarray([0.0, 4.0, 9.0, 15.0, 22.0])
    window = _window(true_state, observed_times, future_times=(27.0, 32.0))
    noisy_x = window.x_meters.copy()
    noisy_y = window.y_meters.copy()
    noisy_x[: len(observed_times)] += np.asarray([0.5, -1.0, 0.8, -0.3, 2.5])
    noisy_y[: len(observed_times)] += np.asarray([-0.4, 0.7, -0.9, 0.5, -2.0])
    noisy_window = dataclasses.replace(window, x_meters=noisy_x, y_meters=noisy_y)
    fitted_initial = forecasting.fit_ctrv_least_squares(
        observed_times,
        noisy_x[: len(observed_times)],
        noisy_y[: len(observed_times)],
    )
    expected_origin = deterministic_model.ctrv_step(
        fitted_initial,
        float(observed_times[-1] - observed_times[0]),
    )

    fitted_origin = forecasting.estimate_ctrv_state(noisy_window)

    assert fitted_origin.x == pytest.approx(expected_origin.x)
    assert fitted_origin.y == pytest.approx(expected_origin.y)
    assert (
        np.hypot(
            fitted_origin.x - noisy_x[len(observed_times) - 1],
            fitted_origin.y - noisy_y[len(observed_times) - 1],
        )
        > 1e-3
    )


def test_direct_fit_improves_representative_noisy_curve_observation_rmse():
    true_state = deterministic_model.CTRVState(20.0, -10.0, 6.0, 0.5, 0.04)
    times = np.arange(9, dtype=float) * 5.0
    positions = _positions(true_state, times)
    positions += np.random.default_rng(42).normal(0.0, 2.0, positions.shape)
    heuristic_parameters = forecasting._initial_ctrv_guess(
        times,
        positions[:, 0],
        positions[:, 1],
    )
    fitted = forecasting.fit_ctrv_least_squares(
        times,
        positions[:, 0],
        positions[:, 1],
    )

    heuristic_rmse = _position_rmse(
        heuristic_parameters,
        times,
        positions[:, 0],
        positions[:, 1],
    )
    fitted_rmse = _position_rmse(
        _state_parameters(fitted),
        times,
        positions[:, 0],
        positions[:, 1],
    )

    assert fitted_rmse <= heuristic_rmse


def test_optimizer_failure_raises_clear_runtime_error(monkeypatch):
    monkeypatch.setattr(
        forecasting,
        "least_squares",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            status=0,
            message="maximum evaluations reached",
            x=np.asarray([0.0, 0.0, 1.0, 0.0, 0.0]),
        ),
    )

    with pytest.raises(RuntimeError, match="least-squares fitting failed"):
        forecasting.fit_ctrv_least_squares(
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
        )
