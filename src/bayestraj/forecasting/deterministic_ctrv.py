"""Reusable deterministic CTRV forecasting helpers."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import bayestraj.models.ctrv as deterministic_model
import bayestraj.observations.window as observation_window

MINIMUM_MOVEMENT_METERS = 1e-6


@dataclass(frozen=True, slots=True)
class DeterministicExperimentConfig:
    """Configuration of one deterministic recorded-trajectory experiment."""

    run_id: int
    start_index: int
    observation_count: int
    prediction_count: int
    position_noise_std_m: float
    position_noise_seed: int


@dataclass(frozen=True, slots=True)
class DeterministicRollingExperimentConfig:
    """Configuration of one deterministic rolling evaluation."""

    run_id: int
    window_mode: str
    observation_count: int
    prediction_count: int
    position_noise_std_m: float
    position_noise_seed: int
    stride: int | None


def estimate_ctrv_state(
    window: observation_window.TrajectoryWindowData,
) -> deterministic_model.CTRVState:
    """Fit one CTRV trajectory and return its state at the forecast origin."""
    observed = window.observed_slice
    time_observed = window.time_seconds[observed]
    x_observed = window.x_meters[observed]
    y_observed = window.y_meters[observed]
    fitted_initial_state = fit_ctrv_least_squares(
        time_observed,
        x_observed,
        y_observed,
    )
    elapsed_time = float(time_observed[-1] - time_observed[0])
    fitted_origin_state = deterministic_model.ctrv_step(
        fitted_initial_state,
        elapsed_time,
    )

    return deterministic_model.CTRVState(
        x=fitted_origin_state.x,
        y=fitted_origin_state.y,
        speed=fitted_origin_state.speed,
        heading=_wrap_angle(fitted_origin_state.heading),
        turn_rate=fitted_origin_state.turn_rate,
    )


def fit_ctrv_least_squares(
    time_seconds,
    x_meters,
    y_meters,
) -> deterministic_model.CTRVState:
    """Fit one deterministic CTRV trajectory to all supplied positions."""
    time_seconds, x_meters, y_meters = _validate_position_history(
        time_seconds,
        x_meters,
        y_meters,
        minimum_count=3,
    )
    elapsed_times = time_seconds - time_seconds[0]
    initial_guess = _initial_ctrv_guess(elapsed_times, x_meters, y_meters)
    lower_bounds = np.asarray([-np.inf, -np.inf, 0.0, -np.inf, -np.inf])
    upper_bounds = np.full(5, np.inf)
    result = least_squares(
        _ctrv_position_residuals,
        initial_guess,
        args=(elapsed_times, x_meters, y_meters),
        bounds=(lower_bounds, upper_bounds),
        loss="linear",
        x_scale="jac",
    )
    parameters = np.asarray(result.x, dtype=float)
    if not result.success:
        raise RuntimeError(
            "Deterministic CTRV least-squares fitting failed "
            f"(status={result.status}): {result.message}"
        )
    if parameters.shape != (5,) or not np.all(np.isfinite(parameters)):
        raise RuntimeError(
            "Deterministic CTRV least-squares fitting returned invalid parameters."
        )
    residuals = _ctrv_position_residuals(
        parameters,
        elapsed_times,
        x_meters,
        y_meters,
    )
    if not np.all(np.isfinite(residuals)):
        raise RuntimeError(
            "Deterministic CTRV least-squares fitting returned invalid residuals."
        )

    return deterministic_model.CTRVState(
        x=parameters[0],
        y=parameters[1],
        speed=parameters[2],
        heading=_wrap_angle(parameters[3]),
        turn_rate=parameters[4],
    )


def _initial_ctrv_guess(
    time_seconds: np.ndarray,
    x_meters: np.ndarray,
    y_meters: np.ndarray,
) -> np.ndarray:
    """Return the previous separate motion estimates as optimizer initials."""
    speed = estimate_speed_from_positions(time_seconds, x_meters, y_meters)

    delta_x = np.diff(x_meters)
    delta_y = np.diff(y_meters)
    moving = np.hypot(delta_x, delta_y) > MINIMUM_MOVEMENT_METERS
    segment_headings = np.unwrap(np.arctan2(delta_y[moving], delta_x[moving]))
    segment_midpoint_times = (0.5 * (time_seconds[:-1] + time_seconds[1:]))[moving]
    if len(segment_headings) >= 2:
        turn_rate, heading_initial = np.polyfit(
            segment_midpoint_times,
            segment_headings,
            deg=1,
        )
    elif len(segment_headings) == 1:
        heading_initial = float(segment_headings[0])
        turn_rate = 0.0
    else:
        heading_initial = 0.0
        turn_rate = 0.0

    return np.asarray(
        [x_meters[0], y_meters[0], speed, heading_initial, turn_rate],
        dtype=float,
    )


def _ctrv_position_residuals(
    parameters,
    elapsed_times,
    x_observed,
    y_observed,
) -> np.ndarray:
    """Return interleaved local x/y residuals for one CTRV trajectory."""
    predicted_positions = _ctrv_positions(parameters, elapsed_times)
    residuals = np.column_stack(
        (
            predicted_positions[:, 0] - x_observed,
            predicted_positions[:, 1] - y_observed,
        )
    )
    return residuals.ravel()


def _ctrv_positions(parameters, elapsed_times) -> np.ndarray:
    """Evaluate one fitted CTRV state at all non-negative elapsed times."""
    x_initial, y_initial, speed, heading_initial, turn_rate = parameters
    initial_state = deterministic_model.CTRVState(
        x=x_initial,
        y=y_initial,
        speed=speed,
        heading=heading_initial,
        turn_rate=turn_rate,
    )
    positions = np.empty((len(elapsed_times), 2), dtype=float)
    for index, elapsed_time in enumerate(elapsed_times):
        if elapsed_time == 0.0:
            state = initial_state
        else:
            state = deterministic_model.ctrv_step(initial_state, float(elapsed_time))
        positions[index] = state.x, state.y
    return positions


def estimate_speed_from_positions(
    time_seconds,
    x_meters,
    y_meters,
) -> float:
    """Estimate speed from a linear fit to all supplied position measurements."""
    time_seconds, x_meters, y_meters = _validate_position_history(
        time_seconds,
        x_meters,
        y_meters,
        minimum_count=2,
    )

    centered_time = time_seconds - np.mean(time_seconds)
    denominator = float(np.dot(centered_time, centered_time))
    if denominator <= 0:
        raise ValueError("Selected position timestamps must span positive time.")
    velocity_x = float(
        np.dot(centered_time, x_meters - np.mean(x_meters)) / denominator
    )
    velocity_y = float(
        np.dot(centered_time, y_meters - np.mean(y_meters)) / denominator
    )
    return float(np.hypot(velocity_x, velocity_y))


def _validate_position_history(
    time_seconds,
    x_meters,
    y_meters,
    *,
    minimum_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matching finite local position vectors with increasing times."""
    time_seconds = np.asarray(time_seconds, dtype=float)
    x_meters = np.asarray(x_meters, dtype=float)
    y_meters = np.asarray(y_meters, dtype=float)
    if (
        time_seconds.ndim != 1
        or x_meters.shape != time_seconds.shape
        or y_meters.shape != time_seconds.shape
        or time_seconds.size < minimum_count
        or not np.all(np.isfinite(time_seconds))
        or not np.all(np.isfinite(x_meters))
        or not np.all(np.isfinite(y_meters))
    ):
        raise ValueError(
            "time_seconds, x_meters, and y_meters must be matching finite "
            f"vectors with at least {minimum_count} values."
        )
    if np.any(np.diff(time_seconds) <= 0):
        raise ValueError("Position timestamps must be strictly increasing.")
    return time_seconds, x_meters, y_meters


def _wrap_angle(value: float) -> float:
    """Wrap one heading to the closed-open interval [-pi, pi)."""
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def build_prediction_table(
    window: observation_window.TrajectoryWindowData,
    initial_state: deterministic_model.CTRVState,
) -> pd.DataFrame:
    """Predict and compare deterministic positions at held-out timestamps."""
    observed_end_time = float(window.time_seconds[window.observation_count - 1])
    prediction = window.prediction_slice
    prediction_times = window.time_seconds[prediction]
    time_steps = np.diff(np.concatenate(([observed_end_time], prediction_times)))
    prediction_interval = float(time_steps[0])
    if (
        not np.all(np.isfinite(time_steps))
        or np.any(time_steps <= 0)
        or not np.allclose(time_steps, prediction_interval)
    ):
        raise ValueError(
            "Deterministic CTRV prediction requires positive, equally spaced "
            "future timestamps."
        )

    predicted_states = deterministic_model.predict_ctrv(
        initial_state,
        dt=prediction_interval,
        steps=window.prediction_count,
    )
    x_predicted = np.array([state.x for state in predicted_states])
    y_predicted = np.array([state.y for state in predicted_states])
    x_actual = window.x_meters[prediction]
    y_actual = window.y_meters[prediction]
    position_error = np.hypot(
        x_predicted - x_actual,
        y_predicted - y_actual,
    )

    return pd.DataFrame(
        {
            "time": window.timestamps[prediction],
            "horizon_seconds": prediction_times - observed_end_time,
            "x_actual": x_actual,
            "y_actual": y_actual,
            "x_predicted": x_predicted,
            "y_predicted": y_predicted,
            "position_error_m": position_error,
        }
    )
