"""Reusable deterministic CTRV forecasting helpers."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import bayestraj.models.deterministic_ctrv as deterministic_model
import bayestraj.observations.window as observation_window

DEFAULT_SPEED_ESTIMATION_POINTS = 5
DEFAULT_HEADING_ESTIMATION_SEGMENTS = 5
MINIMUM_MOVEMENT_METERS = 1e-6


@dataclass(frozen=True, slots=True)
class DeterministicExperimentConfig:
    """Configuration of one deterministic recorded-trajectory experiment."""

    run_id: int
    start_index: int
    observation_count: int
    additional_position_noise_std_m: float
    position_noise_seed: int


@dataclass(frozen=True, slots=True)
class DeterministicRollingExperimentConfig:
    """Configuration of one deterministic rolling evaluation."""

    run_id: int
    window_mode: str
    observation_count: int
    additional_position_noise_std_m: float
    position_noise_seed: int
    stride: int | None


def estimate_ctrv_state(
    window: observation_window.TrajectoryWindowData,
    *,
    speed_estimation_points=DEFAULT_SPEED_ESTIMATION_POINTS,
    heading_estimation_segments=DEFAULT_HEADING_ESTIMATION_SEGMENTS,
) -> deterministic_model.CTRVState:
    """Estimate the final CTRV state from observed positions and times only."""
    if (
        isinstance(speed_estimation_points, bool)
        or not isinstance(speed_estimation_points, int)
        or speed_estimation_points < 2
    ):
        raise ValueError("speed_estimation_points must be an integer of at least 2.")
    if (
        isinstance(heading_estimation_segments, bool)
        or not isinstance(heading_estimation_segments, int)
        or heading_estimation_segments < 2
    ):
        raise ValueError(
            "heading_estimation_segments must be an integer greater than or equal to 2."
        )

    observed = window.observed_slice
    time_observed = window.time_seconds[observed]
    x_observed = window.x_meters[observed]
    y_observed = window.y_meters[observed]
    speed = estimate_speed_from_positions(
        time_observed,
        x_observed,
        y_observed,
        point_count=speed_estimation_points,
    )

    delta_x = np.diff(x_observed)
    delta_y = np.diff(y_observed)
    moving = np.hypot(delta_x, delta_y) > MINIMUM_MOVEMENT_METERS
    segment_headings = np.unwrap(np.arctan2(delta_y[moving], delta_x[moving]))
    segment_midpoint_times = (0.5 * (time_observed[:-1] + time_observed[1:]))[moving]
    if len(segment_headings) < 2:
        raise ValueError(
            "Observed positions must contain at least two moving segments."
        )

    fit_count = min(heading_estimation_segments, len(segment_headings))
    turn_rate, heading_intercept = np.polyfit(
        segment_midpoint_times[-fit_count:],
        segment_headings[-fit_count:],
        deg=1,
    )
    heading = heading_intercept + turn_rate * time_observed[-1]

    return deterministic_model.CTRVState(
        x=float(x_observed[-1]),
        y=float(y_observed[-1]),
        speed=speed,
        heading=float(heading),
        turn_rate=float(turn_rate),
    )


def estimate_speed_from_positions(
    time_seconds,
    x_meters,
    y_meters,
    *,
    point_count=DEFAULT_SPEED_ESTIMATION_POINTS,
) -> float:
    """Estimate speed from a local linear fit to recent position measurements."""
    time_seconds = np.asarray(time_seconds, dtype=float)
    x_meters = np.asarray(x_meters, dtype=float)
    y_meters = np.asarray(y_meters, dtype=float)
    if (
        time_seconds.ndim != 1
        or x_meters.shape != time_seconds.shape
        or y_meters.shape != time_seconds.shape
        or time_seconds.size < 2
        or not np.all(np.isfinite(time_seconds))
        or not np.all(np.isfinite(x_meters))
        or not np.all(np.isfinite(y_meters))
    ):
        raise ValueError(
            "time_seconds, x_meters, and y_meters must be matching finite "
            "vectors with at least two values."
        )
    if isinstance(point_count, bool) or not isinstance(point_count, int):
        raise ValueError("point_count must be an integer of at least 2.")
    if point_count < 2:
        raise ValueError("point_count must be an integer of at least 2.")

    fit_count = min(point_count, time_seconds.size)
    selected_time = time_seconds[-fit_count:]
    selected_x = x_meters[-fit_count:]
    selected_y = y_meters[-fit_count:]
    if np.any(np.diff(selected_time) <= 0):
        raise ValueError("Selected position timestamps must be strictly increasing.")

    centered_time = selected_time - np.mean(selected_time)
    denominator = float(np.dot(centered_time, centered_time))
    if denominator <= 0:
        raise ValueError("Selected position timestamps must span positive time.")
    velocity_x = float(
        np.dot(centered_time, selected_x - np.mean(selected_x)) / denominator
    )
    velocity_y = float(
        np.dot(centered_time, selected_y - np.mean(selected_y)) / denominator
    )
    return float(np.hypot(velocity_x, velocity_y))


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
