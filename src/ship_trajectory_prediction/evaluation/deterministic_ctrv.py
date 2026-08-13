"""Reusable deterministic CTRV evaluation helpers."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ship_trajectory_prediction.models.deterministic_ctrv import CTRVState, predict_ctrv
from ship_trajectory_prediction.trajectory import TrajectoryWindowData

DEFAULT_SPEED_ESTIMATION_POINTS = 5
DEFAULT_HEADING_ESTIMATION_SEGMENTS = 5
MINIMUM_MOVEMENT_METERS = 1e-6


@dataclass(frozen=True, slots=True)
class DeterministicExperimentConfig:
    """Configuration of one deterministic recorded-trajectory experiment."""

    run_id: int
    start_index: int
    observation_count: int
    prediction_count: int
    additional_position_noise_std_m: float
    position_noise_seed: int


def estimate_ctrv_state(
    window: TrajectoryWindowData,
    *,
    speed_estimation_points=DEFAULT_SPEED_ESTIMATION_POINTS,
    heading_estimation_segments=DEFAULT_HEADING_ESTIMATION_SEGMENTS,
) -> CTRVState:
    """Estimate the final CTRV state from observed trajectory values only."""
    if (
        isinstance(speed_estimation_points, bool)
        or not isinstance(speed_estimation_points, int)
        or speed_estimation_points < 1
    ):
        raise ValueError("speed_estimation_points must be a positive integer.")
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
    speed_observed = window.gps_speed_mps[observed]

    valid_speeds = speed_observed[np.isfinite(speed_observed) & (speed_observed >= 0)]
    if len(valid_speeds) == 0:
        raise ValueError("Observed gps_speed must contain a finite non-negative value.")
    speed = float(np.median(valid_speeds[-speed_estimation_points:]))

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

    return CTRVState(
        x=float(x_observed[-1]),
        y=float(y_observed[-1]),
        speed=speed,
        heading=float(heading),
        turn_rate=float(turn_rate),
    )


def build_prediction_table(
    window: TrajectoryWindowData,
    initial_state: CTRVState,
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

    predicted_states = predict_ctrv(
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
