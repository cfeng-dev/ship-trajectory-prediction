"""Run one deterministic CTRV trajectory prediction."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ship_trajectory_prediction.evaluation.reporting import print_prediction_setup
from ship_trajectory_prediction.models.deterministic_ctrv import CTRVState, predict_ctrv
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import (
    TrajectoryWindowData,
    prepare_trajectory_window,
    read_ship_data,
)

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

RUN_ID = 1
START_INDEX = 0
OBSERVATION_COUNT = 20
PREDICTION_COUNT = 5
SPEED_ESTIMATION_POINTS = 5
HEADING_ESTIMATION_SEGMENTS = 5
MINIMUM_MOVEMENT_METERS = 1e-6


def estimate_ctrv_state(
    window: TrajectoryWindowData,
    *,
    speed_estimation_points=SPEED_ESTIMATION_POINTS,
    heading_estimation_segments=HEADING_ESTIMATION_SEGMENTS,
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


def plot_prediction(
    window: TrajectoryWindowData,
    prediction_table: pd.DataFrame,
):
    """Plot observed, held-out, and deterministic CTRV trajectories."""
    observed = window.observed_slice
    prediction_start_index = window.observation_count - 1
    start_x = window.x_meters[prediction_start_index]
    start_y = window.y_meters[prediction_start_index]

    held_out_x = np.concatenate(([start_x], prediction_table["x_actual"]))
    held_out_y = np.concatenate(([start_y], prediction_table["y_actual"]))
    predicted_x = np.concatenate(([start_x], prediction_table["x_predicted"]))
    predicted_y = np.concatenate(([start_y], prediction_table["y_predicted"]))

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.plot(
        window.x_meters[observed],
        window.y_meters[observed],
        color="tab:blue",
        linewidth=2,
        label="Observed trajectory",
    )
    axis.plot(
        held_out_x,
        held_out_y,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Held-out trajectory",
    )
    axis.plot(
        predicted_x,
        predicted_y,
        color="tab:red",
        marker="o",
        linewidth=2,
        label="Deterministic CTRV prediction",
    )
    axis.scatter(
        start_x,
        start_y,
        color="tab:blue",
        zorder=3,
        label="Prediction start",
    )
    axis.set_title("Deterministic CTRV Trajectory Prediction")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plt.show()
    return figure, axis


def main(*, show_plot=True):
    """Estimate one CTRV state and predict a real held-out trajectory."""
    trajectory_data = read_ship_data(DATA_FILE, run_id=RUN_ID)
    window = prepare_trajectory_window(
        trajectory_data,
        observation_count=OBSERVATION_COUNT,
        prediction_count=PREDICTION_COUNT,
        start_index=START_INDEX,
    )
    initial_state = estimate_ctrv_state(window)
    prediction_table = build_prediction_table(window, initial_state)

    print_prediction_setup(
        "Deterministic CTRV Trajectory Prediction",
        data_file=DATA_FILE,
        run_id=RUN_ID,
        window=window,
        extra_rows=[
            ("Estimated speed", f"{initial_state.speed:.3f} m/s"),
            ("Estimated heading", f"{initial_state.heading:.5f} rad"),
            ("Estimated turn rate", f"{initial_state.turn_rate:.6f} rad/s"),
        ],
    )
    print("\nHeld-out prediction:")
    print(prediction_table.round(3).to_string(index=False))
    print(f"\nADE: {prediction_table['position_error_m'].mean():.2f} m")
    print(f"FDE: {prediction_table['position_error_m'].iloc[-1]:.2f} m")

    if show_plot:
        plot_prediction(window, prediction_table)
    return prediction_table


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Print prediction metrics without opening a plot window.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(show_plot=not arguments.no_plot)
