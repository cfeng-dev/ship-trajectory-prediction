"""Fit the Bayesian time-varying motion model to one trajectory window."""

import numpy as np

from ship_trajectory_prediction.models.plotting import (
    plot_time_varying_motion_prediction,
)
from ship_trajectory_prediction.models.time_varying_motion import (
    fit_time_varying_motion_model,
    prepare_trajectory_window,
    summarize_predictions,
)
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory.io import read_ship_data

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

RUN_ID = 1
START_INDEX = 0
OBSERVATION_COUNT = 20
PREDICTION_COUNT = 10

SPEED_PRIOR_LOG_SD = 0.5
HEADING_PRIOR_SCALE = 0.5
ACCELERATION_INITIAL_SCALE = 0.1
ACCELERATION_STATE_SCALE = 0.05
ACCELERATION_DECAY_TIME = 60.0
TURN_RATE_INITIAL_SCALE = 0.01
TURN_RATE_STATE_SCALE = 0.01
TURN_RATE_DECAY_TIME = 120.0
SIGMA_POSITION = 5.0
SIGMA_SPEED = 0.2


def main():
    """Fit the model and evaluate posterior predictions on held-out data."""
    trajectory_data = read_ship_data(DATA_FILE, run_id=RUN_ID)
    window = prepare_trajectory_window(
        trajectory_data,
        observation_count=OBSERVATION_COUNT,
        prediction_count=PREDICTION_COUNT,
        start_index=START_INDEX,
    )

    print("=" * 62)
    print("Bayesian Time-Varying Motion Prediction")
    print("=" * 62)
    print(f"Data file          : {DATA_FILE}")
    print(f"Run ID             : {RUN_ID}")
    print(f"Observed positions : {window.observation_count}")
    print(f"Predicted positions: {window.prediction_count}")
    print(f"Position noise     : {SIGMA_POSITION:.2f} m")
    print(f"Speed noise        : {SIGMA_SPEED:.2f} m/s")

    fit = fit_time_varying_motion_model(
        window,
        speed_prior_log_sd=SPEED_PRIOR_LOG_SD,
        heading_prior_scale=HEADING_PRIOR_SCALE,
        acceleration_initial_scale=ACCELERATION_INITIAL_SCALE,
        acceleration_state_scale=ACCELERATION_STATE_SCALE,
        acceleration_decay_time=ACCELERATION_DECAY_TIME,
        turn_rate_initial_scale=TURN_RATE_INITIAL_SCALE,
        turn_rate_state_scale=TURN_RATE_STATE_SCALE,
        turn_rate_decay_time=TURN_RATE_DECAY_TIME,
        sigma_position=SIGMA_POSITION,
        sigma_speed=SIGMA_SPEED,
    )

    print("\nPosterior initial-state summary:")
    print(
        fit.summary().loc[
            [
                "speed_initial",
                "heading_initial",
                "acceleration_initial",
                "turn_rate_initial",
            ]
        ]
    )

    acceleration_state = fit.stan_variable("acceleration_state")
    turn_rate_state = fit.stan_variable("turn_rate_state")
    speed_state = fit.stan_variable("speed_state")
    print("\nPosterior state medians:")
    print(
        "Acceleration [m/s^2]: "
        f"{np.median(acceleration_state[:, 0]):.4f} -> "
        f"{np.median(acceleration_state[:, -1]):.4f}"
    )
    print(
        "Turn rate [rad/s]   : "
        f"{np.median(turn_rate_state[:, 0]):.5f} -> "
        f"{np.median(turn_rate_state[:, -1]):.5f}"
    )
    print(
        "Speed [m/s]         : "
        f"{np.median(speed_state[:, 0]):.3f} -> "
        f"{np.median(speed_state[:, -1]):.3f}"
    )

    prediction_summary = summarize_predictions(fit, window)
    position_error = np.hypot(
        prediction_summary["x_median"] - prediction_summary["x_actual"],
        prediction_summary["y_median"] - prediction_summary["y_actual"],
    )
    print("\nHeld-out position errors:")
    print(f"Mean error : {position_error.mean():.2f} m")
    print(f"Final error: {position_error.iloc[-1]:.2f} m")

    print("\nPosterior prediction summary:")
    print(prediction_summary.to_string(index=False))

    plot_time_varying_motion_prediction(window, fit)


if __name__ == "__main__":
    main()
