"""Fit the Bayesian CTRA model to one recorded trajectory window."""

import numpy as np

from ship_trajectory_prediction.models.constant_turn_rate_acceleration import (
    fit_constant_turn_rate_acceleration_model,
    prepare_trajectory_window,
    summarize_predictions,
)
from ship_trajectory_prediction.models.plotting import (
    plot_constant_turn_rate_acceleration_prediction,
)
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory.io import read_ship_data

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

RUN_ID = 1
START_INDEX = 0
OBSERVATION_COUNT = 20
PREDICTION_COUNT = 5
SPEED_PRIOR_LOG_SD = 0.5
HEADING_PRIOR_SCALE = 0.5
TURN_RATE_PRIOR_SCALE = 0.01
ACCELERATION_PRIOR_SCALE = 0.05
SIGMA_PRIOR_SCALE = 20.0


def main():
    """Fit the model and plot posterior trajectories against held-out data."""
    trajectory_data = read_ship_data(DATA_FILE, run_id=RUN_ID)
    window = prepare_trajectory_window(
        trajectory_data,
        observation_count=OBSERVATION_COUNT,
        prediction_count=PREDICTION_COUNT,
        start_index=START_INDEX,
    )

    print("=" * 68)
    print("Bayesian Constant-Turn-Rate-and-Acceleration Prediction")
    print("=" * 68)
    print(f"Data file          : {DATA_FILE}")
    print(f"Run ID             : {RUN_ID}")
    print(f"Observed positions : {window.observation_count}")
    print(f"Predicted positions: {window.prediction_count}")
    print(f"Speed prior median : {window.speed_prior_median:.2f} m/s")

    fit = fit_constant_turn_rate_acceleration_model(
        window,
        speed_prior_log_sd=SPEED_PRIOR_LOG_SD,
        heading_prior_scale=HEADING_PRIOR_SCALE,
        turn_rate_prior_scale=TURN_RATE_PRIOR_SCALE,
        acceleration_prior_scale=ACCELERATION_PRIOR_SCALE,
        sigma_prior_scale=SIGMA_PRIOR_SCALE,
    )

    parameter_names = [
        "speed_initial",
        "acceleration",
        "speed_horizon",
        "heading_initial",
        "turn_rate",
        "radius_initial",
        "radius_horizon",
        "sigma",
    ]
    print("\nPosterior parameter summary:")
    print(fit.summary().loc[parameter_names])

    acceleration_samples = fit.stan_variable("acceleration")
    turn_rate_samples = fit.stan_variable("turn_rate")
    print(f"\nAcceleration probability: {np.mean(acceleration_samples > 0):.1%}")
    print(f"Deceleration probability: {np.mean(acceleration_samples < 0):.1%}")
    print(f"Left-turn probability   : {np.mean(turn_rate_samples > 0):.1%}")
    print(f"Right-turn probability  : {np.mean(turn_rate_samples < 0):.1%}")

    prediction_summary = summarize_predictions(fit, window)
    print("\nPosterior prediction summary:")
    print(prediction_summary.to_string(index=False))

    plot_constant_turn_rate_acceleration_prediction(window, fit)


if __name__ == "__main__":
    main()
