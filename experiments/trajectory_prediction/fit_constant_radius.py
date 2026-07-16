"""Fit the Bayesian constant-turn-rate model to one trajectory window."""

import numpy as np

from ship_trajectory_prediction.models.constant_radius import (
    fit_constant_radius_model,
    prepare_trajectory_window,
    summarize_predictions,
)
from ship_trajectory_prediction.models.plotting import (
    plot_constant_radius_prediction,
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
TURN_RATE_PRIOR_SCALE = 0.01
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

    print("=" * 60)
    print("Bayesian Constant-Turn-Rate Trajectory Prediction")
    print("=" * 60)
    print(f"Data file          : {DATA_FILE}")
    print(f"Run ID             : {RUN_ID}")
    print(f"Observed positions : {window.observation_count}")
    print(f"Predicted positions: {window.prediction_count}")
    print(f"Speed prior median : {window.speed_prior_median:.2f} m/s")

    fit = fit_constant_radius_model(
        window,
        speed_prior_log_sd=SPEED_PRIOR_LOG_SD,
        heading_prior_scale=HEADING_PRIOR_SCALE,
        turn_rate_prior_scale=TURN_RATE_PRIOR_SCALE,
        sigma_prior_scale=SIGMA_PRIOR_SCALE,
    )

    print("\nPosterior parameter summary:")
    print(
        fit.summary().loc[["speed", "heading_initial", "turn_rate", "radius", "sigma"]]
    )

    turn_rate_samples = fit.stan_variable("turn_rate")
    radius_samples = fit.stan_variable("radius")
    radius_lower, radius_median, radius_upper = np.quantile(
        radius_samples,
        [0.05, 0.5, 0.95],
    )
    print(f"\nLeft-turn probability : {np.mean(turn_rate_samples > 0):.1%}")
    print(f"Right-turn probability: {np.mean(turn_rate_samples < 0):.1%}")
    print(
        "Derived radius       : "
        f"{radius_median:.1f} m "
        f"(90% interval: {radius_lower:.1f} to {radius_upper:.1f} m)"
    )

    prediction_summary = summarize_predictions(fit, window)
    print("\nPosterior prediction summary:")
    print(prediction_summary.to_string(index=False))

    plot_constant_radius_prediction(window, fit)


if __name__ == "__main__":
    main()
