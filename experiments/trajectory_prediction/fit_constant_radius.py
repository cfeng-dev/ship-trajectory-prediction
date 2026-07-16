"""Fit the Bayesian constant-radius model to one recorded trajectory window."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ship_trajectory_prediction.models.constant_radius import (
    fit_constant_radius_model,
    prepare_trajectory_window,
    summarize_predictions,
)
from ship_trajectory_prediction.trajectory.io import read_ship_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

RUN_ID = 1
START_INDEX = 0
OBSERVATION_COUNT = 20
PREDICTION_COUNT = 10
RADIUS_PRIOR_MEDIAN = 500.0
RADIUS_PRIOR_LOG_SD = 1.0
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
    print("Bayesian Constant-Radius Trajectory Prediction")
    print("=" * 60)
    print(f"Data file          : {DATA_FILE}")
    print(f"Run ID             : {RUN_ID}")
    print(f"Observed positions : {window.observation_count}")
    print(f"Predicted positions: {window.prediction_count}")
    print(f"Estimated speed    : {window.speed_mps:.2f} m/s")
    print(f"Turn direction     : {window.turn_direction:+d}")

    fit = fit_constant_radius_model(
        window,
        radius_prior_median=RADIUS_PRIOR_MEDIAN,
        radius_prior_log_sd=RADIUS_PRIOR_LOG_SD,
        sigma_prior_scale=SIGMA_PRIOR_SCALE,
    )

    print("\nPosterior parameter summary:")
    print(fit.summary().loc[["radius", "sigma"]])

    prediction_summary = summarize_predictions(fit, window)
    print("\nPosterior prediction summary:")
    print(prediction_summary.to_string(index=False))

    plot_prediction(window, fit)


def plot_prediction(window, fit, max_posterior_trajectories=100):
    """Plot observed, held-out, and posterior predicted trajectories."""
    observed = window.observed_slice
    prediction = window.prediction_slice
    x_samples = fit.stan_variable("x_prediction_mean")
    y_samples = fit.stan_variable("y_prediction_mean")

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.plot(
        window.x_meters[observed],
        window.y_meters[observed],
        color="tab:blue",
        linewidth=2,
        label="Observed trajectory",
    )
    axis.plot(
        window.x_meters[prediction],
        window.y_meters[prediction],
        color="black",
        linestyle="--",
        linewidth=2,
        label="Held-out trajectory",
    )

    sample_count = min(max_posterior_trajectories, len(x_samples))
    sample_indices = np.linspace(
        0,
        len(x_samples) - 1,
        num=sample_count,
        dtype=int,
    )
    for sample_index in sample_indices:
        axis.plot(
            x_samples[sample_index],
            y_samples[sample_index],
            color="tab:red",
            alpha=0.05,
            linewidth=1,
        )

    axis.plot(
        np.median(x_samples, axis=0),
        np.median(y_samples, axis=0),
        color="tab:red",
        linewidth=2,
        label="Posterior median",
    )
    axis.scatter(
        window.x_meters[window.observation_count - 1],
        window.y_meters[window.observation_count - 1],
        color="tab:blue",
        zorder=3,
        label="Prediction start",
    )

    axis.set_title("Bayesian Constant-Radius Prediction")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
