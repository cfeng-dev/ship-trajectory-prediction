"""Plotting utilities for Bayesian trajectory evaluation."""

import matplotlib.pyplot as plt
import numpy as np


def plot_prediction(
    window,
    fit,
    *,
    model_name,
    max_posterior_trajectories=100,
):
    """Plot observed, held-out, and posterior trajectories for any model."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")

    observed = window.observed_slice
    prediction = window.prediction_slice
    x_samples = fit.stan_variable("x_prediction_mean")
    y_samples = fit.stan_variable("y_prediction_mean")
    prediction_start_index = window.observation_count - 1
    prediction_start_x = window.x_meters[prediction_start_index]
    prediction_start_y = window.y_meters[prediction_start_index]
    held_out_x = np.concatenate(([prediction_start_x], window.x_meters[prediction]))
    held_out_y = np.concatenate(([prediction_start_y], window.y_meters[prediction]))
    connected_x_samples = np.column_stack(
        (np.full(len(x_samples), prediction_start_x), x_samples)
    )
    connected_y_samples = np.column_stack(
        (np.full(len(y_samples), prediction_start_y), y_samples)
    )

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

    sample_count = min(max_posterior_trajectories, len(x_samples))
    sample_indices = np.linspace(
        0,
        len(x_samples) - 1,
        num=sample_count,
        dtype=int,
    )
    for sample_index in sample_indices:
        axis.plot(
            connected_x_samples[sample_index],
            connected_y_samples[sample_index],
            color="tab:red",
            alpha=0.05,
            linewidth=1,
        )

    axis.plot(
        np.median(connected_x_samples, axis=0),
        np.median(connected_y_samples, axis=0),
        color="tab:red",
        linewidth=2,
        label="Posterior median",
    )
    axis.scatter(
        prediction_start_x,
        prediction_start_y,
        color="tab:blue",
        zorder=3,
        label="Prediction start",
    )

    axis.set_title(f"Bayesian {model_name.strip()} Prediction")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plt.show()
    return figure, axis
