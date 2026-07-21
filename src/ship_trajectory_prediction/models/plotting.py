"""Plotting utilities for Bayesian trajectory predictions."""

import matplotlib.pyplot as plt
import numpy as np


def plot_constant_radius_prediction(
    window,
    fit,
    max_posterior_trajectories=100,
):
    """Plot observed, held-out, and posterior constant-radius trajectories."""
    _plot_prediction(
        window,
        fit,
        max_posterior_trajectories,
        title="Bayesian Constant-Radius Prediction",
    )


def plot_constant_turn_rate_prediction(
    window,
    fit,
    max_posterior_trajectories=100,
):
    """Plot observed, held-out, and posterior constant-turn-rate trajectories."""
    _plot_prediction(
        window,
        fit,
        max_posterior_trajectories,
        title="Bayesian Constant-Turn-Rate Prediction",
    )


def plot_constant_turn_rate_acceleration_prediction(
    window,
    fit,
    max_posterior_trajectories=100,
):
    """Plot observed, held-out, and posterior CTRA trajectories."""
    _plot_prediction(
        window,
        fit,
        max_posterior_trajectories,
        title="Bayesian Constant-Turn-Rate-and-Acceleration Prediction",
    )


def plot_time_varying_motion_prediction(
    window,
    fit,
    max_posterior_trajectories=100,
):
    """Plot observed, held-out, and posterior time-varying trajectories."""
    _plot_prediction(
        window,
        fit,
        max_posterior_trajectories,
        title="Bayesian Time-Varying Motion Prediction",
    )


def _plot_prediction(window, fit, max_posterior_trajectories, *, title):
    """Plot one posterior trajectory prediction with the requested title."""
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

    axis.set_title(title)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plt.show()
