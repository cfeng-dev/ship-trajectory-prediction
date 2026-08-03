"""Plotting utilities for Bayesian trajectory evaluation."""

import matplotlib.pyplot as plt
import numpy as np

from ship_trajectory_prediction.evaluation.reporting import (
    posterior_variable_samples,
)


def plot_prediction(
    window,
    fit,
    *,
    model_name,
    max_posterior_trajectories=100,
    state_prediction_variable_names=("x_prediction_mean", "y_prediction_mean"),
    observed_position_values=None,
    observed_trajectory_label="Observed trajectory",
):
    """Plot observed, held-out, and latent posterior paths for any model."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")
    if (
        not isinstance(state_prediction_variable_names, (tuple, list))
        or len(state_prediction_variable_names) != 2
        or not all(
            isinstance(name, str) and name.strip()
            for name in state_prediction_variable_names
        )
    ):
        raise ValueError(
            "state_prediction_variable_names must contain non-empty x and y names."
        )
    if (
        not isinstance(observed_trajectory_label, str)
        or not observed_trajectory_label.strip()
    ):
        raise ValueError("observed_trajectory_label must be a non-empty string.")

    prediction = window.prediction_slice
    observed_x, observed_y = _resolve_observed_position_values(
        window,
        observed_position_values,
    )
    x_variable_name, y_variable_name = state_prediction_variable_names
    x_samples = posterior_variable_samples(fit, x_variable_name)
    y_samples = posterior_variable_samples(fit, y_variable_name)
    prediction_start_x = observed_x[-1]
    prediction_start_y = observed_y[-1]
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
        observed_x,
        observed_y,
        color="tab:blue",
        linewidth=2,
        label=observed_trajectory_label.strip(),
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


def _resolve_observed_position_values(window, observed_position_values):
    """Return finite observed positions, optionally overriding window values."""
    if observed_position_values is None:
        observed = window.observed_slice
        return (
            np.asarray(window.x_meters[observed], dtype=float),
            np.asarray(window.y_meters[observed], dtype=float),
        )
    if (
        not isinstance(observed_position_values, (tuple, list))
        or len(observed_position_values) != 2
    ):
        raise ValueError(
            "observed_position_values must contain observed x and y values."
        )
    observed_x = np.asarray(observed_position_values[0], dtype=float)
    observed_y = np.asarray(observed_position_values[1], dtype=float)
    expected_shape = (window.observation_count,)
    if (
        observed_x.shape != expected_shape
        or observed_y.shape != expected_shape
        or not np.all(np.isfinite(observed_x))
        or not np.all(np.isfinite(observed_y))
    ):
        raise ValueError(
            "observed_position_values must contain finite x and y vectors "
            "matching window.observation_count."
        )
    return observed_x, observed_y
