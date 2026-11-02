"""Console reporting helpers for trajectory forecast validation."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

import bayestraj.observations.window as observation_window

MIN_SEPARATOR_WIDTH = 72

PREDICTION_COLUMN_LABELS = {
    "horizon_seconds": "Horizon[s]",
    "x_actual": "Actual x[m]",
    "y_actual": "Actual y[m]",
    "x_predicted": "Pred. x[m]",
    "y_predicted": "Pred. y[m]",
    "x_median": "Median x[m]",
    "y_median": "Median y[m]",
    "position_error_m": "Error[m]",
    "prediction_radius_m": "Radius[m]",
    "radial_covered": "Covered",
}


def format_aligned_rows(rows) -> str:
    """Format label-value pairs with one shared separator column."""
    rows = list(rows)
    if not rows:
        return ""
    label_width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{label_width}} : {value}" for label, value in rows)


def format_prediction_table(table: pd.DataFrame, *, columns) -> str:
    """Format selected prediction columns with compact terminal labels."""
    columns = list(columns)
    missing_columns = [column for column in columns if column not in table]
    if missing_columns:
        raise ValueError(f"Missing prediction table columns: {missing_columns}")

    display_table = table.loc[:, columns].rename(columns=PREDICTION_COLUMN_LABELS)
    if "Covered" in display_table:
        display_table["Covered"] = display_table["Covered"].map(
            {True: "yes", False: "no"}
        )
    numeric_formatters = {
        "Horizon[s]": lambda value: f"{value:.1f}",
        "Actual x[m]": lambda value: f"{value:.3f}",
        "Actual y[m]": lambda value: f"{value:.3f}",
        "Pred. x[m]": lambda value: f"{value:.3f}",
        "Pred. y[m]": lambda value: f"{value:.3f}",
        "Median x[m]": lambda value: f"{value:.3f}",
        "Median y[m]": lambda value: f"{value:.3f}",
        "Error[m]": lambda value: f"{value:.3f}",
        "Radius[m]": lambda value: f"{value:.3f}",
    }
    formatters = {
        column: formatter
        for column, formatter in numeric_formatters.items()
        if column in display_table
    }
    return display_table.to_string(index=False, formatters=formatters)


def format_evaluation_report(metric_rows, prediction_table: str) -> str:
    """Format one single-window evaluation like the rolling summary."""
    separator = "=" * MIN_SEPARATOR_WIDTH
    return "\n".join(
        [
            separator,
            "Complete prediction evaluation",
            separator,
            format_aligned_rows(metric_rows),
            "\nPer-horizon evaluation:",
            prediction_table,
        ]
    )


def posterior_variable_samples(fit, variable_name):
    """Return posterior draws from either MCMC or variational inference."""
    if not hasattr(fit, "stan_variable"):
        raise TypeError("fit must provide CmdStan-style posterior variables.")

    if hasattr(fit, "variational_sample"):
        values = fit.stan_variable(variable_name, mean=False)
    else:
        values = fit.stan_variable(variable_name)
    return np.asarray(values, dtype=float)


def summarize_predictions(
    fit: Any,
    window: observation_window.TrajectoryWindowData,
    credible_interval: float = 0.9,
    *,
    prediction_variables: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Summarize future model positions and sensor observations."""
    if not np.isfinite(credible_interval) or not 0 < credible_interval < 1:
        raise ValueError("credible_interval must be between 0 and 1.")
    if prediction_variables is None:
        prediction_variables = {
            "x": "x_prediction",
            "y": "y_prediction",
            "x_observation": "x_observation_prediction",
            "y_observation": "y_observation_prediction",
        }
    lower_probability = (1 - credible_interval) / 2
    upper_probability = 1 - lower_probability
    prediction = window.prediction_slice
    table_data: dict[str, Any] = {
        "time": window.timestamps[prediction],
        "t": window.time_seconds[prediction],
        "x_actual": window.x_meters[prediction],
        "y_actual": window.y_meters[prediction],
    }
    for prefix, variable_name in prediction_variables.items():
        samples = _prediction_samples(fit, variable_name, window.prediction_count)
        table_data[f"{prefix}_median"] = np.median(samples, axis=0)
        table_data[f"{prefix}_lower"] = np.quantile(
            samples,
            lower_probability,
            axis=0,
        )
        table_data[f"{prefix}_upper"] = np.quantile(
            samples,
            upper_probability,
            axis=0,
        )
    return pd.DataFrame(table_data)


def _prediction_samples(fit: Any, variable_name: str, prediction_count: int):
    """Extract one finite posterior prediction matrix."""
    samples = posterior_variable_samples(fit, variable_name)
    if samples.ndim != 2 or samples.shape[1] != prediction_count:
        raise ValueError(
            f"Posterior variable {variable_name!r} has an unexpected shape."
        )
    if samples.shape[0] == 0 or not np.all(np.isfinite(samples)):
        raise ValueError(
            f"Posterior variable {variable_name!r} must contain finite draws."
        )
    return samples
