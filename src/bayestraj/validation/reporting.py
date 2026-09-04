"""Console reporting helpers for trajectory forecast validation."""

from collections.abc import Mapping
from pathlib import Path
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


def posterior_parameter_summary(fit, variable_names, credible_interval=0.9):
    """Summarize selected parameters while retaining MCMC diagnostics."""
    variable_names = list(variable_names)
    if callable(getattr(type(fit), "summary", None)):
        return fit.summary().loc[variable_names]

    if not 0 < credible_interval < 1:
        raise ValueError("credible_interval must be between 0 and 1.")
    lower_probability = (1 - credible_interval) / 2
    upper_probability = 1 - lower_probability
    rows = []
    for variable_name in variable_names:
        samples = posterior_variable_samples(fit, variable_name)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError(
                f"Posterior variable {variable_name!r} must contain scalar draws."
            )
        rows.append(
            {
                "variable": variable_name,
                "Mean": np.mean(samples),
                "StdDev": np.std(samples, ddof=1) if samples.size > 1 else 0.0,
                f"{100 * lower_probability:g}%": np.quantile(
                    samples,
                    lower_probability,
                ),
                "50%": np.median(samples),
                f"{100 * upper_probability:g}%": np.quantile(
                    samples,
                    upper_probability,
                ),
            }
        )
    return pd.DataFrame(rows).set_index("variable")


def variational_elbo_history(fit):
    """Read the ELBO progress table from one CmdStan variational run."""
    if not hasattr(fit, "variational_sample") or not hasattr(fit, "runset"):
        raise TypeError("fit must be a CmdStan variational result.")

    stdout_files = fit.runset.stdout_files
    if not stdout_files:
        raise ValueError("Variational fit does not provide a stdout file.")

    rows = []
    table_started = False
    with Path(stdout_files[0]).open(encoding="utf-8") as stream:
        for line in stream:
            if "delta_ELBO_mean" in line and "delta_ELBO_med" in line:
                table_started = True
                continue
            if not table_started:
                continue

            fields = line.split()
            if len(fields) < 4:
                continue
            try:
                row = {
                    "iteration": int(fields[0]),
                    "elbo": float(fields[1]),
                    "delta_elbo_mean": float(fields[2]),
                    "delta_elbo_median": float(fields[3]),
                }
            except ValueError:
                if rows:
                    break
                continue
            rows.append(row)

    if not rows:
        raise ValueError("CmdStan stdout does not contain an ELBO progress table.")
    return pd.DataFrame(rows)


def print_variational_diagnostics(fit, *, converged=None):
    """Print a concise ELBO convergence report for a variational fit."""
    history = variational_elbo_history(fit)
    final = history.iloc[-1]
    rows = []
    if converged is not None:
        rows.append(("VI converged", converged))
    rows.extend(
        [
            ("ELBO evaluations", len(history)),
            ("Final iteration", int(final["iteration"])),
            ("Final ELBO", f"{final['elbo']:.3f}"),
            ("Mean relative delta", f"{final['delta_elbo_mean']:.6g}"),
            ("Median relative delta", f"{final['delta_elbo_median']:.6g}"),
        ]
    )
    print("\nVariational diagnostics:")
    print(format_aligned_rows(rows))


def print_prediction_setup(
    title,
    *,
    data_file,
    run_id,
    window,
    extra_rows=(),
):
    """Print shared window information and model-specific setup values."""
    rows = [
        ("Data file", data_file),
        ("Run ID", run_id),
        ("Observed positions", window.observation_count),
        ("Predicted positions", window.prediction_count),
        *extra_rows,
    ]
    separator = "=" * max(MIN_SEPARATOR_WIDTH, len(title))

    print(separator)
    print(title)
    print(separator)
    print(format_aligned_rows(rows))


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
