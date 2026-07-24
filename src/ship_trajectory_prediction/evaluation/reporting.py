"""Console reporting helpers for trajectory prediction experiments."""

from pathlib import Path

import numpy as np
import pandas as pd

MIN_SEPARATOR_WIDTH = 60


def posterior_variable_samples(fit, variable_name):
    """Return posterior draws from either MCMC or variational inference."""
    if not hasattr(fit, "stan_variable"):
        raise TypeError("fit must provide CmdStan-style posterior variables.")

    if hasattr(fit, "variational_sample"):
        values = fit.stan_variable(variable_name, mean=False)
    else:
        values = fit.stan_variable(variable_name)
    return np.asarray(values, dtype=float)


def posterior_parameter_summary(fit, variable_names, credible_interval=0.9):
    """Summarize selected parameters while retaining MCMC diagnostics."""
    variable_names = list(variable_names)
    if hasattr(fit, "summary"):
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


def print_variational_diagnostics(fit):
    """Print a concise ELBO convergence report for a variational fit."""
    history = variational_elbo_history(fit)
    final = history.iloc[-1]
    print("\nVariational convergence:")
    print(f"ELBO evaluations : {len(history)}")
    print(f"Final iteration   : {int(final['iteration'])}")
    print(f"Final ELBO        : {final['elbo']:.3f}")
    print(f"Mean relative delta: {final['delta_elbo_mean']:.6g}")
    print(f"Median relative delta: {final['delta_elbo_median']:.6g}")


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
    label_width = max(len(label) for label, _ in rows)

    print(separator)
    print(title)
    print(separator)
    for label, value in rows:
        print(f"{label:<{label_width}}: {value}")
