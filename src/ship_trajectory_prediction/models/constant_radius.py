"""Bayesian constant-radius prediction with inferred turn direction."""

from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanMCMC, CmdStanModel

from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import TrajectoryWindowData
from ship_trajectory_prediction.trajectory.window import (
    estimate_initial_heading,
    estimate_positive_speed_median,
)

STAN_FILE = project_path("stan/models/constant_radius.stan")


def build_stan_data(
    window: TrajectoryWindowData,
    *,
    curvature_prior_scale=0.002,
    sigma_prior_scale=20.0,
):
    """Build data for constant signed-curvature inference.

    Curvature uses 1/m and has a zero-centered prior, allowing its posterior
    sign to determine left or right motion. Radius is derived in Stan as the
    inverse absolute curvature.
    """
    prior_values = {
        "curvature_prior_scale": curvature_prior_scale,
        "sigma_prior_scale": sigma_prior_scale,
    }
    for name, value in prior_values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite value.")

    observed = window.observed_slice
    prediction = window.prediction_slice
    speed_mps = estimate_positive_speed_median(window.gps_speed_mps[observed])
    initial_heading = estimate_initial_heading(
        window.x_meters[observed],
        window.y_meters[observed],
    )

    return {
        "N_observed": window.observation_count,
        "time_observed": window.time_seconds[observed],
        "x_observed": window.x_meters[observed],
        "y_observed": window.y_meters[observed],
        "N_prediction": window.prediction_count,
        "time_prediction": window.time_seconds[prediction],
        "x_initial": float(window.x_meters[0]),
        "y_initial": float(window.y_meters[0]),
        "speed": speed_mps,
        "heading_initial": initial_heading,
        **prior_values,
    }


def compile_constant_radius_model(stan_file=STAN_FILE):
    """Compile and return the constant-radius CmdStan model."""
    stan_file = Path(stan_file)
    if not stan_file.is_file():
        raise FileNotFoundError(f"Stan model not found: {stan_file}")
    return CmdStanModel(stan_file=str(stan_file))


def fit_constant_radius_model(
    window: TrajectoryWindowData,
    *,
    curvature_prior_scale=0.002,
    sigma_prior_scale=20.0,
    chains=4,
    parallel_chains=None,
    iter_warmup=500,
    iter_sampling=1000,
    seed=42,
    show_progress=True,
    inits=None,
):
    """Fit constant curvature while inferring turn direction from its sign."""
    stan_data = build_stan_data(
        window,
        curvature_prior_scale=curvature_prior_scale,
        sigma_prior_scale=sigma_prior_scale,
    )
    model = compile_constant_radius_model()
    if parallel_chains is None:
        parallel_chains = chains
    if inits is None:
        inits = {
            "curvature": 0.0,
            "sigma": sigma_prior_scale / 2,
        }

    return model.sample(
        data=stan_data,
        chains=chains,
        parallel_chains=parallel_chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        seed=seed,
        show_progress=show_progress,
        inits=inits,
    )


def summarize_predictions(fit, window: TrajectoryWindowData, credible_interval=0.9):
    """Summarize posterior predictive positions against held-out observations."""
    if not 0 < credible_interval < 1:
        raise ValueError("credible_interval must be between 0 and 1.")

    x_samples = _prediction_samples(fit, "x_prediction", window.prediction_count)
    y_samples = _prediction_samples(fit, "y_prediction", window.prediction_count)
    lower_probability = (1 - credible_interval) / 2
    upper_probability = 1 - lower_probability
    prediction = window.prediction_slice

    return pd.DataFrame(
        {
            "time": window.timestamps[prediction],
            "t": window.time_seconds[prediction],
            "x_actual": window.x_meters[prediction],
            "y_actual": window.y_meters[prediction],
            "x_median": np.median(x_samples, axis=0),
            "y_median": np.median(y_samples, axis=0),
            "x_lower": np.quantile(x_samples, lower_probability, axis=0),
            "x_upper": np.quantile(x_samples, upper_probability, axis=0),
            "y_lower": np.quantile(y_samples, lower_probability, axis=0),
            "y_upper": np.quantile(y_samples, upper_probability, axis=0),
        }
    )


def _prediction_samples(fit, variable_name, prediction_count):
    """Extract and validate one posterior prediction matrix."""
    if not isinstance(fit, CmdStanMCMC) and not hasattr(fit, "stan_variable"):
        raise TypeError("fit must provide CmdStan-style posterior variables.")

    samples = np.asarray(fit.stan_variable(variable_name), dtype=float)
    if samples.ndim != 2 or samples.shape[1] != prediction_count:
        raise ValueError(
            f"Posterior variable {variable_name!r} has an unexpected shape."
        )
    return samples
