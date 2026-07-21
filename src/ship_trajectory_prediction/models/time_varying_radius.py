"""Bayesian trajectory prediction with linearly time-varying curvature."""

from pathlib import Path

import numpy as np
from cmdstanpy import CmdStanModel

from ship_trajectory_prediction.models.constant_radius import (
    TrajectoryWindow,
    prepare_trajectory_window,
    summarize_predictions,
)
from ship_trajectory_prediction.paths import project_path

STAN_FILE = project_path("stan/models/time_varying_radius.stan")

__all__ = [
    "STAN_FILE",
    "TrajectoryWindow",
    "build_stan_data",
    "compile_time_varying_radius_model",
    "fit_time_varying_radius_model",
    "prepare_trajectory_window",
    "summarize_predictions",
]


def build_stan_data(
    window,
    *,
    radius_prior_median=500.0,
    curvature_initial_prior_scale=0.002,
    curvature_rate_prior_scale=5e-6,
    sigma_prior_scale=20.0,
    integration_substeps=4,
):
    """Build CmdStan data for a linearly changing signed curvature.

    Curvature is measured in 1/m and its rate in 1/(m s). The prior mean
    converts the familiar radius prior into signed curvature by using the turn
    direction inferred exclusively from the observed trajectory.
    """
    positive_values = {
        "radius_prior_median": radius_prior_median,
        "curvature_initial_prior_scale": curvature_initial_prior_scale,
        "curvature_rate_prior_scale": curvature_rate_prior_scale,
        "sigma_prior_scale": sigma_prior_scale,
    }
    for name, value in positive_values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite value.")

    if (
        isinstance(integration_substeps, bool)
        or not isinstance(integration_substeps, int)
        or integration_substeps < 1
    ):
        raise ValueError("integration_substeps must be an integer of at least 1.")

    observed = window.observed_slice
    prediction = window.prediction_slice
    return {
        "N_observed": window.observation_count,
        "time_observed": window.time_seconds[observed],
        "x_observed": window.x_meters[observed],
        "y_observed": window.y_meters[observed],
        "N_prediction": window.prediction_count,
        "time_prediction": window.time_seconds[prediction],
        "x_initial": float(window.x_meters[0]),
        "y_initial": float(window.y_meters[0]),
        "speed": window.speed_mps,
        "heading_initial": window.initial_heading,
        "curvature_prior_mean": window.turn_direction / radius_prior_median,
        "curvature_initial_prior_scale": curvature_initial_prior_scale,
        "curvature_rate_prior_scale": curvature_rate_prior_scale,
        "sigma_prior_scale": sigma_prior_scale,
        "integration_substeps": integration_substeps,
    }


def compile_time_varying_radius_model(stan_file=STAN_FILE):
    """Compile and return the time-varying-radius CmdStan model."""
    stan_file = Path(stan_file)
    if not stan_file.is_file():
        raise FileNotFoundError(f"Stan model not found: {stan_file}")
    return CmdStanModel(stan_file=str(stan_file))


def fit_time_varying_radius_model(
    window,
    *,
    radius_prior_median=500.0,
    curvature_initial_prior_scale=0.002,
    curvature_rate_prior_scale=5e-6,
    sigma_prior_scale=20.0,
    integration_substeps=4,
    chains=4,
    parallel_chains=None,
    iter_warmup=500,
    iter_sampling=1000,
    seed=42,
    show_progress=True,
    adapt_delta=0.95,
    max_treedepth=12,
    inits=None,
):
    """Fit a constant-speed trajectory with smoothly changing curvature.

    ``curvature_initial`` and ``curvature_rate`` define
    ``curvature(t) = curvature_initial + curvature_rate * t``. Radius is a
    derived quantity, so nearly straight motion remains numerically stable even
    when its radius becomes very large.
    """
    stan_data = build_stan_data(
        window,
        radius_prior_median=radius_prior_median,
        curvature_initial_prior_scale=curvature_initial_prior_scale,
        curvature_rate_prior_scale=curvature_rate_prior_scale,
        sigma_prior_scale=sigma_prior_scale,
        integration_substeps=integration_substeps,
    )
    model = compile_time_varying_radius_model()
    if parallel_chains is None:
        parallel_chains = chains
    if inits is None:
        inits = {
            "curvature_initial_raw": 0.0,
            "curvature_rate_raw": 0.0,
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
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        inits=inits,
    )
