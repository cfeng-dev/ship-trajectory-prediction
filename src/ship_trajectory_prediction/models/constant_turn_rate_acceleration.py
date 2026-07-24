"""Bayesian constant-turn-rate-and-acceleration trajectory prediction."""

from pathlib import Path

import numpy as np
from cmdstanpy import CmdStanModel

from ship_trajectory_prediction.models.constant_turn_rate import (
    build_stan_data as _build_constant_turn_rate_stan_data,
)
from ship_trajectory_prediction.models.constant_turn_rate import (
    summarize_predictions,
)
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import TrajectoryWindowData

STAN_FILE = project_path("stan/models/constant_turn_rate_acceleration.stan")

__all__ = [
    "STAN_FILE",
    "build_stan_data",
    "compile_constant_turn_rate_acceleration_model",
    "fit_constant_turn_rate_acceleration_model",
    "summarize_predictions",
]


def build_stan_data(
    window: TrajectoryWindowData,
    *,
    speed_prior_log_sd=0.5,
    heading_prior_scale=0.5,
    turn_rate_prior_scale=0.01,
    acceleration_prior_scale=0.05,
    sigma_prior_scale=20.0,
):
    """Build CmdStan data for constant turn rate and acceleration inference."""
    if not np.isfinite(acceleration_prior_scale) or acceleration_prior_scale <= 0:
        raise ValueError("acceleration_prior_scale must be a positive finite value.")

    stan_data = _build_constant_turn_rate_stan_data(
        window,
        speed_prior_log_sd=speed_prior_log_sd,
        heading_prior_scale=heading_prior_scale,
        turn_rate_prior_scale=turn_rate_prior_scale,
        sigma_prior_scale=sigma_prior_scale,
    )
    stan_data["acceleration_prior_scale"] = acceleration_prior_scale
    stan_data["time_horizon"] = float(window.time_seconds[-1])
    return stan_data


def compile_constant_turn_rate_acceleration_model(stan_file=STAN_FILE):
    """Compile and return the constant-turn-rate-and-acceleration model."""
    stan_file = Path(stan_file)
    if not stan_file.is_file():
        raise FileNotFoundError(f"Stan model not found: {stan_file}")
    return CmdStanModel(stan_file=str(stan_file))


def fit_constant_turn_rate_acceleration_model(
    window: TrajectoryWindowData,
    *,
    speed_prior_log_sd=0.5,
    heading_prior_scale=0.5,
    turn_rate_prior_scale=0.01,
    acceleration_prior_scale=0.05,
    sigma_prior_scale=20.0,
    chains=4,
    parallel_chains=None,
    iter_warmup=500,
    iter_sampling=1000,
    seed=42,
    show_progress=True,
    inits=None,
    inference_method="mcmc",
    variational_options=None,
):
    """Estimate a CTRA trajectory with MCMC or variational inference.

    ``speed_initial`` is the speed at the first position and ``acceleration``
    is a constant tangential acceleration in m/s^2. ``turn_rate`` remains
    constant, so the instantaneous radius changes with speed. Positive endpoint
    speeds keep the linear speed path positive across the complete horizon.
    """
    if inference_method not in {"mcmc", "vi"}:
        raise ValueError("inference_method must be 'mcmc' or 'vi'.")

    stan_data = build_stan_data(
        window,
        speed_prior_log_sd=speed_prior_log_sd,
        heading_prior_scale=heading_prior_scale,
        turn_rate_prior_scale=turn_rate_prior_scale,
        acceleration_prior_scale=acceleration_prior_scale,
        sigma_prior_scale=sigma_prior_scale,
    )
    model = compile_constant_turn_rate_acceleration_model()
    if parallel_chains is None:
        parallel_chains = chains
    if inits is None:
        inits = {
            "speed_initial": stan_data["speed_prior_median"],
            "speed_horizon": stan_data["speed_prior_median"],
            "heading_initial": stan_data["heading_prior_mean"],
            "turn_rate": 0.0,
            "sigma": stan_data["sigma_prior_scale"] / 2,
        }

    if inference_method == "vi":
        options = dict(variational_options or {})
        conflicting = {"data", "seed", "inits"}.intersection(options)
        if conflicting:
            names = ", ".join(sorted(conflicting))
            raise ValueError(f"variational_options must not override: {names}.")
        options.setdefault("algorithm", "meanfield")
        options.setdefault("iter", 20_000)
        options.setdefault("draws", 1_000)
        options.setdefault("tol_rel_obj", 0.05)
        options.setdefault("eval_elbo", 100)
        options.setdefault("show_console", show_progress)
        return model.variational(
            data=stan_data,
            seed=seed,
            inits=inits,
            **options,
        )

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
