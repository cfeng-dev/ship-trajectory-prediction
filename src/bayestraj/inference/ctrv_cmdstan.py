"""CmdStan compilation and batch inference for the Bayesian CTRV model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from cmdstanpy import CmdStanMCMC, CmdStanModel, CmdStanVB

import bayestraj.inference.cmdstan as cmdstan_inference
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.observations.position as observation_support
import bayestraj.observations.window as observation_window


def compile_bayesian_ctrv_model(
    stan_file: str | Path = bayesian_model.STAN_FILE,
) -> CmdStanModel:
    """Compile and return the parametric Bayesian CTRV model."""
    stan_path = Path(stan_file)
    if not stan_path.is_file():
        raise FileNotFoundError(f"Stan model not found: {stan_path}")
    return CmdStanModel(stan_file=str(stan_path))


def fit_bayesian_ctrv_model(
    window: observation_window.TrajectoryWindowData,
    *,
    priors: bayesian_model.BayesianCTRVPriors | None = None,
    position_observations: observation_support.PositionObservations | None = None,
    inference_method: str = "vi",
    algorithm: str = "meanfield",
    iter: int = 20_000,
    grad_samples: int = cmdstan_inference.DEFAULT_MEANFIELD_GRAD_SAMPLES,
    elbo_samples: int = 100,
    eta: float = 1.0,
    adapt_iter: int = cmdstan_inference.DEFAULT_VI_ADAPT_ITER,
    tol_rel_obj: float = 0.01,
    eval_elbo: int = 100,
    draws: int = 1_000,
    chains: int = 4,
    parallel_chains: int | None = None,
    iter_warmup: int = 1_000,
    iter_sampling: int = 1_000,
    adapt_delta: float = 0.9,
    max_treedepth: int = 10,
    seed: int = 42,
    inits: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | float | None = None,
    require_converged: bool = True,
    show_console: bool = False,
    variational_options: Mapping[str, Any] | None = None,
    mcmc_options: Mapping[str, Any] | None = None,
) -> CmdStanVB | CmdStanMCMC:
    """Fit the latent-state CTRV model with VI or MCMC."""
    inference_method = cmdstan_inference.normalize_inference_method(inference_method)
    stan_data = bayesian_model.build_stan_data(
        window,
        priors=priors,
        position_observations=position_observations,
    )
    model = compile_bayesian_ctrv_model()

    if inference_method == "vi":
        if mcmc_options:
            raise ValueError("mcmc_options can only be used with MCMC inference.")
        return cmdstan_inference.run_variational_inference(
            model,
            stan_data,
            algorithm=algorithm,
            iter=iter,
            grad_samples=grad_samples,
            elbo_samples=elbo_samples,
            eta=eta,
            adapt_iter=adapt_iter,
            tol_rel_obj=tol_rel_obj,
            eval_elbo=eval_elbo,
            draws=draws,
            seed=seed,
            inits=inits,
            default_inits_factory=lambda: _default_initial_values(
                stan_data,
                seed=seed,
            ),
            require_converged=require_converged,
            show_console=show_console,
            options=variational_options,
        )

    if variational_options:
        raise ValueError("variational_options can only be used with VI inference.")
    return cmdstan_inference.run_mcmc_inference(
        model,
        stan_data,
        chains=chains,
        parallel_chains=parallel_chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        seed=seed,
        inits=inits,
        default_inits_factory=lambda: [
            _default_initial_values(stan_data, seed=seed + chain_index)
            for chain_index in range(chains)
        ],
        show_console=show_console,
        options=mcmc_options,
    )


def _default_initial_values(stan_data: Mapping[str, Any], *, seed: int):
    """Create seeded latent-state initials from the position history."""
    generator = np.random.default_rng(seed)
    x_true = _smooth_position_initials(stan_data["x_observed"])
    y_true = _smooth_position_initials(stan_data["y_observed"])
    observation_noise_initial = 1.0 / float(
        stan_data["sigma_position_observation_prior_rate"]
    )
    latent_jitter_scale = 0.02 * observation_noise_initial
    x_true += generator.normal(0.0, latent_jitter_scale, x_true.size)
    y_true += generator.normal(0.0, latent_jitter_scale, y_true.size)
    speed, heading_initial, turn_rate = (
        bayesian_model.estimate_constant_motion_from_positions(
            stan_data["time_observed"],
            x_true,
            y_true,
        )
    )
    speed_jitter = 0.02 * float(stan_data["speed_prior_scale"])
    turn_jitter = 0.02 * float(stan_data["turn_rate_prior_scale"])
    angle_limit = np.pi - 1e-6
    return {
        "x_initial": float(x_true[0]),
        "y_initial": float(y_true[0]),
        "speed_state": np.maximum(
            speed + generator.normal(0.0, speed_jitter, x_true.size),
            float(stan_data["speed_state_lower_mps"]),
        ),
        "heading_initial": float(
            np.clip(
                heading_initial + generator.normal(0.0, 0.01),
                -angle_limit,
                angle_limit,
            )
        ),
        "turn_rate_state": turn_rate + generator.normal(0.0, turn_jitter, x_true.size),
        "sigma_position_observation": float(observation_noise_initial),
        "sigma_speed_process": float(0.5 / stan_data["sigma_speed_process_prior_rate"]),
        "sigma_turn_rate_process": float(
            0.5 / stan_data["sigma_turn_rate_process_prior_rate"]
        ),
    }


def _smooth_position_initials(observed) -> np.ndarray:
    """Return a lightly smoothed position history for latent-state initials."""
    observed = np.asarray(observed, dtype=float)
    initial = observed.copy()
    initial[1:-1] = 0.25 * observed[:-2] + 0.5 * observed[1:-1] + 0.25 * observed[2:]
    return initial
