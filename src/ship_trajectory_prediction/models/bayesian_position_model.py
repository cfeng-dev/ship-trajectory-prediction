"""Bayesian local position-displacement model fitted with VI or MCMC."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from cmdstanpy import CmdStanMCMC, CmdStanModel, CmdStanVB

import ship_trajectory_prediction.models.bayesian_inference as inference_support
import ship_trajectory_prediction.models.bayesian_observations as observation_support
import ship_trajectory_prediction.models.paths as model_paths
import ship_trajectory_prediction.observations.window as observation_window

STAN_FILE = model_paths.stan_path("models/bayesian_position_model.stan")
MIN_OBSERVATION_COUNT = 3
REGULAR_TIME_STEP_ATOL_SECONDS = 1e-9
DEFAULT_MEANFIELD_GRAD_SAMPLES = inference_support.DEFAULT_MEANFIELD_GRAD_SAMPLES
DEFAULT_VI_ADAPT_ITER = inference_support.DEFAULT_VI_ADAPT_ITER

PositionObservations = observation_support.PositionObservations
simulate_position_observations = observation_support.simulate_position_observations
variational_converged = inference_support.variational_converged

NOISE_PARAMETER_NAMES = ("sigma_displacement_residual",)
PARAMETER_NAMES = (
    "displacement_scale",
    "rotation_angle",
    "sigma_displacement_residual",
)


@dataclass(frozen=True, slots=True)
class BayesianPositionModelPriors:
    """Calibrated position-only prior scales for local displacement dynamics."""

    log_displacement_scale_prior_scale: float
    rotation_angle_prior_scale: float
    sigma_displacement_residual_prior_scale: float

    def __post_init__(self) -> None:
        """Validate every configured prior scale."""
        for prior_field in fields(self):
            value = getattr(self, prior_field.name)
            observation_support.validate_positive_finite(prior_field.name, value)
            object.__setattr__(self, prior_field.name, float(value))


def build_stan_data(
    window: observation_window.TrajectoryWindowData,
    *,
    priors: BayesianPositionModelPriors,
    position_observations: PositionObservations | None = None,
) -> dict[str, Any]:
    """Build Stan data from the complete regular observation window."""
    if not isinstance(priors, BayesianPositionModelPriors):
        raise TypeError("priors must be a BayesianPositionModelPriors instance.")
    if window.observation_count < MIN_OBSERVATION_COUNT:
        raise ValueError(
            f"window must contain at least {MIN_OBSERVATION_COUNT} observed positions."
        )
    if window.prediction_count < 1:
        raise ValueError("window must contain at least one prediction position.")

    position_observations = observation_support.resolve_position_observations(
        window,
        position_observations,
    )
    time_history = np.asarray(
        position_observations.time_seconds,
        dtype=float,
    )
    _validate_regular_prediction_times(window, time_history)
    x_history = np.asarray(
        position_observations.x_meters,
        dtype=float,
    )
    y_history = np.asarray(
        position_observations.y_meters,
        dtype=float,
    )
    observation_support.validate_finite_vector("x_observed", x_history)
    observation_support.validate_finite_vector("y_observed", y_history)

    return {
        "N_observed": window.observation_count,
        "x_observed": x_history,
        "y_observed": y_history,
        "N_prediction": window.prediction_count,
        "log_displacement_scale_prior_scale": (
            priors.log_displacement_scale_prior_scale
        ),
        "rotation_angle_prior_scale": priors.rotation_angle_prior_scale,
        "sigma_displacement_residual_prior_scale": (
            priors.sigma_displacement_residual_prior_scale
        ),
    }


def compile_bayesian_position_model(
    stan_file: str | Path = STAN_FILE,
) -> CmdStanModel:
    """Compile and return the Bayesian position CmdStan model."""
    stan_path = Path(stan_file)
    if not stan_path.is_file():
        raise FileNotFoundError(f"Stan model not found: {stan_path}")
    return CmdStanModel(stan_file=str(stan_path))


def fit_bayesian_position_model(
    window: observation_window.TrajectoryWindowData,
    *,
    priors: BayesianPositionModelPriors,
    position_observations: PositionObservations | None = None,
    inference_method: str = "vi",
    algorithm: str = "meanfield",
    iter: int = 20_000,
    grad_samples: int = DEFAULT_MEANFIELD_GRAD_SAMPLES,
    elbo_samples: int = 100,
    eta: float = 1.0,
    adapt_iter: int = DEFAULT_VI_ADAPT_ITER,
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
    """Fit the local displacement model with mean-field/full-rank VI or MCMC."""
    inference_method = inference_support.normalize_inference_method(inference_method)
    stan_data = build_stan_data(
        window,
        priors=priors,
        position_observations=position_observations,
    )
    model = compile_bayesian_position_model()

    if inference_method == "vi":
        if mcmc_options:
            raise ValueError("mcmc_options can only be used with MCMC inference.")
        inference_support.validate_variational_arguments(
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
            require_converged=require_converged,
            show_console=show_console,
        )
        options = dict(variational_options or {})
        inference_support.reject_conflicting_options(
            "variational_options",
            options,
            {
                "data",
                "seed",
                "inits",
                "algorithm",
                "iter",
                "grad_samples",
                "elbo_samples",
                "eta",
                "adapt_iter",
                "tol_rel_obj",
                "eval_elbo",
                "draws",
                "require_converged",
                "show_console",
            },
        )
        if inits is None:
            inits = _default_initial_values(stan_data, seed=seed)
        return model.variational(
            data=stan_data,
            seed=seed,
            inits=inits,
            algorithm=algorithm,
            iter=iter,
            grad_samples=grad_samples,
            elbo_samples=elbo_samples,
            eta=eta,
            adapt_iter=adapt_iter,
            tol_rel_obj=tol_rel_obj,
            eval_elbo=eval_elbo,
            draws=draws,
            require_converged=require_converged,
            show_console=show_console,
            **options,
        )

    if variational_options:
        raise ValueError("variational_options can only be used with VI inference.")
    if parallel_chains is None:
        parallel_chains = chains
    inference_support.validate_mcmc_arguments(
        chains=chains,
        parallel_chains=parallel_chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        seed=seed,
        show_console=show_console,
    )
    options = dict(mcmc_options or {})
    inference_support.reject_conflicting_options(
        "mcmc_options",
        options,
        {
            "data",
            "seed",
            "inits",
            "chains",
            "parallel_chains",
            "iter_warmup",
            "iter_sampling",
            "adapt_delta",
            "max_treedepth",
            "show_console",
        },
    )
    if inits is None:
        inits = [
            _default_initial_values(stan_data, seed=seed + chain_index)
            for chain_index in range(chains)
        ]
    return model.sample(
        data=stan_data,
        seed=seed,
        inits=inits,
        chains=chains,
        parallel_chains=parallel_chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        show_console=show_console,
        **options,
    )


def _default_initial_values(stan_data: Mapping[str, Any], *, seed: int):
    """Initialize rotation-scaling parameters from the local displacement pairs."""
    generator = np.random.default_rng(seed)
    x_observed = np.asarray(stan_data["x_observed"], dtype=float)
    y_observed = np.asarray(stan_data["y_observed"], dtype=float)
    displacement = np.column_stack((np.diff(x_observed), np.diff(y_observed)))
    previous = displacement[:-1]
    current = displacement[1:]
    denominator = float(np.sum(previous * previous))
    if denominator <= 1e-12:
        scale = 1.0
        angle = 0.0
    else:
        real_part = float(np.sum(previous * current) / denominator)
        imaginary_part = float(
            np.sum(previous[:, 0] * current[:, 1] - previous[:, 1] * current[:, 0])
            / denominator
        )
        scale = max(np.hypot(real_part, imaginary_part), 1e-6)
        angle = float(np.arctan2(imaginary_part, real_part))

    cosine = np.cos(angle)
    sine = np.sin(angle)
    autoregressive_matrix = scale * np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=float,
    )
    residual = current - previous @ autoregressive_matrix.T
    residual_scale = max(
        float(np.sqrt(np.mean(residual**2))),
        0.5 * float(stan_data["sigma_displacement_residual_prior_scale"]),
    )
    angle_limit = np.pi - 1e-6
    return {
        "log_displacement_scale": float(np.log(scale) + generator.normal(0.0, 1e-3)),
        "rotation_angle": float(
            np.clip(angle + generator.normal(0.0, 1e-3), -angle_limit, angle_limit)
        ),
        "sigma_displacement_residual": float(
            max(residual_scale * np.exp(generator.normal(0.0, 1e-3)), 1e-6)
        ),
    }


def _validate_regular_prediction_times(window, time_history: np.ndarray) -> None:
    """Require one common sampling interval for fitted and forecast steps."""
    prediction_times = np.asarray(
        window.time_seconds[window.prediction_slice],
        dtype=float,
    )
    relevant_times = np.concatenate((time_history, prediction_times))
    time_steps = np.diff(relevant_times)
    if not np.all(np.isfinite(time_steps)) or np.any(time_steps <= 0):
        raise ValueError("Position-model timestamps must be finite and increasing.")
    if not np.allclose(
        time_steps,
        time_steps[0],
        rtol=0.0,
        atol=REGULAR_TIME_STEP_ATOL_SECONDS,
    ):
        raise ValueError(
            "Bayesian Position Model requires one regular sampling interval."
        )
