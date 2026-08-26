"""Parametric Bayesian CTRV model fitted with VI or MCMC."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanMCMC, CmdStanModel, CmdStanVB

import bayestraj.models.bayesian_inference as inference_support
import bayestraj.models.bayesian_observations as observation_support
import bayestraj.models.paths as model_paths
import bayestraj.observations.window as observation_window
import bayestraj.validation.reporting as reporting

STAN_FILE = model_paths.stan_path("models/bayesian_ctrv.stan")

DEFAULT_MEANFIELD_GRAD_SAMPLES = inference_support.DEFAULT_MEANFIELD_GRAD_SAMPLES
DEFAULT_VI_ADAPT_ITER = inference_support.DEFAULT_VI_ADAPT_ITER
DEFAULT_POSITION_OBSERVATION_NOISE_STD_M = (
    observation_support.DEFAULT_POSITION_OBSERVATION_NOISE_STD_M
)
MIN_OBSERVATION_COUNT = 3
SPEED_INITIAL_LOWER_MPS = 0.001

PositionObservations = observation_support.PositionObservations
simulate_position_observations = observation_support.simulate_position_observations
variational_converged = inference_support.variational_converged

PARAMETER_NAMES = (
    "speed",
    "heading_initial",
    "turn_rate",
    "sigma_position_observation",
)


@dataclass(frozen=True, slots=True)
class BayesianCTRVPriors:
    """Provisional priors transferred from the earlier CTRV experiments.

    The kinematic values are useful initial candidates but are not claimed to
    be final calibration results for this constant-parameter CTRV model. The
    observation-noise prior is a separately configurable scenario assumption.
    """

    speed_prior_mean: float = 3.524
    speed_prior_scale: float = 0.365
    turn_rate_prior_mean: float = 0.0
    turn_rate_prior_scale: float = 0.001698
    sigma_position_observation_prior_upper_m: float = 20.0
    sigma_position_observation_prior_tail_probability: float = 0.05

    def __post_init__(self) -> None:
        """Validate and normalize all configured prior values."""
        for prior_field in fields(self):
            name = prior_field.name
            value = getattr(self, name)
            if name == "speed_prior_mean":
                value = observation_support.validate_non_negative_finite(name, value)
            elif name == "turn_rate_prior_mean":
                value = observation_support.validate_finite_scalar(name, value)
            elif name == "sigma_position_observation_prior_tail_probability":
                value = observation_support.validate_finite_scalar(name, value)
                if not 0.0 < value < 1.0:
                    raise ValueError(f"{name} must be strictly between zero and one.")
            else:
                value = observation_support.validate_positive_finite(name, value)
            object.__setattr__(self, name, value)

    @property
    def sigma_position_observation_prior_rate(self) -> float:
        """Return the exponential rate implied by one prior tail statement."""
        return float(
            -np.log(self.sigma_position_observation_prior_tail_probability)
            / self.sigma_position_observation_prior_upper_m
        )


def build_stan_data(
    window: observation_window.TrajectoryWindowData,
    *,
    priors: BayesianCTRVPriors | None = None,
    position_observations: PositionObservations | None = None,
) -> dict[str, Any]:
    """Build position-only Stan data from the complete observed window."""
    if priors is None:
        priors = BayesianCTRVPriors()
    if not isinstance(priors, BayesianCTRVPriors):
        raise TypeError("priors must be a BayesianCTRVPriors instance or None.")
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
    time_observed = np.asarray(
        position_observations.time_seconds,
        dtype=float,
    )
    x_observed = np.asarray(
        position_observations.x_meters,
        dtype=float,
    )
    y_observed = np.asarray(
        position_observations.y_meters,
        dtype=float,
    )
    time_prediction = np.asarray(
        window.time_seconds[window.prediction_slice],
        dtype=float,
    )
    _validate_time_arrays(time_observed, time_prediction)
    observation_support.validate_finite_vector("x_observed", x_observed)
    observation_support.validate_finite_vector("y_observed", y_observed)

    return {
        "N_history": window.observation_count,
        "time_observed": time_observed,
        "x_observed": x_observed,
        "y_observed": y_observed,
        "sigma_position_observation_prior_rate": (
            priors.sigma_position_observation_prior_rate
        ),
        "N_prediction": window.prediction_count,
        "time_prediction": time_prediction,
        "speed_prior_mean": priors.speed_prior_mean,
        "speed_prior_scale": priors.speed_prior_scale,
        "turn_rate_prior_mean": priors.turn_rate_prior_mean,
        "turn_rate_prior_scale": priors.turn_rate_prior_scale,
    }


def compile_bayesian_ctrv_model(
    stan_file: str | Path = STAN_FILE,
) -> CmdStanModel:
    """Compile and return the parametric Bayesian CTRV model."""
    stan_path = Path(stan_file)
    if not stan_path.is_file():
        raise FileNotFoundError(f"Stan model not found: {stan_path}")
    return CmdStanModel(stan_file=str(stan_path))


def fit_bayesian_ctrv_model(
    window: observation_window.TrajectoryWindowData,
    *,
    priors: BayesianCTRVPriors | None = None,
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
    """Fit the three-parameter CTRV model with VI or MCMC."""
    inference_method = inference_support.normalize_inference_method(inference_method)
    stan_data = build_stan_data(
        window,
        priors=priors,
        position_observations=position_observations,
    )
    model = compile_bayesian_ctrv_model()

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


def estimate_constant_motion_from_positions(
    time_seconds,
    x_meters,
    y_meters,
) -> tuple[float, float, float]:
    """Estimate numerical initials for speed, initial heading, and turn rate."""
    time_seconds = np.asarray(time_seconds, dtype=float)
    x_meters = np.asarray(x_meters, dtype=float)
    y_meters = np.asarray(y_meters, dtype=float)
    observation_support.validate_matching_position_time_arrays(
        time_seconds,
        x_meters,
        y_meters,
    )
    time_steps = np.diff(time_seconds)
    if np.any(time_steps <= 0):
        raise ValueError("time_seconds must be strictly increasing.")
    delta_x = np.diff(x_meters)
    delta_y = np.diff(y_meters)
    displacement = np.hypot(delta_x, delta_y)
    speed = float(np.median(displacement / time_steps))
    moving = displacement > 1e-9
    if not np.any(moving):
        return 0.0, 0.0, 0.0

    segment_heading = np.unwrap(np.arctan2(delta_y[moving], delta_x[moving]))
    segment_mid_time = (
        0.5 * (time_seconds[:-1] + time_seconds[1:])[moving] - time_seconds[0]
    )
    if segment_heading.size < 2 or np.ptp(segment_mid_time) <= 0:
        turn_rate = 0.0
        heading_initial = float(segment_heading[0])
    else:
        turn_rate, heading_initial = np.polyfit(
            segment_mid_time,
            segment_heading,
            deg=1,
        )
    heading_initial = float(
        np.arctan2(np.sin(heading_initial), np.cos(heading_initial))
    )
    return speed, heading_initial, float(turn_rate)


def _default_initial_values(stan_data: Mapping[str, Any], *, seed: int):
    """Create seeded initials from the selected position history."""
    generator = np.random.default_rng(seed)
    speed, heading_initial, turn_rate = estimate_constant_motion_from_positions(
        stan_data["time_observed"],
        stan_data["x_observed"],
        stan_data["y_observed"],
    )
    speed_jitter = 0.02 * float(stan_data["speed_prior_scale"])
    turn_jitter = 0.02 * float(stan_data["turn_rate_prior_scale"])
    angle_limit = np.pi - 1e-6
    return {
        "speed": float(
            max(
                speed + generator.normal(0.0, speed_jitter),
                SPEED_INITIAL_LOWER_MPS,
            )
        ),
        "heading_initial": float(
            np.clip(
                heading_initial + generator.normal(0.0, 0.01),
                -angle_limit,
                angle_limit,
            )
        ),
        "turn_rate": float(turn_rate + generator.normal(0.0, turn_jitter)),
        "sigma_position_observation": float(
            1.0 / stan_data["sigma_position_observation_prior_rate"]
        ),
    }


def _validate_time_arrays(time_observed, time_prediction) -> None:
    """Validate selected history and future timestamps."""
    observation_support.validate_finite_vector("time_observed", time_observed)
    observation_support.validate_finite_vector("time_prediction", time_prediction)
    if np.any(np.diff(time_observed) <= 0):
        raise ValueError("Observed timestamps must be strictly increasing.")
    if time_prediction[0] <= time_observed[-1] or np.any(np.diff(time_prediction) <= 0):
        raise ValueError(
            "Prediction timestamps must be strictly increasing and follow "
            "the observed timestamps."
        )


def _prediction_samples(fit: Any, variable_name: str, prediction_count: int):
    """Extract one finite posterior prediction matrix."""
    samples = reporting.posterior_variable_samples(fit, variable_name)
    if samples.ndim != 2 or samples.shape[1] != prediction_count:
        raise ValueError(
            f"Posterior variable {variable_name!r} has an unexpected shape."
        )
    if samples.shape[0] == 0 or not np.all(np.isfinite(samples)):
        raise ValueError(
            f"Posterior variable {variable_name!r} must contain finite draws."
        )
    return samples
