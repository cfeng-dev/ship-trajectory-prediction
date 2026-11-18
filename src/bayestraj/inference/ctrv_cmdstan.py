"""Batch inference and prior-predictive forecasts for the Bayesian CTRV model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from cmdstanpy import CmdStanMCMC, CmdStanModel, CmdStanVB

import bayestraj.inference.cmdstan as cmdstan_inference
import bayestraj.inference.particle_utils as particle_utils
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.numeric_validation as numeric_validation
import bayestraj.observations.position as observation_support
import bayestraj.observations.window as observation_window
from bayestraj.models import ctrv as ctrv_dynamics


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
    algorithm: str = cmdstan_inference.DEFAULT_VI_ALGORITHM,
    iter: int = cmdstan_inference.DEFAULT_VI_ITER,
    grad_samples: int = cmdstan_inference.DEFAULT_VI_GRAD_SAMPLES,
    elbo_samples: int = cmdstan_inference.DEFAULT_VI_ELBO_SAMPLES,
    eta: float = cmdstan_inference.DEFAULT_VI_ETA,
    adapt_iter: int = cmdstan_inference.DEFAULT_VI_ADAPT_ITER,
    tol_rel_obj: float = cmdstan_inference.DEFAULT_VI_TOL_REL_OBJ,
    eval_elbo: int = cmdstan_inference.DEFAULT_VI_EVAL_ELBO,
    draws: int = cmdstan_inference.DEFAULT_VI_DRAWS,
    chains: int = 4,
    parallel_chains: int | None = None,
    iter_warmup: int = 1_000,
    iter_sampling: int = 2_000,
    adapt_delta: float = 0.95,
    max_treedepth: int = 10,
    seed: int = 42,
    inits: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | float | None = None,
    require_converged: bool = True,
    show_console: bool = False,
    variational_options: Mapping[str, Any] | None = None,
    mcmc_options: Mapping[str, Any] | None = None,
) -> CmdStanVB | CmdStanMCMC:
    """Fit the noncentered latent-state CTRV model with VI or MCMC."""
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
    """Express seeded motion-state initials as standardized innovations."""
    generator = np.random.default_rng(seed)
    observed_mask = np.asarray(
        stan_data.get("observed_mask", np.ones(stan_data["N_history"])), dtype=bool
    )
    time = np.asarray(stan_data["time_observed"])
    # Missing-cycle placeholders never enter numerical initialization. Interpolation
    # here constructs optimizer initials only; it does not create observations.
    x_true = _smooth_position_initials(
        np.interp(
            time,
            time[observed_mask],
            np.asarray(stan_data["x_observed"])[observed_mask],
        )
    )
    y_true = _smooth_position_initials(
        np.interp(
            time,
            time[observed_mask],
            np.asarray(stan_data["y_observed"])[observed_mask],
        )
    )
    observation_noise_excess_initial = 1.0 / float(
        stan_data["sigma_position_observation_prior_rate"]
    )
    latent_jitter_scale = 0.02 * observation_noise_excess_initial
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
    transition_count = x_true.size - 1
    speed_state = np.maximum(
        speed + generator.normal(0.0, speed_jitter, transition_count),
        # Stan maps a start on the lower bound to -inf when unconstraining.
        1e-6,
    )
    heading_initial = float(
        np.clip(
            heading_initial + generator.normal(0.0, 0.01),
            -angle_limit,
            angle_limit,
        )
    )
    turn_rate_state = turn_rate + generator.normal(0.0, turn_jitter, transition_count)
    sigma_speed_process = float(0.5 / stan_data["sigma_speed_process_prior_rate"])
    sigma_turn_rate_process = float(
        0.5 / stan_data["sigma_turn_rate_process_prior_rate"]
    )
    time_scale = np.sqrt(
        np.diff(time[:-1]) / stan_data["process_reference_interval_seconds"]
    )
    return {
        "x_initial": float(x_true[0]),
        "y_initial": float(y_true[0]),
        "speed_initial": float(speed_state[0]),
        "speed_innovation": np.diff(speed_state) / (sigma_speed_process * time_scale),
        "heading_initial": heading_initial,
        "turn_rate_initial": float(turn_rate_state[0]),
        "turn_rate_innovation": np.diff(turn_rate_state)
        / (sigma_turn_rate_process * time_scale),
        "sigma_position_observation_excess": float(observation_noise_excess_initial),
        "sigma_speed_process": sigma_speed_process,
        "sigma_turn_rate_process": sigma_turn_rate_process,
    }


def _smooth_position_initials(observed) -> np.ndarray:
    """Return a lightly smoothed position history for latent-state initials."""
    observed = np.asarray(observed, dtype=float)
    initial = observed.copy()
    initial[1:-1] = 0.25 * observed[:-2] + 0.5 * observed[1:-1] + 0.25 * observed[2:]
    return initial


def _physical_fit_draws(fit: object) -> dict[str, np.ndarray]:
    return {
        "x_at_origin": _posterior_origin_position(fit, "x"),
        "y_at_origin": _posterior_origin_position(fit, "y"),
        "speed_at_origin": _posterior_vector(fit, "speed_at_origin"),
        "heading_at_origin": _posterior_vector(fit, "heading_at_origin"),
        "turn_rate_at_origin": _posterior_vector(fit, "turn_rate_at_origin"),
        "sigma_position_observation": _positive_posterior_vector(
            fit, "sigma_position_observation"
        ),
        "sigma_speed_process": _positive_posterior_vector(fit, "sigma_speed_process"),
        "sigma_turn_rate_process": _positive_posterior_vector(
            fit, "sigma_turn_rate_process"
        ),
    }


def _posterior_draws(fit: object, name: str) -> np.ndarray:
    if hasattr(fit, "variational_sample"):
        return np.asarray(fit.stan_variable(name, mean=False), dtype=float)
    return np.asarray(fit.stan_variable(name), dtype=float)


def _posterior_vector(fit: object, name: str) -> np.ndarray:
    try:
        values = _posterior_draws(fit, name)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Posterior variable {name!r} is unavailable.") from error
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"Posterior variable {name!r} must be one-dimensional.")
    if vector.size < 2:
        raise ValueError(
            f"Posterior variable {name!r} must contain at least two draws."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"Posterior variable {name!r} must contain only finite draws.")
    return vector


def _positive_posterior_vector(fit: object, name: str) -> np.ndarray:
    vector = _posterior_vector(fit, name)
    if np.any(vector <= 0.0):
        raise ValueError(f"Posterior variable {name!r} must be strictly positive.")
    return vector


def _posterior_origin_position(fit: object, axis: str) -> np.ndarray:
    explicit_name = f"{axis}_at_origin"
    try:
        return _posterior_vector(fit, explicit_name)
    except ValueError as explicit_error:
        state_name = f"{axis}_state"
        try:
            states = _posterior_draws(fit, state_name)
        except (KeyError, ValueError) as error:
            raise explicit_error from error
        if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 1:
            raise ValueError(
                f"Posterior variable {state_name!r} must be a non-empty draw matrix."
            ) from None
        if not np.all(np.isfinite(states)):
            raise ValueError(
                f"Posterior variable {state_name!r} must contain only finite draws."
            ) from None
        return states[:, -1]


def _forecast_from_origin_draws(
    draws: dict[str, np.ndarray],
    *,
    origin_time_seconds: float,
    future_time_seconds,
    generator: np.random.Generator,
) -> particle_utils.SequentialCTRVFit:
    future_times = particle_utils.validate_future_times(
        future_time_seconds,
        after=origin_time_seconds,
    )
    draw_count = draws["speed_at_origin"].size
    states = np.column_stack(
        (
            draws["x_at_origin"],
            draws["y_at_origin"],
            draws["speed_at_origin"],
            draws["heading_at_origin"],
            draws["turn_rate_at_origin"],
        )
    )
    speed_at_origin = states[:, ctrv_dynamics.STATE_SPEED_INDEX].copy()
    heading_at_origin = states[:, ctrv_dynamics.STATE_HEADING_INDEX].copy()
    turn_rate_at_origin = states[:, ctrv_dynamics.STATE_TURN_RATE_INDEX].copy()
    prediction_count = future_times.size
    x_prediction = np.empty((draw_count, prediction_count))
    y_prediction = np.empty_like(x_prediction)
    x_observation_prediction = np.empty_like(x_prediction)
    y_observation_prediction = np.empty_like(x_prediction)
    current_time = origin_time_seconds
    for prediction_index, prediction_time in enumerate(future_times):
        interval = float(prediction_time - current_time)
        states = ctrv_dynamics.transition_states(states, interval)
        x_prediction[:, prediction_index] = states[:, ctrv_dynamics.STATE_X_INDEX]
        y_prediction[:, prediction_index] = states[:, ctrv_dynamics.STATE_Y_INDEX]
        observation_innovation = generator.normal(
            0.0,
            draws["sigma_position_observation"][:, None],
            size=(draw_count, 2),
        )
        x_observation_prediction[:, prediction_index] = (
            states[:, ctrv_dynamics.STATE_X_INDEX] + observation_innovation[:, 0]
        )
        y_observation_prediction[:, prediction_index] = (
            states[:, ctrv_dynamics.STATE_Y_INDEX] + observation_innovation[:, 1]
        )
        process_scale = ctrv_dynamics.process_time_scale(interval)
        states[:, ctrv_dynamics.STATE_SPEED_INDEX] += generator.normal(
            0.0,
            draws["sigma_speed_process"] * process_scale,
        )
        states[:, ctrv_dynamics.STATE_TURN_RATE_INDEX] += generator.normal(
            0.0,
            draws["sigma_turn_rate_process"] * process_scale,
        )
        ctrv_dynamics.normalize_states(states)
        current_time = float(prediction_time)

    return particle_utils.SequentialCTRVFit(
        {
            "speed_at_origin": speed_at_origin,
            "x_at_origin": draws["x_at_origin"],
            "y_at_origin": draws["y_at_origin"],
            "heading_at_origin": heading_at_origin,
            "turn_rate_at_origin": turn_rate_at_origin,
            "sigma_position_observation": draws["sigma_position_observation"],
            "sigma_speed_process": draws["sigma_speed_process"],
            "sigma_turn_rate_process": draws["sigma_turn_rate_process"],
            "x_prediction": x_prediction,
            "y_prediction": y_prediction,
            "x_observation_prediction": x_observation_prediction,
            "y_observation_prediction": y_observation_prediction,
        }
    )


def forecast_ctrv_prior_predictive(
    *,
    x_at_origin: float,
    y_at_origin: float,
    origin_time_seconds: float,
    future_time_seconds,
    priors: bayesian_model.BayesianCTRVPriors,
    draw_count: int,
    seed: int,
) -> particle_utils.SequentialCTRVFit:
    """Draw a CTRV forecast from the configured priors at one observed origin."""
    if not isinstance(priors, bayesian_model.BayesianCTRVPriors):
        raise TypeError("priors must be a BayesianCTRVPriors instance.")
    if (
        isinstance(draw_count, bool)
        or not isinstance(draw_count, (int, np.integer))
        or draw_count < 2
    ):
        raise ValueError("draw_count must be an integer greater than or equal to 2.")
    seed = numeric_validation.validate_non_negative_integer("seed", seed)
    x_origin = _forecast_origin_scalar(x_at_origin, name="x_at_origin")
    y_origin = _forecast_origin_scalar(y_at_origin, name="y_at_origin")
    origin_time = _forecast_origin_scalar(
        origin_time_seconds, name="origin_time_seconds"
    )
    generator = np.random.default_rng(seed)
    draw_count = int(draw_count)
    draws = {
        "x_at_origin": np.full(draw_count, x_origin),
        "y_at_origin": np.full(draw_count, y_origin),
        "speed_at_origin": np.abs(
            generator.normal(0.0, priors.speed_prior_scale, draw_count)
        ),
        "heading_at_origin": generator.uniform(-np.pi, np.pi, draw_count),
        "turn_rate_at_origin": generator.normal(
            0.0,
            priors.turn_rate_prior_scale,
            draw_count,
        ),
        "sigma_position_observation": generator.exponential(
            1.0 / priors.sigma_position_observation_prior_rate,
            draw_count,
        ),
        "sigma_speed_process": generator.exponential(
            1.0 / priors.sigma_speed_process_prior_rate,
            draw_count,
        ),
        "sigma_turn_rate_process": generator.exponential(
            1.0 / priors.sigma_turn_rate_process_prior_rate,
            draw_count,
        ),
    }
    return _forecast_from_origin_draws(
        draws,
        origin_time_seconds=origin_time,
        future_time_seconds=future_time_seconds,
        generator=generator,
    )


def _forecast_origin_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite scalar.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite scalar.") from error
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite scalar.")
    return numeric
