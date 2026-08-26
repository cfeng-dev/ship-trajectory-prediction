"""Parametric Bayesian CTRV model fitted with VI or MCMC."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import NormalDist
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
MIN_OBSERVATION_COUNT = 3
SPEED_INITIAL_LOWER_MPS = 0.001

PositionObservations = observation_support.PositionObservations
simulate_position_observations = observation_support.simulate_position_observations
variational_converged = inference_support.variational_converged

PARAMETER_NAMES = (
    "speed",
    "heading_initial",
    "turn_rate",
    "sigma_motion_process",
    "sigma_position_observation",
)

_LOG_SPEED_INDEX = 0
_HEADING_INITIAL_INDEX = 1
_TURN_RATE_INDEX = 2
_LOG_MOTION_PROCESS_INDEX = 3
_LOG_OBSERVATION_NOISE_INDEX = 4
_SEQUENTIAL_PARAMETER_COUNT = 5
_POSITION_COUNT = 2
_NUMERICAL_VARIANCE_FLOOR = 1e-9


@dataclass(frozen=True, slots=True)
class BayesianCTRVPriors:
    """Ship-independent priors for constant CTRV motion and observation noise."""

    speed_prior_upper_mps: float = 20.0
    speed_prior_tail_probability: float = 0.05
    turn_rate_prior_abs_heading_change_deg: float = 45.0
    turn_rate_prior_reference_interval_seconds: float = 10.0
    turn_rate_prior_tail_probability: float = 0.05
    sigma_motion_process_prior_upper_m: float = 20.0
    sigma_motion_process_prior_tail_probability: float = 0.05
    sigma_position_observation_prior_upper_m: float = 20.0
    sigma_position_observation_prior_tail_probability: float = 0.05

    def __post_init__(self) -> None:
        """Validate and normalize all configured prior values."""
        for prior_field in fields(self):
            name = prior_field.name
            value = getattr(self, name)
            if name.endswith("_tail_probability"):
                value = observation_support.validate_finite_scalar(name, value)
                if not 0.0 < value < 1.0:
                    raise ValueError(f"{name} must be strictly between zero and one.")
            elif name == "turn_rate_prior_abs_heading_change_deg":
                value = observation_support.validate_positive_finite(name, value)
                if value > 180.0:
                    raise ValueError(f"{name} must not exceed 180 degrees.")
            else:
                value = observation_support.validate_positive_finite(name, value)
            object.__setattr__(self, name, float(value))

    @property
    def speed_prior_scale(self) -> float:
        """Return the half-normal scale implied by the speed tail statement."""
        return _two_sided_normal_scale(
            self.speed_prior_upper_mps,
            self.speed_prior_tail_probability,
        )

    @property
    def turn_rate_prior_scale(self) -> float:
        """Return the normal turn-rate scale implied by a heading-change tail."""
        absolute_upper_rad_s = (
            np.deg2rad(self.turn_rate_prior_abs_heading_change_deg)
            / self.turn_rate_prior_reference_interval_seconds
        )
        return _two_sided_normal_scale(
            absolute_upper_rad_s,
            self.turn_rate_prior_tail_probability,
        )

    @property
    def sigma_position_observation_prior_rate(self) -> float:
        """Return the exponential rate implied by one prior tail statement."""
        return _exponential_rate_from_tail(
            self.sigma_position_observation_prior_upper_m,
            self.sigma_position_observation_prior_tail_probability,
        )

    @property
    def sigma_motion_process_prior_rate(self) -> float:
        """Return the process-noise exponential rate from its tail statement."""
        return _exponential_rate_from_tail(
            self.sigma_motion_process_prior_upper_m,
            self.sigma_motion_process_prior_tail_probability,
        )


@dataclass(frozen=True, slots=True)
class SequentialCTRVFilterConfig:
    """Numerical settings for online Bayesian CTRV particle filtering."""

    particle_count: int = 4_000
    posterior_draw_count: int = 1_000
    resample_ess_fraction: float = 0.5
    rejuvenation_scale: float = 0.05

    def __post_init__(self) -> None:
        """Validate particle-filter sizes and probabilities."""
        for name in ("particle_count", "posterior_draw_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
            if value < 2:
                raise ValueError(f"{name} must be at least two.")
            object.__setattr__(self, name, int(value))

        ess_fraction = observation_support.validate_positive_finite(
            "resample_ess_fraction",
            self.resample_ess_fraction,
        )
        if ess_fraction > 1.0:
            raise ValueError("resample_ess_fraction must not exceed one.")
        rejuvenation_scale = observation_support.validate_non_negative_finite(
            "rejuvenation_scale",
            self.rejuvenation_scale,
        )
        if rejuvenation_scale >= 1.0:
            raise ValueError("rejuvenation_scale must be smaller than one.")
        object.__setattr__(self, "resample_ess_fraction", ess_fraction)
        object.__setattr__(self, "rejuvenation_scale", rejuvenation_scale)


class SequentialCTRVFit:
    """CmdStan-like draws produced by the online Bayesian CTRV filter."""

    __slots__ = ("_variables",)

    def __init__(self, variables: Mapping[str, Any]):
        normalized = {}
        for name, values in variables.items():
            array = np.asarray(values, dtype=float).copy()
            if array.size == 0 or not np.all(np.isfinite(array)):
                raise ValueError(f"Sequential posterior variable {name!r} is invalid.")
            array.setflags(write=False)
            normalized[str(name)] = array
        self._variables = normalized

    def stan_variable(self, variable_name: str) -> np.ndarray:
        """Return posterior draws through the shared reporting interface."""
        try:
            return self._variables[variable_name]
        except KeyError as error:
            raise ValueError(
                f"Unknown sequential posterior variable: {variable_name!r}."
            ) from error


@dataclass(slots=True)
class SequentialBayesianCTRVFilter:
    """Online Rao-Blackwellized filter for CTRV states and parameters."""

    config: SequentialCTRVFilterConfig
    parameter_particles: np.ndarray
    weights: np.ndarray
    state_means: np.ndarray
    state_variances: np.ndarray
    generator: np.random.Generator
    last_observation_time_seconds: float
    processed_observation_count: int
    resample_count: int = 0
    last_effective_sample_size: float | None = None

    @classmethod
    def initialize(
        cls,
        time_seconds,
        x_observed,
        y_observed,
        *,
        priors: BayesianCTRVPriors,
        config: SequentialCTRVFilterConfig | None = None,
        seed: int = 42,
    ) -> SequentialBayesianCTRVFilter:
        """Initialize the posterior and process every supplied position once."""
        if not isinstance(priors, BayesianCTRVPriors):
            raise TypeError("priors must be a BayesianCTRVPriors instance.")
        if config is None:
            config = SequentialCTRVFilterConfig()
        if not isinstance(config, SequentialCTRVFilterConfig):
            raise TypeError(
                "config must be a SequentialCTRVFilterConfig instance or None."
            )
        seed = observation_support.validate_non_negative_integer("seed", seed)
        time_seconds, x_observed, y_observed = _validate_sequential_observations(
            time_seconds,
            x_observed,
            y_observed,
            minimum_count=1,
        )
        generator = np.random.default_rng(seed)
        particle_count = config.particle_count
        parameter_particles = np.empty(
            (particle_count, _SEQUENTIAL_PARAMETER_COUNT),
            dtype=float,
        )
        speed = np.maximum(
            np.abs(generator.normal(0.0, priors.speed_prior_scale, particle_count)),
            SPEED_INITIAL_LOWER_MPS,
        )
        parameter_particles[:, _LOG_SPEED_INDEX] = np.log(speed)
        parameter_particles[:, _HEADING_INITIAL_INDEX] = generator.uniform(
            -np.pi,
            np.pi,
            particle_count,
        )
        parameter_particles[:, _TURN_RATE_INDEX] = generator.normal(
            0.0,
            priors.turn_rate_prior_scale,
            particle_count,
        )
        motion_process = np.maximum(
            generator.exponential(
                1.0 / priors.sigma_motion_process_prior_rate,
                particle_count,
            ),
            1e-6,
        )
        observation_noise = np.maximum(
            generator.exponential(
                1.0 / priors.sigma_position_observation_prior_rate,
                particle_count,
            ),
            1e-6,
        )
        parameter_particles[:, _LOG_MOTION_PROCESS_INDEX] = np.log(motion_process)
        parameter_particles[:, _LOG_OBSERVATION_NOISE_INDEX] = np.log(observation_noise)
        state_means = np.broadcast_to(
            np.asarray([x_observed[0], y_observed[0]], dtype=float),
            (particle_count, _POSITION_COUNT),
        ).copy()
        state_variances = np.maximum(
            observation_noise**2,
            _NUMERICAL_VARIANCE_FLOOR,
        )
        online_filter = cls(
            config=config,
            parameter_particles=parameter_particles,
            weights=np.full(particle_count, 1.0 / particle_count, dtype=float),
            state_means=state_means,
            state_variances=state_variances,
            generator=generator,
            last_observation_time_seconds=float(time_seconds[0]),
            processed_observation_count=1,
        )
        online_filter.update_many(
            time_seconds[1:],
            x_observed[1:],
            y_observed[1:],
        )
        return online_filter

    @property
    def effective_sample_size(self) -> float:
        """Return the current particle-weight effective sample size."""
        return float(1.0 / np.sum(self.weights**2))

    def update_many(self, time_seconds, x_observed, y_observed) -> None:
        """Update the posterior with matching new positions exactly once."""
        time_seconds, x_observed, y_observed = _validate_sequential_observations(
            time_seconds,
            x_observed,
            y_observed,
            minimum_count=0,
        )
        if time_seconds.size and time_seconds[0] <= self.last_observation_time_seconds:
            raise ValueError(
                "Sequential timestamps must follow processed observations."
            )
        for time_value, x_value, y_value in zip(
            time_seconds,
            x_observed,
            y_observed,
            strict=True,
        ):
            self.update(float(time_value), float(x_value), float(y_value))

    def update(
        self,
        time_seconds: float,
        x_observed: float,
        y_observed: float,
    ) -> None:
        """Propagate particles and condition them on one new position."""
        time_seconds = observation_support.validate_finite_scalar(
            "time_seconds",
            time_seconds,
        )
        if time_seconds <= self.last_observation_time_seconds:
            raise ValueError("time_seconds must follow the previous observation.")
        x_observed = observation_support.validate_finite_scalar(
            "x_observed",
            x_observed,
        )
        y_observed = observation_support.validate_finite_scalar(
            "y_observed",
            y_observed,
        )
        speed, heading_initial, turn_rate, motion_process, observation_noise = (
            _sequential_parameter_values(self.parameter_particles)
        )
        dt = time_seconds - self.last_observation_time_seconds
        predicted_means = _ctrv_particle_step(
            self.state_means,
            speed,
            heading_initial,
            turn_rate,
            dt,
        )
        predicted_variances = self.state_variances + motion_process**2
        innovation_variances = predicted_variances + observation_noise**2
        innovations = np.column_stack(
            (
                x_observed - predicted_means[:, 0],
                y_observed - predicted_means[:, 1],
            )
        )
        log_likelihood = (
            -np.log(2.0 * np.pi)
            - np.log(innovation_variances)
            - 0.5 * np.sum(innovations**2, axis=1) / innovation_variances
        )
        with np.errstate(divide="ignore"):
            log_weights = np.log(self.weights) + log_likelihood
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise RuntimeError("Sequential CTRV particle weights collapsed.")
        self.weights = weights / weight_sum
        kalman_gain = predicted_variances / innovation_variances
        self.state_means = predicted_means + kalman_gain[:, None] * innovations
        self.state_variances = np.maximum(
            (1.0 - kalman_gain) * predicted_variances,
            _NUMERICAL_VARIANCE_FLOOR,
        )
        self.parameter_particles[:, _HEADING_INITIAL_INDEX] = _wrap_angles(
            heading_initial + turn_rate * dt
        )
        self.last_observation_time_seconds = time_seconds
        self.processed_observation_count += 1
        self.last_effective_sample_size = self.effective_sample_size
        if (
            self.last_effective_sample_size
            < self.config.resample_ess_fraction * self.config.particle_count
        ):
            self._resample_and_rejuvenate()

    def forecast(self, future_time_seconds, *, seed: int) -> SequentialCTRVFit:
        """Draw future latent and observed trajectories from the posterior."""
        future_time_seconds = _validate_future_times(
            future_time_seconds,
            after=self.last_observation_time_seconds,
        )
        seed = observation_support.validate_non_negative_integer("seed", seed)
        generator = np.random.default_rng(seed)
        draw_count = self.config.posterior_draw_count
        indices = generator.choice(
            self.config.particle_count,
            size=draw_count,
            replace=True,
            p=self.weights,
        )
        parameters = self.parameter_particles[indices]
        state = self.state_means[indices] + generator.normal(
            0.0,
            np.sqrt(self.state_variances[indices])[:, None],
            size=(draw_count, _POSITION_COUNT),
        )
        speed, heading_initial, turn_rate, motion_process, observation_noise = (
            _sequential_parameter_values(parameters)
        )
        prediction_count = future_time_seconds.size
        x_prediction = np.empty((draw_count, prediction_count), dtype=float)
        y_prediction = np.empty_like(x_prediction)
        x_observation_prediction = np.empty_like(x_prediction)
        y_observation_prediction = np.empty_like(x_prediction)
        current_time = self.last_observation_time_seconds
        heading_current = heading_initial.copy()
        for prediction_index, prediction_time in enumerate(future_time_seconds):
            dt = float(prediction_time - current_time)
            expected_position = _ctrv_particle_step(
                state,
                speed,
                heading_current,
                turn_rate,
                dt,
            )
            state = expected_position + generator.normal(
                0.0,
                motion_process[:, None],
                size=expected_position.shape,
            )
            x_prediction[:, prediction_index] = state[:, 0]
            y_prediction[:, prediction_index] = state[:, 1]
            observation_innovation = generator.normal(
                0.0,
                observation_noise[:, None],
                size=(draw_count, _POSITION_COUNT),
            )
            x_observation_prediction[:, prediction_index] = (
                state[:, 0] + observation_innovation[:, 0]
            )
            y_observation_prediction[:, prediction_index] = (
                state[:, 1] + observation_innovation[:, 1]
            )
            heading_current = _wrap_angles(heading_current + turn_rate * dt)
            current_time = float(prediction_time)

        return SequentialCTRVFit(
            {
                "speed": speed,
                "heading_initial": heading_initial,
                "turn_rate": turn_rate,
                "sigma_motion_process": motion_process,
                "sigma_position_observation": observation_noise,
                "x_prediction": x_prediction,
                "y_prediction": y_prediction,
                "x_observation_prediction": x_observation_prediction,
                "y_observation_prediction": y_observation_prediction,
            }
        )

    def _resample_and_rejuvenate(self) -> None:
        """Resample states and apply Liu-West-style parameter rejuvenation."""
        adjusted_parameters, parameter_mean = _unwrap_sequential_parameters(
            self.parameter_particles,
            self.weights,
        )
        centered_parameters = adjusted_parameters - parameter_mean
        parameter_covariance = (
            centered_parameters.T * self.weights
        ) @ centered_parameters
        indices = _systematic_resample(self.weights, self.generator)
        selected_parameters = adjusted_parameters[indices]
        self.state_means = self.state_means[indices].copy()
        self.state_variances = self.state_variances[indices].copy()
        rejuvenation_scale = self.config.rejuvenation_scale
        shrinkage = np.sqrt(1.0 - rejuvenation_scale**2)
        parameter_cholesky = _regularized_cholesky(parameter_covariance)
        parameter_noise = (
            self.generator.normal(
                size=(self.config.particle_count, _SEQUENTIAL_PARAMETER_COUNT)
            )
            @ parameter_cholesky.T
        )
        self.parameter_particles = (
            shrinkage * selected_parameters
            + (1.0 - shrinkage) * parameter_mean
            + rejuvenation_scale * parameter_noise
        )
        self.parameter_particles[:, _HEADING_INITIAL_INDEX] = _wrap_angles(
            self.parameter_particles[:, _HEADING_INITIAL_INDEX]
        )
        if not np.all(np.isfinite(self.parameter_particles)):
            raise RuntimeError("Sequential CTRV parameter rejuvenation became invalid.")
        self.weights = np.full(
            self.config.particle_count,
            1.0 / self.config.particle_count,
            dtype=float,
        )
        self.resample_count += 1


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
        "sigma_motion_process_prior_rate": priors.sigma_motion_process_prior_rate,
        "sigma_position_observation_prior_rate": (
            priors.sigma_position_observation_prior_rate
        ),
        "N_prediction": window.prediction_count,
        "time_prediction": time_prediction,
        "speed_prior_scale": priors.speed_prior_scale,
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
    """Fit the latent-state CTRV model with VI or MCMC."""
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
    speed, heading_initial, turn_rate = estimate_constant_motion_from_positions(
        stan_data["time_observed"],
        x_true,
        y_true,
    )
    speed_jitter = 0.02 * float(stan_data["speed_prior_scale"])
    turn_jitter = 0.02 * float(stan_data["turn_rate_prior_scale"])
    angle_limit = np.pi - 1e-6
    return {
        "x_true": x_true,
        "y_true": y_true,
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
        "sigma_motion_process": float(
            0.5 / stan_data["sigma_motion_process_prior_rate"]
        ),
        "sigma_position_observation": float(observation_noise_initial),
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


def _two_sided_normal_scale(absolute_upper: float, tail_probability: float) -> float:
    """Return a zero-centered normal scale from a two-sided tail statement."""
    quantile = NormalDist().inv_cdf(1.0 - tail_probability / 2.0)
    return float(absolute_upper / quantile)


def _exponential_rate_from_tail(upper: float, tail_probability: float) -> float:
    """Return an exponential rate from one upper-tail probability statement."""
    return float(-np.log(tail_probability) / upper)


def _smooth_position_initials(observed) -> np.ndarray:
    """Return a lightly smoothed position history for latent-state initials."""
    observed = np.asarray(observed, dtype=float)
    initial = observed.copy()
    initial[1:-1] = 0.25 * observed[:-2] + 0.5 * observed[1:-1] + 0.25 * observed[2:]
    return initial


def _validate_sequential_observations(
    time_seconds,
    x_observed,
    y_observed,
    *,
    minimum_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matching finite time and position arrays for online updates."""
    time_seconds = np.asarray(time_seconds, dtype=float)
    x_observed = np.asarray(x_observed, dtype=float)
    y_observed = np.asarray(y_observed, dtype=float)
    if (
        time_seconds.ndim != 1
        or x_observed.shape != time_seconds.shape
        or y_observed.shape != time_seconds.shape
        or time_seconds.size < minimum_count
        or not np.all(np.isfinite(time_seconds))
        or not np.all(np.isfinite(x_observed))
        or not np.all(np.isfinite(y_observed))
        or np.any(np.diff(time_seconds) <= 0.0)
    ):
        raise ValueError(
            "Sequential time/x/y observations must be matching finite vectors "
            f"with at least {minimum_count} values and increasing timestamps."
        )
    return time_seconds, x_observed, y_observed


def _validate_future_times(future_time_seconds, *, after: float) -> np.ndarray:
    """Return finite increasing forecast times after the current filter state."""
    future_time_seconds = np.asarray(future_time_seconds, dtype=float)
    if (
        future_time_seconds.ndim != 1
        or future_time_seconds.size < 1
        or not np.all(np.isfinite(future_time_seconds))
        or future_time_seconds[0] <= after
        or np.any(np.diff(future_time_seconds) <= 0.0)
    ):
        raise ValueError(
            "future_time_seconds must be a finite increasing vector after the "
            "latest observation."
        )
    return future_time_seconds


def _sequential_parameter_values(parameter_particles: np.ndarray):
    """Transform unconstrained particle columns to CTRV parameter arrays."""
    return (
        np.exp(parameter_particles[:, _LOG_SPEED_INDEX]),
        parameter_particles[:, _HEADING_INITIAL_INDEX],
        parameter_particles[:, _TURN_RATE_INDEX],
        np.exp(parameter_particles[:, _LOG_MOTION_PROCESS_INDEX]),
        np.exp(parameter_particles[:, _LOG_OBSERVATION_NOISE_INDEX]),
    )


def _ctrv_particle_step(
    state_particles: np.ndarray,
    speed: np.ndarray,
    heading: np.ndarray,
    turn_rate: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Advance all particle positions with one stable vectorized CTRV step."""
    half_turn = 0.5 * turn_rate * dt
    distance = speed * dt * np.sinc(half_turn / np.pi)
    midpoint_heading = heading + half_turn
    return state_particles + np.column_stack(
        (
            distance * np.cos(midpoint_heading),
            distance * np.sin(midpoint_heading),
        )
    )


def _unwrap_sequential_parameters(
    parameter_particles: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unwrap heading particles around their weighted circular mean."""
    heading = parameter_particles[:, _HEADING_INITIAL_INDEX]
    heading_mean = np.arctan2(
        np.sum(weights * np.sin(heading)),
        np.sum(weights * np.cos(heading)),
    )
    adjusted = parameter_particles.copy()
    adjusted[:, _HEADING_INITIAL_INDEX] = heading_mean + _wrap_angles(
        heading - heading_mean
    )
    mean = np.sum(weights[:, None] * adjusted, axis=0)
    return adjusted, mean


def _systematic_resample(
    weights: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Return systematic-resampling parent indices."""
    positions = (generator.random() + np.arange(len(weights))) / len(weights)
    cumulative_weights = np.cumsum(weights)
    cumulative_weights[-1] = 1.0
    return np.searchsorted(cumulative_weights, positions, side="right")


def _regularized_cholesky(covariance: np.ndarray) -> np.ndarray:
    """Return a Cholesky factor after symmetric eigenvalue regularization."""
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = max(float(np.max(eigenvalues)), _NUMERICAL_VARIANCE_FLOOR)
    eigenvalues = np.maximum(eigenvalues, largest * 1e-10)
    regularized = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return np.linalg.cholesky(regularized)


def _wrap_angles(values: np.ndarray) -> np.ndarray:
    """Wrap radians to the closed-open interval [-pi, pi)."""
    return (values + np.pi) % (2.0 * np.pi) - np.pi


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
