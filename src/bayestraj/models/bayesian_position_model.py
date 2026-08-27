"""Bayesian latent-position autoregressive measurement-error model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
from cmdstanpy import CmdStanMCMC, CmdStanModel, CmdStanVB

import bayestraj.models.bayesian_inference as inference_support
import bayestraj.models.bayesian_observations as observation_support
import bayestraj.models.paths as model_paths
import bayestraj.observations.window as observation_window

STAN_FILE = model_paths.stan_path("models/bayesian_position_model.stan")
MIN_OBSERVATION_COUNT = 5
POSITION_MODEL_REFERENCE_INTERVAL_SECONDS = 10.0
DEFAULT_MEANFIELD_GRAD_SAMPLES = inference_support.DEFAULT_MEANFIELD_GRAD_SAMPLES
DEFAULT_VI_ADAPT_ITER = inference_support.DEFAULT_VI_ADAPT_ITER

PositionObservations = observation_support.PositionObservations
simulate_position_observations = observation_support.simulate_position_observations
variational_converged = inference_support.variational_converged

NOISE_PARAMETER_NAMES = (
    "sigma_position_observation",
    "sigma_motion_residual",
)
PARAMETER_NAMES = (
    "log_displacement_scale_rate",
    "rotation_rate",
    "displacement_scale_at_reference",
    "rotation_angle_at_reference",
    "sigma_position_observation",
    "sigma_motion_residual",
)

_LOG_DISPLACEMENT_SCALE_RATE_INDEX = 0
_ROTATION_RATE_INDEX = 1
_LOG_OBSERVATION_NOISE_INDEX = 2
_LOG_MOTION_NOISE_INDEX = 3
_PARAMETER_COUNT = 4
_STATE_COUNT = 4
_POSITION_COUNT = 2
_NUMERICAL_VARIANCE_FLOOR = 1e-9


@dataclass(frozen=True, slots=True)
class BayesianPositionModelPriors:
    """Ship-independent priors stated over one reference interval."""

    displacement_scale_prior_factor: float = 2.0
    displacement_scale_prior_tail_probability: float = 0.05
    rotation_angle_prior_abs_upper_deg: float = 45.0
    rotation_angle_prior_tail_probability: float = 0.05
    sigma_position_observation_prior_upper_m: float = 20.0
    sigma_position_observation_prior_tail_probability: float = 0.05
    sigma_motion_residual_prior_upper_m: float = 20.0
    sigma_motion_residual_prior_tail_probability: float = 0.05

    def __post_init__(self) -> None:
        """Validate every configured prior scale."""
        for prior_field in fields(self):
            name = prior_field.name
            value = getattr(self, name)
            if name.endswith("_tail_probability"):
                value = observation_support.validate_finite_scalar(name, value)
                if not 0.0 < value < 1.0:
                    raise ValueError(f"{name} must be strictly between zero and one.")
            elif name == "displacement_scale_prior_factor":
                value = observation_support.validate_positive_finite(name, value)
                if value <= 1.0:
                    raise ValueError(f"{name} must be greater than one.")
            elif name == "rotation_angle_prior_abs_upper_deg":
                value = observation_support.validate_positive_finite(name, value)
                if value > 180.0:
                    raise ValueError(f"{name} must not exceed 180 degrees.")
            else:
                value = observation_support.validate_positive_finite(name, value)
            object.__setattr__(self, name, float(value))

    @property
    def log_displacement_scale_rate_prior_scale(self) -> float:
        """Return the log motion-scale rate prior SD in inverse seconds."""
        reference_scale = _two_sided_normal_scale(
            np.log(self.displacement_scale_prior_factor),
            self.displacement_scale_prior_tail_probability,
        )
        return reference_scale / POSITION_MODEL_REFERENCE_INTERVAL_SECONDS

    @property
    def rotation_rate_prior_scale(self) -> float:
        """Return the rotation-rate prior SD in radians per second."""
        reference_scale = _two_sided_normal_scale(
            np.deg2rad(self.rotation_angle_prior_abs_upper_deg),
            self.rotation_angle_prior_tail_probability,
        )
        return reference_scale / POSITION_MODEL_REFERENCE_INTERVAL_SECONDS

    @property
    def sigma_position_observation_prior_rate(self) -> float:
        """Return the observation-noise exponential rate from its tail statement."""
        return _exponential_rate_from_tail(
            self.sigma_position_observation_prior_upper_m,
            self.sigma_position_observation_prior_tail_probability,
        )

    @property
    def sigma_motion_residual_prior_rate(self) -> float:
        """Return the motion-noise exponential rate from its tail statement."""
        return _exponential_rate_from_tail(
            self.sigma_motion_residual_prior_upper_m,
            self.sigma_motion_residual_prior_tail_probability,
        )


@dataclass(frozen=True, slots=True)
class SequentialPositionFilterConfig:
    """Numerical settings for online Rao-Blackwellized particle filtering."""

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


class SequentialPositionFit:
    """CmdStan-like posterior draws produced by the online position filter."""

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
        """Return posterior draws with the interface used by shared reporting."""
        try:
            return self._variables[variable_name]
        except KeyError as error:
            raise ValueError(
                f"Unknown sequential posterior variable: {variable_name!r}."
            ) from error


@dataclass(slots=True)
class SequentialBayesianPositionFilter:
    """Online Bayesian filter for position states and static motion parameters."""

    config: SequentialPositionFilterConfig
    parameter_particles: np.ndarray
    weights: np.ndarray
    state_means: np.ndarray
    state_covariances: np.ndarray
    generator: np.random.Generator
    last_observation_time_seconds: float
    previous_interval_seconds: float
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
        priors: BayesianPositionModelPriors,
        config: SequentialPositionFilterConfig | None = None,
        seed: int = 42,
    ) -> SequentialBayesianPositionFilter:
        """Initialize the online posterior and process each supplied point once."""
        if not isinstance(priors, BayesianPositionModelPriors):
            raise TypeError("priors must be a BayesianPositionModelPriors instance.")
        if config is None:
            config = SequentialPositionFilterConfig()
        if not isinstance(config, SequentialPositionFilterConfig):
            raise TypeError(
                "config must be a SequentialPositionFilterConfig instance or None."
            )
        seed = observation_support.validate_non_negative_integer("seed", seed)
        time_seconds, x_observed, y_observed = _validate_sequential_observations(
            time_seconds,
            x_observed,
            y_observed,
            minimum_count=2,
        )
        generator = np.random.default_rng(seed)
        particle_count = config.particle_count
        parameter_particles = np.empty(
            (particle_count, _PARAMETER_COUNT),
            dtype=float,
        )
        parameter_particles[:, _LOG_DISPLACEMENT_SCALE_RATE_INDEX] = generator.normal(
            0.0,
            priors.log_displacement_scale_rate_prior_scale,
            particle_count,
        )
        parameter_particles[:, _ROTATION_RATE_INDEX] = generator.normal(
            0.0,
            priors.rotation_rate_prior_scale,
            particle_count,
        )
        observation_noise = np.maximum(
            generator.exponential(
                1.0 / priors.sigma_position_observation_prior_rate,
                particle_count,
            ),
            1e-6,
        )
        motion_noise = np.maximum(
            generator.exponential(
                1.0 / priors.sigma_motion_residual_prior_rate,
                particle_count,
            ),
            1e-6,
        )
        parameter_particles[:, _LOG_OBSERVATION_NOISE_INDEX] = np.log(observation_noise)
        parameter_particles[:, _LOG_MOTION_NOISE_INDEX] = np.log(motion_noise)

        state_mean = np.asarray(
            [
                x_observed[1],
                y_observed[1],
                x_observed[1] - x_observed[0],
                y_observed[1] - y_observed[0],
            ],
            dtype=float,
        )
        state_means = np.broadcast_to(
            state_mean,
            (particle_count, _STATE_COUNT),
        ).copy()
        state_covariance_template = np.asarray(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 2.0, 0.0],
                [0.0, 1.0, 0.0, 2.0],
            ],
            dtype=float,
        )
        state_covariances = (
            observation_noise[:, None, None] ** 2
            * state_covariance_template[None, :, :]
        )
        state_covariances += _NUMERICAL_VARIANCE_FLOOR * np.eye(_STATE_COUNT)[None]
        online_filter = cls(
            config=config,
            parameter_particles=parameter_particles,
            weights=np.full(particle_count, 1.0 / particle_count, dtype=float),
            state_means=state_means,
            state_covariances=state_covariances,
            generator=generator,
            last_observation_time_seconds=float(time_seconds[1]),
            previous_interval_seconds=float(time_seconds[1] - time_seconds[0]),
            processed_observation_count=2,
        )
        online_filter.update_many(
            time_seconds[2:],
            x_observed[2:],
            y_observed[2:],
        )
        return online_filter

    @property
    def effective_sample_size(self) -> float:
        """Return the current particle-weight effective sample size."""
        return float(1.0 / np.sum(self.weights**2))

    def update_many(self, time_seconds, x_observed, y_observed) -> None:
        """Update the online posterior with matching new positions exactly once."""
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
        """Condition particles and their Kalman states on one new observation."""
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
        current_interval_seconds = time_seconds - self.last_observation_time_seconds
        transition, process_covariance = _sequential_transition_terms(
            self.parameter_particles,
            current_interval_seconds=current_interval_seconds,
            previous_interval_seconds=self.previous_interval_seconds,
        )
        predicted_means = np.einsum(
            "nij,nj->ni",
            transition,
            self.state_means,
        )
        predicted_covariances = (
            transition @ self.state_covariances @ np.swapaxes(transition, 1, 2)
            + process_covariance
        )

        observation_noise_variance = np.exp(
            2.0 * self.parameter_particles[:, _LOG_OBSERVATION_NOISE_INDEX]
        )
        innovation_covariances = predicted_covariances[:, :2, :2].copy()
        innovation_covariances[:, 0, 0] += observation_noise_variance
        innovation_covariances[:, 1, 1] += observation_noise_variance
        innovation_inverses, log_determinants = _invert_two_by_two_matrices(
            innovation_covariances
        )
        innovations = np.column_stack(
            (
                x_observed - predicted_means[:, 0],
                y_observed - predicted_means[:, 1],
            )
        )
        squared_distances = np.einsum(
            "ni,nij,nj->n",
            innovations,
            innovation_inverses,
            innovations,
        )
        log_likelihood = -np.log(2.0 * np.pi) - 0.5 * (
            log_determinants + squared_distances
        )
        with np.errstate(divide="ignore"):
            log_weights = np.log(self.weights) + log_likelihood
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise RuntimeError("Sequential particle weights collapsed.")
        self.weights = weights / weight_sum

        state_observation_cross_covariance = predicted_covariances[:, :, :2]
        kalman_gain = state_observation_cross_covariance @ innovation_inverses
        self.state_means = predicted_means + np.einsum(
            "nij,nj->ni",
            kalman_gain,
            innovations,
        )
        self.state_covariances = predicted_covariances - (
            kalman_gain @ np.swapaxes(state_observation_cross_covariance, 1, 2)
        )
        self.state_covariances = 0.5 * (
            self.state_covariances + np.swapaxes(self.state_covariances, 1, 2)
        )
        self.state_covariances += _NUMERICAL_VARIANCE_FLOOR * np.eye(_STATE_COUNT)[None]
        self.last_observation_time_seconds = time_seconds
        self.previous_interval_seconds = current_interval_seconds
        self.processed_observation_count += 1
        self.last_effective_sample_size = self.effective_sample_size

        if (
            self.last_effective_sample_size
            < self.config.resample_ess_fraction * self.config.particle_count
        ):
            self._resample_and_rejuvenate()

    def forecast(self, future_time_seconds, *, seed: int) -> SequentialPositionFit:
        """Draw latent and observation trajectories from the current posterior."""
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
        states = _sample_gaussian_states(
            self.state_means[indices],
            self.state_covariances[indices],
            generator,
        )
        log_displacement_scale_rate = parameters[:, _LOG_DISPLACEMENT_SCALE_RATE_INDEX]
        rotation_rate = parameters[:, _ROTATION_RATE_INDEX]
        displacement_scale_at_reference = np.exp(
            log_displacement_scale_rate * POSITION_MODEL_REFERENCE_INTERVAL_SECONDS
        )
        rotation_angle_at_reference = (
            rotation_rate * POSITION_MODEL_REFERENCE_INTERVAL_SECONDS
        )
        observation_noise = np.exp(parameters[:, _LOG_OBSERVATION_NOISE_INDEX])
        motion_noise = np.exp(parameters[:, _LOG_MOTION_NOISE_INDEX])
        model_position = states[:, :2].copy()
        model_displacement = states[:, 2:].copy()
        prediction_count = future_time_seconds.size
        x_model_prediction = np.empty((draw_count, prediction_count), dtype=float)
        y_model_prediction = np.empty((draw_count, prediction_count), dtype=float)
        x_observation_prediction = np.empty_like(x_model_prediction)
        y_observation_prediction = np.empty_like(y_model_prediction)
        previous_time_seconds = self.last_observation_time_seconds
        previous_interval_seconds = self.previous_interval_seconds
        for prediction_index, prediction_time_seconds in enumerate(future_time_seconds):
            current_interval_seconds = float(
                prediction_time_seconds - previous_time_seconds
            )
            displacement_transition = _displacement_transition_matrices(
                log_displacement_scale_rate,
                rotation_rate,
                current_interval_seconds=current_interval_seconds,
                previous_interval_seconds=previous_interval_seconds,
            )
            motion_residual_scale = motion_noise * np.sqrt(
                current_interval_seconds / POSITION_MODEL_REFERENCE_INTERVAL_SECONDS
            )
            motion_innovation = generator.normal(
                0.0,
                motion_residual_scale[:, None],
                size=(draw_count, _POSITION_COUNT),
            )
            model_displacement = (
                np.einsum(
                    "nij,nj->ni",
                    displacement_transition,
                    model_displacement,
                )
                + motion_innovation
            )
            model_position += model_displacement
            x_model_prediction[:, prediction_index] = model_position[:, 0]
            y_model_prediction[:, prediction_index] = model_position[:, 1]
            observation_innovation = generator.normal(
                0.0,
                observation_noise[:, None],
                size=(draw_count, _POSITION_COUNT),
            )
            x_observation_prediction[:, prediction_index] = (
                model_position[:, 0] + observation_innovation[:, 0]
            )
            y_observation_prediction[:, prediction_index] = (
                model_position[:, 1] + observation_innovation[:, 1]
            )
            previous_interval_seconds = current_interval_seconds
            previous_time_seconds = float(prediction_time_seconds)

        return SequentialPositionFit(
            {
                "log_displacement_scale_rate": log_displacement_scale_rate,
                "rotation_rate": rotation_rate,
                "displacement_scale_at_reference": (displacement_scale_at_reference),
                "rotation_angle_at_reference": rotation_angle_at_reference,
                "sigma_position_observation": observation_noise,
                "sigma_motion_residual": motion_noise,
                "x_model_prediction": x_model_prediction,
                "y_model_prediction": y_model_prediction,
                "x_observation_prediction": x_observation_prediction,
                "y_observation_prediction": y_observation_prediction,
            }
        )

    def _resample_and_rejuvenate(self) -> None:
        """Resample weighted particles and apply Liu-West-style shrinkage."""
        parameter_mean = np.sum(
            self.weights[:, None] * self.parameter_particles,
            axis=0,
        )
        centered_parameters = self.parameter_particles - parameter_mean
        parameter_covariance = (
            centered_parameters.T * self.weights
        ) @ centered_parameters
        indices = _systematic_resample(self.weights, self.generator)
        selected_parameters = self.parameter_particles[indices]
        self.state_means = self.state_means[indices].copy()
        self.state_covariances = self.state_covariances[indices].copy()

        rejuvenation_scale = self.config.rejuvenation_scale
        shrinkage = np.sqrt(1.0 - rejuvenation_scale**2)
        parameter_cholesky = _regularized_cholesky(parameter_covariance)
        parameter_noise = (
            self.generator.normal(size=(self.config.particle_count, _PARAMETER_COUNT))
            @ parameter_cholesky.T
        )
        self.parameter_particles = (
            shrinkage * selected_parameters
            + (1.0 - shrinkage) * parameter_mean
            + rejuvenation_scale * parameter_noise
        )
        self.weights = np.full(
            self.config.particle_count,
            1.0 / self.config.particle_count,
            dtype=float,
        )
        self.resample_count += 1


def build_stan_data(
    window: observation_window.TrajectoryWindowData,
    *,
    priors: BayesianPositionModelPriors,
    position_observations: PositionObservations | None = None,
) -> dict[str, Any]:
    """Build Stan data from every position in one complete observed window."""
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
    time_observed = np.asarray(
        position_observations.time_seconds,
        dtype=float,
    )
    time_prediction = np.asarray(
        window.time_seconds[window.prediction_slice],
        dtype=float,
    )
    _validate_model_times(time_observed, time_prediction)
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
        "N_history": window.observation_count,
        "time_observed": time_observed,
        "x_observed": x_history,
        "y_observed": y_history,
        "N_prediction": window.prediction_count,
        "time_prediction": time_prediction,
        "position_model_reference_interval_seconds": (
            POSITION_MODEL_REFERENCE_INTERVAL_SECONDS
        ),
        "log_displacement_scale_rate_prior_scale": (
            priors.log_displacement_scale_rate_prior_scale
        ),
        "rotation_rate_prior_scale": priors.rotation_rate_prior_scale,
        "sigma_position_observation_prior_rate": (
            priors.sigma_position_observation_prior_rate
        ),
        "sigma_motion_residual_prior_rate": priors.sigma_motion_residual_prior_rate,
    }


def compile_bayesian_position_model(
    stan_file: str | Path = STAN_FILE,
) -> CmdStanModel:
    """Compile and return the latent Bayesian position CmdStan model."""
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
    """Fit the latent-position model with mean-field/full-rank VI or MCMC."""
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
    """Initialize latent positions and motion parameters near a smooth history."""
    generator = np.random.default_rng(seed)
    x_observed = np.asarray(stan_data["x_observed"], dtype=float)
    y_observed = np.asarray(stan_data["y_observed"], dtype=float)
    x_true = _smooth_position_initials(x_observed)
    y_true = _smooth_position_initials(y_observed)
    observation_noise_initial = 1.0 / float(
        stan_data["sigma_position_observation_prior_rate"]
    )
    latent_jitter_scale = 0.02 * observation_noise_initial
    x_true += generator.normal(0.0, latent_jitter_scale, x_true.size)
    y_true += generator.normal(0.0, latent_jitter_scale, y_true.size)
    time_observed = np.asarray(stan_data["time_observed"], dtype=float)
    intervals = np.diff(time_observed)
    displacement = np.column_stack((np.diff(x_true), np.diff(y_true)))
    velocity = displacement / intervals[:, None]
    previous_velocity = velocity[:-1]
    current_velocity = velocity[1:]
    current_intervals = intervals[1:]
    denominator = np.sum(previous_velocity * previous_velocity, axis=1)
    valid = denominator > 1e-12
    if not np.any(valid):
        log_scale_rate = 0.0
        rotation_rate = 0.0
    else:
        real_part = (
            np.sum(previous_velocity * current_velocity, axis=1)[valid]
            / (denominator[valid])
        )
        imaginary_part = (
            previous_velocity[:, 0] * current_velocity[:, 1]
            - previous_velocity[:, 1] * current_velocity[:, 0]
        )[valid] / denominator[valid]
        scale = np.maximum(np.hypot(real_part, imaginary_part), 1e-6)
        log_scale_rate = float(np.median(np.log(scale) / current_intervals[valid]))
        rotation_rate = float(
            np.median(np.arctan2(imaginary_part, real_part) / current_intervals[valid])
        )

    displacement_transition = _displacement_transition_matrices(
        np.full(len(current_intervals), log_scale_rate),
        np.full(len(current_intervals), rotation_rate),
        current_interval_seconds=current_intervals,
        previous_interval_seconds=intervals[:-1],
    )
    expected_displacement = np.einsum(
        "nij,nj->ni",
        displacement_transition,
        displacement[:-1],
    )
    residual = displacement[1:] - expected_displacement
    reference_normalized_residual = (
        residual
        * np.sqrt(POSITION_MODEL_REFERENCE_INTERVAL_SECONDS / current_intervals)[
            :, None
        ]
    )
    residual_scale = max(
        float(np.sqrt(np.mean(reference_normalized_residual**2))),
        0.5 / float(stan_data["sigma_motion_residual_prior_rate"]),
    )
    return {
        "x_true": x_true,
        "y_true": y_true,
        "log_displacement_scale_rate": float(
            log_scale_rate
            + generator.normal(
                0.0,
                0.01 * stan_data["log_displacement_scale_rate_prior_scale"],
            )
        ),
        "rotation_rate": float(
            rotation_rate
            + generator.normal(
                0.0,
                0.01 * stan_data["rotation_rate_prior_scale"],
            )
        ),
        "sigma_position_observation": float(
            max(observation_noise_initial * np.exp(generator.normal(0.0, 1e-3)), 1e-6)
        ),
        "sigma_motion_residual": float(
            max(residual_scale * np.exp(generator.normal(0.0, 1e-3)), 1e-6)
        ),
    }


def _smooth_position_initials(observed: np.ndarray) -> np.ndarray:
    """Return a lightly smoothed copy for latent-position initialization."""
    initial = np.asarray(observed, dtype=float).copy()
    initial[1:-1] = 0.25 * observed[:-2] + 0.5 * observed[1:-1] + 0.25 * observed[2:]
    return initial


def _validate_model_times(
    time_observed: np.ndarray,
    time_prediction: np.ndarray,
) -> None:
    """Require finite increasing history and future times without regularity."""
    if (
        time_observed.ndim != 1
        or time_prediction.ndim != 1
        or time_observed.size < MIN_OBSERVATION_COUNT
        or time_prediction.size < 1
        or not np.all(np.isfinite(time_observed))
        or not np.all(np.isfinite(time_prediction))
        or np.any(np.diff(time_observed) <= 0.0)
        or time_prediction[0] <= time_observed[-1]
        or np.any(np.diff(time_prediction) <= 0.0)
    ):
        raise ValueError(
            "Position-model timestamps must be finite and strictly increasing, "
            "with predictions after all observations."
        )


def _exponential_rate_from_tail(upper: float, tail_probability: float) -> float:
    """Return the exponential rate with the configured probability above upper."""
    return float(-np.log(tail_probability) / upper)


def _two_sided_normal_scale(absolute_upper: float, tail_probability: float) -> float:
    """Return a zero-centered normal scale from a two-sided tail statement."""
    quantile = NormalDist().inv_cdf(1.0 - tail_probability / 2.0)
    return float(absolute_upper / quantile)


def _validate_sequential_observations(
    time_seconds,
    x_observed,
    y_observed,
    *,
    minimum_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matching finite time/position arrays for online updates."""
    time_seconds = np.asarray(time_seconds, dtype=float)
    x_observed = np.asarray(x_observed, dtype=float)
    y_observed = np.asarray(y_observed, dtype=float)
    if (
        time_seconds.ndim != 1
        or x_observed.shape != time_seconds.shape
        or y_observed.shape != x_observed.shape
        or x_observed.size < minimum_count
        or not np.all(np.isfinite(time_seconds))
        or not np.all(np.isfinite(x_observed))
        or not np.all(np.isfinite(y_observed))
        or np.any(np.diff(time_seconds) <= 0.0)
    ):
        raise ValueError(
            "Sequential time/x/y observations must be matching finite vectors "
            f"with at least {minimum_count} values and increasing times."
        )
    return time_seconds, x_observed, y_observed


def _validate_future_times(future_time_seconds, *, after: float) -> np.ndarray:
    """Return finite increasing future times after the filter state."""
    future_time_seconds = np.asarray(future_time_seconds, dtype=float)
    if (
        future_time_seconds.ndim != 1
        or future_time_seconds.size < 1
        or not np.all(np.isfinite(future_time_seconds))
        or future_time_seconds[0] <= after
        or np.any(np.diff(future_time_seconds) <= 0.0)
    ):
        raise ValueError(
            "future_time_seconds must be finite, strictly increasing, and "
            "follow the latest observation."
        )
    return future_time_seconds


def _time_scaled_motion_matrices(
    log_displacement_scale_rate: np.ndarray,
    rotation_rate: np.ndarray,
    interval_seconds,
) -> np.ndarray:
    """Return exp(kappa*dt) R(omega*dt) for matching parameters."""
    interval_seconds = np.broadcast_to(
        np.asarray(interval_seconds, dtype=float),
        np.shape(log_displacement_scale_rate),
    )
    displacement_scale = np.exp(log_displacement_scale_rate * interval_seconds)
    rotation_angle = rotation_rate * interval_seconds
    cosine = np.cos(rotation_angle)
    sine = np.sin(rotation_angle)
    matrices = np.empty((len(displacement_scale), 2, 2), dtype=float)
    matrices[:, 0, 0] = displacement_scale * cosine
    matrices[:, 0, 1] = -displacement_scale * sine
    matrices[:, 1, 0] = displacement_scale * sine
    matrices[:, 1, 1] = displacement_scale * cosine
    return matrices


def _displacement_transition_matrices(
    log_displacement_scale_rate: np.ndarray,
    rotation_rate: np.ndarray,
    *,
    current_interval_seconds,
    previous_interval_seconds,
) -> np.ndarray:
    """Return irregular-time displacement transitions for all parameters."""
    current_interval_seconds = np.broadcast_to(
        np.asarray(current_interval_seconds, dtype=float),
        np.shape(log_displacement_scale_rate),
    )
    previous_interval_seconds = np.broadcast_to(
        np.asarray(previous_interval_seconds, dtype=float),
        np.shape(log_displacement_scale_rate),
    )
    if np.any(current_interval_seconds <= 0.0) or np.any(
        previous_interval_seconds <= 0.0
    ):
        raise ValueError("Position-model intervals must be positive.")
    matrices = _time_scaled_motion_matrices(
        log_displacement_scale_rate,
        rotation_rate,
        current_interval_seconds,
    )
    matrices *= (current_interval_seconds / previous_interval_seconds)[:, None, None]
    return matrices


def _sequential_transition_terms(
    parameter_particles: np.ndarray,
    *,
    current_interval_seconds: float,
    previous_interval_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return state-transition matrices and covariances for all particles."""
    particle_count = len(parameter_particles)
    displacement_transition = _displacement_transition_matrices(
        parameter_particles[:, _LOG_DISPLACEMENT_SCALE_RATE_INDEX],
        parameter_particles[:, _ROTATION_RATE_INDEX],
        current_interval_seconds=current_interval_seconds,
        previous_interval_seconds=previous_interval_seconds,
    )
    transition = np.zeros((particle_count, _STATE_COUNT, _STATE_COUNT), dtype=float)
    transition[:, :2, :2] = np.eye(_POSITION_COUNT)
    transition[:, :2, 2:] = displacement_transition
    transition[:, 2:, 2:] = displacement_transition

    motion_variance = np.exp(2.0 * parameter_particles[:, _LOG_MOTION_NOISE_INDEX]) * (
        current_interval_seconds / POSITION_MODEL_REFERENCE_INTERVAL_SECONDS
    )
    process_template = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    process_covariance = motion_variance[:, None, None] * process_template[None]
    return transition, process_covariance


def _invert_two_by_two_matrices(
    matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable inverses and log determinants for positive 2x2 matrices."""
    determinant = (
        matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] * matrices[:, 1, 0]
    )
    determinant = np.maximum(determinant, _NUMERICAL_VARIANCE_FLOOR**2)
    inverses = np.empty_like(matrices)
    inverses[:, 0, 0] = matrices[:, 1, 1] / determinant
    inverses[:, 0, 1] = -matrices[:, 0, 1] / determinant
    inverses[:, 1, 0] = -matrices[:, 1, 0] / determinant
    inverses[:, 1, 1] = matrices[:, 0, 0] / determinant
    return inverses, np.log(determinant)


def _sample_gaussian_states(
    means: np.ndarray,
    covariances: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Draw one state from each matching Gaussian distribution."""
    samples = np.empty_like(means)
    innovations = generator.normal(size=means.shape)
    for index, (mean, covariance) in enumerate(zip(means, covariances, strict=True)):
        samples[index] = mean + _regularized_cholesky(covariance) @ innovations[index]
    return samples


def _regularized_cholesky(covariance: np.ndarray) -> np.ndarray:
    """Return a Cholesky factor after symmetric eigenvalue regularization."""
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = max(float(np.max(eigenvalues)), _NUMERICAL_VARIANCE_FLOOR)
    eigenvalues = np.maximum(eigenvalues, largest * 1e-10)
    regularized = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return np.linalg.cholesky(regularized)


def _systematic_resample(
    weights: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Return systematic-resampling parent indices."""
    positions = (generator.random() + np.arange(len(weights))) / len(weights)
    cumulative_weights = np.cumsum(weights)
    cumulative_weights[-1] = 1.0
    return np.searchsorted(cumulative_weights, positions, side="right")
