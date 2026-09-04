"""Rao-Blackwellized particle filtering for the Bayesian CTRV model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import bayestraj.inference.particle_utils as particle_utils
import bayestraj.models.bayesian_ctrv as ctrv_model
import bayestraj.models.ctrv as ctrv_dynamics
import bayestraj.numeric_validation as numeric_validation

PROCESS_REFERENCE_INTERVAL_SECONDS = ctrv_dynamics.PROCESS_REFERENCE_INTERVAL_SECONDS
SPEED_STATE_LOWER_MPS = ctrv_dynamics.SPEED_STATE_LOWER_MPS

_LOG_OBSERVATION_NOISE_INDEX = 0
_LOG_SPEED_PROCESS_INDEX = 1
_LOG_TURN_RATE_PROCESS_INDEX = 2
_SEQUENTIAL_PARAMETER_COUNT = 3
_POSITION_COUNT = 2
_STATE_X_INDEX = ctrv_dynamics.STATE_X_INDEX
_STATE_Y_INDEX = ctrv_dynamics.STATE_Y_INDEX
_STATE_SPEED_INDEX = ctrv_dynamics.STATE_SPEED_INDEX
_STATE_HEADING_INDEX = ctrv_dynamics.STATE_HEADING_INDEX
_STATE_TURN_RATE_INDEX = ctrv_dynamics.STATE_TURN_RATE_INDEX
_SEQUENTIAL_STATE_COUNT = ctrv_dynamics.STATE_COUNT
_NUMERICAL_VARIANCE_FLOOR = particle_utils.NUMERICAL_VARIANCE_FLOOR


@dataclass(frozen=True, slots=True)
class SequentialCTRVFilterConfig:
    """Numerical settings that are specific to online CTRV particle filtering."""

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

        ess_fraction = numeric_validation.validate_positive_finite(
            "resample_ess_fraction",
            self.resample_ess_fraction,
        )
        if ess_fraction > 1.0:
            raise ValueError("resample_ess_fraction must not exceed one.")
        rejuvenation_scale = numeric_validation.validate_non_negative_finite(
            "rejuvenation_scale",
            self.rejuvenation_scale,
        )
        if rejuvenation_scale >= 1.0:
            raise ValueError("rejuvenation_scale must be smaller than one.")
        object.__setattr__(self, "resample_ess_fraction", ess_fraction)
        object.__setattr__(self, "rejuvenation_scale", rejuvenation_scale)


@dataclass(slots=True)
class SequentialBayesianCTRVFilter:
    """Online Rao-Blackwellized filter for CTRV states and parameters."""

    config: SequentialCTRVFilterConfig
    parameter_particles: np.ndarray
    weights: np.ndarray
    state_means: np.ndarray
    state_covariances: np.ndarray
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
        priors: ctrv_model.BayesianCTRVPriors,
        config: SequentialCTRVFilterConfig | None = None,
        seed: int = 42,
    ) -> SequentialBayesianCTRVFilter:
        """Initialize the posterior and process every supplied position once."""
        if not isinstance(priors, ctrv_model.BayesianCTRVPriors):
            raise TypeError("priors must be a BayesianCTRVPriors instance.")
        if config is None:
            config = SequentialCTRVFilterConfig()
        if not isinstance(config, SequentialCTRVFilterConfig):
            raise TypeError(
                "config must be a SequentialCTRVFilterConfig instance or None."
            )
        seed = numeric_validation.validate_non_negative_integer("seed", seed)
        time_seconds, x_observed, y_observed = (
            particle_utils.validate_sequential_observations(
                time_seconds,
                x_observed,
                y_observed,
                minimum_count=1,
            )
        )
        generator = np.random.default_rng(seed)
        particle_count = config.particle_count
        parameter_particles = np.empty(
            (particle_count, _SEQUENTIAL_PARAMETER_COUNT),
            dtype=float,
        )
        speed = np.maximum(
            np.abs(generator.normal(0.0, priors.speed_prior_scale, particle_count)),
            SPEED_STATE_LOWER_MPS,
        )
        heading = generator.uniform(
            -np.pi,
            np.pi,
            particle_count,
        )
        turn_rate = generator.normal(
            0.0,
            priors.turn_rate_prior_scale,
            particle_count,
        )
        observation_noise = np.maximum(
            generator.exponential(
                1.0 / priors.sigma_position_observation_prior_rate,
                particle_count,
            ),
            1e-6,
        )
        speed_process = np.maximum(
            generator.exponential(
                1.0 / priors.sigma_speed_process_prior_rate,
                particle_count,
            ),
            1e-6,
        )
        turn_rate_process = np.maximum(
            generator.exponential(
                1.0 / priors.sigma_turn_rate_process_prior_rate,
                particle_count,
            ),
            1e-9,
        )
        parameter_particles[:, _LOG_OBSERVATION_NOISE_INDEX] = np.log(observation_noise)
        parameter_particles[:, _LOG_SPEED_PROCESS_INDEX] = np.log(speed_process)
        parameter_particles[:, _LOG_TURN_RATE_PROCESS_INDEX] = np.log(turn_rate_process)
        state_means = np.column_stack(
            (
                np.full(particle_count, x_observed[0], dtype=float),
                np.full(particle_count, y_observed[0], dtype=float),
                speed,
                heading,
                turn_rate,
            )
        )
        state_covariances = np.zeros(
            (particle_count, _SEQUENTIAL_STATE_COUNT, _SEQUENTIAL_STATE_COUNT),
            dtype=float,
        )
        state_covariances[:, _STATE_X_INDEX, _STATE_X_INDEX] = observation_noise**2
        state_covariances[:, _STATE_Y_INDEX, _STATE_Y_INDEX] = observation_noise**2
        state_covariances[:, _STATE_SPEED_INDEX, _STATE_SPEED_INDEX] = speed_process**2
        state_covariances[:, _STATE_HEADING_INDEX, _STATE_HEADING_INDEX] = (
            turn_rate_process * PROCESS_REFERENCE_INTERVAL_SECONDS
        ) ** 2
        state_covariances[:, _STATE_TURN_RATE_INDEX, _STATE_TURN_RATE_INDEX] = (
            turn_rate_process**2
        )
        state_covariances += (
            _NUMERICAL_VARIANCE_FLOOR * np.eye(_SEQUENTIAL_STATE_COUNT)[None]
        )
        online_filter = cls(
            config=config,
            parameter_particles=parameter_particles,
            weights=np.full(particle_count, 1.0 / particle_count, dtype=float),
            state_means=state_means,
            state_covariances=state_covariances,
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
        return particle_utils.effective_sample_size(self.weights)

    def update_many(self, time_seconds, x_observed, y_observed) -> None:
        """Update the posterior with matching new positions exactly once."""
        time_seconds, x_observed, y_observed = (
            particle_utils.validate_sequential_observations(
                time_seconds,
                x_observed,
                y_observed,
                minimum_count=0,
            )
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
        time_seconds = numeric_validation.validate_finite_scalar(
            "time_seconds",
            time_seconds,
        )
        if time_seconds <= self.last_observation_time_seconds:
            raise ValueError("time_seconds must follow the previous observation.")
        x_observed = numeric_validation.validate_finite_scalar(
            "x_observed",
            x_observed,
        )
        y_observed = numeric_validation.validate_finite_scalar(
            "y_observed",
            y_observed,
        )
        observation_noise, speed_process, turn_rate_process = (
            _sequential_parameter_values(self.parameter_particles)
        )
        dt = time_seconds - self.last_observation_time_seconds
        process_variance_scale = dt / PROCESS_REFERENCE_INTERVAL_SECONDS
        pre_transition_covariances = self.state_covariances.copy()
        pre_transition_covariances[:, _STATE_SPEED_INDEX, _STATE_SPEED_INDEX] += (
            speed_process**2 * process_variance_scale
        )
        pre_transition_covariances[
            :, _STATE_TURN_RATE_INDEX, _STATE_TURN_RATE_INDEX
        ] += turn_rate_process**2 * process_variance_scale
        transition_jacobians = ctrv_dynamics.transition_jacobians(
            self.state_means,
            dt,
        )
        predicted_means = ctrv_dynamics.transition_states(self.state_means, dt)
        predicted_covariances = (
            transition_jacobians
            @ pre_transition_covariances
            @ np.swapaxes(transition_jacobians, 1, 2)
        )
        innovation_covariances = predicted_covariances[:, :2, :2].copy()
        innovation_covariances[:, _STATE_X_INDEX, _STATE_X_INDEX] += (
            observation_noise**2
        )
        innovation_covariances[:, _STATE_Y_INDEX, _STATE_Y_INDEX] += (
            observation_noise**2
        )
        innovation_inverses, log_determinants = _invert_two_by_two_matrices(
            innovation_covariances
        )
        innovations = np.column_stack(
            (
                x_observed - predicted_means[:, _STATE_X_INDEX],
                y_observed - predicted_means[:, _STATE_Y_INDEX],
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
            raise RuntimeError("Sequential CTRV particle weights collapsed.")
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
        self.state_covariances += (
            _NUMERICAL_VARIANCE_FLOOR * np.eye(_SEQUENTIAL_STATE_COUNT)[None]
        )
        ctrv_dynamics.normalize_states(self.state_means, self.state_covariances)
        self.last_observation_time_seconds = time_seconds
        self.processed_observation_count += 1
        self.last_effective_sample_size = self.effective_sample_size
        if (
            self.last_effective_sample_size
            < self.config.resample_ess_fraction * self.config.particle_count
        ):
            self._resample_and_rejuvenate()

    def sample_current_posterior(
        self,
        *,
        seed: int,
    ) -> particle_utils.SequentialCTRVFit:
        """Draw the current CTRV state and parameters without forecasting."""
        seed = numeric_validation.validate_non_negative_integer("seed", seed)
        generator = np.random.default_rng(seed)
        indices = generator.choice(
            self.config.particle_count,
            size=self.config.posterior_draw_count,
            replace=True,
            p=self.weights,
        )
        parameters = self.parameter_particles[indices]
        states = _sample_gaussian_states(
            self.state_means[indices],
            self.state_covariances[indices],
            generator,
        )
        ctrv_dynamics.normalize_states(states)
        observation_noise, speed_process, turn_rate_process = (
            _sequential_parameter_values(parameters)
        )
        return particle_utils.SequentialCTRVFit(
            {
                "speed_at_origin": states[:, _STATE_SPEED_INDEX],
                "heading_at_origin": states[:, _STATE_HEADING_INDEX],
                "turn_rate_at_origin": states[:, _STATE_TURN_RATE_INDEX],
                "sigma_position_observation": observation_noise,
                "sigma_speed_process": speed_process,
                "sigma_turn_rate_process": turn_rate_process,
            }
        )

    def forecast(
        self,
        future_time_seconds,
        *,
        seed: int,
    ) -> particle_utils.SequentialCTRVFit:
        """Draw future latent and observed trajectories from the posterior."""
        future_time_seconds = particle_utils.validate_future_times(
            future_time_seconds,
            after=self.last_observation_time_seconds,
        )
        seed = numeric_validation.validate_non_negative_integer("seed", seed)
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
        ctrv_dynamics.normalize_states(states)
        observation_noise, speed_process, turn_rate_process = (
            _sequential_parameter_values(parameters)
        )
        speed_at_origin = states[:, _STATE_SPEED_INDEX].copy()
        heading_at_origin = states[:, _STATE_HEADING_INDEX].copy()
        turn_rate_at_origin = states[:, _STATE_TURN_RATE_INDEX].copy()
        prediction_count = future_time_seconds.size
        x_prediction = np.empty((draw_count, prediction_count), dtype=float)
        y_prediction = np.empty_like(x_prediction)
        x_observation_prediction = np.empty_like(x_prediction)
        y_observation_prediction = np.empty_like(x_prediction)
        current_time = self.last_observation_time_seconds
        for prediction_index, prediction_time in enumerate(future_time_seconds):
            dt = float(prediction_time - current_time)
            process_time_scale = np.sqrt(dt / PROCESS_REFERENCE_INTERVAL_SECONDS)
            states[:, _STATE_SPEED_INDEX] += generator.normal(
                0.0,
                speed_process * process_time_scale,
            )
            states[:, _STATE_TURN_RATE_INDEX] += generator.normal(
                0.0,
                turn_rate_process * process_time_scale,
            )
            ctrv_dynamics.normalize_states(states)
            states = ctrv_dynamics.transition_states(states, dt)
            x_prediction[:, prediction_index] = states[:, _STATE_X_INDEX]
            y_prediction[:, prediction_index] = states[:, _STATE_Y_INDEX]
            observation_innovation = generator.normal(
                0.0,
                observation_noise[:, None],
                size=(draw_count, _POSITION_COUNT),
            )
            x_observation_prediction[:, prediction_index] = (
                states[:, _STATE_X_INDEX] + observation_innovation[:, 0]
            )
            y_observation_prediction[:, prediction_index] = (
                states[:, _STATE_Y_INDEX] + observation_innovation[:, 1]
            )
            current_time = float(prediction_time)

        return particle_utils.SequentialCTRVFit(
            {
                "speed_at_origin": speed_at_origin,
                "heading_at_origin": heading_at_origin,
                "turn_rate_at_origin": turn_rate_at_origin,
                "sigma_position_observation": observation_noise,
                "sigma_speed_process": speed_process,
                "sigma_turn_rate_process": turn_rate_process,
                "x_prediction": x_prediction,
                "y_prediction": y_prediction,
                "x_observation_prediction": x_observation_prediction,
                "y_observation_prediction": y_observation_prediction,
            }
        )

    def _resample_and_rejuvenate(self) -> None:
        """Resample states and apply Liu-West-style parameter rejuvenation."""
        adjusted_parameters, parameter_mean = _sequential_parameter_moments(
            self.parameter_particles,
            self.weights,
        )
        centered_parameters = adjusted_parameters - parameter_mean
        parameter_covariance = (
            centered_parameters.T * self.weights
        ) @ centered_parameters
        indices = particle_utils.systematic_resample(self.weights, self.generator)
        selected_parameters = adjusted_parameters[indices]
        self.state_means = self.state_means[indices].copy()
        self.state_covariances = self.state_covariances[indices].copy()
        rejuvenation_scale = self.config.rejuvenation_scale
        shrinkage = np.sqrt(1.0 - rejuvenation_scale**2)
        parameter_cholesky = particle_utils.regularized_cholesky(parameter_covariance)
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
        if not np.all(np.isfinite(self.parameter_particles)):
            raise RuntimeError("Sequential CTRV parameter rejuvenation became invalid.")
        self.weights = np.full(
            self.config.particle_count,
            1.0 / self.config.particle_count,
            dtype=float,
        )
        self.resample_count += 1


def _sequential_parameter_values(parameter_particles: np.ndarray):
    """Transform unconstrained particles to positive process-scale arrays."""
    return (
        np.exp(parameter_particles[:, _LOG_OBSERVATION_NOISE_INDEX]),
        np.exp(parameter_particles[:, _LOG_SPEED_PROCESS_INDEX]),
        np.exp(parameter_particles[:, _LOG_TURN_RATE_PROCESS_INDEX]),
    )


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
        samples[index] = (
            mean + particle_utils.regularized_cholesky(covariance) @ innovations[index]
        )
    return samples


def _sequential_parameter_moments(
    parameter_particles: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log-scale particles and their weighted mean."""
    mean = np.sum(weights[:, None] * parameter_particles, axis=0)
    return parameter_particles, mean
