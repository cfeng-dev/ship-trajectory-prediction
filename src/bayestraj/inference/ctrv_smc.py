"""Online CTRV bootstrap filtering without Rao-Blackwellization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import bayestraj.inference.particle_utils as particle_utils
import bayestraj.models.bayesian_ctrv as ctrv_model
import bayestraj.models.ctrv_dynamics as ctrv_dynamics
import bayestraj.observations.position as observation_support

_STATE_COUNT = ctrv_dynamics.STATE_COUNT
_PARAMETER_COUNT = 3
_MINIMUM_SCALE = 1e-9
_STATE_X_INDEX = ctrv_dynamics.STATE_X_INDEX
_STATE_Y_INDEX = ctrv_dynamics.STATE_Y_INDEX
_STATE_SPEED_INDEX = ctrv_dynamics.STATE_SPEED_INDEX
_STATE_HEADING_INDEX = ctrv_dynamics.STATE_HEADING_INDEX
_STATE_TURN_RATE_INDEX = ctrv_dynamics.STATE_TURN_RATE_INDEX


@dataclass(frozen=True, slots=True)
class SequentialMonteCarloCTRVConfig:
    """Numerical settings for full-state online CTRV particle filtering."""

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


@dataclass(slots=True)
class SequentialMonteCarloCTRVFilter:
    """Approximate online CTRV posterior with full state particles."""

    config: SequentialMonteCarloCTRVConfig
    parameter_particles: np.ndarray
    state_particles: np.ndarray
    weights: np.ndarray
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
        config: SequentialMonteCarloCTRVConfig | None = None,
        seed: int = 42,
    ) -> SequentialMonteCarloCTRVFilter:
        """Draw prior particles and condition positions on the first fix."""
        if not isinstance(priors, ctrv_model.BayesianCTRVPriors):
            raise TypeError("priors must be a BayesianCTRVPriors instance.")
        if config is None:
            config = SequentialMonteCarloCTRVConfig()
        if not isinstance(config, SequentialMonteCarloCTRVConfig):
            raise TypeError("config must be a SequentialMonteCarloCTRVConfig instance.")
        seed = observation_support.validate_non_negative_integer("seed", seed)
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
        observation_noise = np.maximum(
            generator.exponential(
                1.0 / priors.sigma_position_observation_prior_rate,
                particle_count,
            ),
            _MINIMUM_SCALE,
        )
        speed_process = np.maximum(
            generator.exponential(
                1.0 / priors.sigma_speed_process_prior_rate,
                particle_count,
            ),
            _MINIMUM_SCALE,
        )
        turn_rate_process = np.maximum(
            generator.exponential(
                1.0 / priors.sigma_turn_rate_process_prior_rate,
                particle_count,
            ),
            _MINIMUM_SCALE,
        )
        parameter_particles = np.log(
            np.column_stack((observation_noise, speed_process, turn_rate_process))
        )
        state_particles = np.empty((particle_count, _STATE_COUNT), dtype=float)
        state_particles[:, 0] = generator.normal(x_observed[0], observation_noise)
        state_particles[:, 1] = generator.normal(y_observed[0], observation_noise)
        state_particles[:, 2] = np.maximum(
            np.abs(generator.normal(0.0, priors.speed_prior_scale, particle_count)),
            ctrv_dynamics.SPEED_STATE_LOWER_MPS,
        )
        state_particles[:, 3] = generator.uniform(-np.pi, np.pi, particle_count)
        state_particles[:, 4] = generator.normal(
            0.0,
            priors.turn_rate_prior_scale,
            particle_count,
        )
        online_filter = cls(
            config=config,
            parameter_particles=parameter_particles,
            state_particles=state_particles,
            weights=np.full(particle_count, 1.0 / particle_count, dtype=float),
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
        """Propagate full state particles and condition on one position."""
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
        observation_noise, speed_process, turn_rate_process = _parameter_values(
            self.parameter_particles
        )
        dt = time_seconds - self.last_observation_time_seconds
        process_time_scale = np.sqrt(
            dt / ctrv_dynamics.PROCESS_REFERENCE_INTERVAL_SECONDS
        )
        proposed_states = self.state_particles.copy()
        proposed_states[:, _STATE_SPEED_INDEX] += self.generator.normal(
            0.0,
            speed_process * process_time_scale,
        )
        proposed_states[:, _STATE_TURN_RATE_INDEX] += self.generator.normal(
            0.0,
            turn_rate_process * process_time_scale,
        )
        ctrv_dynamics.normalize_states(proposed_states)
        proposed_states = ctrv_dynamics.transition_states(proposed_states, dt)
        squared_position_error = (x_observed - proposed_states[:, 0]) ** 2 + (
            y_observed - proposed_states[:, 1]
        ) ** 2
        log_likelihood = (
            -np.log(2.0 * np.pi)
            - 2.0 * np.log(observation_noise)
            - 0.5 * squared_position_error / observation_noise**2
        )
        with np.errstate(divide="ignore"):
            log_weights = np.log(self.weights) + log_likelihood
        maximum_log_weight = float(np.max(log_weights))
        if not np.isfinite(maximum_log_weight):
            raise RuntimeError("Sequential CTRV SMC particle weights collapsed.")
        unnormalized_weights = np.exp(log_weights - maximum_log_weight)
        weight_sum = float(np.sum(unnormalized_weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise RuntimeError("Sequential CTRV SMC particle weights collapsed.")
        weights = unnormalized_weights / weight_sum
        effective_sample_size = float(1.0 / np.sum(weights**2))
        parameter_particles = self.parameter_particles
        resampled = effective_sample_size < (
            self.config.resample_ess_fraction * self.config.particle_count
        )
        if resampled:
            proposed_states, parameter_particles, weights = self._resampled_population(
                proposed_states,
                parameter_particles,
                weights,
            )

        self.state_particles = proposed_states
        self.parameter_particles = parameter_particles
        self.weights = weights
        self.last_observation_time_seconds = time_seconds
        self.processed_observation_count += 1
        self.last_effective_sample_size = effective_sample_size
        if resampled:
            self.resample_count += 1

    def sample_current_posterior(
        self,
        *,
        seed: int,
    ) -> particle_utils.SequentialCTRVFit:
        """Draw the current state and parameters without advancing the filter."""
        seed = observation_support.validate_non_negative_integer("seed", seed)
        generator = np.random.default_rng(seed)
        indices = generator.choice(
            self.config.particle_count,
            size=self.config.posterior_draw_count,
            replace=True,
            p=self.weights,
        )
        states = self.state_particles[indices]
        observation_noise, speed_process, turn_rate_process = _parameter_values(
            self.parameter_particles[indices]
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
        """Draw future CTRV trajectories from the weighted SMC posterior."""
        future_time_seconds = particle_utils.validate_future_times(
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
        states = self.state_particles[indices].copy()
        observation_noise, speed_process, turn_rate_process = _parameter_values(
            self.parameter_particles[indices]
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
            process_time_scale = np.sqrt(
                dt / ctrv_dynamics.PROCESS_REFERENCE_INTERVAL_SECONDS
            )
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
                size=(draw_count, 2),
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

    def _resampled_population(
        self,
        state_particles: np.ndarray,
        parameter_particles: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resample joint ancestry and rejuvenate static log parameters."""
        parameter_mean = np.sum(weights[:, None] * parameter_particles, axis=0)
        centered_parameters = parameter_particles - parameter_mean
        parameter_covariance = (centered_parameters.T * weights) @ centered_parameters
        indices = particle_utils.systematic_resample(weights, self.generator)
        selected_states = state_particles[indices].copy()
        selected_parameters = parameter_particles[indices]
        rejuvenation_scale = self.config.rejuvenation_scale
        shrinkage = np.sqrt(1.0 - rejuvenation_scale**2)
        parameter_noise = (
            self.generator.normal(size=(self.config.particle_count, _PARAMETER_COUNT))
            @ particle_utils.regularized_cholesky(parameter_covariance).T
        )
        rejuvenated_parameters = (
            shrinkage * selected_parameters
            + (1.0 - shrinkage) * parameter_mean
            + rejuvenation_scale * parameter_noise
        )
        if not np.all(np.isfinite(rejuvenated_parameters)):
            raise RuntimeError("Sequential CTRV SMC parameter rejuvenation failed.")
        _parameter_values(rejuvenated_parameters)
        return (
            selected_states,
            rejuvenated_parameters,
            np.full(
                self.config.particle_count,
                1.0 / self.config.particle_count,
                dtype=float,
            ),
        )


def _parameter_values(parameter_particles: np.ndarray):
    """Transform log-scale particles to positive observation/process scales."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        values = np.exp(parameter_particles)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise RuntimeError("Sequential CTRV SMC parameters became invalid.")
    return values[:, 0], values[:, 1], values[:, 2]
