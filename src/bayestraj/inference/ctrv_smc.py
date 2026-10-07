"""Observation-guided online CTRV filtering without Rao-Blackwellization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import bayestraj.inference.particle_utils as particle_utils
import bayestraj.models.bayesian_ctrv as ctrv_model
import bayestraj.models.ctrv as ctrv_dynamics
import bayestraj.numeric_validation as numeric_validation

_STATE_COUNT = ctrv_dynamics.STATE_COUNT
_PARAMETER_COUNT = 3
_MINIMUM_SCALE = 1e-9
_MOTION_COUNT = 2
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
    predictive_log_density_threshold: float | None = None

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
        threshold = self.predictive_log_density_threshold
        if threshold is not None:
            threshold = numeric_validation.validate_finite_scalar(
                "predictive_log_density_threshold",
                threshold,
            )
        object.__setattr__(self, "resample_ess_fraction", ess_fraction)
        object.__setattr__(self, "rejuvenation_scale", rejuvenation_scale)
        object.__setattr__(self, "predictive_log_density_threshold", threshold)


@dataclass(slots=True)
class SequentialMonteCarloCTRVFilter:
    """Approximate online CTRV posterior with full state particles.

    ``state_particles`` include motion noise prepared for the next observed
    interval. ``forecast_origin_particles`` retain the latest state whose
    motion was informed by an observed displacement.
    """

    config: SequentialMonteCarloCTRVConfig
    parameter_particles: np.ndarray
    state_particles: np.ndarray
    weights: np.ndarray
    generator: np.random.Generator
    last_observation_time_seconds: float
    processed_observation_count: int
    resample_count: int = 0
    last_effective_sample_size: float | None = None
    forecast_origin_particles: np.ndarray | None = None
    position_observation_noise_floor_m: float = 0.0
    last_log_predictive_density: float | None = None
    predictive_log_density_history: list[float] = field(default_factory=list)
    track_lost: bool | None = None
    last_observed_position: np.ndarray | None = None
    reinitialization_count: int = 0

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
        observation_noise_excess = np.maximum(
            generator.exponential(
                1.0 / priors.sigma_position_observation_prior_rate,
                particle_count,
            ),
            _MINIMUM_SCALE,
        )
        observation_noise = np.hypot(
            observation_noise_excess,
            priors.sigma_position_observation_floor_m,
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
            np.column_stack(
                (observation_noise_excess, speed_process, turn_rate_process)
            )
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
            forecast_origin_particles=state_particles.copy(),
            position_observation_noise_floor_m=(
                priors.sigma_position_observation_floor_m
            ),
            last_observed_position=np.array(
                [x_observed[0], y_observed[0]],
                dtype=float,
            ),
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
        observation_noise, speed_process, turn_rate_process = _parameter_values(
            self.parameter_particles,
            position_observation_noise_floor_m=(
                self.position_observation_noise_floor_m
            ),
        )
        dt = time_seconds - self.last_observation_time_seconds
        process_time_scale = ctrv_dynamics.process_time_scale(dt)
        log_predictive_density = self._one_step_ahead_log_predictive_density(
            dt,
            x_observed,
            y_observed,
            observation_noise,
        )
        self.last_log_predictive_density = log_predictive_density
        self.predictive_log_density_history.append(log_predictive_density)
        threshold = self.config.predictive_log_density_threshold
        self.track_lost = (
            None if threshold is None else log_predictive_density < threshold
        )
        if self.track_lost:
            self._reinitialize_at_observation(
                time_seconds,
                x_observed,
                y_observed,
                dt,
                observation_noise,
                speed_process,
                turn_rate_process,
                process_time_scale,
            )
            return
        if self.processed_observation_count == 1:
            proposed_origins = ctrv_dynamics.transition_states(self.state_particles, dt)
            parameter_particles = self.parameter_particles
            with np.errstate(divide="ignore"):
                base_log_weights = np.log(self.weights)
            log_proposal_correction = np.zeros(self.config.particle_count)
            auxiliary_resampled = False
        else:
            (
                proposed_origins,
                parameter_particles,
                base_log_weights,
                log_proposal_correction,
            ) = self._guided_proposal(
                dt,
                x_observed,
                y_observed,
                speed_process * process_time_scale,
                turn_rate_process * process_time_scale,
            )
            observation_noise, _, _ = _parameter_values(
                parameter_particles,
                position_observation_noise_floor_m=(
                    self.position_observation_noise_floor_m
                ),
            )
            auxiliary_resampled = True
        squared_position_error = (x_observed - proposed_origins[:, 0]) ** 2 + (
            y_observed - proposed_origins[:, 1]
        ) ** 2
        log_likelihood = (
            -np.log(2.0 * np.pi)
            - 2.0 * np.log(observation_noise)
            - 0.5 * squared_position_error / observation_noise**2
        )
        with np.errstate(divide="ignore"):
            log_weights = base_log_weights + log_likelihood + log_proposal_correction
        weights = _normalized_weights(log_weights)
        effective_sample_size = float(1.0 / np.sum(weights**2))
        resampled = effective_sample_size < (
            self.config.resample_ess_fraction * self.config.particle_count
        )
        if resampled:
            proposed_origins, parameter_particles, weights = self._resampled_population(
                proposed_origins,
                parameter_particles,
                weights,
            )

        _, speed_process, turn_rate_process = _parameter_values(
            parameter_particles,
            position_observation_noise_floor_m=self.position_observation_noise_floor_m,
        )
        next_states = proposed_origins.copy()
        next_states[:, _STATE_SPEED_INDEX] += self.generator.normal(
            0.0,
            speed_process * process_time_scale,
        )
        next_states[:, _STATE_TURN_RATE_INDEX] += self.generator.normal(
            0.0,
            turn_rate_process * process_time_scale,
        )
        ctrv_dynamics.normalize_states(next_states)
        self.forecast_origin_particles = proposed_origins
        self.state_particles = next_states
        self.parameter_particles = parameter_particles
        self.weights = weights
        self.last_observation_time_seconds = time_seconds
        self.processed_observation_count += 1
        self.last_effective_sample_size = effective_sample_size
        self.resample_count += int(auxiliary_resampled) + int(resampled)
        self.last_observed_position = np.array([x_observed, y_observed], dtype=float)

    def _reinitialize_at_observation(
        self,
        time_seconds: float,
        x_observed: float,
        y_observed: float,
        dt: float,
        observation_noise: np.ndarray,
        speed_process: np.ndarray,
        turn_rate_process: np.ndarray,
        process_time_scale: float,
    ) -> None:
        """Restart state particles around an observation outside predictive support."""
        particle_count = self.config.particle_count
        recovered_origins = np.empty((particle_count, _STATE_COUNT), dtype=float)
        recovered_origins[:, _STATE_X_INDEX] = self.generator.normal(
            x_observed,
            observation_noise,
        )
        recovered_origins[:, _STATE_Y_INDEX] = self.generator.normal(
            y_observed,
            observation_noise,
        )
        previous_position = self.last_observed_position
        if previous_position is None:
            displacement = np.zeros(2, dtype=float)
        else:
            displacement = np.asarray(previous_position, dtype=float)
            if displacement.shape != (2,) or not np.all(np.isfinite(displacement)):
                raise RuntimeError("Sequential CTRV SMC last observation became invalid.")
            displacement = np.array([x_observed, y_observed]) - displacement
        distance = float(np.hypot(*displacement))
        is_heading_observable = distance > 2.0 * float(np.median(observation_noise))
        speed_scale = np.sqrt(2.0) * observation_noise / dt
        if is_heading_observable:
            recovered_origins[:, _STATE_SPEED_INDEX] = np.maximum(
                np.abs(self.generator.normal(distance / dt, speed_scale)),
                ctrv_dynamics.SPEED_STATE_LOWER_MPS,
            )
            heading = float(np.arctan2(displacement[1], displacement[0]))
            heading_scale = np.minimum(np.pi, speed_scale / (distance / dt))
            recovered_origins[:, _STATE_HEADING_INDEX] = self.generator.normal(
                heading,
                heading_scale,
            )
        else:
            recovered_origins[:, _STATE_SPEED_INDEX] = np.maximum(
                np.abs(self.generator.normal(0.0, speed_scale)),
                ctrv_dynamics.SPEED_STATE_LOWER_MPS,
            )
            recovered_origins[:, _STATE_HEADING_INDEX] = self.generator.uniform(
                -np.pi,
                np.pi,
                particle_count,
            )
        recovered_origins[:, _STATE_TURN_RATE_INDEX] = self.generator.normal(
            0.0,
            turn_rate_process * process_time_scale,
        )
        ctrv_dynamics.normalize_states(recovered_origins)
        next_states = recovered_origins.copy()
        next_states[:, _STATE_SPEED_INDEX] += self.generator.normal(
            0.0,
            speed_process * process_time_scale,
        )
        next_states[:, _STATE_TURN_RATE_INDEX] += self.generator.normal(
            0.0,
            turn_rate_process * process_time_scale,
        )
        ctrv_dynamics.normalize_states(next_states)
        self.forecast_origin_particles = recovered_origins
        self.state_particles = next_states
        self.weights = np.full(particle_count, 1.0 / particle_count, dtype=float)
        self.last_observation_time_seconds = time_seconds
        self.last_observed_position = np.array([x_observed, y_observed], dtype=float)
        self.processed_observation_count += 1
        self.last_effective_sample_size = float(particle_count)
        self.reinitialization_count += 1

    def _one_step_ahead_log_predictive_density(
        self,
        dt: float,
        x_observed: float,
        y_observed: float,
        observation_noise: np.ndarray,
    ) -> float:
        """Evaluate ``p(z_k | z_1:k-1)`` before conditioning on ``z_k``."""
        predicted_positions = ctrv_dynamics.transition_states(
            self.state_particles,
            dt,
        )[:, :2]
        squared_position_error = (x_observed - predicted_positions[:, 0]) ** 2 + (
            y_observed - predicted_positions[:, 1]
        ) ** 2
        log_components = (
            -np.log(2.0 * np.pi)
            - 2.0 * np.log(observation_noise)
            - 0.5 * squared_position_error / observation_noise**2
        )
        with np.errstate(divide="ignore"):
            log_weighted_components = np.log(self.weights) + log_components
        return _logsumexp(log_weighted_components)

    def _guided_proposal(
        self,
        dt: float,
        x_observed: float,
        y_observed: float,
        speed_process_scale: np.ndarray,
        turn_rate_process_scale: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Propose interval motion near the new fix and return log corrections."""
        base_states = self._forecast_origins().copy()
        observation_noise, _, _ = _parameter_values(
            self.parameter_particles,
            position_observation_noise_floor_m=self.position_observation_noise_floor_m,
        )
        # Position fixes cannot identify heading while translation is below the
        # observation scale. Restore its circular support before resampling can
        # turn that numerical ancestry loss into a false physical certainty.
        if np.median(base_states[:, _STATE_SPEED_INDEX]) * dt <= (
            2.0 * np.median(observation_noise)
        ):
            base_states[:, _STATE_HEADING_INDEX] = self.generator.uniform(
                -np.pi,
                np.pi,
                self.config.particle_count,
            )
        prior_motion = base_states[:, [_STATE_SPEED_INDEX, _STATE_TURN_RATE_INDEX]]
        delta_x = x_observed - base_states[:, _STATE_X_INDEX]
        delta_y = y_observed - base_states[:, _STATE_Y_INDEX]
        chord_heading = np.arctan2(delta_y, delta_x)
        half_turn = ctrv_dynamics.wrap_angles(
            chord_heading - base_states[:, _STATE_HEADING_INDEX]
        )
        observed_turn_rate = 2.0 * half_turn / dt
        distance = np.hypot(delta_x, delta_y)
        distance_per_speed = dt * np.sinc(half_turn / np.pi)
        observed_speed = distance / np.maximum(
            np.abs(distance_per_speed), _MINIMUM_SCALE
        )
        observed_motion = np.column_stack((observed_speed, observed_turn_rate))

        linearization_states = base_states.copy()
        linearization_states[:, _STATE_SPEED_INDEX] = observed_speed
        linearization_states[:, _STATE_TURN_RATE_INDEX] = observed_turn_rate
        position_jacobians = ctrv_dynamics.transition_jacobians(
            linearization_states,
            dt,
        )[:, :2, :][:, :, [_STATE_SPEED_INDEX, _STATE_TURN_RATE_INDEX]]
        prior_variances = np.column_stack(
            (speed_process_scale**2, turn_rate_process_scale**2)
        )
        observation_variances = observation_noise**2
        innovation_covariances = np.einsum(
            "nij,nj,nkj->nik",
            position_jacobians,
            prior_variances,
            position_jacobians,
        )
        innovation_covariances[:, 0, 0] += observation_variances
        innovation_covariances[:, 1, 1] += observation_variances
        innovation_inverses, _ = _invert_two_by_two_matrices(innovation_covariances)
        gains = np.einsum(
            "ni,nji,njk->nik",
            prior_variances,
            position_jacobians,
            innovation_inverses,
        )
        proposal_means = prior_motion + np.einsum(
            "nij,njk,nk->ni",
            gains,
            position_jacobians,
            observed_motion - prior_motion,
        )
        proposal_covariances = np.zeros(
            (self.config.particle_count, _MOTION_COUNT, _MOTION_COUNT),
            dtype=float,
        )
        proposal_covariances[:, 0, 0] = prior_variances[:, 0]
        proposal_covariances[:, 1, 1] = prior_variances[:, 1]
        proposal_covariances -= np.einsum(
            "nij,njk,nkl->nil",
            gains,
            position_jacobians,
            proposal_covariances.copy(),
        )
        proposal_covariances = _regularize_two_by_two_covariances(proposal_covariances)

        prior_states = base_states.copy()
        prior_states[:, _STATE_SPEED_INDEX] = prior_motion[:, 0]
        prior_states[:, _STATE_TURN_RATE_INDEX] = prior_motion[:, 1]
        prior_positions = ctrv_dynamics.transition_states(prior_states, dt)[:, :2]
        prediction_residuals = np.column_stack(
            (x_observed - prior_positions[:, 0], y_observed - prior_positions[:, 1])
        )
        _, innovation_log_determinants = _invert_two_by_two_matrices(
            innovation_covariances
        )
        predictive_log_density = (
            -np.log(2.0 * np.pi)
            - 0.5 * innovation_log_determinants
            - 0.5
            * np.einsum(
                "ni,nij,nj->n",
                prediction_residuals,
                innovation_inverses,
                prediction_residuals,
            )
        )
        with np.errstate(divide="ignore"):
            auxiliary_log_weights = np.log(self.weights) + predictive_log_density
        auxiliary_weights = _normalized_weights(auxiliary_log_weights)
        ancestor_indices = particle_utils.systematic_resample(
            auxiliary_weights,
            self.generator,
        )
        base_states = base_states[ancestor_indices]
        prior_motion = prior_motion[ancestor_indices]
        prior_variances = prior_variances[ancestor_indices]
        proposal_means = proposal_means[ancestor_indices]
        proposal_covariances = proposal_covariances[ancestor_indices]
        predictive_log_density = predictive_log_density[ancestor_indices]
        parameter_particles = self.parameter_particles[ancestor_indices].copy()

        proposed_motion = _sample_two_dimensional_gaussians(
            proposal_means,
            proposal_covariances,
            self.generator,
        )
        proposed_motion[:, 0] = np.maximum(
            np.abs(proposed_motion[:, 0]),
            ctrv_dynamics.SPEED_STATE_LOWER_MPS,
        )

        proposal_states = base_states.copy()
        proposal_states[:, _STATE_SPEED_INDEX] = proposed_motion[:, 0]
        proposal_states[:, _STATE_TURN_RATE_INDEX] = proposed_motion[:, 1]
        proposed_origins = ctrv_dynamics.transition_states(proposal_states, dt)
        log_prior = _folded_motion_log_density(
            proposed_motion,
            prior_motion,
            prior_variances,
        )
        log_proposal = _folded_bivariate_log_density(
            proposed_motion,
            proposal_means,
            proposal_covariances,
        )
        return (
            proposed_origins,
            parameter_particles,
            np.zeros(self.config.particle_count),
            log_prior - log_proposal - predictive_log_density,
        )

    def sample_current_posterior(
        self,
        *,
        seed: int,
    ) -> particle_utils.SequentialCTRVFit:
        """Draw the current state and parameters without advancing the filter."""
        seed = numeric_validation.validate_non_negative_integer("seed", seed)
        generator = np.random.default_rng(seed)
        indices = generator.choice(
            self.config.particle_count,
            size=self.config.posterior_draw_count,
            replace=True,
            p=self.weights,
        )
        states = self._forecast_origins()[indices]
        observation_noise, speed_process, turn_rate_process = _parameter_values(
            self.parameter_particles[indices],
            position_observation_noise_floor_m=self.position_observation_noise_floor_m,
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
        seed = numeric_validation.validate_non_negative_integer("seed", seed)
        generator = np.random.default_rng(seed)
        draw_count = self.config.posterior_draw_count
        indices = generator.choice(
            self.config.particle_count,
            size=draw_count,
            replace=True,
            p=self.weights,
        )
        states = self._forecast_origins()[indices].copy()
        observation_noise, speed_process, turn_rate_process = _parameter_values(
            self.parameter_particles[indices],
            position_observation_noise_floor_m=self.position_observation_noise_floor_m,
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
            process_time_scale = ctrv_dynamics.process_time_scale(dt)
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
            states[:, _STATE_SPEED_INDEX] += generator.normal(
                0.0,
                speed_process * process_time_scale,
            )
            states[:, _STATE_TURN_RATE_INDEX] += generator.normal(
                0.0,
                turn_rate_process * process_time_scale,
            )
            ctrv_dynamics.normalize_states(states)
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

    def _forecast_origins(self) -> np.ndarray:
        """Return the latest transition-informed state at an observation time."""
        if self.forecast_origin_particles is None:
            return self.state_particles
        return self.forecast_origin_particles

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
        _parameter_values(
            rejuvenated_parameters,
            position_observation_noise_floor_m=self.position_observation_noise_floor_m,
        )
        return (
            selected_states,
            rejuvenated_parameters,
            np.full(
                self.config.particle_count,
                1.0 / self.config.particle_count,
                dtype=float,
            ),
        )


def _parameter_values(
    parameter_particles: np.ndarray,
    *,
    position_observation_noise_floor_m: float = 0.0,
):
    """Transform log-scale particles to positive observation/process scales."""
    position_observation_noise_floor_m = (
        numeric_validation.validate_non_negative_finite(
            "position_observation_noise_floor_m",
            position_observation_noise_floor_m,
        )
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        values = np.exp(parameter_particles)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise RuntimeError("Sequential CTRV SMC parameters became invalid.")
    values[:, 0] = np.hypot(values[:, 0], position_observation_noise_floor_m)
    return values[:, 0], values[:, 1], values[:, 2]


def _normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    """Normalize log weights or raise when no finite particle remains."""
    maximum_log_weight = float(np.max(log_weights))
    if not np.isfinite(maximum_log_weight):
        raise RuntimeError("Sequential CTRV SMC particle weights collapsed.")
    unnormalized_weights = np.exp(log_weights - maximum_log_weight)
    weight_sum = float(np.sum(unnormalized_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise RuntimeError("Sequential CTRV SMC particle weights collapsed.")
    return unnormalized_weights / weight_sum


def _logsumexp(log_values: np.ndarray) -> float:
    """Return a stable logarithm of a finite positive sum of exponentials."""
    maximum_log_value = float(np.max(log_values))
    if not np.isfinite(maximum_log_value):
        raise RuntimeError("Sequential CTRV SMC predictive density collapsed.")
    shifted_sum = float(np.sum(np.exp(log_values - maximum_log_value)))
    if not np.isfinite(shifted_sum) or shifted_sum <= 0.0:
        raise RuntimeError("Sequential CTRV SMC predictive density collapsed.")
    return maximum_log_value + float(np.log(shifted_sum))


def _invert_two_by_two_matrices(
    matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable inverses and log determinants for positive 2x2 matrices."""
    determinant = (
        matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] * matrices[:, 1, 0]
    )
    determinant = np.maximum(determinant, _MINIMUM_SCALE**2)
    inverses = np.empty_like(matrices)
    inverses[:, 0, 0] = matrices[:, 1, 1] / determinant
    inverses[:, 0, 1] = -matrices[:, 0, 1] / determinant
    inverses[:, 1, 0] = -matrices[:, 1, 0] / determinant
    inverses[:, 1, 1] = matrices[:, 0, 0] / determinant
    return inverses, np.log(determinant)


def _regularize_two_by_two_covariances(covariances: np.ndarray) -> np.ndarray:
    """Return symmetric positive 2x2 covariance matrices."""
    symmetric = 0.5 * (covariances + np.swapaxes(covariances, 1, 2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, _MINIMUM_SCALE)
    return np.einsum(
        "nij,nj,nkj->nik",
        eigenvectors,
        eigenvalues,
        eigenvectors,
    )


def _sample_two_dimensional_gaussians(
    means: np.ndarray,
    covariances: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Draw one sample from every matching two-dimensional Gaussian."""
    cholesky = np.linalg.cholesky(covariances)
    innovations = generator.normal(size=means.shape)
    return means + np.einsum("nij,nj->ni", cholesky, innovations)


def _folded_motion_log_density(
    values: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Evaluate independent Gaussian motion density after speed reflection."""
    speed_scale = np.sqrt(variances[:, 0])
    turn_rate_scale = np.sqrt(variances[:, 1])
    positive_speed = _normal_log_density(values[:, 0], means[:, 0], speed_scale)
    negative_speed = _normal_log_density(-values[:, 0], means[:, 0], speed_scale)
    return np.logaddexp(positive_speed, negative_speed) + _normal_log_density(
        values[:, 1],
        means[:, 1],
        turn_rate_scale,
    )


def _folded_bivariate_log_density(
    values: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """Evaluate correlated Gaussian proposal density after speed reflection."""
    inverses, log_determinants = _invert_two_by_two_matrices(covariances)
    positive_residuals = values - means
    negative_values = values.copy()
    negative_values[:, 0] *= -1.0
    negative_residuals = negative_values - means
    positive_quadratic = np.einsum(
        "ni,nij,nj->n",
        positive_residuals,
        inverses,
        positive_residuals,
    )
    negative_quadratic = np.einsum(
        "ni,nij,nj->n",
        negative_residuals,
        inverses,
        negative_residuals,
    )
    normalization = -np.log(2.0 * np.pi) - 0.5 * log_determinants
    return np.logaddexp(
        normalization - 0.5 * positive_quadratic,
        normalization - 0.5 * negative_quadratic,
    )


def _normal_log_density(
    values: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Return elementwise univariate Normal log densities."""
    standardized = (values - means) / scales
    return -0.5 * np.log(2.0 * np.pi) - np.log(scales) - 0.5 * standardized**2
