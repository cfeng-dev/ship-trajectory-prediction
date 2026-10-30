"""Causal VI, MCMC, and Sequential VI backends for raw radar cycles.

Association uses each backend's own latent predictive distribution. Until a
fit is identifiable, a labelled seed-conditioned prior is propagated without
conditioning on buffered fixes. The first fit starts from the configured
priors and real fixes, discarding that temporary predictive population.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from bayestraj import numeric_validation
from bayestraj.inference import ctrv_cmdstan, ctrv_sequential_vi, particle_utils
from bayestraj.models import ctrv
from bayestraj.models.bayesian_ctrv import BayesianCTRVPriors
from bayestraj.observations.position import PositionObservations
from bayestraj.observations.window import TrajectoryWindowData

RADAR_INFERENCE_METHODS = ("vi", "mcmc", "smc", "sequential_vi")


def normalize_radar_inference_method(value: str) -> str:
    """Validate the explicitly selected radar inference method."""
    if value not in RADAR_INFERENCE_METHODS:
        raise ValueError(
            "inference_method must be one of: " + ", ".join(RADAR_INFERENCE_METHODS)
        )
    return value


def _stream_seed(seed: int, stream: int, index: int = 0) -> int:
    """Derive reproducible independent fitting and propagation streams."""
    return int(np.random.SeedSequence([seed, stream, index]).generate_state(1)[0])


def _states(draws: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [
            draws[name]
            for name in (
                "x_at_origin",
                "y_at_origin",
                "speed_at_origin",
                "heading_at_origin",
                "turn_rate_at_origin",
            )
        ]
    )


def _replace_states(
    draws: dict[str, np.ndarray], states: np.ndarray
) -> dict[str, np.ndarray]:
    names = (
        "x_at_origin",
        "y_at_origin",
        "speed_at_origin",
        "heading_at_origin",
        "turn_rate_at_origin",
    )
    return {
        **draws,
        **{name: states[:, index].copy() for index, name in enumerate(names)},
    }


def _perturb_motion(
    states: np.ndarray,
    draws: dict[str, np.ndarray],
    interval: float,
    generator: np.random.Generator,
) -> np.ndarray:
    """Prepare next-interval motion after the preceding raw cycle interval."""
    states = states.copy()
    scale = ctrv.process_time_scale(interval)
    states[:, ctrv.STATE_SPEED_INDEX] += generator.normal(
        0.0, draws["sigma_speed_process"] * scale
    )
    states[:, ctrv.STATE_TURN_RATE_INDEX] += generator.normal(
        0.0, draws["sigma_turn_rate_process"] * scale
    )
    ctrv.normalize_states(states)
    return states


@dataclass(slots=True)
class _PredictivePopulation:
    """Unweighted draws with separate current and prepared motion states."""

    draws: dict[str, np.ndarray]
    next_states: np.ndarray
    time_seconds: float
    generator: np.random.Generator

    def predict(self, time_seconds: float) -> tuple[np.ndarray, np.ndarray]:
        positions = ctrv.transition_states(
            self.next_states, time_seconds - self.time_seconds
        )[:, :2]
        mean = positions.mean(axis=0)
        centered = positions - mean
        return mean, centered.T @ centered / positions.shape[0]

    def log_predictive_density(
        self, time_seconds: float, position: np.ndarray, covariance: np.ndarray
    ) -> float:
        """Score a new fix using the same latent population as association."""
        positions = ctrv.transition_states(
            self.next_states, time_seconds - self.time_seconds
        )[:, :2]
        return ctrv_sequential_vi._predictive_log_density(
            ctrv_sequential_vi.SequentialVIPredictiveDraws(
                x=positions[:, 0],
                y=positions[:, 1],
                sigma_position_observation=self.draws["sigma_position_observation"],
            ),
            x_observed=position[0],
            y_observed=position[1],
            measurement_covariance=covariance,
        )

    def advance(self, time_seconds: float) -> None:
        interval = time_seconds - self.time_seconds
        origins = ctrv.transition_states(self.next_states, interval)
        self.draws = _replace_states(self.draws, origins)
        self.next_states = _perturb_motion(
            origins, self.draws, interval, self.generator
        )
        self.time_seconds = time_seconds

    def forecast(self, times, *, seed: int) -> particle_utils.SequentialCTRVFit:
        # Preserve the existing Stan/SMC/Sequential VI forecast convention:
        # the first displacement uses transition-informed origin motion;
        # process noise follows it. Association separately uses next_states,
        # whose motion is already prepared for the next raw radar cycle.
        return ctrv_sequential_vi._forecast_from_origin_draws(
            self.draws,
            origin_time_seconds=self.time_seconds,
            future_time_seconds=times,
            generator=np.random.default_rng(seed),
        )


def _seed_population(
    position: np.ndarray,
    covariance: np.ndarray,
    *,
    time_seconds: float,
    priors: BayesianCTRVPriors,
    draw_count: int,
    seed: int,
) -> _PredictivePopulation:
    generator = np.random.default_rng(_stream_seed(seed, 1))
    residual = np.hypot(
        generator.exponential(
            1.0 / priors.sigma_position_observation_prior_rate, draw_count
        ),
        priors.sigma_position_observation_floor_m,
    )
    sensor_innovation = generator.multivariate_normal(
        np.zeros(2), covariance, size=draw_count
    )
    positions = (
        position
        + sensor_innovation
        + generator.normal(size=(draw_count, 2)) * residual[:, None]
    )
    draws = {
        "x_at_origin": positions[:, 0],
        "y_at_origin": positions[:, 1],
        "speed_at_origin": np.maximum(
            np.abs(generator.normal(0.0, priors.speed_prior_scale, draw_count)),
            ctrv.SPEED_STATE_LOWER_MPS,
        ),
        "heading_at_origin": generator.uniform(-np.pi, np.pi, draw_count),
        "turn_rate_at_origin": generator.normal(
            0.0, priors.turn_rate_prior_scale, draw_count
        ),
        "sigma_position_observation": residual,
        "sigma_speed_process": generator.exponential(
            1.0 / priors.sigma_speed_process_prior_rate, draw_count
        ),
        "sigma_turn_rate_process": generator.exponential(
            1.0 / priors.sigma_turn_rate_process_prior_rate, draw_count
        ),
    }
    return _PredictivePopulation(
        draws=draws,
        next_states=_states(draws),
        time_seconds=time_seconds,
        generator=np.random.default_rng(_stream_seed(seed, 2)),
    )


@dataclass(frozen=True, slots=True)
class _CycleFix:
    time_seconds: float
    position: np.ndarray | None
    covariance: np.ndarray | None


def _observation_arrays(
    cycles: list[_CycleFix],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.array([cycle.time_seconds for cycle in cycles])
    positions = np.zeros((len(cycles), 2))
    covariances = np.zeros((len(cycles), 2, 2))
    mask = np.array([cycle.position is not None for cycle in cycles])
    for index, cycle in enumerate(cycles):
        if cycle.position is not None:
            positions[index] = cycle.position
            covariances[index] = cycle.covariance
    return times, positions, covariances, mask


@dataclass(slots=True)
class CausalBR24RadarFilter:
    """Adapt batch or Sequential VI to one raw radar cycle at a time."""

    inference_method: str
    priors: BayesianCTRVPriors
    observation_count: int
    config: Mapping[str, Any] | ctrv_sequential_vi.SequentialVIConfig
    seed: int
    last_observation_time_seconds: float
    _population: _PredictivePopulation = field(repr=False)
    _cycles: list[_CycleFix] = field(repr=False)
    _sequential: ctrv_sequential_vi.SequentialCTRVVI | None = field(
        default=None, repr=False
    )
    processed_observation_count: int = 1
    accepted_measurement_count: int = 1
    posterior_fit_count: int = 0
    fit_status: str = "seed_conditioned_prior"
    last_fit_time_seconds: float | None = None
    fit_converged: bool | None = None
    last_fit_error: str | None = None
    last_fit: object | None = field(default=None, repr=False)
    last_fit_runtime_seconds: float | None = None
    cumulative_fit_runtime_seconds: float = 0.0
    last_log_predictive_density: float | None = None
    predictive_log_density_history: list[float] = field(default_factory=list)

    @classmethod
    def initialize(
        cls,
        time_seconds: float,
        position: np.ndarray,
        measurement_covariance: np.ndarray,
        *,
        inference_method: str,
        priors: BayesianCTRVPriors,
        observation_count: int = 10,
        vi_config: Mapping[str, Any] | None = None,
        mcmc_config: Mapping[str, Any] | None = None,
        sequential_vi_config: ctrv_sequential_vi.SequentialVIConfig | None = None,
        seed: int = 42,
    ) -> CausalBR24RadarFilter:
        normalize_radar_inference_method(inference_method)
        if inference_method == "smc":
            raise ValueError("Use the existing SMC filter for SMC inference.")
        observation_count = numeric_validation.validate_non_negative_integer(
            "observation_count", observation_count
        )
        if observation_count < 3:
            raise ValueError("observation_count must be at least three raw cycles.")
        seed = numeric_validation.validate_non_negative_integer("seed", seed)
        if inference_method == "sequential_vi":
            config = sequential_vi_config or ctrv_sequential_vi.SequentialVIConfig()
            if not isinstance(config, ctrv_sequential_vi.SequentialVIConfig):
                raise TypeError("sequential_vi_config must be a SequentialVIConfig.")
            draw_count = config.draws
        else:
            selected = vi_config if inference_method == "vi" else mcmc_config
            config = dict(selected or {})
            if inference_method == "vi":
                config.setdefault("require_converged", False)
            forbidden = {"seed", "priors", "position_observations", "inference_method"}
            if forbidden.intersection(config):
                raise ValueError(
                    "Inference options must not override radar inputs or seed."
                )
            draw_count = config.get("draws", 1_000)
            if (
                isinstance(draw_count, bool)
                or not isinstance(draw_count, int)
                or draw_count < 2
            ):
                raise ValueError(
                    "draws must be an integer greater than or equal to two."
                )
        return cls(
            inference_method=inference_method,
            priors=priors,
            observation_count=observation_count,
            config=config,
            seed=seed,
            last_observation_time_seconds=float(time_seconds),
            _population=_seed_population(
                position,
                measurement_covariance,
                time_seconds=time_seconds,
                priors=priors,
                draw_count=draw_count,
                seed=seed,
            ),
            _cycles=[
                _CycleFix(
                    float(time_seconds), position.copy(), measurement_covariance.copy()
                )
            ],
        )

    @property
    def is_prior_predictive(self) -> bool:
        return self.posterior_fit_count == 0

    @property
    def converged(self) -> bool | None:
        """Return convergence for VI; MCMC diagnostics remain on last_fit."""
        if self._sequential is not None:
            return self._sequential.bootstrap_converged and all(
                update.converged for update in self._sequential.updates
            )
        return self.fit_converged

    @property
    def sequential_filter(self) -> ctrv_sequential_vi.SequentialCTRVVI | None:
        """Expose the initialized Sequential VI backend for reporting."""
        return self._sequential

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return method-specific fit status without inventing particle ESS."""
        return {
            "inference_method": self.inference_method,
            "fit_status": self.fit_status,
            "is_prior_predictive": self.is_prior_predictive,
            "processed_cycle_count": self.processed_observation_count,
            "accepted_measurement_count": self.accepted_measurement_count,
            "window_cycle_count": len(self._cycles),
            "window_accepted_measurement_count": sum(
                cycle.position is not None for cycle in self._cycles
            ),
            "posterior_fit_count": self.posterior_fit_count,
            "last_fit_time_seconds": self.last_fit_time_seconds,
            "last_fit_runtime_seconds": self.last_fit_runtime_seconds,
            "cumulative_fit_runtime_seconds": self.cumulative_fit_runtime_seconds,
            "converged": self.converged,
            "bootstrap_completed": self._sequential is not None,
            "last_fit_error": self.last_fit_error,
        }

    @property
    def estimated_position(self) -> np.ndarray:
        if self._sequential is not None:
            return self._sequential.estimated_position
        return np.array(
            [
                self._population.draws["x_at_origin"].mean(),
                self._population.draws["y_at_origin"].mean(),
            ]
        )

    @property
    def expected_position_observation_variance(self) -> float:
        if self._sequential is not None:
            return self._sequential.expected_position_observation_variance
        return float(np.mean(self._population.draws["sigma_position_observation"] ** 2))

    def predict_position_distribution(self, time_seconds: float):
        time_seconds = self._validate_next_time(time_seconds)
        if self._sequential is not None:
            return self._sequential.predict_position_distribution(time_seconds)
        return self._population.predict(time_seconds)

    def sample_current_posterior(self, *, seed: int):
        numeric_validation.validate_non_negative_integer("seed", seed)
        if self._sequential is not None:
            return self._sequential.sample_current_posterior(seed=seed)
        return particle_utils.SequentialCTRVFit(self._population.draws)

    def forecast(self, future_time_seconds, *, seed: int):
        numeric_validation.validate_non_negative_integer("seed", seed)
        if self._sequential is not None:
            return self._sequential.forecast(future_time_seconds, seed=seed)
        return self._population.forecast(future_time_seconds, seed=seed)

    def update(self, time_seconds, x_observed, y_observed, *, measurement_covariance):
        position = np.array(
            [
                numeric_validation.validate_finite_scalar("x_observed", x_observed),
                numeric_validation.validate_finite_scalar("y_observed", y_observed),
            ]
        )
        covariance = np.asarray(measurement_covariance, dtype=float)
        if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
            raise ValueError("measurement_covariance must be a finite 2x2 matrix.")
        if (
            not np.allclose(covariance, covariance.T)
            or np.linalg.eigvalsh(covariance)[0] < -1e-9
        ):
            raise ValueError(
                "measurement_covariance must be symmetric positive semidefinite."
            )
        self._process(time_seconds, position, covariance)

    def advance_without_measurement(self, time_seconds):
        self._process(time_seconds, None, None)

    def _validate_next_time(self, time_seconds):
        value = numeric_validation.validate_finite_scalar("time_seconds", time_seconds)
        if value <= self.last_observation_time_seconds:
            raise ValueError("time_seconds must follow the previous radar cycle.")
        return value

    def _process(self, time_seconds, position, covariance):
        value = self._validate_next_time(time_seconds)
        log_predictive_density = None
        if self._sequential is None and position is not None:
            # Evaluate before advance/refit, which can replace the population
            # with a posterior already conditioned on this observation.
            log_predictive_density = self._population.log_predictive_density(
                value, position, covariance
            )
        self._cycles.append(_CycleFix(value, position, covariance))
        if self.inference_method != "sequential_vi" or self._sequential is not None:
            self._cycles = self._cycles[-self.observation_count :]
        if self._sequential is not None:
            if position is None:
                self._sequential.advance_without_measurement(value)
            else:
                record = self._sequential.update(
                    value, *position, measurement_covariance=covariance
                )
                log_predictive_density = self._sequential.last_log_predictive_density
                self.posterior_fit_count += 1
                self.last_fit_time_seconds = value
                self.fit_converged = record.converged
                self.last_fit_runtime_seconds = record.runtime_seconds
                self.cumulative_fit_runtime_seconds += record.runtime_seconds
            self.fit_status = (
                "posterior" if position is not None else "posterior_prediction"
            )
        else:
            self._population.advance(value)
            times, positions, covariances, mask = _observation_arrays(self._cycles)
            minimum = (
                self.config.n_bootstrap
                if self.inference_method == "sequential_vi"
                else 3
            )
            if np.count_nonzero(mask) >= minimum:
                self._fit(times, positions, covariances, mask)
            else:
                self.fit_status = (
                    "seed_conditioned_prior"
                    if self.is_prior_predictive
                    else "posterior_prediction_insufficient_fixes"
                )
        self.last_observation_time_seconds = value
        self.processed_observation_count += 1
        self.accepted_measurement_count += int(position is not None)
        self.last_log_predictive_density = log_predictive_density
        if log_predictive_density is not None:
            self.predictive_log_density_history.append(log_predictive_density)

    def _fit(self, times, positions, covariances, mask):
        fit_seed = _stream_seed(self.seed, 0, self.processed_observation_count)
        started = perf_counter()
        try:
            if self.inference_method == "sequential_vi":
                self._sequential = ctrv_sequential_vi.SequentialCTRVVI.initialize(
                    times,
                    positions[:, 0],
                    positions[:, 1],
                    priors=self.priors,
                    config=self.config,
                    seed=fit_seed,
                    measurement_covariances=covariances,
                    observed_mask=mask,
                    batch_fitter=self._sequential_batch_fit,
                    update_fitter=self._sequential_update_fit,
                )
                self.fit_converged = self._sequential.bootstrap_converged
            else:
                interval = float(times[-1] - times[-2])
                window_times = np.append(times, times[-1] + interval)
                window = TrajectoryWindowData(
                    timestamps=pd.DatetimeIndex(
                        pd.to_datetime(window_times, unit="s", origin="unix", utc=True)
                    ),
                    time_seconds=window_times,
                    x_meters=np.append(positions[:, 0], 0.0),
                    y_meters=np.append(positions[:, 1], 0.0),
                    reference_longitude=None,
                    reference_latitude=None,
                    gps_speed_mps=np.full(window_times.shape, np.nan),
                    observation_count=times.size,
                )
                observations = PositionObservations(
                    time_seconds=times,
                    x_meters=positions[:, 0],
                    y_meters=positions[:, 1],
                    position_noise_std_m=0.0,
                    noise_seed=fit_seed,
                    measurement_covariances=covariances,
                    observed_mask=mask,
                )
                fit = ctrv_cmdstan.fit_bayesian_ctrv_model(
                    window,
                    priors=self.priors,
                    position_observations=observations,
                    inference_method=self.inference_method,
                    seed=fit_seed,
                    **self.config,
                )
                self.last_fit = fit
                draws = ctrv_sequential_vi._physical_fit_draws(fit)
                generator = np.random.default_rng(
                    _stream_seed(self.seed, 2, self.processed_observation_count)
                )
                self._population = _PredictivePopulation(
                    draws=draws,
                    next_states=_perturb_motion(
                        _states(draws), draws, interval, generator
                    ),
                    time_seconds=float(times[-1]),
                    generator=generator,
                )
                self.fit_converged = (
                    ctrv_sequential_vi._variational_fit_converged(fit)
                    if self.inference_method == "vi"
                    else None
                )
        except Exception as error:
            self.fit_status = "fit_failed"
            self.last_fit_error = str(error)
            raise
        self.posterior_fit_count += 1
        self.last_fit_time_seconds = float(times[-1])
        self.fit_status = "posterior"
        self.last_fit_error = None
        self.last_fit_runtime_seconds = perf_counter() - started
        self.cumulative_fit_runtime_seconds += self.last_fit_runtime_seconds

    def _sequential_batch_fit(self, *args, **kwargs):
        fit = ctrv_cmdstan.fit_bayesian_ctrv_model(*args, **kwargs)
        self.last_fit = fit
        return fit

    def _sequential_update_fit(self, *args, **kwargs):
        fit = ctrv_sequential_vi.fit_sequential_update(*args, **kwargs)
        self.last_fit = fit
        return fit
