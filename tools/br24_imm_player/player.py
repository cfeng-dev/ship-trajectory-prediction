"""Presentation-safe online session data for the local BR24 IMM player.

This module owns the raw BR24 input and optional local IMM tracker lifetime.
Reference labels are read only after a forecast has been produced, solely for
display and evaluation values.
"""

from __future__ import annotations

import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

import bayestraj.observations.br24 as br24
import bayestraj.observations.paths as paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMM_PYTHON = PROJECT_ROOT / "external/IMM/br24_imm.py"
DATA_FILE = paths.data_path("processed/BR24Dataset20131107_with_reference.csv")

INITIAL_OBSERVATION_COUNT = 10
PREDICTION_COUNT = 2
PLAYBACK_INTERVAL_MS = 1_000
FOLLOW_SHIP_VIEW_SPAN_METERS = 300.0
CONFIDENCE_LEVELS = (0.5, 0.9)
GAUSS_HERMITE_ORDER = 12
RESIDUAL_POSITION_NOISE_FLOOR_M = 1e-6


@dataclass(frozen=True, slots=True)
class IMMPlayerConfig:
    """Settings used to construct one isolated local IMM playback session."""

    data_file: Path = DATA_FILE
    initial_observation_count: int = INITIAL_OBSERVATION_COUNT
    prediction_count: int = PREDICTION_COUNT
    playback_interval_ms: int = PLAYBACK_INTERVAL_MS
    follow_ship_view_span_meters: float = FOLLOW_SHIP_VIEW_SPAN_METERS

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_file", Path(self.data_file).expanduser())
        for name in (
            "initial_observation_count",
            "prediction_count",
            "playback_interval_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        span = float(self.follow_ship_view_span_meters)
        if not np.isfinite(span) or span <= 0.0:
            raise ValueError(
                "follow_ship_view_span_meters must be a positive finite number."
            )
        object.__setattr__(self, "follow_ship_view_span_meters", span)


@dataclass(frozen=True, slots=True)
class IMMPlayerStage:
    """One online IMM posterior and forecast refreshed at a radar cycle."""

    observation_count: int
    cycle_counter: int
    observation_time_seconds: float
    detection_id: int | None
    tracked_times_seconds: np.ndarray
    tracked_positions: np.ndarray
    forecast: Any


@dataclass(frozen=True, slots=True)
class IMMMovementState:
    """Display-only movement values derived from the IMM track history."""

    position_meters: np.ndarray
    speed_mps: float | None
    heading_degrees: float | None
    turn_rate_degrees_per_second: float | None
    time_seconds: float


@dataclass(frozen=True, slots=True)
class IMMForecastMetrics:
    """Evaluation-only score values for the latest Gaussian IMM forecast."""

    ade_m: float | None
    fde_m: float | None
    rmse_m: float | None
    energy_score_m: float | None
    predictive_log_density: float | None
    joint_coverage_50: float | None
    joint_coverage_90: float | None


@dataclass(frozen=True, slots=True)
class IMMForecastUncertainty:
    """Forecast covariance summaries for the IMM uncertainty chart."""

    radius_50_meters: np.ndarray
    radius_90_meters: np.ndarray
    marginal_standard_deviations_meters: np.ndarray


@dataclass(frozen=True, slots=True)
class IMMDisplayState:
    """All non-mutating values needed by one player display refresh."""

    stage: IMMPlayerStage
    movement: IMMMovementState
    metrics: IMMForecastMetrics
    uncertainty: IMMForecastUncertainty


class IMMPlaybackStore:
    """Advance one local IMM through raw cycles and cache every display stage."""

    def __init__(self, config: IMMPlayerConfig) -> None:
        self.config = config
        self.data_file = config.data_file
        self.cycles = br24.read_br24_tracking_input(self.data_file)
        self.reference = br24.read_br24_tracking_reference(self.data_file)
        if self.reference is not None:
            self._validate_reference_alignment()
        self.times = np.asarray(
            [cycle.time_seconds for cycle in self.cycles], dtype=float
        )
        self.maximum_observation_count = len(self.cycles) - config.prediction_count
        if not 1 <= config.initial_observation_count <= self.maximum_observation_count:
            raise ValueError(
                "initial_observation_count must leave room for the requested "
                "prediction horizon."
            )
        if not self.cycles[0].detections:
            raise ValueError(
                "The earliest BR24 radar cycle has no detection to initialize."
            )

        tracker_type = _load_optional_imm_tracker()
        initial_detection = self.cycles[0].detections[0]
        self._tracker = tracker_type.initialize(
            self.cycles[0], initial_detection_id=initial_detection.detection_id
        )
        self._processed_cycle_count = 1
        self._tracked_positions = [
            np.asarray(self._tracker.estimated_position, dtype=float).copy()
        ]
        self._detection_ids = [initial_detection.detection_id]
        self._stages: dict[int, IMMPlayerStage] = {}
        self._display_states: dict[int, IMMDisplayState] = {}
        self._capture_stage()

    def stage_for(self, observation_count: int) -> IMMPlayerStage:
        """Return the forecast refreshed after exactly ``observation_count`` cycles."""
        observation_count = self._validate_observation_count(observation_count)
        self._advance_to(observation_count)
        return self._stages[observation_count]

    def display_state_for(self, observation_count: int) -> IMMDisplayState:
        """Return one cached stage and its display-only state and metrics."""
        observation_count = self._validate_observation_count(observation_count)
        self._advance_to(observation_count)
        if observation_count not in self._display_states:
            stage = self._stages[observation_count]
            self._display_states[observation_count] = IMMDisplayState(
                stage=stage,
                movement=_movement_state(stage),
                metrics=_forecast_metrics(stage, self.reference),
                uncertainty=_forecast_uncertainty(stage),
            )
        return self._display_states[observation_count]

    def _advance_to(self, observation_count: int) -> None:
        if observation_count in self._stages:
            return
        if observation_count < self._processed_cycle_count:
            raise RuntimeError("An earlier IMM stage was not cached.")
        while self._processed_cycle_count < observation_count:
            step = self._tracker.process_cycle(self.cycles[self._processed_cycle_count])
            self._tracked_positions.append(
                np.asarray(self._tracker.estimated_position, dtype=float).copy()
            )
            self._detection_ids.append(step.detection_id)
            self._processed_cycle_count += 1
            self._capture_stage()

    def _capture_stage(self) -> None:
        observation_count = self._processed_cycle_count
        future_times = self.times[
            observation_count : observation_count + self.config.prediction_count
        ]
        if future_times.shape != (self.config.prediction_count,):
            raise RuntimeError(
                "IMM stage does not leave the configured forecast horizon."
            )
        forecast = self._tracker.forecast(future_times)
        self._stages[observation_count] = IMMPlayerStage(
            observation_count=observation_count,
            cycle_counter=self.cycles[observation_count - 1].cycle_counter,
            observation_time_seconds=float(self._tracker.last_observation_time_seconds),
            detection_id=self._detection_ids[-1],
            tracked_times_seconds=self.times[:observation_count].copy(),
            tracked_positions=np.asarray(self._tracked_positions, dtype=float).copy(),
            forecast=forecast,
        )

    def _validate_observation_count(self, observation_count: int) -> int:
        if isinstance(observation_count, bool) or not isinstance(
            observation_count, int
        ):
            raise ValueError("observation_count must be an integer.")
        if not 1 <= observation_count <= self.maximum_observation_count:
            raise ValueError(
                "observation_count must leave room for the configured forecast."
            )
        return observation_count

    def _validate_reference_alignment(self) -> None:
        """Require labels to align with cycles, while keeping them out of the IMM."""
        cycle_counters = np.asarray(
            [cycle.cycle_counter for cycle in self.cycles], dtype=int
        )
        cycle_times = np.asarray(
            [cycle.time_seconds for cycle in self.cycles], dtype=float
        )
        if not np.array_equal(
            self.reference.cycle_counters, cycle_counters
        ) or not np.array_equal(self.reference.time_seconds, cycle_times):
            raise ValueError(
                "BR24 reference labels must align exactly with radar cycles."
            )


def _movement_state(stage: IMMPlayerStage) -> IMMMovementState:
    """Derive current movement quantities from track positions and cycle times."""
    positions = np.asarray(stage.tracked_positions, dtype=float)
    times = np.asarray(stage.tracked_times_seconds, dtype=float)
    speed_mps = None
    heading_degrees = None
    turn_rate_degrees_per_second = None
    if len(positions) >= 2:
        current_delta = positions[-1] - positions[-2]
        current_interval = float(times[-1] - times[-2])
        if current_interval <= 0.0:
            raise ValueError("IMM track times must be strictly increasing.")
        speed_mps = float(np.linalg.norm(current_delta) / current_interval)
        if speed_mps > 0.0:
            heading_radians = float(np.arctan2(current_delta[1], current_delta[0]))
            heading_degrees = float(np.rad2deg(heading_radians))
    if len(positions) >= 3 and heading_degrees is not None:
        previous_delta = positions[-2] - positions[-3]
        previous_interval = float(times[-2] - times[-3])
        if previous_interval <= 0.0:
            raise ValueError("IMM track times must be strictly increasing.")
        if np.linalg.norm(previous_delta) > 0.0:
            previous_heading = float(np.arctan2(previous_delta[1], previous_delta[0]))
            heading_change = np.arctan2(
                np.sin(np.deg2rad(heading_degrees) - previous_heading),
                np.cos(np.deg2rad(heading_degrees) - previous_heading),
            )
            turn_interval = 0.5 * (previous_interval + current_interval)
            turn_rate_degrees_per_second = float(
                np.rad2deg(heading_change) / turn_interval
            )
    return IMMMovementState(
        position_meters=positions[-1].copy(),
        speed_mps=speed_mps,
        heading_degrees=heading_degrees,
        turn_rate_degrees_per_second=turn_rate_degrees_per_second,
        time_seconds=float(stage.observation_time_seconds),
    )


def _forecast_metrics(stage: IMMPlayerStage, reference) -> IMMForecastMetrics:
    """Score the current Gaussian forecast only after the tracker has updated."""
    if reference is None:
        return IMMForecastMetrics(None, None, None, None, None, None, None)
    forecast = stage.forecast
    means = np.asarray(forecast.position_means, dtype=float)
    covariances = _regularize_covariances(forecast.position_covariances)
    target_indices = np.arange(
        stage.observation_count,
        stage.observation_count + means.shape[0],
    )
    targets = np.column_stack(
        (reference.x_meters[target_indices], reference.y_meters[target_indices])
    )
    errors = np.linalg.norm(means - targets, axis=1)
    log_densities = np.asarray(
        [
            _gaussian_log_density(
                target,
                mean,
                covariance + np.eye(2) * RESIDUAL_POSITION_NOISE_FLOOR_M**2,
            )
            for target, mean, covariance in zip(
                targets, means, covariances, strict=True
            )
        ]
    )
    energy_scores = np.asarray(
        [
            _gaussian_energy_score(target, mean, covariance)
            for target, mean, covariance in zip(
                targets, means, covariances, strict=True
            )
        ]
    )
    squared_mahalanobis = np.asarray(
        [
            _squared_mahalanobis(target - mean, covariance)
            for target, mean, covariance in zip(
                targets, means, covariances, strict=True
            )
        ]
    )
    return IMMForecastMetrics(
        ade_m=float(errors.mean()),
        fde_m=float(errors[-1]),
        rmse_m=float(np.sqrt(np.mean(errors**2))),
        energy_score_m=float(energy_scores.mean()),
        predictive_log_density=float(log_densities.mean()),
        joint_coverage_50=float(np.mean(squared_mahalanobis <= chi2.ppf(0.5, 2))),
        joint_coverage_90=float(np.mean(squared_mahalanobis <= chi2.ppf(0.9, 2))),
    )


def _forecast_uncertainty(stage: IMMPlayerStage) -> IMMForecastUncertainty:
    """Summarize 2D Gaussian covariances for the display-only uncertainty view."""
    covariances = _regularize_covariances(stage.forecast.position_covariances)
    equivalent_radii = np.sqrt(np.sqrt(np.linalg.det(covariances)))
    return IMMForecastUncertainty(
        radius_50_meters=np.sqrt(chi2.ppf(0.5, 2)) * equivalent_radii,
        radius_90_meters=np.sqrt(chi2.ppf(0.9, 2)) * equivalent_radii,
        marginal_standard_deviations_meters=np.sqrt(
            np.diagonal(covariances, axis1=1, axis2=2)
        ),
    )


def _regularize_covariances(covariances) -> np.ndarray:
    """Return symmetric positive-definite position covariances."""
    covariances = np.asarray(covariances, dtype=float)
    if (
        covariances.ndim != 3
        or covariances.shape[1:] != (2, 2)
        or not np.all(np.isfinite(covariances))
    ):
        raise ValueError("IMM forecast covariances must be finite (count, 2, 2).")
    result = np.empty_like(covariances)
    for index, covariance in enumerate(covariances):
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
        result[index] = (eigenvectors * np.maximum(eigenvalues, 1e-12)) @ eigenvectors.T
    return result


def _gaussian_log_density(actual, mean, covariance) -> float:
    """Return the bivariate Gaussian log density at one reference position."""
    difference = np.asarray(actual, dtype=float) - np.asarray(mean, dtype=float)
    solved = np.linalg.solve(covariance, difference)
    _, log_determinant = np.linalg.slogdet(covariance)
    return float(-np.log(2.0 * np.pi) - 0.5 * (log_determinant + difference @ solved))


def _gaussian_energy_score(actual, mean, covariance) -> float:
    """Integrate the bivariate Gaussian energy score by Gauss-Hermite nodes."""
    roots, weights = np.polynomial.hermite.hermgauss(GAUSS_HERMITE_ORDER)
    standard_points = np.sqrt(2.0) * np.stack(
        np.meshgrid(roots, roots, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    quadrature_weights = np.outer(weights, weights).reshape(-1) / np.pi
    transform = np.linalg.cholesky(covariance)
    samples = np.asarray(mean, dtype=float) + standard_points @ transform.T
    distance_to_actual = np.linalg.norm(samples - actual, axis=1)
    independent_difference = standard_points @ (np.sqrt(2.0) * transform).T
    pairwise_distance = np.linalg.norm(independent_difference, axis=1)
    return float(
        quadrature_weights @ distance_to_actual
        - 0.5 * quadrature_weights @ pairwise_distance
    )


def _squared_mahalanobis(difference, covariance) -> float:
    """Return one squared Mahalanobis distance."""
    difference = np.asarray(difference, dtype=float)
    return float(difference @ np.linalg.solve(covariance, difference))


def _load_optional_imm_tracker():
    """Load the deliberately untracked local IMM implementation when available."""
    if not IMM_PYTHON.is_file():
        raise FileNotFoundError(
            "Optional IMM implementation not found: external/IMM/br24_imm.py. "
            "Restore the local external IMM files to open this player."
        )
    module = runpy.run_path(str(IMM_PYTHON))
    try:
        return module["BR24IMMTracker"]
    except KeyError as error:
        raise RuntimeError(
            "The local IMM implementation must expose BR24IMMTracker."
        ) from error
