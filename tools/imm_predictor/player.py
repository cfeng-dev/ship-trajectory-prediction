"""Presentation-safe online session data for the IMM Predictor.

This module owns the raw BR24 input and optional local IMM tracker lifetime.
Reference labels are read only after a forecast has been produced, solely for
display and evaluation values.
"""

from __future__ import annotations

import runpy
import time
from dataclasses import dataclass, fields, replace
from math import fsum
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr
from scipy.stats import chi2

import bayestraj.observations.br24 as br24
import bayestraj.observations.paths as paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMM_PYTHON = PROJECT_ROOT / "external/IMM/br24_imm.py"
DATA_FILE = paths.data_path("processed/BR24Dataset20131107_with_reference.csv")

PREDICTION_COUNT = 2
PLAYBACK_INTERVAL_MS = 1_000
FOLLOW_SHIP_VIEW_SPAN_METERS = 100.0
CONFIDENCE_LEVELS = (0.5, 0.9)
PREDICTION_REGION_STYLE_LABELS = {
    "funnel": "Horizon density region",
    "density_regions": "Per-step density regions",
    "ellipses": "Covariance ellipses",
    "horizon_density_contours": "Horizon density contours",
    "density_contours": "Per-step density contours",
    "none": "No uncertainty region",
}
PLOT_DISPLAY_OPTIONS = {
    "show_legend": "Show legend",
    "show_recorded_trajectory": "Show reference trajectory",
    "show_imm_track": "Show IMM track through N",
    "show_current_position": "Show current position",
    "show_sample_trajectories": "Show sampled IMM trajectories",
    "show_forecast_median": "Show forecast (median)",
    "show_forecast_mean": "Show forecast (mean)",
    "show_prediction_region_50": "Show 50% posterior-predictive region",
    "show_prediction_region_90": "Show 90% posterior-predictive region",
}
GAUSS_HERMITE_ORDER = 12
RESIDUAL_POSITION_NOISE_FLOOR_M = 1e-6


@dataclass(frozen=True, slots=True)
class IMMPlayerConfig:
    """Settings used to construct one isolated local IMM playback session."""

    data_file: Path = DATA_FILE
    prediction_count: int = PREDICTION_COUNT
    playback_interval_ms: int = PLAYBACK_INTERVAL_MS
    follow_ship_view_span_meters: float = FOLLOW_SHIP_VIEW_SPAN_METERS
    prediction_region_style: str = "funnel"
    show_prediction_region_50: bool = True
    show_prediction_region_90: bool = True
    show_legend: bool = True
    show_recorded_trajectory: bool = True
    show_imm_track: bool = True
    show_current_position: bool = True
    show_forecast_median: bool = True
    show_forecast_mean: bool = False
    show_sample_trajectories: bool = True
    prediction_sample_count: int = 20
    forecast_seed: int = 42
    start_index: int = 0
    position_noise_std_m: float = 0.0
    position_noise_seed: int = 2026

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_file", Path(self.data_file).expanduser())
        for name in (
            "start_index",
            "position_noise_seed",
            "prediction_count",
            "prediction_sample_count",
            "forecast_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        noise = float(self.position_noise_std_m)
        if not np.isfinite(noise) or noise < 0:
            raise ValueError("position_noise_std_m must be finite and non-negative.")
        object.__setattr__(self, "position_noise_std_m", noise)
        if self.prediction_region_style not in PREDICTION_REGION_STYLE_LABELS:
            raise ValueError("Unsupported prediction uncertainty style.")
        for name in PLOT_DISPLAY_OPTIONS:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")
        for name in ("playback_interval_ms",):
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
class IMMStatePosterior:
    """Read-only [x, y, heading, turn rate, speed] Gaussian state moments."""

    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float).copy()
        covariance = np.asarray(self.covariance, dtype=float).copy()
        if mean.shape != (5,) or not np.all(np.isfinite(mean)):
            raise ValueError("IMM posterior mean must be a finite five-element vector.")
        if covariance.shape != (5, 5) or not np.all(np.isfinite(covariance)):
            raise ValueError("IMM posterior covariance must be a finite 5-by-5 matrix.")
        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(covariance))))
        if not np.allclose(covariance, covariance.T, rtol=0, atol=tolerance):
            raise ValueError("IMM posterior covariance must be symmetric.")
        if np.min(np.linalg.eigvalsh(covariance)) < -tolerance:
            raise ValueError("IMM posterior covariance must be positive semidefinite.")
        mean.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True, slots=True)
class IMMStateMixture:
    """Cached CV/CTRV/CA moments in the MATLAB common CTRV coordinates."""

    weights: np.ndarray
    components: tuple[IMMStatePosterior, ...]

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float).copy()
        components = tuple(self.components)
        if (
            weights.shape != (3,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-10)
        ):
            raise ValueError(
                "IMM mixture needs three non-negative probabilities summing to one."
            )
        if len(components) != 3 or not all(
            isinstance(component, IMMStatePosterior) for component in components
        ):
            raise ValueError("IMM mixture needs CV, CTRV and CA state moments.")
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "components", components)

    def marginal_parameters(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return active weights, means and standard deviations in native SI units."""
        active = self.weights > 0
        weights = self.weights[active]
        means = np.array([component.mean[index] for component in self.components])[
            active
        ]
        sigmas = np.sqrt(
            np.maximum(
                [component.covariance[index, index] for component in self.components], 0
            )
        )[active]
        if index == 2:
            # Share the plot's local heading branch with scalar summaries.
            center = np.angle(np.sum(weights * np.exp(1j * means)))
            means = center + (means - center + np.pi) % (2 * np.pi) - np.pi
        return weights, means, sigmas

    def marginal_median(self, index: int) -> float:
        """Return the lower 0.5 quantile, retaining discrete probability jumps."""
        weights, means, sigmas = self.marginal_parameters(index)
        continuous = sigmas > 0

        def balance(value: float, *, before_atom=False) -> float:
            # Sign of F(value)-0.5, scaled in log space so tiny Gaussian tails
            # neither cancel against 0.5 nor underflow between distant modes.
            left = means <= value
            if before_atom:
                left &= continuous | (means != value)
            base = float(np.sum(weights[left])) - 0.5
            log_terms = []
            signs = []
            if base:
                log_terms.append(float(np.log(abs(base))))
                signs.append(float(np.sign(base)))
            for weight, mean, sigma, on_left in zip(
                weights[continuous],
                means[continuous],
                sigmas[continuous],
                left[continuous],
                strict=True,
            ):
                log_terms.append(
                    float(np.log(weight) + log_ndtr(-abs(value - mean) / sigma))
                )
                signs.append(-1.0 if on_left else 1.0)
            if not log_terms:
                return 0.0
            scale = max(log_terms)
            if not np.isfinite(scale):
                return 0.0
            return fsum(
                sign * np.exp(term - scale)
                for sign, term in zip(signs, log_terms, strict=True)
            )

        for atom in np.unique(means[~continuous]):
            if balance(float(atom), before_atom=True) <= 0 <= balance(float(atom)):
                return float(atom)
        lower, upper = float(np.min(means)), float(np.max(means))
        if lower == upper:
            return lower
        padding = 8 * float(np.max(sigmas))
        return float(
            brentq(balance, lower - padding, upper + padding, xtol=1e-12, maxiter=200)
        )


@dataclass(frozen=True, slots=True)
class IMMPlayerStage:
    """One online IMM posterior and forecast refreshed at a radar cycle."""

    observation_count: int
    cycle_counter: int
    observation_time_seconds: float
    detection_id: int | None
    nis: float | None
    measurement_covariance: np.ndarray | None
    update_time_seconds: float
    tracked_times_seconds: np.ndarray
    tracked_positions: np.ndarray
    forecast: Any
    sample_positions: np.ndarray | None = None
    posterior: IMMStatePosterior | None = None
    posterior_mixture: IMMStateMixture | None = None
    forecast_median_positions: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class IMMMovementState:
    """Display-only reference motion values, optionally derived from reference data."""

    position_meters: np.ndarray
    speed_mps: float | None
    heading_degrees: float | None
    turn_rate_degrees_per_second: float | None
    time_seconds: float
    turn_rate_is_derived: bool = False
    speed_is_derived: bool = False
    heading_is_derived: bool = False


@dataclass(frozen=True, slots=True)
class IMMPosteriorMedians:
    """Current motion marginal medians, not finite differences of the track."""

    speed_mps: float
    heading_degrees: float
    turn_rate_degrees_per_second: float


@dataclass(frozen=True, slots=True)
class IMMForecastMetrics:
    """Median point errors and Gaussian distribution scores for the IMM forecast."""

    ade_m: float | None
    fde_m: float | None
    rmse_m: float | None
    energy_score_m: float | None
    predictive_log_density: float | None
    joint_coverage_50_count: int
    joint_coverage_90_count: int
    joint_coverage_total: int


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
    posterior_medians: IMMPosteriorMedians | None
    metrics: IMMForecastMetrics
    uncertainty: IMMForecastUncertainty
    reference_state: IMMMovementState | None = None


class IMMPlaybackStore:
    """Advance one local IMM through raw cycles and cache every display stage."""

    def __init__(self, config: IMMPlayerConfig) -> None:
        self.config = config
        self.data_file = config.data_file
        cycles = br24.read_br24_tracking_input(self.data_file)
        if config.start_index >= len(cycles):
            raise ValueError(
                f"Start index {config.start_index} must be smaller than {len(cycles)} radar cycles."
            )
        self.cycles = cycles[config.start_index :]
        self.reference = br24.read_br24_tracking_reference(self.data_file)
        if self.reference is not None:
            self.reference = replace(
                self.reference,
                **{
                    field.name: (
                        None
                        if getattr(self.reference, field.name) is None
                        else getattr(self.reference, field.name)[config.start_index :]
                    )
                    for field in fields(self.reference)
                },
            )
            self._validate_reference_alignment()
        self.cycles = _add_position_noise(self.cycles, config)
        self.times = np.asarray(
            [cycle.time_seconds for cycle in self.cycles], dtype=float
        )
        self.maximum_observation_count = len(self.cycles) - config.prediction_count
        if self.maximum_observation_count < 1:
            raise ValueError(
                "The BR24 input must leave room for the requested prediction horizon."
            )
        if not self.cycles[0].detections:
            raise ValueError(
                "The selected starting BR24 radar cycle has no detection to initialize."
            )

        tracker_type = _load_optional_imm_tracker()
        initial_detection = self.cycles[0].detections[0]
        initial_update_started = time.perf_counter()
        self._tracker = tracker_type.initialize(
            self.cycles[0], initial_detection_id=initial_detection.detection_id
        )
        self._processed_cycle_count = 1
        self._tracked_positions = [
            np.asarray(self._tracker.estimated_position, dtype=float).copy()
        ]
        self._detection_ids = [initial_detection.detection_id]
        self._nis: list[float | None] = [None]
        self._measurement_covariances: list[np.ndarray | None] = [
            br24.br24_measurement_covariance(
                range_m=initial_detection.range_m,
                azimuth_rad=initial_detection.azimuth_rad,
                std_range_m=initial_detection.std_range_m,
                std_azimuth_rad=initial_detection.std_azimuth_rad,
            )
        ]
        self._stages: dict[int, IMMPlayerStage] = {}
        self._display_states: dict[int, IMMDisplayState] = {}
        self._capture_stage(
            update_time_seconds=time.perf_counter() - initial_update_started
        )

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
                posterior_medians=_posterior_medians(stage),
                metrics=_forecast_metrics(
                    stage,
                    self.reference,
                    coverage_stages=tuple(
                        self._stages[count] for count in range(1, observation_count + 1)
                    ),
                ),
                uncertainty=_forecast_uncertainty(stage),
                reference_state=_reference_state(stage, self.reference),
            )
        return self._display_states[observation_count]

    def _advance_to(self, observation_count: int) -> None:
        if observation_count in self._stages:
            return
        if observation_count < self._processed_cycle_count:
            raise RuntimeError("An earlier IMM stage was not cached.")
        while self._processed_cycle_count < observation_count:
            update_started = time.perf_counter()
            step = self._tracker.process_cycle(self.cycles[self._processed_cycle_count])
            self._tracked_positions.append(
                np.asarray(self._tracker.estimated_position, dtype=float).copy()
            )
            self._detection_ids.append(step.detection_id)
            self._nis.append(step.nis)
            self._measurement_covariances.append(
                None
                if step.measurement_covariance is None
                else np.asarray(step.measurement_covariance, dtype=float).copy()
            )
            self._processed_cycle_count += 1
            self._capture_stage(
                update_time_seconds=time.perf_counter() - update_started
            )

    def _capture_stage(self, *, update_time_seconds: float) -> None:
        observation_count = self._processed_cycle_count
        posterior_reader = getattr(self._tracker, "posterior_state", None)
        posterior = posterior_reader() if callable(posterior_reader) else None
        mixture_reader = getattr(self._tracker, "posterior_mixture", None)
        posterior_mixture = mixture_reader() if callable(mixture_reader) else None
        future_times = self.times[
            observation_count : observation_count + self.config.prediction_count
        ]
        if future_times.shape != (self.config.prediction_count,):
            raise RuntimeError(
                "IMM stage does not leave the configured forecast horizon."
            )
        forecast = (
            self._tracker.forecast(future_times)
            if self.config.prediction_count
            else None
        )
        forecast_medians = _forecast_position_medians(self._tracker, forecast)
        sample_positions = None
        if forecast is not None and self.config.prediction_sample_count:
            seed = int(
                np.random.SeedSequence(
                    [
                        self.config.forecast_seed,
                        self.cycles[observation_count - 1].cycle_counter,
                    ]
                ).generate_state(1)[0]
            )
            sample_positions = self._tracker.sample_forecast(
                future_times,
                sample_count=self.config.prediction_sample_count,
                seed=seed,
            )
            sample_positions.setflags(write=False)
        self._stages[observation_count] = IMMPlayerStage(
            observation_count=observation_count,
            cycle_counter=self.cycles[observation_count - 1].cycle_counter,
            observation_time_seconds=float(self._tracker.last_observation_time_seconds),
            detection_id=self._detection_ids[-1],
            nis=self._nis[-1],
            measurement_covariance=(
                None
                if self._measurement_covariances[-1] is None
                else self._measurement_covariances[-1].copy()
            ),
            update_time_seconds=float(update_time_seconds),
            tracked_times_seconds=self.times[:observation_count].copy(),
            tracked_positions=np.asarray(self._tracked_positions, dtype=float).copy(),
            forecast=forecast,
            sample_positions=sample_positions,
            posterior=posterior,
            posterior_mixture=posterior_mixture,
            forecast_median_positions=forecast_medians,
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


def _add_position_noise(cycles, config: IMMPlayerConfig):
    """Perturb radar positions only, retaining nominal sensor uncertainties.

    This is an input robustness experiment, not a retuning of the IMM noise
    model. Reference positions never participate in generating observations.
    """
    if config.position_noise_std_m == 0:
        return cycles
    generator = np.random.default_rng(config.position_noise_seed)
    noisy_cycles = []
    for cycle in cycles:
        detections = []
        for detection in cycle.detections:
            position = br24.polar_to_cartesian(
                detection.range_m, detection.azimuth_rad
            )[0]
            position = position + generator.normal(
                0.0, config.position_noise_std_m, size=2
            )
            detections.append(
                replace(
                    detection,
                    range_m=float(np.linalg.norm(position)),
                    azimuth_rad=float(np.arctan2(position[1], position[0])),
                )
            )
        noisy_cycles.append(replace(cycle, detections=tuple(detections)))
    return tuple(noisy_cycles)


def _reference_state(stage: IMMPlayerStage, reference) -> IMMMovementState | None:
    """Read aligned evaluation labels only after the IMM stage is complete."""
    if reference is None:
        return None
    index = stage.observation_count - 1
    turn_rate = float(np.rad2deg(reference.resolved_turn_rate_rad_s[index]))
    speed = float(reference.resolved_velocity_mps[index])
    heading = float(np.rad2deg(reference.resolved_yaw_rad[index]))
    return IMMMovementState(
        position_meters=np.array(
            [reference.x_meters[index], reference.y_meters[index]]
        ),
        speed_mps=speed if np.isfinite(speed) else None,
        heading_degrees=heading if np.isfinite(heading) else None,
        turn_rate_degrees_per_second=turn_rate if np.isfinite(turn_rate) else None,
        time_seconds=float(reference.time_seconds[index]),
        turn_rate_is_derived=reference.turn_rate_is_derived,
        speed_is_derived=reference.speed_is_derived,
        heading_is_derived=reference.heading_is_derived,
    )


def _posterior_medians(stage: IMMPlayerStage) -> IMMPosteriorMedians | None:
    """Summarize exactly the cached distribution displayed in the motion panels."""
    if stage.posterior_mixture is not None:
        values = [stage.posterior_mixture.marginal_median(index) for index in (4, 2, 3)]
    elif stage.posterior is not None:
        # For the labelled fused-only Gaussian fallback, median equals mean.
        values = stage.posterior.mean[[4, 2, 3]]
    else:
        return None
    heading = float(np.rad2deg((values[1] + np.pi) % (2 * np.pi) - np.pi))
    return IMMPosteriorMedians(float(values[0]), heading, float(np.rad2deg(values[2])))


def _forecast_position_medians(tracker, forecast) -> np.ndarray | None:
    """Read coordinatewise mixture medians independently of display path samples."""
    if forecast is None:
        return None
    reader = getattr(tracker, "forecast_mixtures", None)
    mixtures = reader(forecast.time_seconds) if callable(reader) else None
    if mixtures is None:
        # Older backends expose only a fused Gaussian: its marginal mean = median.
        positions = np.asarray(forecast.position_means, dtype=float).copy()
    else:
        positions = np.array(
            [
                [mixture.marginal_median(index) for index in (0, 1)]
                for mixture in mixtures
            ]
        )
    if positions.shape != np.shape(forecast.position_means) or not np.all(
        np.isfinite(positions)
    ):
        raise ValueError(
            "IMM forecast medians must be finite matching (time, xy) positions."
        )
    positions.setflags(write=False)
    return positions


def _forecast_metrics(
    stage: IMMPlayerStage, reference, *, coverage_stages
) -> IMMForecastMetrics:
    """Score cached median positions while retaining Gaussian distribution scores."""
    if reference is None or stage.forecast is None:
        return IMMForecastMetrics(None, None, None, None, None, 0, 0, 0)
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
    # Legacy fused-Gaussian stages have equal marginal means and medians.
    median_positions = (
        means
        if stage.forecast_median_positions is None
        else np.asarray(stage.forecast_median_positions, dtype=float)
    )
    errors = np.linalg.norm(median_positions - targets, axis=1)
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
    covered_50_count, covered_90_count, total_count = _joint_coverage_counts(
        coverage_stages,
        reference,
    )
    return IMMForecastMetrics(
        ade_m=float(errors.mean()),
        fde_m=float(errors[-1]),
        rmse_m=float(np.sqrt(np.mean(errors**2))),
        energy_score_m=float(energy_scores.mean()),
        predictive_log_density=float(log_densities.mean()),
        joint_coverage_50_count=covered_50_count,
        joint_coverage_90_count=covered_90_count,
        joint_coverage_total=total_count,
    )


def _joint_coverage_counts(stages, reference) -> tuple[int, int, int]:
    """Aggregate 2D 50% and 90% coverage over each forecast through the stage."""
    covered_50_count, covered_90_count, total_count = 0, 0, 0
    threshold_50 = chi2.ppf(0.5, 2)
    threshold_90 = chi2.ppf(0.9, 2)
    for stage in stages:
        forecast = stage.forecast
        if forecast is None:
            continue
        means = np.asarray(forecast.position_means, dtype=float)
        covariances = _regularize_covariances(forecast.position_covariances)
        target_indices = np.arange(
            stage.observation_count,
            stage.observation_count + means.shape[0],
        )
        targets = np.column_stack(
            (reference.x_meters[target_indices], reference.y_meters[target_indices])
        )
        squared_mahalanobis = np.asarray(
            [
                _squared_mahalanobis(target - mean, covariance)
                for target, mean, covariance in zip(
                    targets, means, covariances, strict=True
                )
            ]
        )
        covered_50_count += int(np.count_nonzero(squared_mahalanobis <= threshold_50))
        covered_90_count += int(np.count_nonzero(squared_mahalanobis <= threshold_90))
        total_count += len(squared_mahalanobis)
    return covered_50_count, covered_90_count, total_count


def _forecast_uncertainty(stage: IMMPlayerStage) -> IMMForecastUncertainty:
    """Summarize 2D Gaussian covariances for the display-only uncertainty view."""
    if stage.forecast is None:
        return IMMForecastUncertainty(np.empty(0), np.empty(0), np.empty((0, 2)))
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


def _load_optional_imm_tracker(*, imm_python: Path | None = None):
    """Load the deliberately untracked local IMM implementation when available."""
    imm_python = IMM_PYTHON if imm_python is None else Path(imm_python)
    if not imm_python.is_file():
        raise FileNotFoundError(
            "Optional IMM implementation not found: external/IMM/br24_imm.py. "
            "Restore the local external IMM files to open this player."
        )
    module = runpy.run_path(str(imm_python))
    try:
        tracker_type = module["BR24IMMTracker"]
    except KeyError as error:
        raise RuntimeError(
            "The local IMM implementation must expose BR24IMMTracker."
        ) from error
    fuse = module.get("_mix_to_ctrv")
    if not callable(fuse):
        return tracker_type
    to_ctrv = module.get("_to_ctrv")

    def state_mixture(state) -> IMMStateMixture | None:
        if not callable(to_ctrv):
            return None
        return IMMStateMixture(
            state.mode_probabilities,
            (
                IMMStatePosterior(*to_ctrv(state.cv_state, state.cv_covariance)),
                IMMStatePosterior(state.ctrv_state, state.ctrv_covariance),
                IMMStatePosterior(*to_ctrv(state.ca_state, state.ca_covariance)),
            ),
        )

    class DisplayTracker(tracker_type):
        """Expose output moments without modifying the optional filter implementation."""

        def posterior_state(self) -> IMMStatePosterior:
            mean, covariance = fuse(self._state, self._state.mode_probabilities)
            return IMMStatePosterior(mean, covariance)

        def posterior_mixture(self) -> IMMStateMixture | None:
            """Read the same model conversions as output fusion, without collapsing."""
            return state_mixture(self._state)

        def forecast_mixtures(
            self, future_time_seconds
        ) -> tuple[IMMStateMixture, ...] | None:
            """Read model components through the unchanged backend forecast equations."""
            to_world = module.get("_state_in_world_coordinates")
            predict = module.get("_predict_imm")
            state_type = module.get("_IMMState")
            validate_times = module.get("_validate_future_times")
            if not all(
                callable(function)
                for function in (
                    to_ctrv,
                    to_world,
                    predict,
                    state_type,
                    validate_times,
                )
            ):
                return None
            times = validate_times(
                future_time_seconds, after=self._last_observation_time_seconds
            )
            state = to_world(self._state, self._sensor_position)
            current_time = self._last_observation_time_seconds
            mixtures = []
            for future_time in times:
                predictions, _, probabilities = predict(
                    state, float(future_time - current_time), np.zeros(2)
                )
                cv, ctrv, ca = predictions
                state = state_type(*cv, *ctrv, *ca, probabilities)
                mixtures.append(state_mixture(state))
                current_time = float(future_time)
            return tuple(mixtures)

    return DisplayTracker
