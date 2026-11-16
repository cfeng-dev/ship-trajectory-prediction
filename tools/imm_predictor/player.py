"""Presentation-safe online session data for the IMM Predictor.

This module owns local position or BR24 measurements and the optional IMM tracker.
Reference values remain separate from observations and serve display and evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

import bayestraj.inference.imm_adapter as imm_adapter
import bayestraj.observations.br24 as br24
import bayestraj.observations.io as observations_io
import bayestraj.validation.metrics as validation_metrics
from bayestraj.inference.imm_adapter import IMMStateMixture, IMMStatePosterior
from bayestraj.inference.imm_adapter import (
    forecast_position_medians as _forecast_position_medians,
)
from bayestraj.paths import project_path

IMM_PYTHON = project_path("external/IMM/br24_imm.py")
DATA_FILE = project_path("data/processed/BR24_imm_tracked_trajectory.csv")

PREDICTION_COUNT = 3
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
    "show_reference_points": "Show reference points",
    "show_imm_track": "Show IMM track through N",
    "show_current_position": "Show current position",
    "show_sample_trajectories": "Show sampled IMM trajectories",
    "show_forecast_median": "Show forecast (median)",
    "show_forecast_mean": "Show forecast (mean)",
    "show_forecast_points": "Show forecast points",
    "show_prediction_region_50": "Show 50% posterior-predictive region",
    "show_prediction_region_90": "Show 90% posterior-predictive region",
}
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
    show_reference_points: bool = False
    show_imm_track: bool = True
    show_current_position: bool = True
    show_forecast_median: bool = False
    show_forecast_mean: bool = True
    show_forecast_points: bool = True
    show_sample_trajectories: bool = True
    prediction_sample_count: int = 20
    forecast_seed: int = 42
    start_index: int = 0
    position_noise_std_m: float = 0.0
    position_noise_seed: int = 2026
    measurement_noise_std_m: float = 0.0

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
        measurement_noise = float(self.measurement_noise_std_m)
        if not np.isfinite(measurement_noise) or measurement_noise < 0:
            raise ValueError("measurement_noise_std_m must be finite and non-negative.")
        object.__setattr__(self, "measurement_noise_std_m", measurement_noise)
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
class IMMPlayerStage:
    """One online IMM posterior and forecast refreshed at an input step."""

    observation_count: int
    cycle_counter: int
    observation_time_seconds: float
    detection_id: int | None
    nis: float | None
    measurement_covariance: np.ndarray | None
    fit_runtime_seconds: float
    tracked_times_seconds: np.ndarray
    tracked_positions: np.ndarray
    forecast: Any
    sample_positions: np.ndarray | None = None
    posterior: IMMStatePosterior | None = None
    posterior_mixture: IMMStateMixture | None = None
    forecast_median_positions: np.ndarray | None = None
    associated_input: bool = False
    position_input: bool = False

    @property
    def association_status(self) -> str:
        """Describe the input step independently of whether NIS is available."""
        if self.position_input:
            return "position_csv"
        if self.detection_id is None:
            return "prediction_only"
        if self.observation_count == 1:
            return "initialization"
        return "provided" if self.associated_input else "accepted"


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
class IMMStateEstimate:
    """Current fused IMM motion state, not finite differences of the track."""

    speed_mps: float
    heading_degrees: float
    turn_rate_degrees_per_second: float


@dataclass(frozen=True, slots=True)
class IMMForecastMetrics:
    """Mean point errors and Gaussian distribution scores for the IMM forecast."""

    ade_m: float | None
    fde_m: float | None
    rmse_m: float | None
    energy_score_m: float | None
    negative_log_predictive_density: float | None
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
    state_estimate: IMMStateEstimate | None
    metrics: IMMForecastMetrics
    uncertainty: IMMForecastUncertainty
    reference_state: IMMMovementState | None = None


@dataclass(frozen=True, slots=True)
class _PositionMeasurement:
    """A local position with known uncertainty, without invented radar IDs."""

    time_seconds: float
    cycle_counter: int
    position: np.ndarray
    measurement_covariance: np.ndarray
    detection_id: None = None


def _read_position_input(config):
    data = observations_io.read_processed_trajectory_with_reference(config.data_file)
    supplied = set(observations_io.REFERENCE_POSITION_COLUMNS).intersection(
        data.columns
    )
    if supplied and supplied != set(observations_io.REFERENCE_POSITION_COLUMNS):
        raise ValueError(
            "Reference positions require both ground_truth_x and ground_truth_y."
        )
    times = data["time"].to_numpy(dtype=float)
    indices = np.arange(1, len(data) + 1)
    # Zero is an exact-position assumption; only a numerical variance is retained.
    variance = max(
        config.measurement_noise_std_m**2, RESIDUAL_POSITION_NOISE_FLOOR_M**2
    )
    covariance = np.eye(2) * variance
    covariance.setflags(write=False)
    measurements = []
    for index, position in enumerate(data[["x", "y"]].to_numpy(dtype=float)):
        position.setflags(write=False)
        measurements.append(
            _PositionMeasurement(times[index], indices[index], position, covariance)
        )
    motion = {
        quantity: next(
            (data[name].to_numpy(dtype=float) for name in names if name in data), None
        )
        for quantity, names in observations_io.REFERENCE_MOTION_COLUMNS.items()
    }
    reference = br24.BR24TrackingReference(
        cycle_counters=indices,
        time_seconds=times,
        x_meters=data["ground_truth_x" if supplied else "x"].to_numpy(dtype=float),
        y_meters=data["ground_truth_y" if supplied else "y"].to_numpy(dtype=float),
        yaw_rad=motion["heading"],
        velocity_mps=motion["speed"],
        turn_rate_rad_s=motion["turn_rate"],
    )
    return tuple(measurements), reference


class IMMPlaybackStore:
    """Advance one IMM through positions or BR24 input and cache stages."""

    def __init__(self, config: IMMPlayerConfig) -> None:
        self.config = config
        self.data_file = config.data_file
        if br24.is_br24_tracking_input(self.data_file):
            cycles = br24.read_br24_tracking_input(self.data_file)
            reference = br24.read_br24_tracking_reference(self.data_file)
        elif br24.is_br24_associated_input(self.data_file):
            cycles = br24.read_br24_associated_input(self.data_file)
            reference = br24.read_br24_tracking_reference(self.data_file)
        else:
            cycles, reference = _read_position_input(config)
        if config.start_index >= len(cycles):
            raise ValueError(
                f"Start index {config.start_index} must be smaller than {len(cycles)} input steps."
            )
        self.cycles = cycles[config.start_index :]
        self.reference = reference
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
        if isinstance(self.cycles[0], _PositionMeasurement):
            origin = self.cycles[0].time_seconds
            self.cycles = tuple(
                replace(cycle, time_seconds=cycle.time_seconds - origin)
                for cycle in self.cycles
            )
            self.reference = replace(
                self.reference, time_seconds=self.reference.time_seconds - origin
            )
        self.cycles = _add_position_noise(self.cycles, config)
        self.times = np.asarray(
            [cycle.time_seconds for cycle in self.cycles], dtype=float
        )
        self.maximum_observation_count = len(self.cycles) - config.prediction_count
        if self.maximum_observation_count < 1:
            raise ValueError(
                "The input must leave room for the requested prediction horizon."
            )
        initial = self.cycles[0]
        associated_input = isinstance(
            initial, (br24.BR24AssociatedMeasurement, _PositionMeasurement)
        )
        if not associated_input and not initial.detections:
            raise ValueError(
                "The selected starting BR24 radar cycle has no detection to initialize."
            )

        tracker_type = _load_optional_imm_tracker()
        if associated_input:
            initial_detection_id = initial.detection_id
            initial_covariance = initial.measurement_covariance
            self._tracker = tracker_type.initialize_position(
                initial.time_seconds, initial.position, initial_covariance
            )
        else:
            detection = initial.detections[0]
            initial_detection_id = detection.detection_id
            initial_covariance = br24.br24_measurement_covariance(
                range_m=detection.range_m,
                azimuth_rad=detection.azimuth_rad,
                std_range_m=detection.std_range_m,
                std_azimuth_rad=detection.std_azimuth_rad,
            )
            self._tracker = tracker_type.initialize(
                initial, initial_detection_id=initial_detection_id
            )
        self._processed_cycle_count = 1
        self._tracked_positions = [
            np.asarray(self._tracker.estimated_position, dtype=float).copy()
        ]
        self._detection_ids = [initial_detection_id]
        self._nis: list[float | None] = [None]
        self._measurement_covariances: list[np.ndarray | None] = [initial_covariance]
        self._stages: dict[int, IMMPlayerStage] = {}
        self._display_states: dict[int, IMMDisplayState] = {}
        self._capture_stage(fit_runtime_seconds=0.0)

    def stage_for(self, observation_count: int) -> IMMPlayerStage:
        """Return the forecast after exactly ``observation_count`` input steps."""
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
                state_estimate=_state_estimate(stage),
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
            cycle = self.cycles[self._processed_cycle_count]
            if isinstance(
                cycle, (br24.BR24AssociatedMeasurement, _PositionMeasurement)
            ):
                self._tracker.update_position(
                    cycle.time_seconds, cycle.position, cycle.measurement_covariance
                )
                detection_id, nis, covariance = (
                    cycle.detection_id,
                    None,
                    cycle.measurement_covariance,
                )
            else:
                step = self._tracker.process_cycle(cycle)
                detection_id, nis, covariance = (
                    step.detection_id,
                    step.nis,
                    step.measurement_covariance,
                )
            fit_runtime_seconds = time.perf_counter() - update_started
            self._tracked_positions.append(
                np.asarray(self._tracker.estimated_position, dtype=float).copy()
            )
            self._detection_ids.append(detection_id)
            self._nis.append(nis)
            self._measurement_covariances.append(
                None
                if covariance is None
                else np.asarray(covariance, dtype=float).copy()
            )
            self._processed_cycle_count += 1
            self._capture_stage(fit_runtime_seconds=fit_runtime_seconds)

    def _capture_stage(self, *, fit_runtime_seconds: float) -> None:
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
        forecast = None
        if self.config.prediction_count:
            forecast_runtime_started = time.perf_counter()
            forecast = self._tracker.forecast(future_times)
            fit_runtime_seconds += time.perf_counter() - forecast_runtime_started
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
            fit_runtime_seconds=float(fit_runtime_seconds),
            tracked_times_seconds=self.times[:observation_count].copy(),
            tracked_positions=np.asarray(self._tracked_positions, dtype=float).copy(),
            forecast=forecast,
            sample_positions=sample_positions,
            posterior=posterior,
            posterior_mixture=posterior_mixture,
            forecast_median_positions=forecast_medians,
            associated_input=isinstance(
                self.cycles[observation_count - 1], br24.BR24AssociatedMeasurement
            ),
            position_input=isinstance(
                self.cycles[observation_count - 1], _PositionMeasurement
            ),
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
    if isinstance(cycles[0], _PositionMeasurement):
        noise_x = generator.normal(0.0, config.position_noise_std_m, len(cycles))
        noise_y = generator.normal(0.0, config.position_noise_std_m, len(cycles))
        return tuple(
            replace(cycle, position=cycle.position + np.array([x, y]))
            for cycle, x, y in zip(cycles, noise_x, noise_y, strict=True)
        )
    noisy_cycles = []
    for cycle in cycles:
        if isinstance(cycle, br24.BR24AssociatedMeasurement):
            noisy_cycles.append(
                replace(
                    cycle,
                    position=cycle.position
                    + generator.normal(0.0, config.position_noise_std_m, size=2),
                )
            )
            continue
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
    """Read aligned reference values with elapsed time for the completed stage."""
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
        time_seconds=float(reference.time_seconds[index] - reference.time_seconds[0]),
        turn_rate_is_derived=reference.turn_rate_is_derived,
        speed_is_derived=reference.speed_is_derived,
        heading_is_derived=reference.heading_is_derived,
    )


def _state_estimate(stage: IMMPlayerStage) -> IMMStateEstimate | None:
    """Read the cached fused state, matching the MATLAB output estimate."""
    if stage.posterior is None:
        return None
    values = stage.posterior.mean[[4, 2, 3]]
    heading = float(np.rad2deg((values[1] + np.pi) % (2 * np.pi) - np.pi))
    return IMMStateEstimate(float(values[0]), heading, float(np.rad2deg(values[2])))


def _forecast_metrics(
    stage: IMMPlayerStage, reference, *, coverage_stages
) -> IMMForecastMetrics:
    """Score cached mean positions and their Gaussian distribution."""
    if reference is None or stage.forecast is None:
        return IMMForecastMetrics(None, None, None, None, None, 0, 0, 0)
    forecast = stage.forecast
    means = np.asarray(forecast.position_means, dtype=float)
    covariances = np.asarray(forecast.position_covariances, dtype=float)
    target_indices = np.arange(
        stage.observation_count,
        stage.observation_count + means.shape[0],
    )
    targets = np.column_stack(
        (reference.x_meters[target_indices], reference.y_meters[target_indices])
    )
    errors = np.linalg.norm(means - targets, axis=1)
    scores = validation_metrics.gaussian_position_metrics(targets, means, covariances)
    covered_50_count, covered_90_count, total_count = _joint_coverage_counts(
        coverage_stages,
        reference,
    )
    return IMMForecastMetrics(
        ade_m=float(errors.mean()),
        fde_m=float(errors[-1]),
        rmse_m=float(np.sqrt(np.mean(errors**2))),
        energy_score_m=float(np.mean(scores["energy_score_m"])),
        negative_log_predictive_density=float(
            np.mean(scores["negative_log_predictive_density"])
        ),
        joint_coverage_50_count=covered_50_count,
        joint_coverage_90_count=covered_90_count,
        joint_coverage_total=total_count,
    )


def _joint_coverage_counts(stages, reference) -> tuple[int, int, int]:
    """Aggregate 2D 50% and 90% coverage over each forecast through the stage."""
    covered_50_count, covered_90_count, total_count = 0, 0, 0
    for stage in stages:
        forecast = stage.forecast
        if forecast is None:
            continue
        means = np.asarray(forecast.position_means, dtype=float)
        covariances = np.asarray(forecast.position_covariances, dtype=float)
        target_indices = np.arange(
            stage.observation_count,
            stage.observation_count + means.shape[0],
        )
        targets = np.column_stack(
            (reference.x_meters[target_indices], reference.y_meters[target_indices])
        )
        coverage = validation_metrics.gaussian_position_coverage(
            targets, means, covariances
        )
        covered_50_count += int(np.count_nonzero(coverage[0.5]))
        covered_90_count += int(np.count_nonzero(coverage[0.9]))
        total_count += len(targets)
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
    """Use the same covariance floors as scoring and Gaussian ellipse plots."""
    return validation_metrics.regularize_position_covariances(covariances)


def _gaussian_log_density(actual, mean, covariance) -> float:
    """Use the shared latent-position Gaussian score and numerical floor."""
    return float(
        validation_metrics.gaussian_position_log_predictive_density(
            np.asarray(actual)[None, :],
            np.asarray(mean)[None, :],
            np.asarray(covariance)[None, :, :],
        )[0]
    )


def _gaussian_energy_score(actual, mean, covariance) -> float:
    """Use the common Gaussian Energy Score quadrature."""
    return float(
        validation_metrics.gaussian_position_metrics(
            np.asarray(actual)[None, :],
            np.asarray(mean)[None, :],
            np.asarray(covariance)[None, :, :],
        )["energy_score_m"][0]
    )


def _squared_mahalanobis(difference, covariance) -> float:
    """Return one squared Mahalanobis distance."""
    difference = np.asarray(difference, dtype=float)
    return float(difference @ np.linalg.solve(covariance, difference))


def _load_optional_imm_tracker(*, imm_python: Path | None = None):
    """Supply the application's backend path to the shared read-only adapter."""
    imm_python = IMM_PYTHON if imm_python is None else Path(imm_python)
    return imm_adapter.load_imm_tracker(imm_python)
