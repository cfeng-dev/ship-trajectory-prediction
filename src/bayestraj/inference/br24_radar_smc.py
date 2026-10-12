"""Online Bayesian CTRV tracking from raw BR24 radar cycles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from bayestraj.inference.ctrv_smc import (
    SequentialMonteCarloCTRVConfig,
    SequentialMonteCarloCTRVFilter,
)
from bayestraj.models.bayesian_ctrv import BayesianCTRVPriors
from bayestraj.observations.br24 import (
    ASSOCIATION_GATE_NIS,
    BR24Detection,
    BR24RadarCycle,
    br24_measurement_covariance,
    br24_measurement_position,
    read_br24_tracking_input,
)


@dataclass(frozen=True, slots=True)
class BR24TrackerStep:
    """One online tracking decision made without target-reference input."""

    time_seconds: float
    cycle_counter: int
    detection_id: int | None
    nis: float | None
    measurement_covariance: np.ndarray | None


@dataclass(slots=True)
class BR24BayesianRadarTracker:
    """Associate raw BR24 detections and update a Bayesian CTRV SMC filter."""

    filter: SequentialMonteCarloCTRVFilter
    association_gate_nis: float = ASSOCIATION_GATE_NIS

    @property
    def estimated_position(self) -> np.ndarray:
        """Return the current weighted posterior position in local metres."""
        origins = self.filter.forecast_origin_particles
        if origins is None:
            origins = self.filter.state_particles
        return np.sum(self.filter.weights[:, None] * origins[:, :2], axis=0)

    @classmethod
    def initialize(
        cls,
        cycle: BR24RadarCycle,
        *,
        initial_detection_id: int,
        priors: BayesianCTRVPriors,
        config: SequentialMonteCarloCTRVConfig | None = None,
        seed: int = 42,
        residual_observation_noise_floor_m: float = 1e-6,
    ) -> BR24BayesianRadarTracker:
        """Start the target track from the documented manual BR24 seed."""
        detection = _detection_by_id(cycle.detections, initial_detection_id)
        position = _measurement_position(cycle, detection)
        covariance = _measurement_covariance(detection)
        if config is None:
            config = SequentialMonteCarloCTRVConfig(
                predictive_log_density_threshold=None,
            )
        online_filter = SequentialMonteCarloCTRVFilter.initialize(
            np.array([cycle.time_seconds]),
            np.array([position[0]]),
            np.array([position[1]]),
            priors=replace(
                priors,
                sigma_position_observation_floor_m=(residual_observation_noise_floor_m),
            ),
            config=config,
            seed=seed,
            initial_measurement_covariance=covariance,
        )
        return cls(filter=online_filter)

    def process_cycle(self, cycle: BR24RadarCycle) -> BR24TrackerStep:
        """Associate and assimilate one later cycle, or propagate on a miss."""
        if not cycle.detections:
            self.filter.advance_without_measurement(cycle.time_seconds)
            return BR24TrackerStep(
                time_seconds=cycle.time_seconds,
                cycle_counter=cycle.cycle_counter,
                detection_id=None,
                nis=None,
                measurement_covariance=None,
            )

        predicted_position, predicted_covariance = (
            self.filter.predict_position_distribution(cycle.time_seconds)
        )
        residual_variance = self.filter.expected_position_observation_variance
        candidates = [
            (
                detection,
                _measurement_position(cycle, detection),
                _measurement_covariance(detection),
            )
            for detection in cycle.detections
        ]
        accepted = _nearest_gated_detection(
            predicted_position=predicted_position,
            predicted_covariance=predicted_covariance,
            residual_variance=residual_variance,
            candidates=candidates,
            gate_nis=self.association_gate_nis,
        )
        if accepted is None:
            self.filter.advance_without_measurement(cycle.time_seconds)
            return BR24TrackerStep(
                time_seconds=cycle.time_seconds,
                cycle_counter=cycle.cycle_counter,
                detection_id=None,
                nis=None,
                measurement_covariance=None,
            )

        detection, position, covariance, nis = accepted
        self.filter.update(
            cycle.time_seconds,
            position[0],
            position[1],
            measurement_covariance=covariance,
        )
        return BR24TrackerStep(
            time_seconds=cycle.time_seconds,
            cycle_counter=cycle.cycle_counter,
            detection_id=detection.detection_id,
            nis=nis,
            measurement_covariance=covariance,
        )


def track_br24_csv(
    input_csv: str | Path,
    *,
    initial_cycle_counter: int,
    initial_detection_id: int,
    priors: BayesianCTRVPriors,
    config: SequentialMonteCarloCTRVConfig | None = None,
    seed: int = 42,
    residual_observation_noise_floor_m: float = 1e-6,
) -> tuple[BR24BayesianRadarTracker, tuple[BR24TrackerStep, ...]]:
    """Track a manually seeded target through later BR24 radar cycles online."""
    cycles = read_br24_tracking_input(input_csv)
    initial_index = next(
        (
            index
            for index, cycle in enumerate(cycles)
            if cycle.cycle_counter == initial_cycle_counter
        ),
        None,
    )
    if initial_index is None:
        raise ValueError("initial_cycle_counter is not present in the BR24 input.")
    initial_cycle = cycles[initial_index]
    tracker = BR24BayesianRadarTracker.initialize(
        initial_cycle,
        initial_detection_id=initial_detection_id,
        priors=priors,
        config=config,
        seed=seed,
        residual_observation_noise_floor_m=residual_observation_noise_floor_m,
    )
    steps = [
        BR24TrackerStep(
            time_seconds=initial_cycle.time_seconds,
            cycle_counter=initial_cycle.cycle_counter,
            detection_id=initial_detection_id,
            nis=None,
            measurement_covariance=_measurement_covariance(
                _detection_by_id(initial_cycle.detections, initial_detection_id)
            ),
        )
    ]
    steps.extend(tracker.process_cycle(cycle) for cycle in cycles[initial_index + 1 :])
    return tracker, tuple(steps)


def _detection_by_id(
    detections: tuple[BR24Detection, ...],
    detection_id: int,
) -> BR24Detection:
    for detection in detections:
        if detection.detection_id == detection_id:
            return detection
    raise ValueError("initial_detection_id is not present in the initial cycle.")


def _measurement_position(
    cycle: BR24RadarCycle, detection: BR24Detection
) -> np.ndarray:
    """Use the BR24 world-frame convention of sensor position plus polar fix."""
    return br24_measurement_position(
        sensor_position=cycle.sensor_position,
        range_m=detection.range_m,
        azimuth_rad=detection.azimuth_rad,
    )


def _measurement_covariance(detection: BR24Detection) -> np.ndarray:
    return br24_measurement_covariance(
        range_m=detection.range_m,
        azimuth_rad=detection.azimuth_rad,
        std_range_m=detection.std_range_m,
        std_azimuth_rad=detection.std_azimuth_rad,
    )


def _nearest_gated_detection(
    *,
    predicted_position: np.ndarray,
    predicted_covariance: np.ndarray,
    residual_variance: float,
    candidates: list[tuple[BR24Detection, np.ndarray, np.ndarray]],
    gate_nis: float,
) -> tuple[BR24Detection, np.ndarray, np.ndarray, float] | None:
    """Select the lowest-NIS raw detection using its own measurement covariance."""
    best: tuple[BR24Detection, np.ndarray, np.ndarray, float] | None = None
    for detection, position, covariance in candidates:
        innovation = position - predicted_position
        innovation_covariance = (
            predicted_covariance + covariance + np.eye(2) * (residual_variance)
        )
        try:
            nis = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "BR24 innovation covariance must be invertible."
            ) from error
        if nis < gate_nis and (best is None or nis < best[3]):
            best = (detection, position, covariance, nis)
    return best
