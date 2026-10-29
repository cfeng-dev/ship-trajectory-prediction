"""Position-observation data shared by Bayesian trajectory models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import bayestraj.numeric_validation as numeric_validation
import bayestraj.observations.window as observation_window


@dataclass(frozen=True, slots=True)
class PositionObservations:
    """Immutable observed positions supplied to one Bayesian fit."""

    time_seconds: np.ndarray
    x_meters: np.ndarray
    y_meters: np.ndarray
    position_noise_std_m: float
    noise_seed: int
    measurement_covariances: np.ndarray | None = None
    observed_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Copy, validate, and make all observation arrays read-only."""
        time_seconds = np.asarray(self.time_seconds, dtype=float).copy()
        x_meters = np.asarray(self.x_meters, dtype=float).copy()
        y_meters = np.asarray(self.y_meters, dtype=float).copy()
        validate_matching_position_time_arrays(time_seconds, x_meters, y_meters)
        if np.any(np.diff(time_seconds) <= 0):
            raise ValueError("time_seconds must be strictly increasing.")
        position_noise_std_m = numeric_validation.validate_non_negative_finite(
            "position_noise_std_m",
            self.position_noise_std_m,
        )
        noise_seed = numeric_validation.validate_non_negative_integer(
            "noise_seed",
            self.noise_seed,
        )
        measurement_covariances = None
        if self.measurement_covariances is not None:
            measurement_covariances = validate_measurement_covariances(
                self.measurement_covariances, count=time_seconds.size
            )
        observed_mask = None
        if self.observed_mask is not None:
            observed_mask = np.asarray(self.observed_mask)
            if observed_mask.shape != time_seconds.shape or observed_mask.dtype != bool:
                raise ValueError("observed_mask must be a matching boolean vector.")
            observed_mask = observed_mask.copy()
            observed_mask.setflags(write=False)
        for values in (time_seconds, x_meters, y_meters):
            values.setflags(write=False)
        object.__setattr__(self, "time_seconds", time_seconds)
        object.__setattr__(self, "x_meters", x_meters)
        object.__setattr__(self, "y_meters", y_meters)
        object.__setattr__(
            self,
            "position_noise_std_m",
            position_noise_std_m,
        )
        object.__setattr__(self, "noise_seed", noise_seed)
        object.__setattr__(self, "measurement_covariances", measurement_covariances)
        object.__setattr__(self, "observed_mask", observed_mask)


def validate_measurement_covariances(values, *, count: int) -> np.ndarray:
    """Copy finite symmetric positive-semidefinite measurement matrices."""
    covariance = np.asarray(values, dtype=float).copy()
    if covariance.shape != (count, 2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("measurement_covariances must have finite shape (n, 2, 2).")
    transpose = covariance.swapaxes(-1, -2)
    if not np.allclose(covariance, transpose, rtol=1e-12, atol=1e-12):
        raise ValueError("measurement_covariances must be symmetric.")
    covariance = 0.5 * (covariance + transpose)
    if np.any(np.linalg.eigvalsh(covariance) < 0.0):
        raise ValueError("measurement_covariances must be positive semidefinite.")
    covariance.setflags(write=False)
    return covariance


def simulate_position_observations(
    window: observation_window.TrajectoryWindowData,
    *,
    position_noise_std_m: float = 0.0,
    seed: int = 2026,
) -> PositionObservations:
    """Create reproducible full-window position observations."""
    position_noise_std_m = numeric_validation.validate_non_negative_finite(
        "position_noise_std_m",
        position_noise_std_m,
    )
    seed = numeric_validation.validate_non_negative_integer("seed", seed)
    if window.observation_count < 2:
        raise ValueError("window must contain at least two observed positions.")

    observed = window.observed_slice
    time_seconds = np.asarray(window.time_seconds[observed], dtype=float).copy()
    x_reference = np.asarray(window.x_meters[observed], dtype=float).copy()
    y_reference = np.asarray(window.y_meters[observed], dtype=float).copy()
    x_meters = x_reference
    y_meters = y_reference
    if position_noise_std_m > 0:
        generator = np.random.default_rng(seed)
        x_meters = generator.normal(
            loc=x_reference,
            scale=position_noise_std_m,
        )
        y_meters = generator.normal(
            loc=y_reference,
            scale=position_noise_std_m,
        )

    return PositionObservations(
        time_seconds=time_seconds,
        x_meters=x_meters,
        y_meters=y_meters,
        position_noise_std_m=position_noise_std_m,
        noise_seed=seed,
    )


def resolve_position_observations(
    window: observation_window.TrajectoryWindowData,
    position_observations: PositionObservations | None,
) -> PositionObservations:
    """Return observations aligned with the complete observed window."""
    if position_observations is None:
        position_observations = simulate_position_observations(
            window,
            position_noise_std_m=0.0,
            seed=0,
        )
    if not isinstance(position_observations, PositionObservations):
        raise TypeError(
            "position_observations must be a PositionObservations instance or None."
        )

    expected_time = np.asarray(
        window.time_seconds[window.observed_slice],
        dtype=float,
    )
    if (
        position_observations.time_seconds.shape != expected_time.shape
        or not np.array_equal(position_observations.time_seconds, expected_time)
    ):
        raise ValueError(
            "position_observations must match the observed timestamps in window."
        )
    return position_observations


def validate_matching_position_time_arrays(time_seconds, x_meters, y_meters) -> None:
    """Validate matching finite position and time vectors with two points."""
    for name, values in (
        ("time_seconds", time_seconds),
        ("x_meters", x_meters),
        ("y_meters", y_meters),
    ):
        numeric_validation.validate_finite_vector(name, values)
    if (
        time_seconds.size < 2
        or x_meters.shape != time_seconds.shape
        or y_meters.shape != time_seconds.shape
    ):
        raise ValueError(
            "time_seconds, x_meters, and y_meters must be matching vectors "
            "with at least two values."
        )
