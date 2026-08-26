"""Shared position-observation data for Bayesian trajectory models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import bayestraj.observations.window as observation_window

DEFAULT_POSITION_OBSERVATION_NOISE_STD_M = 2.0


@dataclass(frozen=True, slots=True)
class PositionObservations:
    """Immutable observed positions supplied to one Bayesian fit."""

    time_seconds: np.ndarray
    x_meters: np.ndarray
    y_meters: np.ndarray
    position_noise_std_m: float
    noise_seed: int
    observation_noise_std_m: float = DEFAULT_POSITION_OBSERVATION_NOISE_STD_M

    def __post_init__(self) -> None:
        """Copy, validate, and make all observation arrays read-only."""
        time_seconds = np.asarray(self.time_seconds, dtype=float).copy()
        x_meters = np.asarray(self.x_meters, dtype=float).copy()
        y_meters = np.asarray(self.y_meters, dtype=float).copy()
        validate_matching_position_time_arrays(time_seconds, x_meters, y_meters)
        if np.any(np.diff(time_seconds) <= 0):
            raise ValueError("time_seconds must be strictly increasing.")
        position_noise_std_m = validate_non_negative_finite(
            "position_noise_std_m",
            self.position_noise_std_m,
        )
        noise_seed = validate_non_negative_integer("noise_seed", self.noise_seed)
        observation_noise_std_m = validate_positive_finite(
            "observation_noise_std_m",
            self.observation_noise_std_m,
        )
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
        object.__setattr__(
            self,
            "observation_noise_std_m",
            observation_noise_std_m,
        )


def simulate_position_observations(
    window: observation_window.TrajectoryWindowData,
    *,
    position_noise_std_m: float = 0.0,
    seed: int = 2026,
) -> PositionObservations:
    """Create reproducible full-window position observations."""
    position_noise_std_m = validate_non_negative_finite(
        "position_noise_std_m",
        position_noise_std_m,
    )
    seed = validate_non_negative_integer("seed", seed)
    if window.observation_count < 2:
        raise ValueError("window must contain at least two observed positions.")

    observed = window.observed_slice
    time_seconds = np.asarray(window.time_seconds[observed], dtype=float).copy()
    x_meters = np.asarray(window.x_meters[observed], dtype=float).copy()
    y_meters = np.asarray(window.y_meters[observed], dtype=float).copy()
    if position_noise_std_m > 0:
        generator = np.random.default_rng(seed)
        x_meters += generator.normal(0.0, position_noise_std_m, x_meters.size)
        y_meters += generator.normal(0.0, position_noise_std_m, y_meters.size)

    return PositionObservations(
        time_seconds=time_seconds,
        x_meters=x_meters,
        y_meters=y_meters,
        position_noise_std_m=position_noise_std_m,
        noise_seed=seed,
        observation_noise_std_m=(
            position_noise_std_m
            if position_noise_std_m > 0
            else DEFAULT_POSITION_OBSERVATION_NOISE_STD_M
        ),
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
        validate_finite_vector(name, values)
    if (
        time_seconds.size < 2
        or x_meters.shape != time_seconds.shape
        or y_meters.shape != time_seconds.shape
    ):
        raise ValueError(
            "time_seconds, x_meters, and y_meters must be matching vectors "
            "with at least two values."
        )


def validate_finite_vector(name: str, values) -> None:
    """Validate one non-empty, one-dimensional, finite array."""
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a non-empty finite vector.")


def validate_non_negative_integer(name: str, value: int) -> int:
    """Validate and return a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def validate_non_negative_finite(name: str, value: float) -> float:
    """Validate and return a non-negative finite scalar."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a non-negative finite value.") from error
    if not np.isfinite(numeric_value) or numeric_value < 0:
        raise ValueError(f"{name} must be a non-negative finite value.")
    return numeric_value


def validate_finite_scalar(name: str, value: float) -> float:
    """Validate and return a signed finite scalar."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite value.") from error
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be a finite value.")
    return numeric_value


def validate_positive_finite(name: str, value: float) -> float:
    """Validate and return a positive finite scalar."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive finite value.") from error
    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be a positive finite value.")
    return numeric_value
