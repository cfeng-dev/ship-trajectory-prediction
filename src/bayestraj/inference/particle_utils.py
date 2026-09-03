"""Shared validation and numerical utilities for particle filters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

NUMERICAL_VARIANCE_FLOOR = 1e-9


class SequentialCTRVFit:
    """CmdStan-like draws produced by online CTRV particle filters."""

    __slots__ = ("_variables",)

    def __init__(self, variables: Mapping[str, Any]):
        normalized = {}
        for name, values in variables.items():
            array = np.asarray(values, dtype=float).copy()
            if array.size == 0 or not np.all(np.isfinite(array)):
                raise ValueError(f"Sequential posterior variable {name!r} is invalid.")
            array.setflags(write=False)
            normalized[str(name)] = array
        self._variables = normalized

    def stan_variable(self, variable_name: str) -> np.ndarray:
        """Return posterior draws through the shared reporting interface."""
        try:
            return self._variables[variable_name]
        except KeyError as error:
            raise ValueError(
                f"Unknown sequential posterior variable: {variable_name!r}."
            ) from error


def validate_sequential_observations(
    time_seconds,
    x_observed,
    y_observed,
    *,
    minimum_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matching finite time and position arrays for online updates."""
    time_seconds = np.asarray(time_seconds, dtype=float)
    x_observed = np.asarray(x_observed, dtype=float)
    y_observed = np.asarray(y_observed, dtype=float)
    if (
        time_seconds.ndim != 1
        or x_observed.shape != time_seconds.shape
        or y_observed.shape != time_seconds.shape
        or time_seconds.size < minimum_count
        or not np.all(np.isfinite(time_seconds))
        or not np.all(np.isfinite(x_observed))
        or not np.all(np.isfinite(y_observed))
        or np.any(np.diff(time_seconds) <= 0.0)
    ):
        raise ValueError(
            "Sequential time/x/y observations must be matching finite vectors "
            f"with at least {minimum_count} values and increasing timestamps."
        )
    return time_seconds, x_observed, y_observed


def validate_future_times(future_time_seconds, *, after: float) -> np.ndarray:
    """Return finite increasing forecast times after the current filter state."""
    future_time_seconds = np.asarray(future_time_seconds, dtype=float)
    if (
        future_time_seconds.ndim != 1
        or future_time_seconds.size < 1
        or not np.all(np.isfinite(future_time_seconds))
        or future_time_seconds[0] <= after
        or np.any(np.diff(future_time_seconds) <= 0.0)
    ):
        raise ValueError(
            "future_time_seconds must be a finite increasing vector after the "
            "latest observation."
        )
    return future_time_seconds


def effective_sample_size(weights: np.ndarray) -> float:
    """Return the effective number of weighted particles."""
    return float(1.0 / np.sum(weights**2))


def systematic_resample(
    weights: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Return systematic-resampling parent indices."""
    positions = (generator.random() + np.arange(len(weights))) / len(weights)
    cumulative_weights = np.cumsum(weights)
    cumulative_weights[-1] = 1.0
    return np.searchsorted(cumulative_weights, positions, side="right")


def regularized_cholesky(covariance: np.ndarray) -> np.ndarray:
    """Return a Cholesky factor after symmetric eigenvalue regularization."""
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = max(float(np.max(eigenvalues)), NUMERICAL_VARIANCE_FLOOR)
    eigenvalues = np.maximum(eigenvalues, largest * 1e-10)
    regularized = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return np.linalg.cholesky(regularized)
