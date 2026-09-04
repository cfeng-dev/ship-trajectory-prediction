"""Reusable validation for numeric function arguments."""

import numpy as np


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
