"""Accuracy metrics for posterior ship-trajectory predictions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ship_trajectory_prediction.evaluation.reporting import (
    posterior_variable_samples,
)


@dataclass(frozen=True)
class PositionEvaluation:
    """Point and uncertainty metrics for one held-out trajectory window."""

    prediction_table: pd.DataFrame
    errors_m: np.ndarray
    ade_m: float
    fde_m: float
    radial_coverage: float
    mean_prediction_radius_m: float
    mean_marginal_interval_width_m: float
    credible_interval: float


@dataclass(frozen=True, slots=True)
class EmpiricalCovarianceRegion:
    """One joint empirical posterior region in two coordinates."""

    center: tuple[float, float]
    width: float
    height: float
    angle_degrees: float
    squared_radius: float
    precision_xx: float
    precision_xy: float
    precision_yy: float

    @property
    def equivalent_radius(self) -> float:
        """Return the radius of a circle with the same area as the ellipse."""
        return 0.5 * float(np.sqrt(self.width * self.height))

    def squared_distance(self, x_value: float, y_value: float) -> float:
        """Return squared Mahalanobis distance from the region center."""
        delta_x = float(x_value) - self.center[0]
        delta_y = float(y_value) - self.center[1]
        return float(
            self.precision_xx * delta_x**2
            + 2 * self.precision_xy * delta_x * delta_y
            + self.precision_yy * delta_y**2
        )

    def contains(self, x_value: float, y_value: float) -> bool:
        """Return whether one point lies inside or on this ellipse."""
        squared_distance = self.squared_distance(x_value, y_value)
        return bool(
            squared_distance <= self.squared_radius
            or np.isclose(squared_distance, self.squared_radius)
        )


def empirical_covariance_regions(
    x_values,
    y_values,
    *,
    probabilities=(0.5, 0.9),
) -> dict[float, EmpiricalCovarianceRegion]:
    """Construct joint empirical regions from paired posterior coordinates.

    Orientation and eccentricity come from the regularized two-dimensional
    sample covariance. Region sizes use empirical quantiles of the corresponding
    Mahalanobis distances rather than a Gaussian chi-square assumption.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if (
        x_values.ndim != 1
        or y_values.shape != x_values.shape
        or x_values.size == 0
        or not np.all(np.isfinite(x_values))
        or not np.all(np.isfinite(y_values))
    ):
        raise ValueError("x_values and y_values must be matching finite vectors.")
    probabilities = tuple(
        _validate_credible_interval(probability) for probability in probabilities
    )
    if not probabilities:
        raise ValueError("probabilities must contain at least one value.")

    samples = np.column_stack((x_values, y_values))
    center = np.median(samples, axis=0)
    centered_samples = samples - center
    covariance = centered_samples.T @ centered_samples / max(len(samples) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest_eigenvalue = max(float(np.max(eigenvalues)), 1e-6)
    eigenvalues = np.maximum(eigenvalues, largest_eigenvalue * 1e-6)
    projected_samples = centered_samples @ eigenvectors
    squared_distances = np.sum(projected_samples**2 / eigenvalues, axis=1)
    precision = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T
    angle_degrees = float(
        np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    )

    regions = {}
    for probability in probabilities:
        squared_radius = max(
            float(np.quantile(squared_distances, probability)),
            1e-12,
        )
        half_axes = np.sqrt(squared_radius * eigenvalues)
        regions[probability] = EmpiricalCovarianceRegion(
            center=(float(center[0]), float(center[1])),
            width=2 * float(half_axes[1]),
            height=2 * float(half_axes[0]),
            angle_degrees=angle_degrees,
            squared_radius=squared_radius,
            precision_xx=float(precision[0, 0]),
            precision_xy=float(precision[0, 1]),
            precision_yy=float(precision[1, 1]),
        )
    return regions


def evaluate_position_predictions(
    fit,
    window,
    credible_interval=0.9,
    *,
    position_variable_names=("x_prediction", "y_prediction"),
):
    """Evaluate posterior position draws against one held-out trajectory.

    ADE and FDE use the Euclidean distance between the posterior-median
    position and the held-out position. Coverage uses the same joint empirical
    covariance ellipse that is drawn around the posterior samples. Its size is
    the requested empirical Mahalanobis-distance quantile at each horizon.
    """
    credible_interval = _validate_credible_interval(credible_interval)
    x_variable_name, y_variable_name = _validate_position_variable_names(
        position_variable_names
    )
    prediction_count = window.prediction_count
    if prediction_count < 1:
        raise ValueError("window must contain at least one held-out prediction.")
    x_samples = _prediction_samples(fit, x_variable_name, prediction_count)
    y_samples = _prediction_samples(fit, y_variable_name, prediction_count)
    if x_samples.shape != y_samples.shape:
        raise ValueError("Position prediction variables must have matching shapes.")

    prediction = window.prediction_slice
    x_actual = np.asarray(window.x_meters[prediction], dtype=float)
    y_actual = np.asarray(window.y_meters[prediction], dtype=float)
    if x_actual.shape != (prediction_count,) or y_actual.shape != (prediction_count,):
        raise ValueError("Held-out positions do not match prediction_count.")
    if not np.all(np.isfinite(x_actual)) or not np.all(np.isfinite(y_actual)):
        raise ValueError("Held-out positions must contain only finite values.")

    lower_probability = (1 - credible_interval) / 2
    upper_probability = 1 - lower_probability
    x_median = np.median(x_samples, axis=0)
    y_median = np.median(y_samples, axis=0)
    x_lower = np.quantile(x_samples, lower_probability, axis=0)
    x_upper = np.quantile(x_samples, upper_probability, axis=0)
    y_lower = np.quantile(y_samples, lower_probability, axis=0)
    y_upper = np.quantile(y_samples, upper_probability, axis=0)

    errors_m = np.hypot(x_median - x_actual, y_median - y_actual)
    prediction_regions = tuple(
        empirical_covariance_regions(
            x_samples[:, time_index],
            y_samples[:, time_index],
            probabilities=(credible_interval,),
        )[credible_interval]
        for time_index in range(prediction_count)
    )
    prediction_radius_m = np.asarray(
        [region.equivalent_radius for region in prediction_regions],
        dtype=float,
    )
    squared_mahalanobis_distance = np.asarray(
        [
            region.squared_distance(actual_x, actual_y)
            for region, actual_x, actual_y in zip(
                prediction_regions,
                x_actual,
                y_actual,
                strict=True,
            )
        ],
        dtype=float,
    )
    squared_mahalanobis_radius = np.asarray(
        [region.squared_radius for region in prediction_regions],
        dtype=float,
    )
    covered = np.asarray(
        [
            region.contains(actual_x, actual_y)
            for region, actual_x, actual_y in zip(
                prediction_regions,
                x_actual,
                y_actual,
                strict=True,
            )
        ],
        dtype=bool,
    )
    mean_marginal_interval_width_m = 0.5 * ((x_upper - x_lower) + (y_upper - y_lower))

    prediction_times = np.asarray(window.time_seconds[prediction], dtype=float)
    prediction_start_time = float(window.time_seconds[window.observation_count - 1])
    horizon_seconds = prediction_times - prediction_start_time
    if (
        prediction_times.shape != (prediction_count,)
        or not np.all(np.isfinite(horizon_seconds))
        or np.any(horizon_seconds <= 0)
        or np.any(np.diff(horizon_seconds) <= 0)
    ):
        raise ValueError(
            "Prediction horizons must be finite, positive, and strictly increasing."
        )

    prediction_table = pd.DataFrame(
        {
            "time": window.timestamps[prediction],
            "horizon_seconds": horizon_seconds,
            "x_actual": x_actual,
            "y_actual": y_actual,
            "x_median": x_median,
            "y_median": y_median,
            "x_lower": x_lower,
            "x_upper": x_upper,
            "y_lower": y_lower,
            "y_upper": y_upper,
            "position_error_m": errors_m,
            "prediction_radius_m": prediction_radius_m,
            "squared_mahalanobis_distance": squared_mahalanobis_distance,
            "squared_mahalanobis_radius": squared_mahalanobis_radius,
            "radial_covered": covered,
            "mean_marginal_interval_width_m": mean_marginal_interval_width_m,
        }
    )

    return PositionEvaluation(
        prediction_table=prediction_table,
        errors_m=errors_m,
        ade_m=float(np.mean(errors_m)),
        fde_m=float(errors_m[-1]),
        radial_coverage=float(np.mean(covered)),
        mean_prediction_radius_m=float(np.mean(prediction_radius_m)),
        mean_marginal_interval_width_m=float(np.mean(mean_marginal_interval_width_m)),
        credible_interval=credible_interval,
    )


def format_position_evaluation(evaluation):
    """Format one position evaluation as a concise console report."""
    if not isinstance(evaluation, PositionEvaluation):
        raise TypeError("evaluation must be a PositionEvaluation instance.")

    interval_percent = 100 * evaluation.credible_interval
    table = evaluation.prediction_table[
        [
            "horizon_seconds",
            "x_actual",
            "y_actual",
            "x_median",
            "y_median",
            "position_error_m",
            "prediction_radius_m",
            "radial_covered",
        ]
    ].copy()
    numeric_columns = table.select_dtypes(include=[np.number]).columns
    table[numeric_columns] = table[numeric_columns].round(2)

    metric_rows = [
        ("ADE", f"{evaluation.ade_m:.2f} m"),
        ("FDE", f"{evaluation.fde_m:.2f} m"),
        (
            f"Joint 2D {interval_percent:g}% coverage",
            f"{evaluation.radial_coverage:.1%}",
        ),
        (
            "Mean equivalent region radius",
            f"{evaluation.mean_prediction_radius_m:.2f} m",
        ),
        (
            "Mean marginal interval width",
            f"{evaluation.mean_marginal_interval_width_m:.2f} m",
        ),
    ]
    label_width = max(len(label) for label, _ in metric_rows)
    metric_lines = [f"{label:<{label_width}} : {value}" for label, value in metric_rows]

    return "\n".join(
        [
            "Held-out position accuracy:",
            *metric_lines,
            "\nPer-horizon accuracy:",
            table.to_string(index=False),
        ]
    )


def print_position_evaluation(evaluation):
    """Print one shared position-accuracy report."""
    print()
    print(format_position_evaluation(evaluation))


def _prediction_samples(fit, variable_name, prediction_count):
    """Extract and validate one posterior prediction matrix."""
    if not hasattr(fit, "stan_variable"):
        raise TypeError("fit must provide CmdStan-style posterior variables.")

    samples = posterior_variable_samples(fit, variable_name)
    if samples.ndim != 2 or samples.shape[1] != prediction_count:
        raise ValueError(
            f"Posterior variable {variable_name!r} has an unexpected shape."
        )
    if samples.shape[0] == 0 or not np.all(np.isfinite(samples)):
        raise ValueError(
            f"Posterior variable {variable_name!r} must contain finite draws."
        )
    return samples


def _validate_credible_interval(credible_interval):
    """Return one validated credible-interval probability."""
    try:
        credible_interval = float(credible_interval)
    except (TypeError, ValueError) as error:
        raise ValueError("credible_interval must be between 0 and 1.") from error
    if not np.isfinite(credible_interval) or not 0 < credible_interval < 1:
        raise ValueError("credible_interval must be between 0 and 1.")
    return credible_interval


def _validate_position_variable_names(position_variable_names):
    """Return two explicit posterior variable names for x and y predictions."""
    if (
        not isinstance(position_variable_names, (tuple, list))
        or len(position_variable_names) != 2
        or not all(
            isinstance(name, str) and name.strip() for name in position_variable_names
        )
    ):
        raise ValueError(
            "position_variable_names must contain non-empty x and y names."
        )
    return tuple(name.strip() for name in position_variable_names)
