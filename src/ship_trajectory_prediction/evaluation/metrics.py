"""Accuracy metrics for posterior ship-trajectory predictions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


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


def evaluate_position_predictions(fit, window, credible_interval=0.9):
    """Evaluate posterior position draws against one held-out trajectory.

    ADE and FDE use the Euclidean distance between the posterior-median
    position and the held-out position. Uncertainty coverage uses a radial
    posterior-predictive region centered on that median. The region radius at
    each horizon is the requested quantile of posterior draw distances from
    the median.
    """
    credible_interval = _validate_credible_interval(credible_interval)
    prediction_count = window.prediction_count
    if prediction_count < 1:
        raise ValueError("window must contain at least one held-out prediction.")
    x_samples = _prediction_samples(fit, "x_prediction", prediction_count)
    y_samples = _prediction_samples(fit, "y_prediction", prediction_count)
    if x_samples.shape != y_samples.shape:
        raise ValueError("x_prediction and y_prediction must have matching shapes.")

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
    draw_distances_m = np.hypot(
        x_samples - x_median,
        y_samples - y_median,
    )
    prediction_radius_m = np.quantile(
        draw_distances_m,
        credible_interval,
        axis=0,
    )
    covered = errors_m <= prediction_radius_m
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
            f"Radial {interval_percent:g}% coverage",
            f"{evaluation.radial_coverage:.1%}",
        ),
        (
            "Mean prediction radius",
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

    samples = np.asarray(fit.stan_variable(variable_name), dtype=float)
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
