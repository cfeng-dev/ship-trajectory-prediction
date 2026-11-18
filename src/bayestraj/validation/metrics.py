"""Accuracy metrics for posterior ship-trajectory predictions."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import beta
from scipy.stats import chi2, norm, t

import bayestraj.validation.reporting as reporting

NLPD_MIN_VARIANCE_M2 = 1e-12
NLPD_RELATIVE_VARIANCE_FLOOR = 1e-12
ENERGY_QUADRATURE_ORDER = 64
STUDENT_T_DEGREES_OF_FREEDOM = 5
GAUSSIAN_SCORE_CSV_NAMES = {
    "negative_log_predictive_density": "gaussian_negative_log_predictive_density",
    "energy_score_m": "gaussian_energy_score_m",
}
POSITION_SCORE_COLUMNS = (
    "negative_log_predictive_density",
    "student_t_negative_log_predictive_density",
    "energy_score_m",
    "student_t_energy_score_m",
)


def _gaussian_geometry(actual_positions, means, covariances):
    """Score latent 2D positions under a joint Gaussian in metre coordinates.

    Only numerical regularization is added: each covariance eigenvalue has an
    absolute floor of 1e-12 m^2 and a relative floor of 1e-12 of its largest
    eigenvalue. No sensor or observation noise is added to the forecast.
    """
    actual = np.asarray(actual_positions, dtype=float)
    means = np.asarray(means, dtype=float)
    covariance = np.asarray(covariances, dtype=float)
    if (
        actual.ndim != 2
        or actual.shape[1:] != (2,)
        or len(actual) == 0
        or means.shape != actual.shape
        or covariance.shape != (len(actual), 2, 2)
        or not np.isfinite(actual).all()
        or not np.isfinite(means).all()
        or not np.isfinite(covariance).all()
    ):
        raise ValueError(
            "Gaussian positions need finite matching (n, 2) means/targets and (n, 2, 2) covariances."
        )
    symmetric = 0.5 * (covariance + covariance.swapaxes(-1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    floor = np.maximum(
        NLPD_MIN_VARIANCE_M2, np.max(eigenvalues, axis=1) * NLPD_RELATIVE_VARIANCE_FLOOR
    )
    if np.any(eigenvalues < -floor[:, None]):
        raise ValueError("Gaussian position covariances must be positive semidefinite.")
    eigenvalues = np.maximum(eigenvalues, floor[:, None])
    return actual, means, eigenvalues, eigenvectors


def gaussian_position_log_predictive_density(actual_positions, means, covariances):
    """Return joint latent Gaussian log densities with common numerical floors."""
    actual, means, eigenvalues, eigenvectors = _gaussian_geometry(
        actual_positions, means, covariances
    )
    projected = np.einsum("nji,nj->ni", eigenvectors, actual - means)
    squared_distance = np.sum(projected**2 / eigenvalues, axis=1)
    return -np.log(2 * np.pi) - 0.5 * (
        np.log(eigenvalues).sum(axis=1) + squared_distance
    )


def latent_position_moments(x_samples, y_samples):
    """Return mean and unbiased covariance of paired latent forecast draws.

    Covariance describes forecast-position spread, not uncertainty of the sample
    mean. A single draw has zero empirical covariance and uses the numerical floor.
    """
    x_samples, y_samples = (
        np.asarray(values, dtype=float) for values in (x_samples, y_samples)
    )
    if (
        x_samples.ndim != 2
        or y_samples.shape != x_samples.shape
        or x_samples.shape[0] == 0
        or x_samples.shape[1] == 0
        or not np.isfinite(x_samples).all()
        or not np.isfinite(y_samples).all()
    ):
        raise ValueError(
            "Latent forecasts need finite paired (draw, horizon) positions."
        )
    positions = np.stack((x_samples, y_samples), axis=-1)
    means = positions.mean(axis=0)
    centered = positions - means
    covariance = np.einsum("shi,shj->hij", centered, centered) / max(
        len(positions) - 1, 1
    )
    return means, covariance


def latent_position_log_predictive_density(x_actual, y_actual, x_samples, y_samples):
    """Score Gaussian moments of paired latent draws without sensor noise."""
    x_actual, y_actual = (
        np.asarray(x_actual, dtype=float),
        np.asarray(y_actual, dtype=float),
    )
    if x_actual.ndim != 1 or y_actual.shape != x_actual.shape:
        raise ValueError("Latent targets must be matching finite vectors.")
    means, covariance = latent_position_moments(x_samples, y_samples)
    return gaussian_position_log_predictive_density(
        np.column_stack((x_actual, y_actual)), means, covariance
    )


def regularize_position_covariances(covariances):
    """Apply the same absolute and relative covariance floors for every method."""
    covariances = np.asarray(covariances, dtype=float)
    if covariances.ndim != 3 or covariances.shape[1:] != (2, 2):
        raise ValueError("Position covariances must be finite (count, 2, 2) arrays.")
    count = len(covariances)
    _, _, eigenvalues, eigenvectors = _gaussian_geometry(
        np.zeros((count, 2)), np.zeros((count, 2)), covariances
    )
    return (eigenvectors * eigenvalues[:, None, :]) @ eigenvectors.swapaxes(-1, -2)


def _gaussian_squared_distances(actual, means, eigenvalues, eigenvectors):
    projected = np.einsum("nji,nj->ni", eigenvectors, actual - means)
    return np.sum(projected**2 / eigenvalues, axis=1)


def gaussian_position_coverage(
    actual_positions, means, covariances, *, probabilities=(0.5, 0.9)
):
    """Return per-step Gaussian-ellipse membership without scoring overhead."""
    actual, means, eigenvalues, eigenvectors = _gaussian_geometry(
        actual_positions, means, covariances
    )
    distances = _gaussian_squared_distances(actual, means, eigenvalues, eigenvectors)
    return {
        probability: distances <= chi2.ppf(_validate_credible_interval(probability), 2)
        for probability in probabilities
    }


@lru_cache(maxsize=1)
def _energy_quadrature():
    roots, weights = np.polynomial.legendre.leggauss(ENERGY_QUADRATURE_ORDER)
    return (roots + 1) / 2, weights / 2


def _gaussian_expected_norm(projected_mean, variances):
    """Integrate E||N(mean, diag(variances))|| using its squared-norm Laplace transform.

    sqrt(q) = integral_0^infinity (1-exp(-t*q)) / (2*sqrt(pi)*t**1.5) dt.
    Substitute t=(u/(1-u))**2 and normalize the RMS length to avoid scale-dependent
    integration errors. This one-dimensional integral is rotation invariant and
    remains accurate for nearly rank-one covariances. expm1 preserves small terms.
    """
    scale = np.sqrt(np.sum(projected_mean**2 + variances, axis=1))
    variance = variances / scale[:, None] ** 2
    mean_squared = (projected_mean / scale[:, None]) ** 2
    nodes, weights = _energy_quadrature()
    t = (nodes / (1 - nodes)) ** 2
    scaled_variance = 2 * t[None, :, None] * variance[:, None, :]
    log_laplace = -0.5 * np.log1p(scaled_variance).sum(axis=2) - (
        t[None, :, None] * mean_squared[:, None, :] / (1 + scaled_variance)
    ).sum(axis=2)
    return (
        scale
        / np.sqrt(np.pi)
        * np.sum(
            weights[None, :] * -np.expm1(log_laplace) / nodes[None, :] ** 2, axis=1
        )
    )


def _student_t_position_scores(projected_errors, variances):
    """Score a bivariate Student t with the supplied mean and covariance."""
    df = STUDENT_T_DEGREES_OF_FREEDOM
    shape_variances = variances * ((df - 2) / df)
    squared_distance = np.sum(projected_errors**2 / shape_variances, axis=1)
    nlpd = (
        np.log(2 * np.pi)
        + 0.5 * np.log(shape_variances).sum(axis=1)
        + 0.5 * (df + 2) * np.log1p(squared_distance / df)
    )

    # The energy score in 2D is half the integral of projected CRPS over [0, pi].
    # Split where the projected error vanishes to resolve narrow forecasts.
    nodes, weights = _energy_quadrature()
    zero_angle = np.arctan2(-projected_errors[:, 0], projected_errors[:, 1]) % np.pi
    boundaries = np.column_stack(
        (np.zeros_like(zero_angle), zero_angle, np.full_like(zero_angle, np.pi))
    )
    widths = np.diff(boundaries, axis=1)
    angles = boundaries[:, :-1, None] + widths[:, :, None] * nodes
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    scales = np.sqrt(np.einsum("naki,ni->nak", directions**2, shape_variances))
    standardized = np.einsum("naki,ni->nak", directions, projected_errors) / scales

    # Closed-form Student-t CRPS, Jordan, Krüger and Lerch (2019), Appendix A.
    half_pair_distance = (
        2 * np.sqrt(df) * beta(0.5, df - 0.5) / ((df - 1) * beta(0.5, df / 2) ** 2)
    )
    crps = scales * (
        standardized * (2 * t.cdf(standardized, df) - 1)
        + 2 * t.pdf(standardized, df) * (df + standardized**2) / (df - 1)
        - half_pair_distance
    )
    energy = 0.5 * np.sum(widths * (crps @ weights), axis=1)
    return {
        "student_t_energy_score_m": energy,
        "student_t_negative_log_predictive_density": nlpd,
    }


def gaussian_position_metrics(
    actual_positions,
    means,
    covariances,
    *,
    credible_interval=0.9,
    student_t_scores=False,
):
    """Score Gaussian moment forecasts under one common evaluation protocol.

    All probability scores use latent positions in metres, with no extra sensor
    variance. Energy uses fixed 64-node Gauss-Legendre integration. Coverage uses
    chi-square ellipses; marginal intervals use Gaussian quantiles. Point-forecast
    medians are evaluated separately. Optionally include Student-t Energy Score
    and NLPD using the same mean and covariance.
    """
    credible_interval = _validate_credible_interval(credible_interval)
    actual, means, eigenvalues, eigenvectors = _gaussian_geometry(
        actual_positions, means, covariances
    )
    covariance = (eigenvectors * eigenvalues[:, None, :]) @ eigenvectors.swapaxes(
        -1, -2
    )
    squared_distance = _gaussian_squared_distances(
        actual, means, eigenvalues, eigenvectors
    )
    threshold = float(chi2.ppf(credible_interval, 2))
    deviation = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    quantile = float(norm.ppf((1 + credible_interval) / 2))
    lower, upper = means - quantile * deviation, means + quantile * deviation
    projected = np.einsum("nji,nj->ni", eigenvectors, actual - means)
    energy = _gaussian_expected_norm(
        projected, eigenvalues
    ) - 0.5 * _gaussian_expected_norm(np.zeros_like(projected), 2 * eigenvalues)
    scores = {
        "negative_log_predictive_density": np.log(2 * np.pi)
        + 0.5 * (np.log(eigenvalues).sum(axis=1) + squared_distance),
        "energy_score_m": np.asarray(energy),
    }
    if student_t_scores:
        scores.update(_student_t_position_scores(projected, eigenvalues))
    return {
        "x_lower": lower[:, 0],
        "x_upper": upper[:, 0],
        "y_lower": lower[:, 1],
        "y_upper": upper[:, 1],
        **{name: scores[name] for name in POSITION_SCORE_COLUMNS if name in scores},
        "prediction_radius_m": np.sqrt(threshold)
        * np.prod(eigenvalues, axis=1) ** 0.25,
        "squared_mahalanobis_distance": squared_distance,
        "squared_mahalanobis_radius": np.full(len(actual), threshold),
        "radial_covered": squared_distance <= threshold,
        "joint_covered_50": squared_distance <= chi2.ppf(0.5, 2),
        "joint_covered_90": squared_distance <= chi2.ppf(0.9, 2),
        "mean_marginal_interval_width_m": np.mean(upper - lower, axis=1),
    }


@dataclass(frozen=True)
class PositionEvaluation:
    """Point and uncertainty metrics for one held-out trajectory window."""

    prediction_table: pd.DataFrame
    errors_m: np.ndarray
    ade_m: float
    fde_m: float
    energy_score_m: float
    radial_coverage: float
    joint_coverage_50: float
    joint_coverage_90: float
    mean_prediction_radius_m: float
    mean_marginal_interval_width_m: float
    total_negative_log_predictive_density: float
    mean_negative_log_predictive_density: float
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


def gaussian_covariance_regions(x_values, y_values, *, probabilities=(0.5, 0.9)):
    """Construct the same Gaussian moment ellipses used for scored coverage."""
    x_values, y_values = (
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
    )
    if x_values.ndim != 1 or y_values.shape != x_values.shape:
        raise ValueError("Forecast coordinates must be matching finite vectors.")
    means, covariances = latent_position_moments(
        np.asarray(x_values, dtype=float)[:, None],
        np.asarray(y_values, dtype=float)[:, None],
    )
    _, means, variances, directions = _gaussian_geometry(means, means, covariances)
    precision = (directions[0] * (1 / variances[0])) @ directions[0].T
    angle = float(np.degrees(np.arctan2(directions[0, 1, 1], directions[0, 0, 1])))
    regions = {}
    for probability in probabilities:
        radius = float(chi2.ppf(_validate_credible_interval(probability), 2))
        axes = np.sqrt(radius * variances[0])
        regions[probability] = EmpiricalCovarianceRegion(
            center=tuple(means[0]),
            width=2 * float(axes[1]),
            height=2 * float(axes[0]),
            angle_degrees=angle,
            squared_radius=radius,
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
    student_t_scores=False,
):
    """Evaluate posterior position draws against one held-out trajectory.

    Point errors evaluate the posterior position mean. The default probability
    metrics use a Gaussian moment approximation of latent positions. Optional
    Student-t scores use the same moments.
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

    x_median = np.median(x_samples, axis=0)
    y_median = np.median(y_samples, axis=0)
    means, covariances = latent_position_moments(x_samples, y_samples)
    errors_m = np.linalg.norm(means - np.column_stack((x_actual, y_actual)), axis=1)
    scores = gaussian_position_metrics(
        np.column_stack((x_actual, y_actual)),
        means,
        covariances,
        credible_interval=credible_interval,
        student_t_scores=student_t_scores,
    )

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
            "x_mean": means[:, 0],
            "y_mean": means[:, 1],
            "point_estimate": "mean",
            "position_error_m": errors_m,
            **scores,
        }
    )

    return PositionEvaluation(
        prediction_table=prediction_table,
        errors_m=errors_m,
        ade_m=float(np.mean(errors_m)),
        fde_m=float(errors_m[-1]),
        energy_score_m=float(np.mean(scores["energy_score_m"])),
        radial_coverage=float(np.mean(scores["radial_covered"])),
        joint_coverage_50=float(np.mean(scores["joint_covered_50"])),
        joint_coverage_90=float(np.mean(scores["joint_covered_90"])),
        mean_prediction_radius_m=float(np.mean(scores["prediction_radius_m"])),
        mean_marginal_interval_width_m=float(
            np.mean(scores["mean_marginal_interval_width_m"])
        ),
        total_negative_log_predictive_density=float(
            np.sum(scores["negative_log_predictive_density"])
        ),
        mean_negative_log_predictive_density=float(
            np.mean(scores["negative_log_predictive_density"])
        ),
        credible_interval=credible_interval,
    )


def format_position_evaluation(evaluation, *, computation_time_seconds=None):
    """Format one position evaluation as a concise console report."""
    if not isinstance(evaluation, PositionEvaluation):
        raise TypeError("evaluation must be a PositionEvaluation instance.")
    if computation_time_seconds is not None and (
        isinstance(computation_time_seconds, bool)
        or not np.isfinite(computation_time_seconds)
        or computation_time_seconds < 0
    ):
        raise ValueError("computation_time_seconds must be finite and non-negative.")

    interval_percent = 100 * evaluation.credible_interval
    table = reporting.format_prediction_table(
        evaluation.prediction_table,
        columns=[
            "horizon_seconds",
            "x_actual",
            "y_actual",
            "x_mean",
            "y_mean",
            "position_error_m",
            "prediction_radius_m",
            "radial_covered",
        ],
    )

    metric_rows = [
        ("ADE", f"{evaluation.ade_m:.2f} m"),
        ("FDE", f"{evaluation.fde_m:.2f} m"),
        ("Energy Score", f"{evaluation.energy_score_m:.2f} m"),
        (
            "Total negative log predictive density",
            f"{evaluation.total_negative_log_predictive_density:.3f}",
        ),
        (
            "Mean negative log predictive density",
            f"{evaluation.mean_negative_log_predictive_density:.3f}",
        ),
        ("Joint 2D 50% coverage", f"{evaluation.joint_coverage_50:.1%}"),
        ("Joint 2D 90% coverage", f"{evaluation.joint_coverage_90:.1%}"),
        (
            "Mean equivalent region radius",
            f"{evaluation.mean_prediction_radius_m:.2f} m",
        ),
        (
            "Mean marginal interval width",
            f"{evaluation.mean_marginal_interval_width_m:.2f} m",
        ),
    ]
    if evaluation.credible_interval not in {0.5, 0.9}:
        metric_rows.insert(
            6,
            (
                f"Joint 2D {interval_percent:g}% coverage",
                f"{evaluation.radial_coverage:.1%}",
            ),
        )
    if computation_time_seconds is not None:
        metric_rows.append(("Computation time", f"{computation_time_seconds:.3f} s"))
    return reporting.format_evaluation_report(metric_rows, table)


def print_position_evaluation(evaluation, *, computation_time_seconds=None):
    """Print one shared position-accuracy report."""
    print()
    print(
        format_position_evaluation(
            evaluation,
            computation_time_seconds=computation_time_seconds,
        )
    )


def _prediction_samples(fit, variable_name, prediction_count):
    """Extract and validate one posterior prediction matrix."""
    if not hasattr(fit, "stan_variable"):
        raise TypeError("fit must provide CmdStan-style posterior variables.")

    samples = reporting.posterior_variable_samples(fit, variable_name)
    if samples.ndim != 2 or samples.shape[1] != prediction_count:
        raise ValueError(
            f"Posterior variable {variable_name!r} has an unexpected shape."
        )
    if samples.shape[0] == 0 or not np.all(np.isfinite(samples)):
        raise ValueError(
            f"Posterior variable {variable_name!r} must contain finite draws."
        )
    return samples


def _posterior_scalar_samples(fit, variable_name, *, draw_count):
    """Extract one finite positive scalar posterior draw vector."""
    samples = reporting.posterior_variable_samples(fit, variable_name)
    if samples.ndim != 1 or samples.shape != (draw_count,):
        raise ValueError(
            f"Posterior variable {variable_name!r} has an unexpected shape."
        )
    if not np.all(np.isfinite(samples)) or np.any(samples <= 0):
        raise ValueError(
            f"Posterior variable {variable_name!r} must contain finite positive draws."
        )
    return samples


def joint_log_predictive_density(
    x_actual,
    y_actual,
    x_samples,
    y_samples,
    observation_noise_samples,
):
    """Return the held-out 2D observation log density at each horizon.

    The posterior predictive density is a Monte-Carlo mixture over draws of
    the latent future state and the inferred isotropic observation noise.
    This observation-target helper is separate from latent-position NLPD.
    """
    x_actual = np.asarray(x_actual, dtype=float)
    y_actual = np.asarray(y_actual, dtype=float)
    x_samples = np.asarray(x_samples, dtype=float)
    y_samples = np.asarray(y_samples, dtype=float)
    observation_noise_samples = np.asarray(
        observation_noise_samples,
        dtype=float,
    )
    if (
        x_actual.ndim != 1
        or y_actual.shape != x_actual.shape
        or x_samples.ndim != 2
        or y_samples.shape != x_samples.shape
        or x_samples.shape[1] != x_actual.size
        or observation_noise_samples.shape != (x_samples.shape[0],)
        or not np.all(np.isfinite(x_actual))
        or not np.all(np.isfinite(y_actual))
        or not np.all(np.isfinite(x_samples))
        or not np.all(np.isfinite(y_samples))
        or not np.all(np.isfinite(observation_noise_samples))
        or np.any(observation_noise_samples <= 0.0)
    ):
        raise ValueError(
            "Actual positions, paired draw matrices, and positive observation "
            "noise draws must have matching finite shapes."
        )
    squared_error = (x_samples - x_actual) ** 2 + (y_samples - y_actual) ** 2
    log_component_density = (
        -np.log(2 * np.pi * observation_noise_samples[:, np.newaxis] ** 2)
        - 0.5 * squared_error / observation_noise_samples[:, np.newaxis] ** 2
    )
    maximum_log_density = np.max(log_component_density, axis=0)
    return maximum_log_density + np.log(
        np.mean(np.exp(log_component_density - maximum_log_density), axis=0)
    )


def _joint_energy_scores(x_actual, y_actual, x_samples, y_samples):
    """Return the joint 2D energy score at each prediction horizon."""
    distance_to_actual = np.hypot(x_samples - x_actual, y_samples - y_actual)
    energy_scores = []
    for time_index in range(x_samples.shape[1]):
        x_values = x_samples[:, time_index]
        y_values = y_samples[:, time_index]
        pairwise_distances = np.hypot(
            x_values[:, np.newaxis] - x_values,
            y_values[:, np.newaxis] - y_values,
        )
        energy_scores.append(
            float(np.mean(distance_to_actual[:, time_index]))
            - 0.5 * float(np.mean(pairwise_distances))
        )
    return np.asarray(energy_scores, dtype=float)


def _joint_region_coverage(
    prediction_region_sets,
    *,
    probability,
    x_actual,
    y_actual,
):
    """Return membership of held-out positions in one joint region level."""
    return np.asarray(
        [
            regions[probability].contains(actual_x, actual_y)
            for regions, actual_x, actual_y in zip(
                prediction_region_sets,
                x_actual,
                y_actual,
                strict=True,
            )
        ],
        dtype=bool,
    )


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
