"""Rolling BR24 IMM evaluation with median point forecasts and Gaussian scores."""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

import bayestraj.inference.imm_adapter as imm_adapter
import bayestraj.observations.br24 as br24
import bayestraj.validation.plotting as validation_plotting
import bayestraj.validation.reporting as reporting
import bayestraj.validation.rolling as rolling
from bayestraj.validation.cli import IMMEvaluationOptions

# The IMM forecast covariance is the complete predictive position uncertainty.
# This small numerical floor only makes the Gaussian log density well defined.
RESIDUAL_POSITION_NOISE_FLOOR_M = 1e-6
GAUSS_HERMITE_ORDER = 12


@dataclass(frozen=True, slots=True)
class IMMRollingSummary:
    """Aggregate forecast metrics with the Bayesian rolling-report semantics."""

    window_count: int
    forecast_count: int
    ade_m: float
    fde_m: float
    mean_energy_score_m: float
    mean_window_runtime_seconds: float
    median_window_runtime_seconds: float
    total_computation_time_seconds: float
    radial_coverage: float
    joint_coverage_50: float
    joint_coverage_90: float
    mean_prediction_radius_m: float
    mean_marginal_interval_width_m: float
    elpd: float
    mean_log_predictive_density: float
    per_horizon_table: pd.DataFrame


def run_imm_evaluation(
    *,
    data_file: str | Path,
    imm_python: str | Path,
    options: IMMEvaluationOptions,
    credible_interval: float,
    gaussian_plot_draws: int,
    plot_seed: int,
) -> tuple[pd.DataFrame, IMMRollingSummary]:
    """Track raw radar cycles online and evaluate only after each forecast."""
    data_file = Path(data_file)
    imm_python = Path(imm_python)
    tracker_type = _load_optional_imm_tracker(imm_python)
    cycles = br24.read_br24_tracking_input(data_file)
    reference = br24.read_br24_tracking_reference(data_file)
    if reference is None:
        raise ValueError("IMM evaluation requires ground_truth_* columns.")
    _validate_reference_alignment(cycles, reference)
    if not cycles[0].detections:
        raise ValueError(
            "The earliest BR24 radar cycle has no detection to initialize."
        )

    windows = rolling.build_online_forecast_specs(
        len(cycles),
        initial_observation_count=options.observation_count,
        prediction_count=options.prediction_count,
        stride=options.stride,
    )
    if options.max_windows is not None:
        windows = windows[: options.max_windows]
    if not windows:
        raise ValueError("No IMM forecast windows remain after the configured limit.")

    initial_detection = cycles[0].detections[0]
    tracker = tracker_type.initialize(
        cycles[0], initial_detection_id=initial_detection.detection_id
    )
    cycle_times = np.asarray([cycle.time_seconds for cycle in cycles], dtype=float)
    tracked_positions = [tracker.estimated_position]
    processed_cycle_count = 1
    show_rolling_plot = options.show_plot and options.stride == options.prediction_count
    plot_groups = []
    prediction_tables = []

    _print_configuration(
        options,
        data_file=data_file,
        window_count=len(windows),
        initial_detection_id=initial_detection.detection_id,
    )
    for number, specification in enumerate(windows, start=1):
        runtime_started = time.perf_counter()
        while processed_cycle_count < specification.forecast_start_index:
            tracker.process_cycle(cycles[processed_cycle_count])
            tracked_positions.append(tracker.estimated_position)
            processed_cycle_count += 1

        forecast_stop_index = (
            specification.forecast_start_index + specification.prediction_count
        )
        future_times = cycle_times[
            specification.forecast_start_index : forecast_stop_index
        ]
        forecast = tracker.forecast(
            future_times,
            seed=plot_seed + specification.window_index,
        )
        # Share the predictor's exact marginal medians, not a sample median.
        forecast_medians = imm_adapter.forecast_position_medians(tracker, forecast)
        forecast_runtime_seconds = time.perf_counter() - runtime_started

        # Evaluation labels are read only after the raw-radar tracker forecast.
        table, covariances = _evaluate_forecast(
            forecast,
            credible_interval=credible_interval,
            forecast_median_positions=forecast_medians,
            reference=reference,
            specification=specification,
            forecast_origin_time=tracker.last_observation_time_seconds,
            forecast_origin_position=tracker.estimated_position,
            window_runtime_seconds=time.perf_counter() - runtime_started,
            forecast_runtime_seconds=forecast_runtime_seconds,
        )
        prediction_tables.append(table)
        if show_rolling_plot:
            position_samples = _draw_gaussian_position_samples(
                forecast.position_means,
                covariances,
                sample_count=gaussian_plot_draws,
                seed=plot_seed + specification.window_index,
            )
            plot_groups.append(
                validation_plotting.RollingPosteriorPlotData(
                    forecast_origin_x=float(tracker.estimated_position[0]),
                    forecast_origin_y=float(tracker.estimated_position[1]),
                    x_samples=position_samples[:, :, 0],
                    y_samples=position_samples[:, :, 1],
                    forecast_positions=forecast_medians,
                    forecast_time_seconds=np.concatenate(
                        (
                            [0.0],
                            forecast.time_seconds
                            - tracker.last_observation_time_seconds,
                        )
                    ),
                )
            )
        print(
            f"Forecast {number}/{len(windows)}: "
            f"ADE={table['position_error_m'].mean():.2f} m, "
            f"FDE={table['position_error_m'].iloc[-1]:.2f} m, "
            f"runtime={table['window_runtime_seconds'].iloc[0]:.3f} s"
        )

    predictions = pd.concat(prediction_tables, ignore_index=True)
    summary = _summarize_predictions(predictions)
    _print_summary(summary)
    if options.show_plot and not show_rolling_plot:
        print("\nRolling plot skipped because forecast origins overlap.")
        print(
            "Use --stride equal to --predictions to display non-overlapping "
            "IMM forecast paths."
        )
    elif show_rolling_plot:
        tracked_positions = np.asarray(tracked_positions)
        validation_plotting.plot_bayesian_rolling_predictions(
            reference.x_meters,
            reference.y_meters,
            plot_groups,
            initial_observation_count=options.observation_count,
            inference_mode="online",
            sample_trajectories_per_forecast=0,
            observed_route_x=tracked_positions[:, 0],
            observed_route_y=tracked_positions[:, 1],
            observed_trajectory_label="IMM-Trackpositionen",
            title_prefix="Python IMM BR24",
            forecast_label="IMM-Prognosemediane",
            show_time_labels=False,
        )
    return predictions, summary


def _load_optional_imm_tracker(imm_python: Path):
    """Return the local tracker class without making the package depend on it."""
    if not imm_python.is_file():
        raise FileNotFoundError(
            "Optional IMM implementation not found: external/IMM/br24_imm.py. "
            "Restore the local external IMM files to run this evaluation."
        )
    return imm_adapter.load_imm_tracker(imm_python)


def _validate_reference_alignment(cycles, reference) -> None:
    """Require one finite evaluation label for every radar cycle."""
    cycle_counters = np.asarray([cycle.cycle_counter for cycle in cycles], dtype=int)
    cycle_times = np.asarray([cycle.time_seconds for cycle in cycles], dtype=float)
    if not np.array_equal(
        reference.cycle_counters, cycle_counters
    ) or not np.array_equal(reference.time_seconds, cycle_times):
        raise ValueError("BR24 reference labels must align exactly with radar cycles.")


def _evaluate_forecast(
    forecast,
    *,
    credible_interval,
    forecast_median_positions,
    reference,
    specification,
    forecast_origin_time,
    forecast_origin_position,
    window_runtime_seconds,
    forecast_runtime_seconds,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Score median positions while retaining Gaussian distribution metrics."""
    prediction_count = specification.prediction_count
    expected_shape = (prediction_count, 2)
    means = np.asarray(forecast.position_means, dtype=float)
    medians = np.asarray(forecast_median_positions, dtype=float)
    covariances = _regularize_covariances(forecast.position_covariances)
    modes = np.asarray(forecast.mode_probabilities, dtype=float)
    if (
        means.shape != expected_shape
        or medians.shape != expected_shape
        or covariances.shape != (prediction_count, 2, 2)
        or modes.shape != (prediction_count, 3)
        or not np.all(np.isfinite(means))
        or not np.all(np.isfinite(medians))
        or not np.all(np.isfinite(modes))
        or not np.allclose(modes.sum(axis=1), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("IMM forecast arrays are not aligned finite distributions.")

    target_indices = np.arange(
        specification.forecast_start_index,
        specification.forecast_start_index + prediction_count,
    )
    actual_positions = np.column_stack(
        (reference.x_meters[target_indices], reference.y_meters[target_indices])
    )
    covariance_with_floor = covariances + np.eye(2) * (
        RESIDUAL_POSITION_NOISE_FLOOR_M**2
    )
    errors = np.linalg.norm(medians - actual_positions, axis=1)
    log_densities = np.asarray(
        [
            _gaussian_log_density(actual, mean, covariance)
            for actual, mean, covariance in zip(
                actual_positions,
                means,
                covariance_with_floor,
                strict=True,
            )
        ]
    )
    energy_scores = np.asarray(
        [
            _gaussian_energy_score(actual, mean, covariance)
            for actual, mean, covariance in zip(
                actual_positions,
                means,
                covariances,
                strict=True,
            )
        ]
    )
    lower_probability = (1.0 - credible_interval) / 2.0
    upper_probability = 1.0 - lower_probability
    marginal_standard_deviations = np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))
    z_score = float(norm.ppf(upper_probability))
    x_lower = means[:, 0] - z_score * marginal_standard_deviations[:, 0]
    x_upper = means[:, 0] + z_score * marginal_standard_deviations[:, 0]
    y_lower = means[:, 1] - z_score * marginal_standard_deviations[:, 1]
    y_upper = means[:, 1] + z_score * marginal_standard_deviations[:, 1]
    squared_mahalanobis = np.asarray(
        [
            _squared_mahalanobis(actual - mean, covariance)
            for actual, mean, covariance in zip(
                actual_positions,
                means,
                covariances,
                strict=True,
            )
        ]
    )
    threshold_50 = float(chi2.ppf(0.5, df=2))
    threshold_90 = float(chi2.ppf(0.9, df=2))
    threshold_interval = float(chi2.ppf(credible_interval, df=2))
    prediction_radius = np.sqrt(threshold_interval) * np.sqrt(
        np.sqrt(np.linalg.det(covariances))
    )
    future_times = np.asarray(forecast.time_seconds, dtype=float)
    if future_times.shape != (prediction_count,):
        raise ValueError("IMM forecast times do not match the requested horizon.")
    horizons = future_times - float(forecast_origin_time)
    if not np.all(np.isfinite(horizons)) or np.any(horizons <= 0.0):
        raise ValueError("IMM forecast horizons must be finite and positive.")

    table = pd.DataFrame(
        {
            "window_index": specification.window_index,
            "window_start_index": specification.start_index,
            "forecast_start_index": specification.forecast_start_index,
            "target_index": target_indices,
            "horizon_step": np.arange(1, prediction_count + 1),
            "time": pd.to_datetime(future_times, unit="s", origin="unix", utc=True),
            "horizon_seconds": horizons,
            "x_actual": actual_positions[:, 0],
            "y_actual": actual_positions[:, 1],
            "x_median": medians[:, 0],
            "y_median": medians[:, 1],
            "x_lower": x_lower,
            "x_upper": x_upper,
            "y_lower": y_lower,
            "y_upper": y_upper,
            "position_error_m": errors,
            "energy_score_m": energy_scores,
            "log_predictive_density": log_densities,
            "prediction_radius_m": prediction_radius,
            "squared_mahalanobis_distance": squared_mahalanobis,
            "squared_mahalanobis_radius": threshold_interval,
            "radial_covered": squared_mahalanobis <= threshold_interval,
            "joint_covered_50": squared_mahalanobis <= threshold_50,
            "joint_covered_90": squared_mahalanobis <= threshold_90,
            "mean_marginal_interval_width_m": 0.5
            * ((x_upper - x_lower) + (y_upper - y_lower)),
            "observation_count": specification.observation_count,
            "prediction_count": prediction_count,
            "inference_mode": "online",
            "inference_method": "imm",
            "model_variant": "imm",
            "converged": None,
            "mcmc_diagnostics_ok": None,
            "position_observation_noise_std_m": RESIDUAL_POSITION_NOISE_FLOOR_M,
            "forecast_origin_time": pd.Timestamp(
                forecast_origin_time, unit="s", tz="UTC"
            ),
            "forecast_origin_seconds": forecast_origin_time,
            "forecast_origin_x_route": float(forecast_origin_position[0]),
            "forecast_origin_y_route": float(forecast_origin_position[1]),
            "x_actual_route": actual_positions[:, 0],
            "y_actual_route": actual_positions[:, 1],
            "x_median_route": medians[:, 0],
            "y_median_route": medians[:, 1],
            "cov_xx": covariances[:, 0, 0],
            "cov_xy": covariances[:, 0, 1],
            "cov_yy": covariances[:, 1, 1],
            "mode_cv": modes[:, 0],
            "mode_ctrv": modes[:, 1],
            "mode_ca": modes[:, 2],
            "fit_runtime_seconds": forecast_runtime_seconds,
            "window_runtime_seconds": window_runtime_seconds,
        }
    )
    return table, covariances


def _regularize_covariances(covariances) -> np.ndarray:
    """Return symmetric positive-definite 2D forecast covariances."""
    covariances = np.asarray(covariances, dtype=float)
    if (
        covariances.ndim != 3
        or covariances.shape[1:] != (2, 2)
        or not np.all(np.isfinite(covariances))
    ):
        raise ValueError(
            "IMM position covariances must be finite (count, 2, 2) arrays."
        )
    regularized = np.empty_like(covariances)
    for index, covariance in enumerate(covariances):
        symmetric_covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric_covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        regularized[index] = (eigenvectors * eigenvalues) @ eigenvectors.T
    return regularized


def _gaussian_log_density(actual, mean, covariance) -> float:
    """Return the exact log density of one regularized bivariate Gaussian."""
    difference = np.asarray(actual, dtype=float) - np.asarray(mean, dtype=float)
    solved = np.linalg.solve(covariance, difference)
    _, log_determinant = np.linalg.slogdet(covariance)
    return float(-np.log(2.0 * np.pi) - 0.5 * (log_determinant + difference @ solved))


def _gaussian_energy_score(actual, mean, covariance) -> float:
    """Numerically integrate the bivariate-Gaussian energy score."""
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
    """Return a squared Mahalanobis distance for one Gaussian forecast."""
    difference = np.asarray(difference, dtype=float)
    return float(difference @ np.linalg.solve(covariance, difference))


def _draw_gaussian_position_samples(
    means, covariances, *, sample_count, seed
) -> np.ndarray:
    """Draw independent marginal samples solely to render IMM covariance regions."""
    generator = np.random.default_rng(seed)
    samples = np.empty((sample_count, len(means), 2), dtype=float)
    for index, (mean, covariance) in enumerate(zip(means, covariances, strict=True)):
        samples[:, index] = generator.multivariate_normal(
            mean,
            covariance,
            size=sample_count,
        )
    return samples


def _summarize_predictions(predictions: pd.DataFrame) -> IMMRollingSummary:
    """Aggregate tables using the same definitions printed by Bayesian CTRV."""
    per_horizon = (
        predictions.groupby("horizon_step", sort=True, as_index=False)
        .agg(
            forecast_count=("position_error_m", "size"),
            mean_horizon_seconds=("horizon_seconds", "mean"),
            ade_m=("position_error_m", "mean"),
            median_error_m=("position_error_m", "median"),
            mean_energy_score_m=("energy_score_m", "mean"),
            radial_coverage=("radial_covered", "mean"),
            joint_coverage_50=("joint_covered_50", "mean"),
            joint_coverage_90=("joint_covered_90", "mean"),
            mean_prediction_radius_m=("prediction_radius_m", "mean"),
            mean_marginal_interval_width_m=(
                "mean_marginal_interval_width_m",
                "mean",
            ),
            mean_log_predictive_density=("log_predictive_density", "mean"),
        )
        .reset_index(drop=True)
    )
    runtimes = predictions.groupby("window_index", sort=True)[
        "window_runtime_seconds"
    ].first()
    return IMMRollingSummary(
        window_count=int(predictions["window_index"].nunique()),
        forecast_count=len(predictions),
        ade_m=float(predictions["position_error_m"].mean()),
        fde_m=float(per_horizon.iloc[-1]["ade_m"]),
        mean_energy_score_m=float(predictions["energy_score_m"].mean()),
        mean_window_runtime_seconds=float(runtimes.mean()),
        median_window_runtime_seconds=float(runtimes.median()),
        total_computation_time_seconds=float(runtimes.sum()),
        radial_coverage=float(predictions["radial_covered"].mean()),
        joint_coverage_50=float(predictions["joint_covered_50"].mean()),
        joint_coverage_90=float(predictions["joint_covered_90"].mean()),
        mean_prediction_radius_m=float(predictions["prediction_radius_m"].mean()),
        mean_marginal_interval_width_m=float(
            predictions["mean_marginal_interval_width_m"].mean()
        ),
        elpd=float(predictions["log_predictive_density"].sum()),
        mean_log_predictive_density=float(predictions["log_predictive_density"].mean()),
        per_horizon_table=per_horizon,
    )


def _print_configuration(
    options, *, data_file, window_count, initial_detection_id
) -> None:
    """Print data and model choices before starting the online tracker."""
    print("=" * 72)
    print("Python IMM BR24 Radar Rolling Evaluation")
    print("=" * 72)
    print(
        reporting.format_aligned_rows(
            (
                ("Data file", str(data_file)),
                ("Trajectory schema", "raw BR24 radar cycles with evaluation labels"),
                ("Model", "local Python CV/CTRV/CA IMM tracker"),
                ("Inference mode", "ONLINE"),
                ("Initial observations", str(options.observation_count)),
                ("Initial detection ID", str(initial_detection_id)),
                ("Prediction horizon", str(options.prediction_count)),
                ("Stride", str(options.stride)),
                ("Forecast origins", str(window_count)),
                ("Reference labels", "evaluation only"),
                ("Point forecast", "coordinate-wise posterior median"),
                ("Uncertainty", "IMM Gaussian position covariance"),
            )
        )
    )


def _print_summary(summary: IMMRollingSummary) -> None:
    """Print the same central quantities as the Bayesian rolling evaluator."""
    print("\n" + "=" * 72)
    print("Complete rolling evaluation")
    print("=" * 72)
    print(
        reporting.format_aligned_rows(
            (
                ("Forecast origins", str(summary.window_count)),
                ("Inference mode", "ONLINE"),
                ("Inference method", "IMM"),
                ("Forecasted positions", str(summary.forecast_count)),
                ("Overall ADE", f"{summary.ade_m:.2f} m"),
                ("Mean maximum-horizon FDE", f"{summary.fde_m:.2f} m"),
                ("Mean Energy Score", f"{summary.mean_energy_score_m:.2f} m"),
                ("ELPD", f"{summary.elpd:.3f}"),
                (
                    "Mean log predictive density",
                    f"{summary.mean_log_predictive_density:.3f}",
                ),
                ("Joint 2D 50% coverage", f"{summary.joint_coverage_50:.1%}"),
                ("Joint 2D 90% coverage", f"{summary.joint_coverage_90:.1%}"),
                ("Mean window runtime", f"{summary.mean_window_runtime_seconds:.3f} s"),
                (
                    "Median window runtime",
                    f"{summary.median_window_runtime_seconds:.3f} s",
                ),
                (
                    "Total computation time",
                    rolling.format_computation_time(
                        summary.total_computation_time_seconds
                    ),
                ),
            )
        )
    )
    print("\nPer-horizon evaluation:")
    print(rolling.format_per_horizon_table(summary.per_horizon_table))
