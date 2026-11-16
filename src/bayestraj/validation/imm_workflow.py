"""Rolling BR24 IMM evaluation with mean point forecasts and Gaussian scores."""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import bayestraj.inference.imm_adapter as imm_adapter
import bayestraj.observations.br24 as br24
import bayestraj.validation.cli as cli
import bayestraj.validation.metrics as metrics
import bayestraj.validation.reporting as reporting
import bayestraj.validation.rolling as rolling
import bayestraj.validation.runtime_plotting as runtime_plotting
from bayestraj.validation.cli import IMMEvaluationOptions

# The IMM forecast covariance is the complete predictive position uncertainty.
# This small numerical floor only makes the Gaussian log density well defined.
RESIDUAL_POSITION_NOISE_FLOOR_M = 1e-6


@dataclass(frozen=True, slots=True)
class IMMRollingSummary:
    """Aggregate forecast metrics with the Bayesian rolling-report semantics."""

    window_count: int
    forecast_count: int
    ade_m: float
    fde_m: float
    mean_energy_score_m: float
    mean_fit_runtime_seconds: float
    median_fit_runtime_seconds: float
    total_computation_time_seconds: float
    radial_coverage: float
    joint_coverage_50: float
    joint_coverage_90: float
    mean_prediction_radius_m: float
    mean_marginal_interval_width_m: float
    total_negative_log_predictive_density: float
    mean_negative_log_predictive_density: float
    per_horizon_table: pd.DataFrame


def run_evaluation(
    *,
    data_file: str | Path,
    imm_python: str | Path,
    output_csv: str | Path,
    prediction_count: int,
    max_windows: int | None,
    show_rolling_diagnostics_plot: bool,
    description: str | None = None,
    argv=None,
) -> tuple[pd.DataFrame, IMMRollingSummary]:
    """Parse script settings, evaluate IMM, export metrics, and show diagnostics."""
    options = cli.parse_imm_evaluation_arguments(
        description=description,
        prediction_count=prediction_count,
        max_windows=max_windows,
        argv=argv,
    )
    predictions, summary = run_imm_evaluation(
        data_file=data_file,
        imm_python=imm_python,
        options=options,
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"\nSaved per-forecast IMM metrics: {output_path}")
    if options.show_plot and show_rolling_diagnostics_plot:
        runtime_plotting.plot_imm_rolling_diagnostics(predictions)
    return predictions, summary


def run_imm_evaluation(
    *,
    data_file: str | Path,
    imm_python: str | Path,
    options: IMMEvaluationOptions,
    credible_interval: float = 0.9,
    forecast_seed: int = 42,
) -> tuple[pd.DataFrame, IMMRollingSummary]:
    """Evaluate raw-radar tracking or known Cartesian associations.

    Scoring uses 90-percent intervals by default. Keep the historical forecast
    seed sequence independently of visualization in the predictor tools.
    """
    data_file = Path(data_file)
    imm_python = Path(imm_python)
    tracker_type = _load_optional_imm_tracker(imm_python)
    associated_input = (
        data_file.is_file()
        and not br24.is_br24_tracking_input(data_file)
        and br24.is_br24_associated_input(data_file)
    )
    cycles = (
        br24.read_br24_associated_input(data_file)
        if associated_input
        else br24.read_br24_tracking_input(data_file)
    )
    reference = br24.read_br24_tracking_reference(data_file)
    if reference is None:
        raise ValueError("IMM evaluation requires ground_truth_* columns.")
    _validate_reference_alignment(cycles, reference)
    if not associated_input and not cycles[0].detections:
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

    if associated_input:
        first = cycles[0]
        initial_detection_id = first.detection_id
        tracker = tracker_type.initialize_position(
            first.time_seconds,
            first.position,
            first.measurement_covariance,
        )
    else:
        initial_detection_id = cycles[0].detections[0].detection_id
        tracker = tracker_type.initialize(
            cycles[0],
            initial_detection_id=initial_detection_id,
        )
    cycle_times = np.asarray([cycle.time_seconds for cycle in cycles], dtype=float)
    processed_cycle_count = 1
    prediction_tables = []

    _print_configuration(
        options,
        data_file=data_file,
        window_count=len(windows),
        initial_detection_id=initial_detection_id,
        associated_input=associated_input,
    )
    for number, specification in enumerate(windows, start=1):
        fit_runtime_started = time.perf_counter()
        while processed_cycle_count < specification.forecast_start_index:
            cycle = cycles[processed_cycle_count]
            if associated_input:
                tracker.update_position(
                    cycle.time_seconds,
                    cycle.position,
                    cycle.measurement_covariance,
                )
            else:
                tracker.process_cycle(cycle)
            processed_cycle_count += 1

        forecast_stop_index = (
            specification.forecast_start_index + specification.prediction_count
        )
        future_times = cycle_times[
            specification.forecast_start_index : forecast_stop_index
        ]
        forecast = tracker.forecast(
            future_times,
            seed=forecast_seed + specification.window_index,
        )
        forecast_runtime_seconds = time.perf_counter() - fit_runtime_started
        # Retain exact medians as separate CSV data; point errors use the mean.
        forecast_medians = imm_adapter.forecast_position_medians(tracker, forecast)

        # Evaluation labels are read only after the raw-radar tracker forecast.
        table, _ = _evaluate_forecast(
            forecast,
            credible_interval=credible_interval,
            forecast_median_positions=forecast_medians,
            reference=reference,
            specification=specification,
            forecast_origin_time=tracker.last_observation_time_seconds,
            forecast_origin_position=tracker.estimated_position,
            forecast_runtime_seconds=forecast_runtime_seconds,
        )
        prediction_tables.append(table)
        print(
            f"Forecast {number}/{len(windows)}: "
            f"ADE={table['position_error_m'].mean():.2f} m, "
            f"FDE={table['position_error_m'].iloc[-1]:.2f} m, "
            f"runtime={forecast_runtime_seconds:.3f} s"
        )

    predictions = pd.concat(prediction_tables, ignore_index=True)
    summary = _summarize_predictions(predictions)
    _print_summary(summary)
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
    forecast_runtime_seconds,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Score fused mean positions and export separately labelled mixture medians."""
    prediction_count = specification.prediction_count
    expected_shape = (prediction_count, 2)
    means = np.asarray(forecast.position_means, dtype=float)
    medians = np.asarray(forecast_median_positions, dtype=float)
    covariances = np.asarray(forecast.position_covariances, dtype=float)
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
    errors = np.linalg.norm(means - actual_positions, axis=1)
    scores = metrics.gaussian_position_metrics(
        actual_positions, means, covariances, credible_interval=credible_interval
    )
    covariances = metrics.regularize_position_covariances(covariances)
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
            "x_mean": means[:, 0],
            "y_mean": means[:, 1],
            "point_estimate": "mean",
            "position_error_m": errors,
            **scores,
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
            "x_mean_route": means[:, 0],
            "y_mean_route": means[:, 1],
            "cov_xx": covariances[:, 0, 0],
            "cov_xy": covariances[:, 0, 1],
            "cov_yy": covariances[:, 1, 1],
            "mode_cv": modes[:, 0],
            "mode_ctrv": modes[:, 1],
            "mode_ca": modes[:, 2],
            "fit_runtime_seconds": forecast_runtime_seconds,
        }
    )
    return table, covariances


def _regularize_covariances(covariances) -> np.ndarray:
    """Use the same covariance floors as scoring and Gaussian ellipse plots."""
    return metrics.regularize_position_covariances(covariances)


def _gaussian_log_density(actual, mean, covariance) -> float:
    """Use the shared latent-position Gaussian score and numerical floor."""
    return float(
        metrics.gaussian_position_log_predictive_density(
            np.asarray(actual)[None, :],
            np.asarray(mean)[None, :],
            np.asarray(covariance)[None, :, :],
        )[0]
    )


def _gaussian_energy_score(actual, mean, covariance) -> float:
    """Use the common Gaussian Energy Score quadrature."""
    return float(
        metrics.gaussian_position_metrics(
            np.asarray(actual)[None, :],
            np.asarray(mean)[None, :],
            np.asarray(covariance)[None, :, :],
        )["energy_score_m"][0]
    )


def _squared_mahalanobis(difference, covariance) -> float:
    """Return a squared Mahalanobis distance for one Gaussian forecast."""
    difference = np.asarray(difference, dtype=float)
    return float(difference @ np.linalg.solve(covariance, difference))


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
            mean_negative_log_predictive_density=(
                "negative_log_predictive_density",
                "mean",
            ),
        )
        .reset_index(drop=True)
    )
    runtime_summary = rolling.summarize_window_runtimes(predictions)
    return IMMRollingSummary(
        window_count=int(predictions["window_index"].nunique()),
        forecast_count=len(predictions),
        ade_m=float(predictions["position_error_m"].mean()),
        fde_m=float(per_horizon.iloc[-1]["ade_m"]),
        mean_energy_score_m=float(predictions["energy_score_m"].mean()),
        mean_fit_runtime_seconds=runtime_summary.mean_seconds,
        median_fit_runtime_seconds=runtime_summary.median_seconds,
        total_computation_time_seconds=runtime_summary.total_seconds,
        radial_coverage=float(predictions["radial_covered"].mean()),
        joint_coverage_50=float(predictions["joint_covered_50"].mean()),
        joint_coverage_90=float(predictions["joint_covered_90"].mean()),
        mean_prediction_radius_m=float(predictions["prediction_radius_m"].mean()),
        mean_marginal_interval_width_m=float(
            predictions["mean_marginal_interval_width_m"].mean()
        ),
        total_negative_log_predictive_density=float(
            predictions["negative_log_predictive_density"].sum()
        ),
        mean_negative_log_predictive_density=float(
            predictions["negative_log_predictive_density"].mean()
        ),
        per_horizon_table=per_horizon,
    )


def _print_configuration(
    options, *, data_file, window_count, initial_detection_id, associated_input=False
) -> None:
    """Print data and model choices before starting the online tracker."""
    print("=" * 72)
    print("Python IMM BR24 Radar Rolling Evaluation")
    print("=" * 72)
    print(
        reporting.format_aligned_rows(
            (
                ("Data file", str(data_file)),
                (
                    "Trajectory schema",
                    "associated BR24 Cartesian fixes with evaluation labels"
                    if associated_input
                    else "raw BR24 radar cycles with evaluation labels",
                ),
                ("Model", "local Python CV/CTRV/CA IMM tracker"),
                ("Inference mode", "ONLINE"),
                ("Initial observations", str(options.observation_count)),
                ("Initial detection ID", str(initial_detection_id)),
                ("Prediction horizon", str(options.prediction_count)),
                ("Stride", str(options.stride)),
                ("Forecast origins", str(window_count)),
                ("Reference labels", "evaluation only"),
                ("Point forecast", "fused IMM mean"),
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
                (
                    "Total negative log predictive density",
                    f"{summary.total_negative_log_predictive_density:.3f}",
                ),
                (
                    "Mean negative log predictive density",
                    f"{summary.mean_negative_log_predictive_density:.3f}",
                ),
                ("Joint 2D 50% coverage", f"{summary.joint_coverage_50:.1%}"),
                ("Joint 2D 90% coverage", f"{summary.joint_coverage_90:.1%}"),
                (
                    "Mean fit/forecast runtime",
                    f"{summary.mean_fit_runtime_seconds:.3f} s",
                ),
                (
                    "Median fit/forecast runtime",
                    f"{summary.median_fit_runtime_seconds:.3f} s",
                ),
                (
                    "Total fit/forecast time",
                    rolling.format_computation_time(
                        summary.total_computation_time_seconds
                    ),
                ),
            )
        )
    )
    print("\nPer-horizon evaluation:")
    print(rolling.format_per_horizon_table(summary.per_horizon_table))
