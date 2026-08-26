"""Rolling evaluation workflow for deterministic CTRV forecasts."""

import dataclasses
import time

import numpy as np
import pandas as pd

import bayestraj.forecasting.deterministic_ctrv as forecasting
import bayestraj.observations.coordinates as coordinates
import bayestraj.observations.io as observations_io
import bayestraj.observations.window as observation_window
import bayestraj.validation.cli as validation_cli
import bayestraj.validation.plotting as plotting
import bayestraj.validation.rolling as rolling_validation


@dataclasses.dataclass(frozen=True, slots=True)
class DeterministicRollingSummary:
    """Aggregate deterministic position errors over rolling windows."""

    window_count: int
    forecast_count: int
    ade_m: float
    fde_m: float
    mean_window_runtime_seconds: float
    median_window_runtime_seconds: float
    total_computation_time_seconds: float
    per_horizon_table: pd.DataFrame


def run_deterministic_ctrv_evaluation(
    *,
    data_file,
    experiment: forecasting.DeterministicRollingExperimentConfig,
    options: validation_cli.DeterministicCTRVEvaluationOptions,
):
    """Apply runtime options and evaluate deterministic rolling forecasts."""
    configured_experiment = dataclasses.replace(
        experiment,
        window_mode=options.window_mode,
        observation_count=options.observation_count,
        prediction_count=options.prediction_count,
        stride=options.stride,
        additional_position_noise_std_m=options.additional_position_noise_std_m,
        position_noise_seed=options.position_noise_seed,
    )
    return _run_deterministic_ctrv_evaluation(
        data_file=data_file,
        experiment=configured_experiment,
        speed_estimation_points=options.speed_estimation_points,
        heading_estimation_segments=options.heading_estimation_segments,
        max_windows=options.max_windows,
        show_plot=options.show_plot,
    )


def _run_deterministic_ctrv_evaluation(
    *,
    data_file,
    experiment,
    speed_estimation_points,
    heading_estimation_segments,
    max_windows,
    show_plot,
):
    """Estimate and evaluate deterministic CTRV in every rolling window."""
    window_mode = experiment.window_mode
    observation_count = experiment.observation_count
    prediction_count = experiment.prediction_count
    stride = experiment.stride
    position_noise_std_m = experiment.additional_position_noise_std_m
    position_noise_seed = experiment.position_noise_seed
    trajectory_data = observations_io.read_ship_data(
        data_file,
        run_id=experiment.run_id,
    )
    trajectory_data = trajectory_data.sort_values("time").reset_index(drop=True)
    if trajectory_data.empty:
        raise ValueError(f"No trajectory rows found for run_id={experiment.run_id}.")

    windows = rolling_validation.build_rolling_window_specs(
        len(trajectory_data),
        initial_observation_count=observation_count,
        prediction_count=prediction_count,
        stride=stride,
        window_mode=window_mode,
    )
    if max_windows is not None:
        if isinstance(max_windows, bool) or not isinstance(max_windows, int):
            raise ValueError("max_windows must be a positive integer or None.")
        if max_windows < 1:
            raise ValueError("max_windows must be a positive integer or None.")
        windows = windows[:max_windows]

    route_x, route_y, longitude, latitude = _prepare_route_coordinates(trajectory_data)
    route_noise_x, route_noise_y = _simulate_route_position_noise(
        len(trajectory_data),
        standard_deviation_m=position_noise_std_m,
        seed=position_noise_seed,
    )
    effective_stride = prediction_count if stride is None else stride
    _print_setup(
        data_file=data_file,
        run_id=experiment.run_id,
        window_mode=window_mode,
        observation_count=observation_count,
        prediction_count=prediction_count,
        stride=effective_stride,
        window_count=len(windows),
        speed_estimation_points=speed_estimation_points,
        heading_estimation_segments=heading_estimation_segments,
        position_noise_std_m=position_noise_std_m,
        position_noise_seed=position_noise_seed,
    )

    prediction_tables = []
    for number, specification in enumerate(windows, start=1):
        window = observation_window.prepare_trajectory_window(
            trajectory_data,
            observation_count=specification.observation_count,
            prediction_count=specification.prediction_count,
            start_index=specification.start_index,
        )
        estimation_window = _add_observation_noise(
            window,
            route_start_index=specification.start_index,
            route_noise_x=route_noise_x,
            route_noise_y=route_noise_y,
        )
        runtime_started = time.perf_counter()
        initial_state = forecasting.estimate_ctrv_state(
            estimation_window,
            speed_estimation_points=speed_estimation_points,
            heading_estimation_segments=heading_estimation_segments,
        )
        local_table = forecasting.build_prediction_table(
            estimation_window,
            initial_state,
        )
        table = _build_route_prediction_table(
            local_table,
            specification=specification,
            window=window,
            initial_state=initial_state,
            route_x=route_x,
            route_y=route_y,
            longitude=longitude,
            latitude=latitude,
            position_noise_std_m=position_noise_std_m,
            position_noise_seed=position_noise_seed,
        )
        window_runtime_seconds = time.perf_counter() - runtime_started
        table["window_runtime_seconds"] = window_runtime_seconds
        prediction_tables.append(table)
        print(
            f"Window {number}/{len(windows)}: "
            f"ADE={table['position_error_m'].mean():.2f} m, "
            f"FDE={table['position_error_m'].iloc[-1]:.2f} m, "
            f"runtime={window_runtime_seconds:.3f} s, "
            f"heading={initial_state.heading:+.4f} rad, "
            f"turn rate={initial_state.turn_rate:+.5f} rad/s"
        )

    predictions = pd.concat(prediction_tables, ignore_index=True)
    summary = summarize_deterministic_predictions(predictions)
    _print_summary(summary)
    if show_plot:
        plotting.plot_deterministic_rolling_predictions(
            route_x,
            route_y,
            predictions,
            initial_observation_count=observation_count,
            window_mode=window_mode,
            observed_route_x=route_x + route_noise_x,
            observed_route_y=route_y + route_noise_y,
            position_noise_std_m=position_noise_std_m,
        )
    return predictions, summary


def summarize_deterministic_predictions(predictions):
    """Aggregate deterministic rolling errors overall and by horizon."""
    required_columns = {
        "window_index",
        "horizon_step",
        "horizon_seconds",
        "position_error_m",
        "window_runtime_seconds",
    }
    missing = sorted(required_columns.difference(predictions.columns))
    if missing:
        raise ValueError(f"Missing deterministic rolling columns: {missing}")
    if predictions.empty:
        raise ValueError("predictions must contain at least one forecast.")

    error = predictions["position_error_m"].to_numpy(dtype=float)
    horizon = predictions["horizon_seconds"].to_numpy(dtype=float)
    if not np.all(np.isfinite(error)) or np.any(error < 0):
        raise ValueError("position_error_m must contain finite non-negative values.")
    if not np.all(np.isfinite(horizon)) or np.any(horizon <= 0):
        raise ValueError("horizon_seconds must contain finite positive values.")

    per_horizon = (
        predictions.groupby("horizon_step", sort=True, as_index=False)
        .agg(
            forecast_count=("position_error_m", "size"),
            mean_horizon_seconds=("horizon_seconds", "mean"),
            ade_m=("position_error_m", "mean"),
            median_error_m=("position_error_m", "median"),
        )
        .reset_index(drop=True)
    )
    runtime_summary = rolling_validation.summarize_window_runtimes(predictions)
    return DeterministicRollingSummary(
        window_count=int(predictions["window_index"].nunique()),
        forecast_count=len(predictions),
        ade_m=float(np.mean(error)),
        fde_m=float(per_horizon.iloc[-1]["ade_m"]),
        mean_window_runtime_seconds=runtime_summary.mean_seconds,
        median_window_runtime_seconds=runtime_summary.median_seconds,
        total_computation_time_seconds=runtime_summary.total_seconds,
        per_horizon_table=per_horizon,
    )


def _prepare_route_coordinates(trajectory_data):
    """Return route coordinates and GPS values in chronological order."""
    longitude = pd.to_numeric(
        trajectory_data["gps_longitude"], errors="coerce"
    ).to_numpy(dtype=float)
    latitude = pd.to_numeric(trajectory_data["gps_latitude"], errors="coerce").to_numpy(
        dtype=float
    )
    route_x, route_y = coordinates.gps_to_local_coordinates(
        longitude,
        latitude,
        unit="m",
    )
    return route_x, route_y, longitude, latitude


def _simulate_route_position_noise(position_count, *, standard_deviation_m, seed):
    """Generate one reproducible position perturbation for the complete route."""
    if (
        isinstance(standard_deviation_m, bool)
        or not np.isfinite(standard_deviation_m)
        or standard_deviation_m < 0
    ):
        raise ValueError("position_noise_std_m must be finite and non-negative.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("position_noise_seed must be a non-negative integer.")
    if isinstance(position_count, bool) or not isinstance(
        position_count, (int, np.integer)
    ):
        raise ValueError("position_count must be a positive integer.")
    if position_count < 1:
        raise ValueError("position_count must be a positive integer.")
    if standard_deviation_m == 0:
        zeros = np.zeros(int(position_count), dtype=float)
        return zeros, zeros.copy()
    noise = np.random.default_rng(int(seed)).normal(
        0.0,
        float(standard_deviation_m),
        size=(int(position_count), 2),
    )
    return noise[:, 0], noise[:, 1]


def _add_observation_noise(
    window,
    *,
    route_start_index,
    route_noise_x,
    route_noise_y,
):
    """Return a window whose observed positions use the route-wide noise."""
    route_stop_index = route_start_index + window.observation_count
    route_slice = slice(route_start_index, route_stop_index)
    if route_start_index < 0 or route_stop_index > len(route_noise_x):
        raise ValueError("Route position noise does not cover the rolling window.")
    if np.shape(route_noise_x) != np.shape(route_noise_y):
        raise ValueError("Route x/y position noise must have matching shapes.")

    x_meters = np.asarray(window.x_meters, dtype=float).copy()
    y_meters = np.asarray(window.y_meters, dtype=float).copy()
    x_meters[window.observed_slice] += route_noise_x[route_slice]
    y_meters[window.observed_slice] += route_noise_y[route_slice]
    return dataclasses.replace(window, x_meters=x_meters, y_meters=y_meters)


def _build_route_prediction_table(
    local_table,
    *,
    specification,
    window,
    initial_state,
    route_x,
    route_y,
    longitude,
    latitude,
    position_noise_std_m,
    position_noise_seed,
):
    """Attach rolling metadata and convert predictions to one route frame."""
    table = local_table.copy()
    predicted_longitude, predicted_latitude = coordinates.local_to_gps_coordinates(
        table["x_predicted"],
        table["y_predicted"],
        reference_longitude=window.reference_longitude,
        reference_latitude=window.reference_latitude,
        unit="m",
    )
    x_predicted_route, y_predicted_route = coordinates.gps_to_local_coordinates(
        np.concatenate(([longitude[0]], predicted_longitude)),
        np.concatenate(([latitude[0]], predicted_latitude)),
        unit="m",
    )
    x_predicted_route = x_predicted_route[1:]
    y_predicted_route = y_predicted_route[1:]
    target_indices = np.arange(
        specification.forecast_start_index,
        specification.forecast_start_index + specification.prediction_count,
    )
    forecast_origin_index = specification.forecast_start_index - 1

    table.insert(0, "window_index", specification.window_index)
    table.insert(1, "window_start_index", specification.start_index)
    table.insert(2, "forecast_start_index", specification.forecast_start_index)
    table.insert(3, "target_index", target_indices)
    table.insert(4, "horizon_step", np.arange(1, len(table) + 1))
    table["model_variant"] = "deterministic"
    table["observation_count"] = specification.observation_count
    table["prediction_count"] = specification.prediction_count
    table["additional_position_noise_std_m"] = position_noise_std_m
    table["position_noise_seed"] = position_noise_seed
    table["estimated_speed_mps"] = initial_state.speed
    table["estimated_heading_rad"] = initial_state.heading
    table["estimated_turn_rate_rad_s"] = initial_state.turn_rate
    table["forecast_origin_x_route"] = route_x[forecast_origin_index]
    table["forecast_origin_y_route"] = route_y[forecast_origin_index]
    table["x_actual_route"] = route_x[target_indices]
    table["y_actual_route"] = route_y[target_indices]
    table["x_predicted_route"] = x_predicted_route
    table["y_predicted_route"] = y_predicted_route
    table["position_error_m"] = np.hypot(
        x_predicted_route - route_x[target_indices],
        y_predicted_route - route_y[target_indices],
    )
    return table


def _print_setup(**values):
    """Print the deterministic rolling configuration."""
    print("=" * 72)
    print("Deterministic CTRV Rolling-Window Evaluation")
    print("=" * 72)
    print(f"Data file             : {values['data_file']}")
    print(f"Run ID                : {values['run_id']}")
    print(f"Window mode           : {values['window_mode']}")
    print(f"Initial observations  : {values['observation_count']}")
    print(f"Prediction horizon    : {values['prediction_count']}")
    print(f"Stride                : {values['stride']}")
    print(f"Rolling windows       : {values['window_count']}")
    print(f"Speed fit positions   : {values['speed_estimation_points']}")
    print(f"Heading segments      : {values['heading_estimation_segments']}")
    noise = (
        f"{values['position_noise_std_m']:g} m (seed={values['position_noise_seed']})"
        if values["position_noise_std_m"] > 0
        else "disabled"
    )
    print(f"Additional pos. noise : {noise}")


def _print_summary(summary):
    """Print aggregate deterministic rolling metrics."""
    print("\n" + "=" * 72)
    print("Complete deterministic rolling evaluation")
    print("=" * 72)
    rows = [
        ("Evaluated windows", str(summary.window_count)),
        ("Forecasted positions", str(summary.forecast_count)),
        ("Overall ADE", f"{summary.ade_m:.2f} m"),
        ("Mean maximum-horizon FDE", f"{summary.fde_m:.2f} m"),
        (
            "Mean window runtime",
            f"{summary.mean_window_runtime_seconds:.3f} s",
        ),
        (
            "Median window runtime",
            f"{summary.median_window_runtime_seconds:.3f} s",
        ),
        (
            "Total computation time",
            rolling_validation.format_computation_time(
                summary.total_computation_time_seconds
            ),
        ),
    ]
    label_width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{label_width}} : {value}")
    print("\nPer-horizon evaluation:")
    print(rolling_validation.format_per_horizon_table(summary.per_horizon_table))
