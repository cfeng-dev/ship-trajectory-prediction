"""Evaluate deterministic CTRV forecasts across rolling windows."""

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ship_trajectory_prediction.coordinates import (
    gps_to_local_coordinates,
    local_to_gps_coordinates,
)
from ship_trajectory_prediction.evaluation import build_rolling_window_specs
from ship_trajectory_prediction.evaluation.deterministic_ctrv import (
    build_prediction_table,
    estimate_ctrv_state,
)
from ship_trajectory_prediction.evaluation.plotting import (
    plot_deterministic_rolling_predictions,
)
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import (
    prepare_trajectory_window,
    read_ship_data,
)

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)
RUN_ID = 1
WINDOW_MODE = "sliding"
OBSERVATION_COUNT = 20
PREDICTION_COUNT = 3
STRIDE = None
POSITION_NOISE_STD_M = 2.0
POSITION_NOISE_SEED = 2026
MAX_WINDOWS = None
SPEED_ESTIMATION_POINTS = 5
HEADING_ESTIMATION_SEGMENTS = 5


@dataclass(frozen=True, slots=True)
class DeterministicRollingSummary:
    """Aggregate deterministic position errors over rolling windows."""

    window_count: int
    forecast_count: int
    ade_m: float
    fde_m: float
    per_horizon_table: pd.DataFrame


def main(
    *,
    window_mode=WINDOW_MODE,
    observation_count=OBSERVATION_COUNT,
    prediction_count=PREDICTION_COUNT,
    stride=STRIDE,
    speed_estimation_points=SPEED_ESTIMATION_POINTS,
    heading_estimation_segments=HEADING_ESTIMATION_SEGMENTS,
    position_noise_std_m=POSITION_NOISE_STD_M,
    position_noise_seed=POSITION_NOISE_SEED,
    max_windows=MAX_WINDOWS,
    show_plot=True,
):
    """Estimate and evaluate deterministic CTRV in every rolling window."""
    trajectory_data = read_ship_data(DATA_FILE, run_id=RUN_ID)
    trajectory_data = trajectory_data.sort_values("time").reset_index(drop=True)
    if trajectory_data.empty:
        raise ValueError(f"No trajectory rows found for run_id={RUN_ID}.")

    windows = build_rolling_window_specs(
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
        window = prepare_trajectory_window(
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
        initial_state = estimate_ctrv_state(
            estimation_window,
            speed_estimation_points=speed_estimation_points,
            heading_estimation_segments=heading_estimation_segments,
        )
        local_table = build_prediction_table(estimation_window, initial_state)
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
        prediction_tables.append(table)
        print(
            f"Window {number}/{len(windows)}: "
            f"ADE={table['position_error_m'].mean():.2f} m, "
            f"FDE={table['position_error_m'].iloc[-1]:.2f} m, "
            f"heading={initial_state.heading:+.4f} rad, "
            f"turn rate={initial_state.turn_rate:+.5f} rad/s"
        )

    predictions = pd.concat(prediction_tables, ignore_index=True)
    summary = summarize_deterministic_predictions(predictions)
    _print_summary(summary)
    if show_plot:
        plot_deterministic_rolling_predictions(
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
    return DeterministicRollingSummary(
        window_count=int(predictions["window_index"].nunique()),
        forecast_count=len(predictions),
        ade_m=float(np.mean(error)),
        fde_m=float(per_horizon.iloc[-1]["ade_m"]),
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
    route_x, route_y = gps_to_local_coordinates(longitude, latitude, unit="m")
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
    return replace(window, x_meters=x_meters, y_meters=y_meters)


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
    predicted_longitude, predicted_latitude = local_to_gps_coordinates(
        table["x_predicted"],
        table["y_predicted"],
        reference_longitude=window.reference_longitude,
        reference_latitude=window.reference_latitude,
        unit="m",
    )
    x_predicted_route, y_predicted_route = gps_to_local_coordinates(
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
    print(f"Data file             : {DATA_FILE}")
    print(f"Run ID                : {RUN_ID}")
    print(f"Window mode           : {values['window_mode']}")
    print(f"Initial observations  : {values['observation_count']}")
    print(f"Prediction horizon    : {values['prediction_count']}")
    print(f"Stride                : {values['stride']}")
    print(f"Rolling windows       : {values['window_count']}")
    print(f"Speed history points  : {values['speed_estimation_points']}")
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
    print(f"Evaluated windows        : {summary.window_count}")
    print(f"Forecasted positions     : {summary.forecast_count}")
    print(f"Overall ADE              : {summary.ade_m:.2f} m")
    print(f"Mean maximum-horizon FDE : {summary.fde_m:.2f} m")
    print("\nPer-horizon evaluation:")
    print(summary.per_horizon_table.round(3).to_string(index=False))


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-mode",
        choices=("sliding", "expanding"),
        default=WINDOW_MODE,
    )
    parser.add_argument("--observations", type=int, default=OBSERVATION_COUNT)
    parser.add_argument("--predictions", type=int, default=PREDICTION_COUNT)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument(
        "--speed-estimation-points",
        type=int,
        default=SPEED_ESTIMATION_POINTS,
    )
    parser.add_argument(
        "--heading-estimation-segments",
        type=int,
        default=HEADING_ESTIMATION_SEGMENTS,
    )
    parser.add_argument(
        "--position-noise-std-m",
        type=float,
        default=POSITION_NOISE_STD_M,
    )
    parser.add_argument(
        "--position-noise-seed",
        type=int,
        default=POSITION_NOISE_SEED,
    )
    parser.add_argument("--max-windows", type=int, default=MAX_WINDOWS)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Print metrics without opening the route-wide plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(
        window_mode=arguments.window_mode,
        observation_count=arguments.observations,
        prediction_count=arguments.predictions,
        stride=arguments.stride,
        speed_estimation_points=arguments.speed_estimation_points,
        heading_estimation_segments=arguments.heading_estimation_segments,
        position_noise_std_m=arguments.position_noise_std_m,
        position_noise_seed=arguments.position_noise_seed,
        max_windows=arguments.max_windows,
        show_plot=not arguments.no_plot,
    )
