"""Rolling-origin window definitions and aggregate validation metrics."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

WindowMode = Literal["sliding", "expanding"]


@dataclass(frozen=True, slots=True)
class RollingWindowSpec:
    """Indices and sizes for one leakage-free rolling forecast window."""

    window_index: int
    forecast_start_index: int
    start_index: int
    observation_count: int
    prediction_count: int


@dataclass(frozen=True)
class RollingPositionSummary:
    """Aggregate position metrics over multiple rolling forecast windows."""

    inference_method: str
    window_count: int
    forecast_count: int
    ade_m: float
    fde_m: float
    mean_window_runtime_seconds: float
    median_window_runtime_seconds: float
    total_computation_time_seconds: float
    radial_coverage: float
    mean_prediction_radius_m: float
    mean_marginal_interval_width_m: float
    vi_convergence_rate: float | None
    mcmc_diagnostics_pass_rate: float | None
    per_horizon_table: pd.DataFrame


@dataclass(frozen=True, slots=True)
class WindowRuntimeSummary:
    """Unweighted computation-time metrics across rolling windows."""

    mean_seconds: float
    median_seconds: float
    total_seconds: float


def summarize_window_runtimes(prediction_table: pd.DataFrame) -> WindowRuntimeSummary:
    """Summarize one consistently repeated runtime value per rolling window."""
    required_columns = {"window_index", "window_runtime_seconds"}
    missing_columns = sorted(required_columns.difference(prediction_table.columns))
    if missing_columns:
        raise ValueError(f"Missing rolling runtime columns: {missing_columns}")
    if prediction_table.empty:
        raise ValueError("prediction_table must contain at least one forecast.")

    runtime_counts = prediction_table.groupby("window_index")[
        "window_runtime_seconds"
    ].nunique(dropna=False)
    if (runtime_counts != 1).any():
        raise ValueError(
            "window_runtime_seconds must be constant within each rolling window."
        )

    runtimes = (
        prediction_table.groupby("window_index", sort=True)["window_runtime_seconds"]
        .first()
        .to_numpy(dtype=float)
    )
    if not np.all(np.isfinite(runtimes)) or np.any(runtimes < 0):
        raise ValueError(
            "window_runtime_seconds must contain finite non-negative values."
        )
    return WindowRuntimeSummary(
        mean_seconds=float(np.mean(runtimes)),
        median_seconds=float(np.median(runtimes)),
        total_seconds=float(np.sum(runtimes)),
    )


def format_computation_time(seconds: float) -> str:
    """Format a computation duration in seconds or minutes."""
    if seconds >= 60:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.3f} s"


def format_per_horizon_table(table: pd.DataFrame) -> str:
    """Return a compact terminal representation of per-horizon metrics."""
    display_table = table.rename(
        columns={
            "horizon_step": "Step",
            "forecast_count": "Forecasts",
            "mean_horizon_seconds": "Horizon[s]",
            "ade_m": "ADE[m]",
            "median_error_m": "Median[m]",
            "radial_coverage": "Coverage",
            "mean_prediction_radius_m": "Radius[m]",
            "mean_marginal_interval_width_m": "Width[m]",
        }
    )
    return display_table.to_string(
        index=False,
        formatters={
            "Horizon[s]": lambda value: f"{value:.1f}",
            "ADE[m]": lambda value: f"{value:.3f}",
            "Median[m]": lambda value: f"{value:.3f}",
            "Coverage": lambda value: f"{value:.1%}",
            "Radius[m]": lambda value: f"{value:.3f}",
            "Width[m]": lambda value: f"{value:.3f}",
        },
    )


def build_rolling_window_specs(
    row_count: int,
    *,
    initial_observation_count: int,
    prediction_count: int,
    stride: int | None = None,
    window_mode: WindowMode = "sliding",
) -> tuple[RollingWindowSpec, ...]:
    """Return windows covering the complete future part of one trajectory.

    A sliding window retains exactly ``initial_observation_count`` observations.
    An expanding window starts at the first row and includes all observations
    available before the current forecast origin. The final prediction window is
    shortened when fewer than ``prediction_count`` rows remain.
    """
    row_count = _positive_integer(row_count, name="row_count")
    initial_observation_count = _positive_integer(
        initial_observation_count,
        name="initial_observation_count",
    )
    prediction_count = _positive_integer(
        prediction_count,
        name="prediction_count",
    )
    if stride is None:
        stride = prediction_count
    stride = _positive_integer(stride, name="stride")

    if window_mode not in {"sliding", "expanding"}:
        raise ValueError("window_mode must be 'sliding' or 'expanding'.")
    if initial_observation_count >= row_count:
        raise ValueError(
            "row_count must exceed initial_observation_count so that at least "
            "one future position can be predicted."
        )
    if stride > prediction_count:
        raise ValueError(
            "stride must be less than or equal to prediction_count to avoid "
            "gaps in trajectory coverage."
        )

    windows = []
    for window_index, forecast_start_index in enumerate(
        range(initial_observation_count, row_count, stride)
    ):
        current_prediction_count = min(
            prediction_count,
            row_count - forecast_start_index,
        )
        if window_mode == "sliding":
            start_index = forecast_start_index - initial_observation_count
            observation_count = initial_observation_count
        else:
            start_index = 0
            observation_count = forecast_start_index

        windows.append(
            RollingWindowSpec(
                window_index=window_index,
                forecast_start_index=forecast_start_index,
                start_index=start_index,
                observation_count=observation_count,
                prediction_count=current_prediction_count,
            )
        )

    return tuple(windows)


def summarize_rolling_predictions(
    prediction_table: pd.DataFrame,
) -> RollingPositionSummary:
    """Aggregate accuracy, uncertainty, and convergence over rolling windows."""
    required_columns = {
        "window_index",
        "horizon_step",
        "horizon_seconds",
        "position_error_m",
        "prediction_radius_m",
        "radial_covered",
        "mean_marginal_interval_width_m",
        "inference_method",
        "converged",
        "mcmc_diagnostics_ok",
        "window_runtime_seconds",
    }
    missing_columns = sorted(required_columns.difference(prediction_table.columns))
    if missing_columns:
        raise ValueError(f"Missing rolling prediction columns: {missing_columns}")
    if prediction_table.empty:
        raise ValueError("prediction_table must contain at least one forecast.")

    numeric_columns = (
        "horizon_seconds",
        "position_error_m",
        "prediction_radius_m",
        "mean_marginal_interval_width_m",
    )
    for column in numeric_columns:
        values = prediction_table[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{column} must contain only finite values.")
    if (prediction_table["horizon_seconds"] <= 0).any():
        raise ValueError("horizon_seconds must contain only positive values.")
    if (prediction_table["position_error_m"] < 0).any():
        raise ValueError("position_error_m must contain only non-negative values.")

    horizon_steps = prediction_table["horizon_step"].to_numpy()
    if not np.issubdtype(horizon_steps.dtype, np.integer) or (horizon_steps < 1).any():
        raise ValueError("horizon_step must contain positive integers.")

    per_horizon = (
        prediction_table.groupby("horizon_step", sort=True, as_index=False)
        .agg(
            forecast_count=("position_error_m", "size"),
            mean_horizon_seconds=("horizon_seconds", "mean"),
            ade_m=("position_error_m", "mean"),
            median_error_m=("position_error_m", "median"),
            radial_coverage=("radial_covered", "mean"),
            mean_prediction_radius_m=("prediction_radius_m", "mean"),
            mean_marginal_interval_width_m=(
                "mean_marginal_interval_width_m",
                "mean",
            ),
        )
        .reset_index(drop=True)
    )
    inference_methods = prediction_table["inference_method"].dropna().unique()
    if len(inference_methods) != 1 or inference_methods[0] not in {
        "vi",
        "mcmc",
        "sequential",
    }:
        raise ValueError("inference_method must contain exactly one supported value.")
    inference_method = str(inference_methods[0])
    runtime_summary = summarize_window_runtimes(prediction_table)
    diagnostics_by_window = prediction_table.groupby("window_index")[
        [
            "converged",
            "mcmc_diagnostics_ok",
        ]
    ].first()
    if inference_method == "vi":
        if diagnostics_by_window["converged"].isna().any():
            raise ValueError("VI predictions require convergence values.")
        vi_convergence_rate = float(
            diagnostics_by_window["converged"].astype(bool).mean()
        )
        mcmc_diagnostics_pass_rate = None
    elif inference_method == "mcmc":
        if diagnostics_by_window["mcmc_diagnostics_ok"].isna().any():
            raise ValueError("MCMC predictions require sampler diagnostic values.")
        vi_convergence_rate = None
        mcmc_diagnostics_pass_rate = float(
            diagnostics_by_window["mcmc_diagnostics_ok"].astype(bool).mean()
        )
    else:
        vi_convergence_rate = None
        mcmc_diagnostics_pass_rate = None

    return RollingPositionSummary(
        inference_method=inference_method,
        window_count=int(prediction_table["window_index"].nunique()),
        forecast_count=len(prediction_table),
        ade_m=float(prediction_table["position_error_m"].mean()),
        fde_m=float(per_horizon.iloc[-1]["ade_m"]),
        mean_window_runtime_seconds=runtime_summary.mean_seconds,
        median_window_runtime_seconds=runtime_summary.median_seconds,
        total_computation_time_seconds=runtime_summary.total_seconds,
        radial_coverage=float(prediction_table["radial_covered"].mean()),
        mean_prediction_radius_m=float(prediction_table["prediction_radius_m"].mean()),
        mean_marginal_interval_width_m=float(
            prediction_table["mean_marginal_interval_width_m"].mean()
        ),
        vi_convergence_rate=vi_convergence_rate,
        mcmc_diagnostics_pass_rate=mcmc_diagnostics_pass_rate,
        per_horizon_table=per_horizon,
    )


def _positive_integer(value: int, *, name: str) -> int:
    """Return one positive integer or raise a clear error."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)
