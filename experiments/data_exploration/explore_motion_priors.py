"""Explore empirical motion distributions for Bayesian CTRV prior design."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ship_trajectory_prediction.coordinates import gps_to_local_coordinates
from ship_trajectory_prediction.models.ctrv import CTRVState, ctrv_step
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import read_ship_data
from ship_trajectory_prediction.trajectory.window import (
    KILOMETERS_PER_HOUR_TO_METERS_PER_SECOND,
)

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

# Data selection and preprocessing
# Keep calibration runs separate from the final evaluation. ``None`` selects
# every run and should therefore only be used for a descriptive overview.
CALIBRATION_RUN_IDS = None
MIN_COURSE_DISPLACEMENT_METERS = 1.0
MAX_TIME_GAP_SECONDS = 15.0

# Terminal report
PRINT_PER_RUN_SUMMARY = False
REPORT_LABEL_WIDTH = 23

# Histogram data display
PLOT_CENTRAL_QUANTILE = 0.995
HISTOGRAM_MODE = "density"  # Choose "density" or "frequency".

# Plot typography
PLOT_TITLE_PAD = 16
PLOT_TITLE_FONT_SIZE = 13
PLOT_TITLE_FONT_WEIGHT = "bold"
AXIS_LABEL_FONT_SIZE = 13
AXIS_TICK_FONT_SIZE = 11

# Histogram appearance
HISTOGRAM_COLOR = "#4C78A8"
HISTOGRAM_ALPHA = 0.75
HISTOGRAM_EDGE_COLOR = "#2F3E46"
HISTOGRAM_EDGE_LINE_WIDTH = 0.5


@dataclass(frozen=True, slots=True)
class MotionPriorSamples:
    """Empirical SI-unit samples derived independently within each run."""

    speed_mps: np.ndarray
    turn_rate_rad_s: np.ndarray
    turn_rate_innovation: np.ndarray
    position_innovation: np.ndarray
    per_run_summary: pd.DataFrame


@dataclass(frozen=True, slots=True)
class PriorSuggestions:
    """Robust descriptive scales that can inform, but do not set, priors."""

    speed_median_mps: float
    speed_robust_scale_mps: float
    turn_rate_center_rad_s: float
    turn_rate_robust_scale_rad_s: float
    turn_rate_process_scale: float
    position_process_scale: float


def main():
    """Load calibration runs, report robust scales, and show prior plots."""
    data = read_ship_data(DATA_FILE)
    samples = collect_motion_prior_samples(
        data,
        run_ids=CALIBRATION_RUN_IDS,
        min_course_displacement_m=MIN_COURSE_DISPLACEMENT_METERS,
        max_time_gap_s=MAX_TIME_GAP_SECONDS,
    )
    suggestions = suggest_prior_scales(samples)
    _print_report(
        samples,
        suggestions,
        print_per_run_summary=PRINT_PER_RUN_SUMMARY,
    )
    figures, axes = plot_motion_prior_distributions(
        samples,
        suggestions,
        central_quantile=PLOT_CENTRAL_QUANTILE,
        histogram_mode=HISTOGRAM_MODE,
        show_sequentially=True,
    )
    return samples, suggestions, figures, axes


def collect_motion_prior_samples(
    data,
    *,
    run_ids=None,
    min_course_displacement_m=MIN_COURSE_DISPLACEMENT_METERS,
    max_time_gap_s=MAX_TIME_GAP_SECONDS,
) -> MotionPriorSamples:
    """Derive speed, course-rate, and one-step CTRV innovations per run."""
    if run_ids is not None and not isinstance(run_ids, (str, bytes)):
        run_ids = tuple(run_ids)
    _validate_exploration_arguments(
        data,
        run_ids=run_ids,
        min_course_displacement_m=min_course_displacement_m,
        max_time_gap_s=max_time_gap_s,
    )
    selected = data.copy()
    if run_ids is not None:
        selected = selected[selected["run_id"].isin(run_ids)].copy()
    if selected.empty:
        raise ValueError("No rows found for the selected calibration run IDs.")

    speed_samples = []
    turn_rate_samples = []
    turn_rate_innovations = []
    position_innovations = []
    summary_rows = []
    for run_id, run_data in selected.groupby("run_id", sort=True):
        run_samples = _derive_run_samples(
            run_data,
            min_course_displacement_m=min_course_displacement_m,
            max_time_gap_s=max_time_gap_s,
        )
        speed_samples.append(run_samples["speed_mps"])
        turn_rate_samples.append(run_samples["turn_rate_rad_s"])
        turn_rate_innovations.append(run_samples["turn_rate_innovation"])
        position_innovations.append(run_samples["position_innovation"])
        summary_rows.append(
            _summarize_run(
                run_id,
                row_count=len(run_data),
                **run_samples,
            )
        )

    return MotionPriorSamples(
        speed_mps=_concatenate_nonempty(speed_samples, "speed"),
        turn_rate_rad_s=_concatenate_nonempty(turn_rate_samples, "turn rate"),
        turn_rate_innovation=_concatenate_nonempty(
            turn_rate_innovations,
            "turn-rate innovation",
        ),
        position_innovation=_concatenate_nonempty(
            position_innovations,
            "position innovation",
        ),
        per_run_summary=pd.DataFrame(summary_rows),
    )


def suggest_prior_scales(samples: MotionPriorSamples) -> PriorSuggestions:
    """Return robust empirical centers and scales for scientific review."""
    speed_center, speed_scale = _robust_location_scale(samples.speed_mps)
    turn_center, turn_scale = _robust_location_scale(samples.turn_rate_rad_s)
    _, turn_process_scale = _robust_location_scale(samples.turn_rate_innovation)
    _, position_process_scale = _robust_location_scale(samples.position_innovation)
    return PriorSuggestions(
        speed_median_mps=speed_center,
        speed_robust_scale_mps=speed_scale,
        turn_rate_center_rad_s=turn_center,
        turn_rate_robust_scale_rad_s=turn_scale,
        turn_rate_process_scale=turn_process_scale,
        position_process_scale=position_process_scale,
    )


def plot_motion_prior_distributions(
    samples: MotionPriorSamples,
    suggestions: PriorSuggestions,
    *,
    central_quantile=PLOT_CENTRAL_QUANTILE,
    histogram_mode=HISTOGRAM_MODE,
    show_sequentially=False,
):
    """Create four figures and optionally display them one after another."""
    if not np.isfinite(central_quantile) or not 0.5 < central_quantile <= 1:
        raise ValueError("central_quantile must be in the interval (0.5, 1].")
    if histogram_mode not in {"density", "frequency"}:
        raise ValueError("histogram_mode must be 'density' or 'frequency'.")

    plot_specs = (
        (
            _plot_speed_distribution,
            {
                "values": samples.speed_mps,
                "central_quantile": central_quantile,
                "histogram_mode": histogram_mode,
            },
        ),
        (
            _plot_signed_distribution,
            {
                "values": samples.turn_rate_rad_s,
                "title": "Drehrate",
                "xlabel": "Drehrate [rad/s]",
                "central_quantile": central_quantile,
                "histogram_mode": histogram_mode,
                "reference_center": suggestions.turn_rate_center_rad_s,
            },
        ),
        (
            _plot_signed_distribution,
            {
                "values": samples.turn_rate_innovation,
                "title": "Prozessinnovationen der Drehrate",
                "xlabel": "Drehratenänderung / √Δt [rad/s/√s]",
                "central_quantile": central_quantile,
                "histogram_mode": histogram_mode,
                "reference_center": 0.0,
            },
        ),
        (
            _plot_signed_distribution,
            {
                "values": samples.position_innovation,
                "title": "CTRV-Positionsresiduen (Ein-Schritt)",
                "xlabel": "Positionsresiduum / √Δt [m/√s]",
                "central_quantile": central_quantile,
                "histogram_mode": histogram_mode,
                "reference_center": 0.0,
            },
        ),
    )
    figures = []
    axes = []
    for plot_index, (plot_function, plot_arguments) in enumerate(plot_specs):
        figure, axis = plt.subplots(figsize=(8, 5))
        plot_function(axis, **plot_arguments)
        if plot_index == 1:
            secondary_axis = axis.secondary_xaxis(
                "top",
                functions=(np.degrees, np.radians),
            )
            secondary_axis.set_xlabel(
                "Drehrate [°/s]",
                fontsize=AXIS_LABEL_FONT_SIZE,
            )
            secondary_axis.tick_params(axis="x", labelsize=AXIS_TICK_FONT_SIZE)
        axis.xaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
        axis.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
        axis.tick_params(axis="both", labelsize=AXIS_TICK_FONT_SIZE)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figures.append(figure)
        axes.append(axis)
        if show_sequentially:
            plt.show(block=True)
            plt.close(figure)
    return tuple(figures), tuple(axes)


def _derive_run_samples(
    run_data,
    *,
    min_course_displacement_m,
    max_time_gap_s,
):
    """Derive aligned samples from one time-ordered run."""
    ordered = run_data.sort_values("time").reset_index(drop=True)
    timestamps = pd.to_datetime(ordered["time"], utc=True, format="mixed")
    time_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy()
    longitude = pd.to_numeric(ordered["gps_longitude"], errors="coerce").to_numpy()
    latitude = pd.to_numeric(ordered["gps_latitude"], errors="coerce").to_numpy()
    x_meters, y_meters = gps_to_local_coordinates(longitude, latitude, unit="m")
    speed_mps = (
        pd.to_numeric(ordered["gps_speed"], errors="coerce").to_numpy(dtype=float)
        * KILOMETERS_PER_HOUR_TO_METERS_PER_SECOND
    )
    finite_speed = speed_mps[np.isfinite(speed_mps) & (speed_mps >= 0)]

    delta_time = np.diff(time_seconds)
    delta_x = np.diff(x_meters)
    delta_y = np.diff(y_meters)
    distance = np.hypot(delta_x, delta_y)
    valid_segment = (
        np.isfinite(delta_time)
        & (delta_time > 0)
        & (delta_time <= max_time_gap_s)
        & np.isfinite(distance)
        & (distance >= min_course_displacement_m)
    )
    heading = np.arctan2(delta_y, delta_x)
    segment_time = 0.5 * (time_seconds[:-1] + time_seconds[1:])

    adjacent_valid = valid_segment[:-1] & valid_segment[1:]
    heading_change = _wrap_angle(heading[1:] - heading[:-1])
    heading_time_difference = np.diff(segment_time)
    valid_turn = (
        adjacent_valid
        & np.isfinite(heading_time_difference)
        & (heading_time_difference > 0)
    )
    turn_indices = np.flatnonzero(valid_turn)
    turn_rate = heading_change[valid_turn] / heading_time_difference[valid_turn]
    turn_time = 0.5 * (segment_time[:-1] + segment_time[1:])[valid_turn]

    consecutive_turn = np.diff(turn_indices) == 1
    turn_time_difference = np.diff(turn_time)
    valid_turn_innovation = (
        consecutive_turn
        & np.isfinite(turn_time_difference)
        & (turn_time_difference > 0)
        & (turn_time_difference <= max_time_gap_s)
    )
    turn_rate_innovation = np.diff(turn_rate)[valid_turn_innovation] / np.sqrt(
        turn_time_difference[valid_turn_innovation]
    )
    position_innovation = _ctrv_position_innovations(
        x_meters,
        y_meters,
        speed_mps,
        time_seconds,
        heading,
        segment_time,
        valid_segment,
    )
    return {
        "speed_mps": finite_speed,
        "turn_rate_rad_s": turn_rate[np.isfinite(turn_rate)],
        "turn_rate_innovation": turn_rate_innovation[np.isfinite(turn_rate_innovation)],
        "position_innovation": position_innovation[np.isfinite(position_innovation)],
    }


def _ctrv_position_innovations(
    x_meters,
    y_meters,
    speed_mps,
    time_seconds,
    heading,
    segment_time,
    valid_segment,
):
    """Return leakage-free one-step residual components scaled by sqrt(dt)."""
    innovations = []
    for position_index in range(2, len(time_seconds) - 1):
        required_segments = slice(position_index - 2, position_index + 1)
        if not np.all(valid_segment[required_segments]):
            continue
        speed = speed_mps[position_index]
        if not np.isfinite(speed) or speed < 0:
            continue
        heading_dt = segment_time[position_index - 1] - segment_time[position_index - 2]
        prediction_dt = time_seconds[position_index + 1] - time_seconds[position_index]
        if heading_dt <= 0 or prediction_dt <= 0:
            continue
        turn_rate = (
            _wrap_angle(heading[position_index - 1] - heading[position_index - 2])
            / heading_dt
        )
        heading_at_position = heading[position_index - 1] + turn_rate * (
            time_seconds[position_index] - segment_time[position_index - 1]
        )
        predicted = ctrv_step(
            CTRVState(
                x=float(x_meters[position_index]),
                y=float(y_meters[position_index]),
                speed=float(speed),
                heading=float(heading_at_position),
                turn_rate=float(turn_rate),
            ),
            float(prediction_dt),
        )
        scale = np.sqrt(prediction_dt)
        innovations.extend(
            (
                (x_meters[position_index + 1] - predicted.x) / scale,
                (y_meters[position_index + 1] - predicted.y) / scale,
            )
        )
    return np.asarray(innovations, dtype=float)


def _summarize_run(
    run_id,
    *,
    row_count,
    speed_mps,
    turn_rate_rad_s,
    turn_rate_innovation,
    position_innovation,
):
    """Return compact robust statistics for one calibration run."""
    return {
        "run_id": run_id,
        "row_count": row_count,
        "speed_count": len(speed_mps),
        "speed_median_mps": _safe_quantile(speed_mps, 0.5),
        "speed_q95_mps": _safe_quantile(speed_mps, 0.95),
        "turn_rate_count": len(turn_rate_rad_s),
        "turn_rate_median_rad_s": _safe_quantile(turn_rate_rad_s, 0.5),
        "turn_rate_q90_abs_rad_s": _safe_quantile(
            np.abs(turn_rate_rad_s),
            0.9,
        ),
        "turn_rate_q99_abs_rad_s": _safe_quantile(
            np.abs(turn_rate_rad_s),
            0.99,
        ),
        "turn_rate_innovation_count": len(turn_rate_innovation),
        "position_innovation_count": len(position_innovation),
    }


def _print_report(samples, suggestions, *, print_per_run_summary):
    """Print compact evidence and optionally the complete per-run table."""
    run_values = samples.per_run_summary["run_id"].tolist()
    print("=" * 78)
    print("Motion Prior Exploration")
    print("=" * 78)
    _print_report_row("Data file", DATA_FILE)
    _print_report_row("Calibration runs", _format_run_selection(run_values))
    _print_report_row(
        "Usable samples",
        f"speed={len(samples.speed_mps)}, "
        f"turn rate={len(samples.turn_rate_rad_s)}, "
        f"turn innovation={len(samples.turn_rate_innovation)}, "
        f"position innovation={len(samples.position_innovation)}",
    )
    _print_report_row(
        "IMPORTANT",
        "Keep these runs separate from final evaluation.",
    )
    if print_per_run_summary:
        print("\nPer-run empirical summary:")
        print(samples.per_run_summary.round(6).to_string(index=False))
    print("\nRobust prior candidates (not applied automatically):")
    _print_report_row("Speed center", f"{suggestions.speed_median_mps:.4f} m/s")
    _print_report_row(
        "Speed robust scale",
        f"{suggestions.speed_robust_scale_mps:.4f} m/s",
    )
    _print_report_row(
        "Turn-rate center",
        f"{suggestions.turn_rate_center_rad_s:+.6f} rad/s",
    )
    _print_report_row(
        "Turn-rate robust scale",
        f"{suggestions.turn_rate_robust_scale_rad_s:.6f} rad/s",
    )
    _print_report_row(
        "Turn process scale",
        f"{suggestions.turn_rate_process_scale:.6f} rad/s/sqrt(s)",
    )
    _print_report_row(
        "Position residual scale",
        f"{suggestions.position_process_scale:.4f} m/sqrt(s)",
    )
    _print_report_row(
        "NOTE",
        "Position residuals contain motion-model and GPS measurement error; "
        "they are not a direct process-noise estimate.",
    )


def _print_report_row(label, value):
    """Print one terminal report row with a shared label width."""
    print(f"{label:<{REPORT_LABEL_WIDTH}}: {value}")


def _plot_speed_distribution(axis, values, *, central_quantile, histogram_mode):
    """Plot the empirical speed distribution without a parametric fit."""
    plotted = _central_values(values, central_quantile, nonnegative=True)
    axis.hist(
        plotted,
        bins=_histogram_bin_count(plotted),
        density=histogram_mode == "density",
        alpha=HISTOGRAM_ALPHA,
        color=HISTOGRAM_COLOR,
        edgecolor=HISTOGRAM_EDGE_COLOR,
        linewidth=HISTOGRAM_EDGE_LINE_WIDTH,
        label="Empirische Dichte",
    )
    axis.set_title(
        "GPS-Geschwindigkeit des Schiffs",
        pad=PLOT_TITLE_PAD,
        fontsize=PLOT_TITLE_FONT_SIZE,
        fontweight=PLOT_TITLE_FONT_WEIGHT,
    )
    axis.set_xlabel("Geschwindigkeit [m/s]")
    axis.set_ylabel(_histogram_y_label(histogram_mode))


def _plot_signed_distribution(
    axis,
    values,
    *,
    title,
    xlabel,
    central_quantile,
    histogram_mode,
    reference_center,
):
    """Plot one signed empirical distribution and an unlabelled reference."""
    plotted = _central_values(values, central_quantile)
    absolute_limit = max(float(np.max(np.abs(plotted))), 1e-12)
    axis.hist(
        plotted,
        bins=_histogram_bin_count(plotted),
        density=histogram_mode == "density",
        alpha=HISTOGRAM_ALPHA,
        color=HISTOGRAM_COLOR,
        edgecolor=HISTOGRAM_EDGE_COLOR,
        linewidth=HISTOGRAM_EDGE_LINE_WIDTH,
        label="Empirische Dichte",
    )
    axis.axvline(
        reference_center,
        color="black",
        linestyle=":",
        label="_nolegend_",
    )
    axis.set_xlim(-absolute_limit, absolute_limit)
    axis.set_title(
        title,
        pad=PLOT_TITLE_PAD,
        fontsize=PLOT_TITLE_FONT_SIZE,
        fontweight=PLOT_TITLE_FONT_WEIGHT,
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(_histogram_y_label(histogram_mode))


def _central_values(values, quantile, *, nonnegative=False):
    """Trim display-only extremes while retaining all values in statistics."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot plot an empty empirical distribution.")
    if nonnegative:
        upper = float(np.quantile(values, quantile))
        selected = values[(values >= 0) & (values <= upper)]
    else:
        upper = float(np.quantile(np.abs(values), quantile))
        selected = values[np.abs(values) <= upper]
    return selected if selected.size else values


def _robust_location_scale(values):
    """Return median and Gaussian-consistent MAD scale."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Robust statistics require finite non-empty values.")
    median = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - median)))
    return median, scale


def _wrap_angle(values):
    """Wrap radians to the principal interval without converting to degrees."""
    return np.arctan2(np.sin(values), np.cos(values))


def _histogram_bin_count(values):
    """Keep density plots legible for both short and long calibration runs."""
    return int(np.clip(np.sqrt(len(values)), 12, 60))


def _histogram_y_label(histogram_mode):
    """Return the German y-axis label for one histogram mode."""
    return "Dichte" if histogram_mode == "density" else "Häufigkeit"


def _format_run_selection(run_values):
    """Return one compact run label for figures and terminal output."""
    if len(run_values) == 1:
        return f"run {run_values[0]}"
    if len(run_values) <= 12:
        return "run IDs " + ", ".join(str(value) for value in run_values)
    return f"{len(run_values)} selected runs"


def _safe_quantile(values, probability):
    """Return one quantile or NaN when a run has no usable samples."""
    return float(np.quantile(values, probability)) if len(values) else np.nan


def _concatenate_nonempty(arrays, name):
    """Concatenate available run samples or fail with a focused message."""
    available = [np.asarray(values, dtype=float) for values in arrays if len(values)]
    if not available:
        raise ValueError(f"No valid {name} samples found in calibration runs.")
    return np.concatenate(available)


def _validate_exploration_arguments(
    data,
    *,
    run_ids,
    min_course_displacement_m,
    max_time_gap_s,
):
    """Validate data selection and physical filtering controls."""
    required = {
        "time",
        "run_id",
        "gps_latitude",
        "gps_longitude",
        "gps_speed",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if run_ids is not None:
        if isinstance(run_ids, (str, bytes)) or not tuple(run_ids):
            raise ValueError("run_ids must contain at least one run ID or be None.")
    for name, value in {
        "min_course_displacement_m": min_course_displacement_m,
        "max_time_gap_s": max_time_gap_s,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")


if __name__ == "__main__":
    main()
