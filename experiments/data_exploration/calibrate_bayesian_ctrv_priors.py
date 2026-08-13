"""Calibrate position-only empirical priors for the Bayesian CTRV model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import erf, pi, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from ship_trajectory_prediction.coordinates import (
    calculate_signed_turn_rate_from_gps,
    gps_to_local_coordinates,
)
from ship_trajectory_prediction.models.bayesian_ctrv import (
    DEFAULT_INITIAL_SPEED_POINT_COUNT,
    MAX_TURN_RATE_PRIOR_SCALE,
    MIN_COURSE_DISPLACEMENT_METERS,
    MIN_INITIAL_SPEED_PRIOR_SCALE_MPS,
    MIN_TURN_RATE_PRIOR_SCALE,
    ROBUST_MAD_SCALE_FACTOR,
    TURN_RATE_PRIOR_SCALE_MULTIPLIER,
    estimate_initial_speed_from_positions,
)
from ship_trajectory_prediction.models.deterministic_ctrv import CTRVState, ctrv_step
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import read_ship_data

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

# The stop value is exclusive: this default requests run IDs 0 through 99.
RUN_ID_RANGE = range(0, 100)

INITIAL_SPEED_POINT_COUNT = DEFAULT_INITIAL_SPEED_POINT_COUNT
INITIAL_SPEED_HISTOGRAM_BIN_WIDTH_MPS = 0.1
MAX_TIME_GAP_SECONDS = 15.0
POSITION_COLUMNS = ("time", "run_id", "gps_latitude", "gps_longitude")
REPORT_LABEL_WIDTH = 28
PLOT_CENTRAL_QUANTILE = 0.99
SHOW_RECOMMENDED_PRIOR_DENSITY = True

HISTOGRAM_COLOR = "#4C78A8"
PRIOR_COLOR = "#B44C43"
MEDIAN_COLOR = "#222222"
QUARTILE_COLOR = "#666666"
FIGURE_SIZE = (10.5, 5.2)

# Match the scientific typography used by ``explore_ship_data.py``.
TITLE_PAD = 16
TITLE_FONT_SIZE = 13
TITLE_FONT_WEIGHT = "bold"
AXIS_LABEL_FONT_SIZE = 13
AXIS_TICK_FONT_SIZE = 11


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Descriptive and robust statistics for one finite empirical sample."""

    count: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    median: float
    q25: float
    q75: float
    mad: float
    robust_scale: float


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Position-only per-run samples and pooled calibration diagnostics."""

    run_start: int
    run_stop: int
    per_run: pd.DataFrame
    skipped_runs: tuple[tuple[int, str], ...]
    initial_speed_mps: np.ndarray
    turn_rate_rad_s: np.ndarray
    position_innovation: np.ndarray
    speed_innovation: np.ndarray
    turn_rate_innovation: np.ndarray
    initial_speed_summary: DistributionSummary
    turn_rate_summary: DistributionSummary

    @property
    def requested_run_count(self) -> int:
        """Return the number of run IDs in the exclusive range."""
        return self.run_stop - self.run_start

    @property
    def valid_run_count(self) -> int:
        """Return the number of runs with a valid initial-speed estimate."""
        return self.requested_run_count - len(self.skipped_runs)


def main(
    *,
    run_start: int = RUN_ID_RANGE.start,
    run_stop: int = RUN_ID_RANGE.stop,
    show_prior_density: bool = SHOW_RECOMMENDED_PRIOR_DENSITY,
) -> CalibrationResult:
    """Run calibration, print the report, and show plots sequentially."""
    run_ids = _validate_run_range(run_start, run_stop)
    _print_requested_range(run_start, run_stop)

    # The shared loader validates the repository CSV schema. All non-position
    # columns are discarded before any calibration calculation.
    loaded_data = read_ship_data(DATA_FILE, run_id=run_ids)
    position_data = loaded_data.loc[:, POSITION_COLUMNS].copy()
    result = calibrate_position_only_priors(
        position_data,
        run_start=run_start,
        run_stop=run_stop,
    )

    _print_skipped_runs(result.valid_run_count, result.skipped_runs)
    _print_detailed_statistics(result)
    _print_compact_summary(result)
    for _, figure in _iter_calibration_figures(
        result,
        show_prior_density=show_prior_density,
    ):
        plt.show(block=True)
        plt.close(figure)
    return result


def calibrate_position_only_priors(
    data: pd.DataFrame,
    *,
    run_start: int,
    run_stop: int,
) -> CalibrationResult:
    """Derive per-run priors and pooled innovations from GPS positions."""
    run_ids = _validate_run_range(run_start, run_stop)
    _validate_position_data(data)
    groups = {int(run_id): group for run_id, group in data.groupby("run_id")}

    rows = []
    skipped_runs = []
    initial_speeds = []
    pooled_turn_rates = []
    pooled_position_innovations = []
    pooled_speed_innovations = []
    pooled_turn_rate_innovations = []

    for run_id in run_ids:
        run_data = groups.get(run_id)
        if run_data is None or run_data.empty:
            reason = "no rows found"
            skipped_runs.append((run_id, reason))
            rows.append(_skipped_run_row(run_id, reason))
            continue

        try:
            ordered, time_seconds, x_meters, y_meters = _prepare_run(run_data)
            initial_speed = _estimate_run_initial_speed(
                time_seconds,
                x_meters,
                y_meters,
            )
        except (TypeError, ValueError, OverflowError) as error:
            reason = str(error)
            skipped_runs.append((run_id, reason))
            rows.append(
                _skipped_run_row(
                    run_id,
                    reason,
                    n_positions=_count_usable_positions(run_data),
                )
            )
            continue

        turn_rate = calculate_signed_turn_rate_from_gps(
            ordered,
            min_displacement_m=MIN_COURSE_DISPLACEMENT_METERS,
            max_time_gap_s=MAX_TIME_GAP_SECONDS,
        )
        finite_turn_rate = turn_rate[np.isfinite(turn_rate)]
        process = _derive_process_innovations(
            time_seconds,
            x_meters,
            y_meters,
            turn_rate,
        )

        initial_speeds.append(initial_speed)
        pooled_turn_rates.append(finite_turn_rate)
        pooled_position_innovations.append(process["position"])
        pooled_speed_innovations.append(process["speed"])
        pooled_turn_rate_innovations.append(process["turn_rate"])
        rows.append(
            _valid_run_row(
                run_id,
                n_positions=len(ordered),
                initial_speed=initial_speed,
                turn_rate=finite_turn_rate,
                process=process,
            )
        )

    if not initial_speeds:
        raise ValueError(
            "No valid runs contain the required first "
            f"{INITIAL_SPEED_POINT_COUNT} usable position samples."
        )

    speed_values = np.asarray(initial_speeds, dtype=float)
    turn_rate_values = _concatenate_available(pooled_turn_rates)
    if turn_rate_values.size == 0:
        raise ValueError("No valid turn-rate samples found in the selected runs.")

    return CalibrationResult(
        run_start=run_start,
        run_stop=run_stop,
        per_run=pd.DataFrame(rows),
        skipped_runs=tuple(skipped_runs),
        initial_speed_mps=speed_values,
        turn_rate_rad_s=turn_rate_values,
        position_innovation=_concatenate_available(pooled_position_innovations),
        speed_innovation=_concatenate_available(pooled_speed_innovations),
        turn_rate_innovation=_concatenate_available(pooled_turn_rate_innovations),
        initial_speed_summary=_summarize_distribution(speed_values),
        turn_rate_summary=_summarize_distribution(turn_rate_values),
    )


def create_calibration_figures(
    result: CalibrationResult,
    *,
    show_prior_density: bool = SHOW_RECOMMENDED_PRIOR_DENSITY,
) -> dict[str, plt.Figure]:
    """Create clean prior and process-diagnostic figures."""
    return dict(
        _iter_calibration_figures(
            result,
            show_prior_density=show_prior_density,
        )
    )


def _iter_calibration_figures(
    result: CalibrationResult,
    *,
    show_prior_density: bool,
):
    """Yield calibration figures in their interactive display order."""
    run_label = f"Run-IDs {result.run_start}-{result.run_stop - 1}"
    yield (
        "initial_speed_prior",
        _plot_initial_speed(
            result,
            run_label,
            show_prior_density=show_prior_density,
        ),
    )
    yield (
        "turn_rate_prior",
        _plot_turn_rate(
            result,
            run_label,
            show_prior_density=show_prior_density,
        ),
    )
    process_specs = (
        (
            "position_process",
            result.position_innovation,
            "Diagnostik der Positionsprozess-Innovationen",
            "Positionsinnovation [m/√s]",
        ),
        (
            "speed_process",
            result.speed_innovation,
            "Diagnostik der Geschwindigkeitsprozess-Innovationen",
            "Geschwindigkeitsinnovation [(m/s)/√s]",
        ),
        (
            "turn_rate_process",
            result.turn_rate_innovation,
            "Diagnostik der Drehratenprozess-Innovationen",
            "Drehrateninnovation [(rad/s)/√s]",
        ),
    )
    for name, values, title, x_label in process_specs:
        if _candidate_scale(values) is not None:
            yield (
                name,
                _plot_process_diagnostic(
                    values,
                    title=f"{title} ({run_label})",
                    x_label=x_label,
                ),
            )


def _prepare_run(run_data):
    """Return finite unique positions in stable chronological order."""
    prepared = run_data.loc[:, POSITION_COLUMNS].copy()
    prepared["time"] = pd.to_datetime(
        prepared["time"],
        utc=True,
        format="mixed",
        errors="coerce",
    )
    for column in ("gps_latitude", "gps_longitude"):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    valid = (
        prepared["time"].notna()
        & np.isfinite(prepared["gps_latitude"])
        & np.isfinite(prepared["gps_longitude"])
        & prepared["gps_latitude"].between(-90, 90)
        & prepared["gps_longitude"].between(-180, 180)
    )
    prepared = (
        prepared.loc[valid]
        .sort_values("time", kind="stable")
        .drop_duplicates(subset="time", keep="first")
        .reset_index(drop=True)
    )
    if len(prepared) < INITIAL_SPEED_POINT_COUNT:
        raise ValueError(
            f"only {len(prepared)} usable positions; "
            f"need at least {INITIAL_SPEED_POINT_COUNT}"
        )

    timestamps = pd.DatetimeIndex(prepared["time"])
    time_seconds = (timestamps - timestamps[0]).total_seconds().to_numpy(dtype=float)
    x_meters, y_meters = gps_to_local_coordinates(
        prepared["gps_longitude"].to_numpy(dtype=float),
        prepared["gps_latitude"].to_numpy(dtype=float),
        unit="m",
    )
    return prepared, time_seconds, x_meters, y_meters


def _estimate_run_initial_speed(time_seconds, x_meters, y_meters):
    """Estimate one speed from the first contiguous valid positions."""
    initial_time = time_seconds[:INITIAL_SPEED_POINT_COUNT]
    time_steps = np.diff(initial_time)
    if np.any(time_steps <= 0):
        raise ValueError("first usable positions are not strictly chronological")
    if np.any(time_steps > MAX_TIME_GAP_SECONDS):
        largest_gap = float(np.max(time_steps))
        raise ValueError(
            "first usable positions contain a time gap of "
            f"{largest_gap:g} s (maximum {MAX_TIME_GAP_SECONDS:g} s)"
        )
    return estimate_initial_speed_from_positions(
        time_seconds,
        x_meters,
        y_meters,
        point_count=INITIAL_SPEED_POINT_COUNT,
    )


def _derive_process_innovations(
    time_seconds,
    x_meters,
    y_meters,
    turn_rate,
):
    """Return causal position-only innovations scaled by ``sqrt(dt)``."""
    delta_time = np.diff(time_seconds)
    delta_x = np.diff(x_meters)
    delta_y = np.diff(y_meters)
    displacement = np.hypot(delta_x, delta_y)
    valid_time = (
        np.isfinite(delta_time)
        & (delta_time > 0)
        & (delta_time <= MAX_TIME_GAP_SECONDS)
    )
    valid_segment = (
        valid_time
        & np.isfinite(displacement)
        & (displacement >= MIN_COURSE_DISPLACEMENT_METERS)
    )
    segment_speed = np.divide(
        displacement,
        delta_time,
        out=np.full_like(displacement, np.nan),
        where=valid_segment,
    )
    segment_heading = np.arctan2(delta_y, delta_x)
    segment_time = 0.5 * (time_seconds[:-1] + time_seconds[1:])

    speed_time_difference = np.diff(segment_time)
    valid_speed_change = (
        valid_segment[:-1]
        & valid_segment[1:]
        & np.isfinite(speed_time_difference)
        & (speed_time_difference > 0)
        & (speed_time_difference <= MAX_TIME_GAP_SECONDS)
    )
    speed_innovation = np.diff(segment_speed)[valid_speed_change] / np.sqrt(
        speed_time_difference[valid_speed_change]
    )

    turn_indices = np.flatnonzero(np.isfinite(turn_rate))
    if turn_indices.size >= 2:
        turn_values = turn_rate[turn_indices]
        turn_time_difference = np.diff(time_seconds[turn_indices])
        valid_turn_change = (
            (np.diff(turn_indices) == 1)
            & np.isfinite(turn_time_difference)
            & (turn_time_difference > 0)
            & (turn_time_difference <= MAX_TIME_GAP_SECONDS)
        )
        turn_rate_innovation = np.diff(turn_values)[valid_turn_change] / np.sqrt(
            turn_time_difference[valid_turn_change]
        )
    else:
        turn_rate_innovation = np.asarray([], dtype=float)

    position_innovation = []
    for index in range(2, len(time_seconds) - 1):
        if not (
            valid_segment[index - 2] and valid_segment[index - 1] and valid_time[index]
        ):
            continue
        heading_time_difference = segment_time[index - 1] - segment_time[index - 2]
        if heading_time_difference <= 0:
            continue
        heading_change = _wrap_angle(
            segment_heading[index - 1] - segment_heading[index - 2]
        )
        reconstructed_turn_rate = heading_change / heading_time_difference
        heading_at_position = segment_heading[index - 1] + reconstructed_turn_rate * (
            time_seconds[index] - segment_time[index - 1]
        )
        predicted = ctrv_step(
            CTRVState(
                x=x_meters[index],
                y=y_meters[index],
                speed=segment_speed[index - 1],
                heading=heading_at_position,
                turn_rate=reconstructed_turn_rate,
            ),
            delta_time[index],
        )
        diffusion_denominator = np.sqrt(delta_time[index])
        position_innovation.extend(
            (
                (x_meters[index + 1] - predicted.x) / diffusion_denominator,
                (y_meters[index + 1] - predicted.y) / diffusion_denominator,
            )
        )

    return {
        "position": _finite_values(position_innovation),
        "speed": _finite_values(speed_innovation),
        "turn_rate": _finite_values(turn_rate_innovation),
    }


def _valid_run_row(
    run_id,
    *,
    n_positions,
    initial_speed,
    turn_rate,
    process,
):
    """Build one successful per-run output row."""
    turn_summary = _summarize_optional_distribution(turn_rate)
    return {
        "run_id": run_id,
        "status": "used",
        "skip_reason": "",
        "n_positions": n_positions,
        "initial_speed_mps": initial_speed,
        "turn_rate_count": len(turn_rate),
        "median_turn_rate_rad_s": _summary_value(turn_summary, "median"),
        "turn_rate_mad_rad_s": _summary_value(turn_summary, "mad"),
        "turn_rate_q90_abs_rad_s": _safe_quantile(np.abs(turn_rate), 0.9),
        "position_innovation_count": len(process["position"]),
        "position_diffusion_candidate_m_per_sqrt_s": _candidate_scale(
            process["position"]
        ),
        "speed_innovation_count": len(process["speed"]),
        "speed_diffusion_candidate_mps_per_sqrt_s": _candidate_scale(process["speed"]),
        "turn_rate_innovation_count": len(process["turn_rate"]),
        "turn_rate_diffusion_candidate_rad_s_per_sqrt_s": _candidate_scale(
            process["turn_rate"]
        ),
    }


def _skipped_run_row(run_id, reason, *, n_positions=0):
    """Build one skipped per-run output row with explicit missing values."""
    return {
        "run_id": run_id,
        "status": "skipped",
        "skip_reason": reason,
        "n_positions": n_positions,
        "initial_speed_mps": np.nan,
        "turn_rate_count": 0,
        "median_turn_rate_rad_s": np.nan,
        "turn_rate_mad_rad_s": np.nan,
        "turn_rate_q90_abs_rad_s": np.nan,
        "position_innovation_count": 0,
        "position_diffusion_candidate_m_per_sqrt_s": np.nan,
        "speed_innovation_count": 0,
        "speed_diffusion_candidate_mps_per_sqrt_s": np.nan,
        "turn_rate_innovation_count": 0,
        "turn_rate_diffusion_candidate_rad_s_per_sqrt_s": np.nan,
    }


def _summarize_distribution(values) -> DistributionSummary:
    """Return complete descriptive statistics for finite values."""
    values = _finite_values(values)
    if values.size == 0:
        raise ValueError("Distribution statistics require finite values.")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return DistributionSummary(
        count=int(values.size),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        mean=float(np.mean(values)),
        standard_deviation=standard_deviation,
        median=median,
        q25=float(np.quantile(values, 0.25)),
        q75=float(np.quantile(values, 0.75)),
        mad=mad,
        robust_scale=float(ROBUST_MAD_SCALE_FACTOR * mad),
    )


def _summarize_optional_distribution(values):
    """Return a summary when at least one finite sample is available."""
    values = _finite_values(values)
    return _summarize_distribution(values) if values.size else None


def _candidate_scale(values):
    """Return a robust diffusion candidate only for at least two samples."""
    values = _finite_values(values)
    if values.size < 2:
        return None
    return _summarize_distribution(values).robust_scale


def _plot_initial_speed(
    result,
    run_label,
    *,
    show_prior_density,
):
    """Plot per-run initial speeds and the recommended truncated Normal."""
    summary = result.initial_speed_summary
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.hist(
        result.initial_speed_mps,
        bins=_initial_speed_histogram_bin_edges(result.initial_speed_mps),
        density=True,
        color=HISTOGRAM_COLOR,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.6,
        label="Historische Schätzungen",
    )
    if show_prior_density:
        prior_scale = max(summary.robust_scale, MIN_INITIAL_SPEED_PRIOR_SCALE_MPS)
        x_max = max(summary.maximum, summary.median + 4 * prior_scale)
        x_values = np.linspace(0, max(x_max, 1e-6), 500)
        axis.plot(
            x_values,
            _lower_truncated_normal_density(
                x_values,
                center=summary.median,
                scale=prior_scale,
            ),
            color=PRIOR_COLOR,
            linewidth=2,
            label="Empfohlener Prior\n(Normal, Untergrenze 0)",
        )
    axis.axvline(
        summary.median,
        color=MEDIAN_COLOR,
        linewidth=1.6,
        label=f"Median = {summary.median:.3f} m/s",
    )
    axis.axvline(
        summary.q25,
        color=QUARTILE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label="Q25 / Q75",
    )
    axis.axvline(
        summary.q75,
        color=QUARTILE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label="_nolegend_",
    )
    _style_axis(
        axis,
        title=f"Historische Verteilung der Anfangsgeschwindigkeit ({run_label})",
        x_label="Anfangsgeschwindigkeit [m/s]",
        y_label="Dichte",
        x_limits=(0, None),
    )
    axis.xaxis.set_major_locator(MultipleLocator(0.5))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.grid(axis="y", alpha=0.2)
    _add_legend(axis)
    figure.tight_layout()
    return figure


def _plot_turn_rate(
    result,
    run_label,
    *,
    show_prior_density,
):
    """Plot the pooled empirical turn-rate distribution and suggested prior."""
    summary = result.turn_rate_summary
    plotted = _central_signed_values(result.turn_rate_rad_s)
    display_limit = max(float(np.max(np.abs(plotted))), np.finfo(float).eps)
    if show_prior_density:
        prior_scale = max(summary.robust_scale, np.finfo(float).eps)
        display_limit = max(display_limit, prior_scale)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.hist(
        plotted,
        bins=_histogram_bin_count(plotted),
        density=True,
        color=HISTOGRAM_COLOR,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.6,
        label="Empirische Drehraten\n(zentrale 99 %)",
    )
    if show_prior_density:
        x_values = np.linspace(-display_limit, display_limit, 500)
        axis.plot(
            x_values,
            _normal_density(x_values, summary.median, prior_scale),
            color=PRIOR_COLOR,
            linewidth=2,
            label="Empfohlener Normal-Prior",
        )
    axis.axvline(0, color=QUARTILE_COLOR, linestyle=":", label="Null")
    axis.axvline(
        summary.median,
        color=MEDIAN_COLOR,
        linewidth=1.6,
        label=f"Median = {summary.median:+.5f} rad/s",
    )
    _style_axis(
        axis,
        title=f"Historische Drehratenverteilung ({run_label})",
        x_label="Drehrate [rad/s]",
        y_label="Dichte",
        x_limits=(-display_limit, display_limit),
    )
    axis.grid(axis="y", alpha=0.2)
    _add_legend(axis)
    figure.tight_layout()
    return figure


def _plot_process_diagnostic(values, *, title, x_label):
    """Plot one process-innovation sample without implying identification."""
    values = _finite_values(values)
    summary = _summarize_distribution(values)
    plotted = _central_signed_values(values)
    display_limit = max(float(np.max(np.abs(plotted))), np.finfo(float).eps)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.hist(
        plotted,
        bins=_histogram_bin_count(plotted),
        density=True,
        color=HISTOGRAM_COLOR,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.6,
        label="Rekonstruierte Innovationen\n(zentrale 99 %)",
    )
    axis.axvline(0, color=QUARTILE_COLOR, linestyle=":", label="Null")
    axis.axvline(
        summary.median,
        color=MEDIAN_COLOR,
        linewidth=1.5,
        label=f"Median = {summary.median:+.4g}",
    )
    _style_axis(
        axis,
        title=title,
        x_label=x_label,
        y_label="Dichte",
        x_limits=(-display_limit, display_limit),
    )
    axis.grid(axis="y", alpha=0.2)
    _add_legend(axis)
    figure.tight_layout()
    return figure


def _style_axis(axis, *, title, x_label, y_label, x_limits):
    """Apply the scientific typography shared with ``explore_ship_data.py``."""
    axis.set_title(
        title,
        pad=TITLE_PAD,
        fontsize=TITLE_FONT_SIZE,
        fontweight=TITLE_FONT_WEIGHT,
    )
    axis.set_xlabel(x_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axis.set_ylabel(y_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axis.set_xlim(*x_limits)
    axis.tick_params(axis="both", labelsize=AXIS_TICK_FONT_SIZE)


def _add_legend(axis):
    """Place a compact bordered legend inside the upper-right corner."""
    legend = axis.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        borderaxespad=0.0,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#4A4A4A",
    )
    legend.get_frame().set_linewidth(0.8)


def _print_requested_range(run_start, run_stop):
    """Print requested run counts before processing begins."""
    print(f"Run-ID range requested : {run_start} ... {run_stop - 1}")
    print(f"Requested runs         : {run_stop - run_start}")


def _print_skipped_runs(valid_run_count, skipped_runs):
    """Report invalid runs without flooding the terminal."""
    print(f"Valid runs used        : {valid_run_count}")
    print(f"Skipped runs           : {len(skipped_runs)}")
    if not skipped_runs:
        return
    for run_id, reason in skipped_runs[:10]:
        print(f"  Run {run_id}: {reason}")
    if len(skipped_runs) > 10:
        print(f"  ... {len(skipped_runs) - 10} additional skipped runs")


def _print_detailed_statistics(result):
    """Print complete speed and turn-rate empirical summaries."""
    speed = result.initial_speed_summary
    turn = result.turn_rate_summary
    speed_prior_scale = max(
        speed.robust_scale,
        MIN_INITIAL_SPEED_PRIOR_SCALE_MPS,
    )
    turn_prior_scale = max(turn.robust_scale, np.finfo(float).eps)

    print("\nInitial-speed prior calibration")
    print("-" * 31)
    _print_row("Historical runs", result.requested_run_count)
    _print_row("Valid estimates", speed.count)
    _print_row("Minimum", f"{speed.minimum:.3f} m/s")
    _print_row("Maximum", f"{speed.maximum:.3f} m/s")
    _print_row("Mean", f"{speed.mean:.3f} m/s")
    _print_row("Standard deviation", f"{speed.standard_deviation:.3f} m/s")
    _print_row("Median", f"{speed.median:.3f} m/s")
    _print_row("Q25", f"{speed.q25:.3f} m/s")
    _print_row("Q75", f"{speed.q75:.3f} m/s")
    _print_row("MAD", f"{speed.mad:.3f} m/s")
    _print_row("Robust scale", f"{speed.robust_scale:.3f} m/s")
    print("\nSuggested prior:")
    print(f"speed_initial_prior_mean  = {speed.median:.3f}")
    print(f"speed_initial_prior_scale = {speed_prior_scale:.3f}")
    print(
        f"v_1 ~ Normal({speed.median:.3f}, {speed_prior_scale:.3f}), "
        "constrained to v_1 >= 0"
    )
    if speed.median > 0:
        print("This is a lower-truncated Normal, not a Half-Normal distribution.")

    print("\nHistorical turn-rate distribution")
    print("-" * 33)
    _print_row("Valid samples", turn.count)
    _print_row("Minimum", f"{turn.minimum:+.6f} rad/s")
    _print_row("Maximum", f"{turn.maximum:+.6f} rad/s")
    _print_row("Mean", f"{turn.mean:+.6f} rad/s")
    _print_row("Standard deviation", f"{turn.standard_deviation:.6f} rad/s")
    _print_row("Median", f"{turn.median:+.6f} rad/s")
    _print_row("Q25", f"{turn.q25:+.6f} rad/s")
    _print_row("Q75", f"{turn.q75:+.6f} rad/s")
    _print_row("MAD", f"{turn.mad:.6f} rad/s")
    _print_row("Robust scale", f"{turn.robust_scale:.6f} rad/s")
    _print_row(
        "90% |turn rate|",
        f"{np.quantile(np.abs(result.turn_rate_rad_s), 0.9):.6f} rad/s",
    )
    print("\nSuggested empirical prior:")
    _print_row(
        "Suggested prior",
        _format_turn_rate_prior(turn.median, turn_prior_scale),
    )
    print("This is an unbounded Normal distribution.")
    print("\nCurrent model settings for comparison (not empirical results):")
    _print_row(
        "Scale rule",
        f"{TURN_RATE_PRIOR_SCALE_MULTIPLIER:g} x robust scale",
    )
    _print_row(
        "Scale clipping",
        f"{MIN_TURN_RATE_PRIOR_SCALE:.3f} ... {MAX_TURN_RATE_PRIOR_SCALE:.3f} rad/s",
    )


def _print_compact_summary(result):
    """Print the final screenshot-friendly calibration summary."""
    speed = result.initial_speed_summary
    turn = result.turn_rate_summary
    speed_scale = max(speed.robust_scale, MIN_INITIAL_SPEED_PRIOR_SCALE_MPS)
    speed_standard_scale = max(
        speed.standard_deviation,
        MIN_INITIAL_SPEED_PRIOR_SCALE_MPS,
    )
    turn_scale = max(turn.robust_scale, np.finfo(float).eps)
    turn_standard_scale = max(turn.standard_deviation, np.finfo(float).eps)

    print("\n" + "=" * 72)
    print("Bayesian CTRV Historical Prior Calibration")
    print("=" * 72)
    _print_row("Data file", DATA_FILE)
    _print_row("Run IDs", f"{result.run_start} ... {result.run_stop - 1}")
    _print_row("Requested runs", result.requested_run_count)
    _print_row("Valid runs", result.valid_run_count)
    _print_row("Skipped runs", len(result.skipped_runs))
    print("\nInitial speed")
    print("-" * 13)
    _print_row("Median", f"{speed.median:.3f} m/s")
    _print_row("MAD", f"{speed.mad:.3f} m/s")
    _print_row("Robust scale (1.4826 · MAD)", f"{speed_scale:.3f} m/s")
    _print_row("Standard deviation", f"{speed.standard_deviation:.3f} m/s")
    _print_row(
        "Suggested prior (robust)",
        f"Normal({speed.median:.3f}, {speed_scale:.3f}), lower bound 0",
    )
    _print_row(
        "Sensitivity alternative (SD)",
        f"Normal({speed.median:.3f}, {speed_standard_scale:.3f}), lower bound 0",
    )
    print("\nTurn rate")
    print("-" * 9)
    _print_row("Median", f"{turn.median:+.6f} rad/s")
    _print_row("MAD", f"{turn.mad:.6f} rad/s")
    _print_row("Robust scale (1.4826 · MAD)", f"{turn_scale:.6f} rad/s")
    _print_row("Standard deviation", f"{turn.standard_deviation:.6f} rad/s")
    _print_row(
        "90% |turn rate|",
        f"{np.quantile(np.abs(result.turn_rate_rad_s), 0.9):.6f} rad/s",
    )
    _print_row(
        "Suggested prior (robust)",
        _format_turn_rate_prior(turn.median, turn_scale),
    )
    _print_row(
        "Sensitivity alternative (SD)",
        _format_turn_rate_prior(turn.median, turn_standard_scale),
    )

    candidates = _process_candidates(result)
    if candidates:
        print("\nProcess-noise calibration candidates")
        print("-" * 36)
        if "position" in candidates:
            _print_row(
                "Position diffusion",
                f"{candidates['position']:.6f} m/sqrt(s)",
            )
        if "speed" in candidates:
            _print_row(
                "Speed diffusion",
                f"{candidates['speed']:.6f} (m/s)/sqrt(s)",
            )
        if "turn_rate" in candidates:
            _print_row(
                "Turn-rate diffusion",
                f"{candidates['turn_rate']:.6f} (rad/s)/sqrt(s)",
            )
    print("\nDiagnostics use reconstructed kinematics and mix process mismatch with")
    print("GPS measurement error; they are empirical candidates, not ground truth.")
    print("Heading residuals pool overlapping, correlated windows within each run.")
    print("sigma_position_gps is intentionally not calibrated from these trajectories.")
    print("Keep calibration runs disjoint from later evaluation runs.")
    print("=" * 72)


def _process_candidates(result):
    """Return only process candidates supported by at least two values."""
    candidates = {}
    for name, values in (
        ("position", result.position_innovation),
        ("speed", result.speed_innovation),
        ("turn_rate", result.turn_rate_innovation),
    ):
        scale = _candidate_scale(values)
        if scale is not None:
            candidates[name] = scale
    return candidates


def _format_turn_rate_prior(center, scale):
    """Format the unbounded Normal turn-rate prior for terminal summaries."""
    return f"Normal({center:.6f}, {scale:.6f}) rad/s"


def _normal_density(values, center, scale):
    """Evaluate a Normal density without adding a SciPy dependency."""
    standardized = (values - center) / scale
    return np.exp(-0.5 * standardized**2) / (scale * sqrt(2 * pi))


def _lower_truncated_normal_density(values, *, center, scale):
    """Evaluate a Normal density conditioned on non-negative support."""
    normalizing_probability = 0.5 * (1 + erf(center / (scale * sqrt(2))))
    density = _normal_density(values, center, scale) / normalizing_probability
    return np.where(values >= 0, density, 0.0)


def _histogram_bin_count(values):
    """Keep histograms legible for both short and large empirical samples."""
    return int(np.clip(np.sqrt(len(values)), 10, 60))


def _initial_speed_histogram_bin_edges(values):
    """Return 0.1 m/s-aligned bin edges for per-run initial speeds."""
    values = _finite_values(values)
    bin_width = INITIAL_SPEED_HISTOGRAM_BIN_WIDTH_MPS
    lower_edge = float(np.floor(np.min(values) / bin_width) * bin_width)
    upper_edge = float(np.ceil(np.max(values) / bin_width) * bin_width)
    if upper_edge <= lower_edge:
        upper_edge = lower_edge + bin_width
    bin_count = int(round((upper_edge - lower_edge) / bin_width))
    return lower_edge + bin_width * np.arange(bin_count + 1)


def _central_signed_values(values):
    """Trim display-only signed extremes while retaining all statistics."""
    values = _finite_values(values)
    limit = float(np.quantile(np.abs(values), PLOT_CENTRAL_QUANTILE))
    selected = values[np.abs(values) <= limit]
    return selected if selected.size else values


def _wrap_angle(value):
    """Wrap one course difference to the principal radian interval."""
    return float(_wrap_angles(value))


def _wrap_angles(values):
    """Wrap scalar or vector angles to the principal radian interval."""
    values = np.asarray(values, dtype=float)
    return np.arctan2(np.sin(values), np.cos(values))


def _finite_values(values):
    """Return a flat float array containing finite values only."""
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def _concatenate_available(arrays):
    """Concatenate non-empty finite arrays without failing on sparse runs."""
    available = [_finite_values(values) for values in arrays]
    available = [values for values in available if values.size]
    return np.concatenate(available) if available else np.asarray([], dtype=float)


def _safe_quantile(values, probability):
    """Return one finite quantile or NaN for an empty sample."""
    values = _finite_values(values)
    return float(np.quantile(values, probability)) if values.size else np.nan


def _summary_value(summary, name):
    """Read an optional summary field or return NaN."""
    return getattr(summary, name) if summary is not None else np.nan


def _count_usable_positions(run_data):
    """Count finite, unique timestamp-position rows for skip reporting."""
    try:
        prepared, *_ = _prepare_run(run_data)
    except ValueError as error:
        message = str(error)
        if message.startswith("only "):
            return int(message.split()[1])
        return 0
    return len(prepared)


def _validate_position_data(data):
    """Validate that calculations receive only the four allowed columns."""
    missing = sorted(set(POSITION_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"Missing required position columns: {missing}")
    extra = sorted(set(data.columns).difference(POSITION_COLUMNS))
    if extra:
        raise ValueError(
            "Calibration data must contain only time, run_id, gps_latitude, "
            f"and gps_longitude; extra columns: {extra}"
        )


def _validate_run_range(run_start, run_stop):
    """Return a non-empty range with an exclusive stop value."""
    for name, value in (("run_start", run_start), ("run_stop", run_stop)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer.")
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
    if run_stop <= run_start:
        raise ValueError("run_stop must be greater than run_start (stop is exclusive).")
    return range(run_start, run_stop)


def _print_row(label, value):
    """Print one aligned terminal-summary row."""
    print(f"{label:<{REPORT_LABEL_WIDTH}}: {value}")


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-start",
        type=int,
        default=RUN_ID_RANGE.start,
        help="First calibration run ID (inclusive).",
    )
    parser.add_argument(
        "--run-stop",
        type=int,
        default=RUN_ID_RANGE.stop,
        help="Final calibration run boundary (exclusive).",
    )
    prior_density_group = parser.add_mutually_exclusive_group()
    prior_density_group.add_argument(
        "--show-prior-density",
        dest="show_prior_density",
        action="store_true",
        help="Show the recommended prior density curves (default).",
    )
    prior_density_group.add_argument(
        "--hide-prior-density",
        dest="show_prior_density",
        action="store_false",
        help="Hide the recommended prior density curves.",
    )
    parser.set_defaults(show_prior_density=SHOW_RECOMMENDED_PRIOR_DENSITY)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(
        run_start=arguments.run_start,
        run_stop=arguments.run_stop,
        show_prior_density=arguments.show_prior_density,
    )
