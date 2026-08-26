"""Explore provisional priors for the parametric Bayesian CTRV model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.observations.coordinates as coordinates
import bayestraj.observations.io as observations_io
import bayestraj.observations.paths as paths

DATA_FILE = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)
RUN_ID_RANGE = range(0, 100)
OBSERVATION_COUNT = 20
MAX_TIME_GAP_SECONDS = 15.0
POSITION_COLUMNS = ("time", "run_id", "gps_latitude", "gps_longitude")
ROBUST_NORMAL_SCALE = 1.4826
MIN_SPEED_SCALE_MPS = 0.01
MIN_TURN_RATE_SCALE_RAD_S = 1e-6


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Robust empirical location and scale for one parameter sample."""

    count: int
    median: float
    robust_scale: float
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Position-only constant-motion estimates for disjoint observation windows."""

    run_start: int
    run_stop: int
    observation_count: int
    estimates: pd.DataFrame
    skipped_runs: tuple[tuple[int, str], ...]
    speed_summary: DistributionSummary
    turn_rate_summary: DistributionSummary


def main(argv=None) -> CalibrationResult:
    """Print empirical candidates that still require forecast validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-start", type=int, default=RUN_ID_RANGE.start)
    parser.add_argument("--run-stop", type=int, default=RUN_ID_RANGE.stop)
    parser.add_argument(
        "--observations",
        type=int,
        default=OBSERVATION_COUNT,
        help="Estimate constant motion from disjoint observation windows.",
    )
    arguments = parser.parse_args(argv)
    run_ids = _validate_run_range(arguments.run_start, arguments.run_stop)
    data = observations_io.read_ship_data(DATA_FILE, run_id=run_ids)
    result = calibrate_parametric_ctrv_priors(
        data.loc[:, POSITION_COLUMNS],
        run_start=arguments.run_start,
        run_stop=arguments.run_stop,
        observation_count=arguments.observations,
    )
    _print_report(result)
    return result


def calibrate_parametric_ctrv_priors(
    data: pd.DataFrame,
    *,
    run_start: int,
    run_stop: int,
    observation_count: int,
) -> CalibrationResult:
    """Estimate one speed and turn rate per disjoint position-only window."""
    run_ids = _validate_run_range(run_start, run_stop)
    if isinstance(observation_count, bool) or not isinstance(
        observation_count,
        (int, np.integer),
    ):
        raise ValueError("observation_count must be an integer.")
    observation_count = int(observation_count)
    if observation_count < 3:
        raise ValueError("observation_count must be at least 3.")
    _validate_position_data(data)

    grouped = {int(run_id): group for run_id, group in data.groupby("run_id")}
    rows: list[dict[str, float | int]] = []
    skipped_runs: list[tuple[int, str]] = []
    for run_id in run_ids:
        run_data = grouped.get(run_id)
        if run_data is None or run_data.empty:
            skipped_runs.append((run_id, "no rows found"))
            continue
        try:
            time_seconds, x_meters, y_meters = _prepare_run(run_data)
            window_count = 0
            for start in range(
                0,
                len(time_seconds) - observation_count + 1,
                observation_count,
            ):
                stop = start + observation_count
                selected_time = time_seconds[start:stop]
                if np.max(np.diff(selected_time)) > MAX_TIME_GAP_SECONDS:
                    continue
                speed, heading_initial, turn_rate = (
                    bayesian_model.estimate_constant_motion_from_positions(
                        selected_time,
                        x_meters[start:stop],
                        y_meters[start:stop],
                    )
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "window_start": start,
                        "speed_mps": speed,
                        "heading_initial_rad": heading_initial,
                        "turn_rate_rad_s": turn_rate,
                    }
                )
                window_count += 1
            if window_count == 0:
                skipped_runs.append(
                    (
                        run_id,
                        "no complete observation window with acceptable time gaps",
                    )
                )
        except (TypeError, ValueError, OverflowError) as error:
            skipped_runs.append((run_id, str(error)))

    estimates = pd.DataFrame(rows)
    if estimates.empty:
        raise ValueError("No valid constant-motion observation windows were found.")
    return CalibrationResult(
        run_start=run_start,
        run_stop=run_stop,
        observation_count=observation_count,
        estimates=estimates,
        skipped_runs=tuple(skipped_runs),
        speed_summary=_summarize(
            estimates["speed_mps"],
            minimum_scale=MIN_SPEED_SCALE_MPS,
        ),
        turn_rate_summary=_summarize(
            estimates["turn_rate_rad_s"],
            minimum_scale=MIN_TURN_RATE_SCALE_RAD_S,
        ),
    )


def _prepare_run(run_data):
    """Convert one finite chronological GPS run to local metric coordinates."""
    prepared = run_data.loc[:, POSITION_COLUMNS].copy()
    prepared["time"] = pd.to_datetime(
        prepared["time"],
        utc=True,
        format="mixed",
        errors="coerce",
    )
    for name in ("gps_latitude", "gps_longitude"):
        prepared[name] = pd.to_numeric(prepared[name], errors="coerce")
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
    if len(prepared) < 3:
        raise ValueError("fewer than three usable positions")
    timestamps = pd.DatetimeIndex(prepared["time"])
    time_seconds = (timestamps - timestamps[0]).total_seconds().to_numpy(dtype=float)
    x_meters, y_meters = coordinates.gps_to_local_coordinates(
        prepared["gps_longitude"].to_numpy(dtype=float),
        prepared["gps_latitude"].to_numpy(dtype=float),
        unit="m",
    )
    return time_seconds, x_meters, y_meters


def _summarize(values, *, minimum_scale):
    """Return a robust normal-prior candidate from finite estimates."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Parameter estimates must be a finite non-empty vector.")
    median = float(np.median(values))
    robust_scale = max(
        ROBUST_NORMAL_SCALE * float(np.median(np.abs(values - median))),
        minimum_scale,
    )
    return DistributionSummary(
        count=values.size,
        median=median,
        robust_scale=robust_scale,
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
    )


def _print_report(result: CalibrationResult) -> None:
    """Print candidates without presenting them as final model calibration."""
    print("=" * 72)
    print("Parametric Bayesian CTRV prior candidates")
    print("=" * 72)
    print(f"Run IDs               : {result.run_start}-{result.run_stop - 1}")
    print(f"Observations per fit  : {result.observation_count}")
    print(f"Valid windows         : {len(result.estimates)}")
    print(f"Skipped runs          : {len(result.skipped_runs)}")
    print(
        "Speed candidate       : "
        f"Normal({result.speed_summary.median:.6f}, "
        f"{result.speed_summary.robust_scale:.6f}) m/s"
    )
    print(
        "Turn-rate candidate   : "
        f"Normal({result.turn_rate_summary.median:.8f}, "
        f"{result.turn_rate_summary.robust_scale:.8f}) rad/s"
    )
    print("Heading prior         : Uniform(-pi, pi), not calibrated here")
    print("Status                : empirical candidates; held-out validation pending")


def _validate_run_range(run_start, run_stop):
    """Return one non-empty integer run-ID range."""
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in (run_start, run_stop)
    ):
        raise ValueError("run_start and run_stop must be integers.")
    if run_stop <= run_start:
        raise ValueError("run_stop must be greater than run_start.")
    return range(int(run_start), int(run_stop))


def _validate_position_data(data):
    """Require only the position columns used by this diagnostic."""
    missing = sorted(set(POSITION_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"Missing required position columns: {', '.join(missing)}")


if __name__ == "__main__":
    main()
