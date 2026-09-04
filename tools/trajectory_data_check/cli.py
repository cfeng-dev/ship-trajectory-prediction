"""Command-line interface for trajectory CSV validation."""

import argparse
import math
import sys
from collections.abc import Sequence

import pandas as pd

from .checker import TrajectoryDataReport, check_trajectory_csv

DEFAULT_RUN_SUMMARY_LIMIT = 10


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one trajectory CSV and return a process exit code."""
    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        report = check_trajectory_csv(
            arguments.csv_path,
            gps_speed_unit=arguments.gps_speed_unit,
            max_time_gap_seconds=arguments.max_time_gap_seconds,
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print_report(report, show_all_runs=arguments.all_runs)
    return 0 if report.is_valid else 1


def print_report(
    report: TrajectoryDataReport,
    *,
    show_all_runs: bool = False,
) -> None:
    """Print a readable validation result and compact per-run summaries."""
    print("Trajectory data check")
    print("=" * 60)
    print(f"File                  : {report.source}")
    print(f"Rows                  : {report.row_count}")
    print(f"Columns               : {report.column_count}")
    print(f"Runs                  : {len(report.runs)}")
    print(f"GPS speed unit        : {report.gps_speed_unit}")
    print(f"Maximum expected gap  : {report.max_time_gap_seconds:g} s")

    if report.errors:
        print("\nErrors:")
        for message in report.errors:
            print(f"  - {message}")
    if report.warnings:
        print("\nWarnings:")
        for message in report.warnings:
            print(f"  - {message}")

    if report.runs:
        print("\nRun summaries:")
        displayed_runs = (
            report.runs if show_all_runs else report.runs[:DEFAULT_RUN_SUMMARY_LIMIT]
        )
        for run in displayed_runs:
            print(
                f"  Run {run.run_id}: rows={run.row_count}, "
                f"duration={_format_number(run.duration_seconds)} s, "
                f"median_dt={_format_number(run.median_interval_seconds)} s, "
                f"max_dt={_format_number(run.maximum_interval_seconds)} s, "
                f"large_gaps={run.large_gap_count}"
            )
            print(
                f"    distance={_format_number(run.distance_meters)} m, "
                f"median_recorded_speed={_format_number(run.recorded_speed_median)} "
                f"{report.gps_speed_unit}, "
                f"median_derived_speed={_format_number(run.derived_speed_median)} "
                f"{report.gps_speed_unit}, "
                f"median_abs_difference={_format_number(run.median_speed_difference)} "
                f"{report.gps_speed_unit}"
            )
        omitted_run_count = len(report.runs) - len(displayed_runs)
        if omitted_run_count:
            print(
                f"  ... {omitted_run_count} additional run summaries omitted; "
                "use --all-runs to display them."
            )

    if report.is_valid and report.warnings:
        result = "VALID WITH WARNINGS"
    elif report.is_valid:
        result = "VALID"
    else:
        result = "INVALID"
    print(f"\nResult: {result}")


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate and summarize a BayesTraj trajectory CSV file."
    )
    parser.add_argument("csv_path", help="Path to the trajectory CSV file.")
    parser.add_argument(
        "--gps-speed-unit",
        choices=("km/h", "m/s"),
        default="km/h",
        help="Unit used by values in the gps_speed column (default: km/h).",
    )
    parser.add_argument(
        "--max-time-gap-seconds",
        type=_positive_finite_float,
        default=15.0,
        help="Warn when consecutive samples exceed this gap (default: 15).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Display every run summary instead of only the first ten.",
    )
    return parser


def _positive_finite_float(value: str) -> float:
    """Parse one positive finite command-line floating-point value."""
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive finite number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _format_number(value: float | None) -> str:
    """Format optional report numbers compactly."""
    return "n/a" if value is None else f"{value:.3f}"
