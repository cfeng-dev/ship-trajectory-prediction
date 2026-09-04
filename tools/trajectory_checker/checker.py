"""Validation and summary calculations used by the trajectory checker."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import bayestraj.observations.coordinates as coordinates

REQUIRED_TRAJECTORY_COLUMNS = (
    "time",
    "run_id",
    "gps_latitude",
    "gps_longitude",
    "gps_speed",
)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Compact quality and movement summary for one trajectory run."""

    run_id: str
    row_count: int
    start_time: pd.Timestamp | None
    end_time: pd.Timestamp | None
    duration_seconds: float | None
    median_interval_seconds: float | None
    maximum_interval_seconds: float | None
    large_gap_count: int
    distance_meters: float | None
    recorded_speed_median: float | None
    derived_speed_median: float | None
    median_speed_difference: float | None


@dataclass(frozen=True, slots=True)
class TrajectoryDataReport:
    """Complete validation result for one trajectory data source."""

    source: str
    row_count: int
    column_count: int
    gps_speed_unit: str
    max_time_gap_seconds: float
    runs: tuple[RunSummary, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether no validation errors were found."""
        return not self.errors


def check_trajectory_csv(
    csv_path,
    *,
    gps_speed_unit="km/h",
    max_time_gap_seconds=15.0,
):
    """Read, validate, and summarize one trajectory CSV file."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory CSV file not found: {path}")

    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        data = pd.DataFrame()

    return check_trajectory_data(
        data,
        source=str(path.resolve()),
        gps_speed_unit=gps_speed_unit,
        max_time_gap_seconds=max_time_gap_seconds,
    )


def check_trajectory_data(
    data,
    *,
    source="<dataframe>",
    gps_speed_unit="km/h",
    max_time_gap_seconds=15.0,
):
    """Validate a trajectory frame and calculate per-run summaries."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas.DataFrame.")
    if gps_speed_unit not in {"km/h", "m/s"}:
        raise ValueError("gps_speed_unit must be 'km/h' or 'm/s'.")
    max_time_gap_seconds = _validate_max_time_gap(max_time_gap_seconds)

    errors = []
    warnings = []
    duplicate_columns = tuple(data.columns[data.columns.duplicated()])
    if duplicate_columns:
        errors.append(f"Duplicate columns: {list(duplicate_columns)}")

    missing_columns = sorted(set(REQUIRED_TRAJECTORY_COLUMNS).difference(data.columns))
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    if data.empty:
        errors.append("The trajectory data contains no rows.")

    if duplicate_columns or missing_columns or data.empty:
        return _build_report(
            data=data,
            source=source,
            gps_speed_unit=gps_speed_unit,
            max_time_gap_seconds=max_time_gap_seconds,
            runs=(),
            errors=errors,
            warnings=warnings,
        )

    parsed_time = pd.to_datetime(
        data["time"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    numeric_values = {
        column: pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
        for column in ("gps_latitude", "gps_longitude", "gps_speed")
    }

    invalid_time_count = int(parsed_time.isna().sum())
    if invalid_time_count:
        errors.append(f"Unparseable or missing timestamps: {invalid_time_count}")

    missing_run_id_count = int(data["run_id"].isna().sum())
    if missing_run_id_count:
        errors.append(f"Missing run_id values: {missing_run_id_count}")

    for column, values in numeric_values.items():
        invalid_count = int(np.count_nonzero(~np.isfinite(values)))
        if invalid_count:
            errors.append(f"Non-numeric or non-finite {column} values: {invalid_count}")

    latitude = numeric_values["gps_latitude"]
    longitude = numeric_values["gps_longitude"]
    speed = numeric_values["gps_speed"]
    invalid_latitude_count = int(
        np.count_nonzero(np.isfinite(latitude) & ((latitude < -90) | (latitude > 90)))
    )
    invalid_longitude_count = int(
        np.count_nonzero(
            np.isfinite(longitude) & ((longitude < -180) | (longitude > 180))
        )
    )
    negative_speed_count = int(np.count_nonzero(np.isfinite(speed) & (speed < 0)))
    if invalid_latitude_count:
        errors.append(f"Latitude values outside [-90, 90]: {invalid_latitude_count}")
    if invalid_longitude_count:
        errors.append(
            f"Longitude values outside [-180, 180]: {invalid_longitude_count}"
        )
    if negative_speed_count:
        errors.append(f"Negative gps_speed values: {negative_speed_count}")

    prepared_data = pd.DataFrame(
        {
            "run_id": data["run_id"],
            "time": parsed_time,
            **numeric_values,
        },
        index=data.index,
    )
    runs = []
    for run_id, run_data in prepared_data.groupby(
        "run_id",
        sort=False,
        dropna=False,
    ):
        run_label = _format_run_id(run_id)
        run_times = run_data["time"].dropna()
        source_time_steps = run_times.diff().dt.total_seconds().to_numpy()[1:]
        if np.any(source_time_steps < 0):
            warnings.append(f"Run {run_label} is not ordered by timestamp.")

        duplicate_timestamp_count = int(run_times.duplicated().sum())
        if duplicate_timestamp_count:
            errors.append(
                f"Run {run_label} contains duplicate timestamps: "
                f"{duplicate_timestamp_count}"
            )
        if len(run_data) < 3:
            warnings.append(
                f"Run {run_label} contains only {len(run_data)} rows; "
                "at least three are needed for a trajectory window."
            )

        summary = _summarize_run(
            run_id=run_label,
            data=run_data,
            gps_speed_unit=gps_speed_unit,
            max_time_gap_seconds=max_time_gap_seconds,
        )
        if summary.large_gap_count:
            warnings.append(
                f"Run {run_label} contains {summary.large_gap_count} time gap(s) "
                f"larger than {max_time_gap_seconds:g} seconds."
            )
        runs.append(summary)

    return _build_report(
        data=data,
        source=source,
        gps_speed_unit=gps_speed_unit,
        max_time_gap_seconds=max_time_gap_seconds,
        runs=runs,
        errors=errors,
        warnings=warnings,
    )


def _summarize_run(
    *,
    run_id,
    data,
    gps_speed_unit,
    max_time_gap_seconds,
):
    """Calculate one run summary from already parsed columns."""
    ordered = data.sort_values("time", na_position="last").reset_index(drop=True)
    valid_times = ordered["time"].dropna()
    start_time = valid_times.iloc[0] if not valid_times.empty else None
    end_time = valid_times.iloc[-1] if not valid_times.empty else None
    duration_seconds = (
        float((end_time - start_time).total_seconds())
        if start_time is not None and end_time is not None
        else None
    )

    intervals = valid_times.diff().dt.total_seconds().to_numpy()[1:]
    positive_intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    median_interval_seconds = _finite_median(positive_intervals)
    maximum_interval_seconds = _finite_maximum(positive_intervals)
    large_gap_count = int(np.count_nonzero(positive_intervals > max_time_gap_seconds))

    movement_rows = ordered.dropna(
        subset=["time", "gps_latitude", "gps_longitude", "gps_speed"]
    )
    finite_movement_mask = np.all(
        np.isfinite(
            movement_rows[["gps_latitude", "gps_longitude", "gps_speed"]].to_numpy(
                dtype=float
            )
        ),
        axis=1,
    )
    movement_rows = movement_rows.loc[finite_movement_mask]

    distance_meters = None
    derived_speed = np.full(len(movement_rows), np.nan, dtype=float)
    if len(movement_rows) >= 2:
        distances = coordinates.calculate_gps_distances(
            movement_rows["gps_longitude"].to_numpy(dtype=float),
            movement_rows["gps_latitude"].to_numpy(dtype=float),
        )
        distance_meters = float(np.sum(distances))
        derived_speed = coordinates.calculate_speed_from_gps(
            movement_rows,
            unit=gps_speed_unit,
        )

    recorded_speed = movement_rows["gps_speed"].to_numpy(dtype=float)
    recorded_speed_median = _finite_median(recorded_speed)
    derived_speed_median = _finite_median(derived_speed)
    comparison_mask = np.isfinite(derived_speed) & np.isfinite(recorded_speed)
    median_speed_difference = _finite_median(
        np.abs(derived_speed[comparison_mask] - recorded_speed[comparison_mask])
    )

    return RunSummary(
        run_id=run_id,
        row_count=len(data),
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration_seconds,
        median_interval_seconds=median_interval_seconds,
        maximum_interval_seconds=maximum_interval_seconds,
        large_gap_count=large_gap_count,
        distance_meters=distance_meters,
        recorded_speed_median=recorded_speed_median,
        derived_speed_median=derived_speed_median,
        median_speed_difference=median_speed_difference,
    )


def _build_report(
    *,
    data,
    source,
    gps_speed_unit,
    max_time_gap_seconds,
    runs,
    errors,
    warnings,
):
    """Construct an immutable report with stable tuple fields."""
    return TrajectoryDataReport(
        source=str(source),
        row_count=len(data),
        column_count=len(data.columns),
        gps_speed_unit=gps_speed_unit,
        max_time_gap_seconds=max_time_gap_seconds,
        runs=tuple(runs),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _validate_max_time_gap(value):
    """Return a positive finite time-gap threshold."""
    if isinstance(value, (bool, str, bytes)):
        raise ValueError("max_time_gap_seconds must be a positive finite number.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "max_time_gap_seconds must be a positive finite number."
        ) from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError("max_time_gap_seconds must be a positive finite number.")
    return value


def _format_run_id(run_id):
    """Format common numeric and missing run identifiers compactly."""
    if pd.isna(run_id):
        return "<missing>"
    if isinstance(run_id, (int, np.integer)):
        return str(int(run_id))
    if isinstance(run_id, (float, np.floating)) and float(run_id).is_integer():
        return str(int(run_id))
    return str(run_id)


def _finite_median(values):
    """Return the median of finite values or None when unavailable."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def _finite_maximum(values):
    """Return the maximum of finite values or None when unavailable."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else None
