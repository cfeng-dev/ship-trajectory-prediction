"""Utilities for loading and preprocessing ship trajectory data."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

import bayestraj.observations.coordinates as coordinates
import bayestraj.observations.window as observation_window

DEFAULT_SUMMARY_LABEL_WIDTH = 20


def read_ship_data(csv_path, run_id=None, start_time=None, end_time=None):
    """
    Read ship trajectory data from a CSV file and optionally filter it.

    The CSV file must contain the shared trajectory columns ``time``,
    ``run_id``, ``gps_latitude``, ``gps_longitude``, and ``gps_speed``.
    Additional columns, such as propulsion or simulation values, are preserved.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the CSV file.
    run_id : int, iterable of int, or None, optional
        Selected run ID or IDs. If None, all runs are loaded.
    start_time : str or None, optional
        Start time for filtering, e.g. "2026-01-09 23:00:00".
        If None, no lower time limit is applied.
    end_time : str or None, optional
        End time for filtering, e.g. "2026-01-10 01:00:00".
        If None, no upper time limit is applied.

    Returns
    -------
    pandas.DataFrame
        Loaded and optionally filtered ship data with parsed timestamps.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = pd.read_csv(csv_path)

    required_columns = [
        "time",
        "run_id",
        "gps_latitude",
        "gps_longitude",
        "gps_speed",
    ]

    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data["time"] = pd.to_datetime(data["time"], utc=True)

    if run_id is not None:
        if isinstance(run_id, Iterable) and not isinstance(run_id, (str, bytes)):
            data = data[data["run_id"].isin(tuple(run_id))]
        else:
            data = data[data["run_id"] == run_id]

    if start_time is not None:
        start_time = pd.to_datetime(start_time, utc=True)
        data = data[data["time"] >= start_time]

    if end_time is not None:
        end_time = pd.to_datetime(end_time, utc=True)
        data = data[data["time"] <= end_time]

    return data.copy()


def export_processed_trajectory(input_csv, run_id, output_dir):
    """Export one recorded ship run as local position observations in seconds."""
    if run_id is None or (
        isinstance(run_id, Iterable) and not isinstance(run_id, (str, bytes))
    ):
        raise ValueError("run_id must select exactly one run.")

    trajectory_data = read_ship_data(input_csv, run_id=run_id)
    if trajectory_data.empty:
        raise ValueError(f"run_id {run_id} was not found in {input_csv}.")
    if len(trajectory_data) < 2:
        raise ValueError("The selected run must contain at least two samples.")

    trajectory_data = trajectory_data.sort_values("time").reset_index(drop=True)

    timestamps = trajectory_data["time"]
    if timestamps.isna().any():
        raise ValueError("Source timestamps must be valid timezone-aware datetimes.")
    elapsed_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds()
    elapsed_seconds_array = elapsed_seconds.to_numpy(dtype=float)
    if not np.all(np.isfinite(elapsed_seconds_array)):
        raise ValueError("Generated time values must be finite.")
    if elapsed_seconds_array[0] != 0.0:
        raise ValueError("Generated time values must start at 0.0 seconds.")
    if np.any(np.diff(elapsed_seconds_array) <= 0.0):
        raise ValueError("Source timestamps must be strictly increasing after sorting.")

    longitude = pd.to_numeric(
        trajectory_data["gps_longitude"], errors="coerce"
    ).to_numpy(dtype=float)
    latitude = pd.to_numeric(trajectory_data["gps_latitude"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.all(np.isfinite(longitude)) or not np.all(np.isfinite(latitude)):
        raise ValueError("GPS coordinates must be finite.")

    x_coordinates, y_coordinates = coordinates.gps_to_local_coordinates(
        longitude,
        latitude,
        unit="m",
    )
    x_coordinates = np.asarray(x_coordinates, dtype=float)
    y_coordinates = np.asarray(y_coordinates, dtype=float)
    if len(x_coordinates) != len(elapsed_seconds_array) or len(y_coordinates) != len(
        elapsed_seconds_array
    ):
        raise ValueError("Generated x/y coordinates must match the time sample count.")
    if not np.all(np.isfinite(x_coordinates)) or not np.all(np.isfinite(y_coordinates)):
        raise ValueError("Generated x/y coordinates must be finite.")

    exported_data = pd.DataFrame(
        {
            "time": elapsed_seconds_array,
            "x": x_coordinates,
            "y": y_coordinates,
        }
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ship_trajectory_run_{run_id}.csv"
    exported_data.to_csv(output_path, index=False)
    return output_path


def resample_trajectory_data(data, interval_seconds=10):
    """Reduce trajectory data to one sample per regular time window.

    Each run is processed independently when a ``run_id`` column is present.
    The first existing row in every non-empty time window is kept; missing
    windows are not interpolated or filled.

    Parameters
    ----------
    data : pandas.DataFrame
        Trajectory data containing a ``time`` column.
    interval_seconds : float, optional
        Width of each sampling window in seconds.

    Returns
    -------
    pandas.DataFrame
        Resampled trajectory data with original timestamps and columns.
    """
    if "time" not in data.columns:
        raise ValueError("Trajectory data must contain a 'time' column.")

    try:
        interval = pd.to_timedelta(interval_seconds, unit="s")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "interval_seconds must be a positive finite number."
        ) from error

    if pd.isna(interval) or interval <= pd.Timedelta(0):
        raise ValueError("interval_seconds must be a positive finite number.")

    prepared_data = data.copy()
    prepared_data["time"] = pd.to_datetime(
        prepared_data["time"],
        utc=True,
        format="mixed",
    )

    if prepared_data.empty:
        return prepared_data

    if "run_id" in prepared_data.columns:
        groups = (
            group
            for _, group in prepared_data.groupby(
                "run_id",
                sort=False,
                dropna=False,
            )
        )
    else:
        groups = (prepared_data,)

    resampled_groups = [_resample_trajectory_group(group, interval) for group in groups]
    return pd.concat(resampled_groups, ignore_index=True)


def _resample_trajectory_group(data, interval):
    """Resample one trajectory run without inventing missing observations."""
    sorted_data = data.sort_values("time").copy()
    sorted_data["_source_time"] = sorted_data["time"]

    resampled_data = (
        sorted_data.set_index("time")
        .resample(interval, origin=sorted_data["time"].iloc[0])
        .first()
        .dropna(subset=["_source_time"])
    )
    resampled_data["time"] = resampled_data.pop("_source_time")

    return resampled_data.reset_index(drop=True)[data.columns]


def print_ship_data_summary(
    data,
    label_width=DEFAULT_SUMMARY_LABEL_WIDTH,
    *,
    gps_speed_unit="km/h",
    propulsion_speed_unit="rpm",
):
    """
    Print a short summary of the loaded ship trajectory data.

    Parameters
    ----------
    data : pandas.DataFrame
        Ship trajectory data.
    label_width : int, optional
        Shared width for labels before the value separator.
    gps_speed_unit : {"m/s", "km/h"}, optional
        Display unit for GPS speed. Source values are interpreted as km/h and
        converted when m/s is selected.
    propulsion_speed_unit : str, optional
        Display unit for shaft and thruster speed. No conversion is applied.
    """
    if gps_speed_unit not in {"m/s", "km/h"}:
        raise ValueError("gps_speed_unit must be 'm/s' or 'km/h'.")

    speed_labels = (
        f"GPS speed [{gps_speed_unit}]",
        f"Shaft speed [{propulsion_speed_unit}]",
        f"Thruster speed [{propulsion_speed_unit}]",
    )
    label_width = max(
        label_width,
        len("Longitude [deg]"),
        *(len(label) for label in speed_labels),
    )

    print("Ship data summary")
    print("=" * 50)

    _print_summary_row("Number of rows", len(data), label_width=label_width)

    if data.empty:
        print("No data available for the selected filters.")
        return

    _print_summary_row(
        "Number of runs",
        data["run_id"].nunique(),
        label_width=label_width,
    )

    print("\nTime range:")
    _print_summary_row("Start", data["time"].min(), label_width=label_width)
    _print_summary_row("End", data["time"].max(), label_width=label_width)

    print("\nGPS position range:")
    _print_summary_row(
        "Latitude [deg]",
        f"{data['gps_latitude'].min()} to {data['gps_latitude'].max()}",
        label_width=label_width,
    )
    _print_summary_row(
        "Longitude [deg]",
        f"{data['gps_longitude'].min()} to {data['gps_longitude'].max()}",
        label_width=label_width,
    )

    gps_speed = data["gps_speed"]
    if gps_speed_unit == "m/s":
        gps_speed = (
            gps_speed * observation_window.KILOMETERS_PER_HOUR_TO_METERS_PER_SECOND
        )

    print("\nSpeed range:")
    _print_summary_row(
        speed_labels[0],
        f"{gps_speed.min()} to {gps_speed.max()}",
        label_width=label_width,
    )
    _print_summary_row(
        speed_labels[1],
        f"{data['shaft_speed'].min()} to {data['shaft_speed'].max()}",
        label_width=label_width,
    )
    _print_summary_row(
        speed_labels[2],
        f"{data['thruster_speed'].min()} to {data['thruster_speed'].max()}",
        label_width=label_width,
    )


def _print_summary_row(label, value, *, label_width):
    """Print one summary row with a shared label width."""
    print(f"{label:<{label_width}}: {value}")
