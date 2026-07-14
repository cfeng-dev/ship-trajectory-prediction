"""Utilities for loading and preprocessing ship trajectory data."""

from pathlib import Path

import pandas as pd


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
    run_id : int or None, optional
        Selected run ID. If None, all runs are loaded.
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
        data = data[data["run_id"] == run_id]

    if start_time is not None:
        start_time = pd.to_datetime(start_time, utc=True)
        data = data[data["time"] >= start_time]

    if end_time is not None:
        end_time = pd.to_datetime(end_time, utc=True)
        data = data[data["time"] <= end_time]

    return data.copy()


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


def print_ship_data_summary(data):
    """
    Print a short summary of the loaded ship trajectory data.

    Parameters
    ----------
    data : pandas.DataFrame
        Ship trajectory data.
    """
    print("Ship data summary")
    print("=" * 50)

    print(f"Number of rows: {len(data)}")

    if data.empty:
        print("No data available for the selected filters.")
        return

    print(f"Number of runs: {data['run_id'].nunique()}")

    print("\nTime range:")
    print(f"Start: {data['time'].min()}")
    print(f"End:   {data['time'].max()}")

    print("\nGPS position range:")
    print(f"Latitude:  {data['gps_latitude'].min()} to {data['gps_latitude'].max()}")
    print(f"Longitude: {data['gps_longitude'].min()} to {data['gps_longitude'].max()}")

    print("\nSpeed range:")
    print(f"GPS speed:      {data['gps_speed'].min()} to {data['gps_speed'].max()}")
    print(f"Shaft speed:    {data['shaft_speed'].min()} to {data['shaft_speed'].max()}")
    print(
        f"Thruster speed: {data['thruster_speed'].min()} to {data['thruster_speed'].max()}"
    )
