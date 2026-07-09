"""
@file ship_data_io.py
@description Provides helper functions to read and preprocess real ship trajectory data.
@date Created on: 09.07.2026
@author C.Feng
"""

from pathlib import Path

import pandas as pd


def read_ship_data(csv_path):
    """
    Read ship trajectory data from a CSV file.

    The CSV file is expected to contain columns such as:
    time, run_id, gps_latitude, gps_longitude, gps_speed,
    shaft_speed, thruster_speed.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded ship data with parsed timestamps.
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
        "shaft_speed",
        "thruster_speed",
    ]

    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data["time"] = pd.to_datetime(data["time"], utc=True)

    return data


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
