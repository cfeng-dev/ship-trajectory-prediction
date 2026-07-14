"""DataFrame creation and CSV export for simulated trajectory data."""

from pathlib import Path

import numpy as np
import pandas as pd

from ship_trajectory_prediction.simulation.core import add_observation_noise
from ship_trajectory_prediction.trajectory.coordinates import (
    local_to_gps_coordinates,
)

METERS_PER_SECOND_TO_KILOMETERS_PER_HOUR = 3.6

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Directory for simulated data
DATA_DIR = PROJECT_ROOT / "data" / "simulated"


def _format_utc_timestamp(timestamp):
    """Format a timestamp in UTC with fixed hundredths of a second."""
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    timestamp = timestamp.round("10ms")
    hundredths = timestamp.microsecond // 10_000
    return f"{timestamp:%Y-%m-%d %H:%M:%S}.{hundredths:02d}+00:00"


def create_simulation_dataframe(
    simulator,
    random_seed=42,
    start_time=None,
    *,
    reference_longitude,
    reference_latitude,
):
    """
    Create a DataFrame from a completed ship simulation.

    Parameters
    ----------
    simulator : ShipSimulator
        Simulator instance containing stored trajectory data.
    random_seed : int, optional
        Random seed for reproducible observation noise.
    start_time : datetime-like or None, optional
        UTC time corresponding to ``t = 0``. The current UTC time is used if
        no start time is provided.
    reference_longitude, reference_latitude : float
        GPS coordinates represented by the local simulation origin ``(0, 0)``.

    Returns
    -------
    trajectory_df : pd.DataFrame
        Simulated trajectory data including true positions, noisy GPS-like
        observations, GPS speed in km/h, and simulated speed in m/s.
    """
    x_true = np.array(simulator.x_all)
    y_true = np.array(simulator.y_all)
    elapsed_time = np.array(simulator.t_all)
    speed_mps = np.array(simulator.v_all)

    if start_time is None:
        start_timestamp = pd.Timestamp.now(tz="UTC").floor("s")
    else:
        start_timestamp = pd.Timestamp(start_time)
        if start_timestamp.tzinfo is None:
            start_timestamp = start_timestamp.tz_localize("UTC")
        else:
            start_timestamp = start_timestamp.tz_convert("UTC")

    timestamps = start_timestamp + pd.to_timedelta(elapsed_time, unit="s")

    x_obs, y_obs = add_observation_noise(
        x=x_true,
        y=y_true,
        sigma=simulator.sigma,
        random_seed=random_seed,
    )

    if len(x_obs) == 0:
        gps_longitude = np.array([])
        gps_latitude = np.array([])
    else:
        gps_longitude, gps_latitude = local_to_gps_coordinates(
            x_obs,
            y_obs,
            reference_longitude=reference_longitude,
            reference_latitude=reference_latitude,
        )

    trajectory_df = pd.DataFrame(
        {
            "time": timestamps,
            "gps_latitude": gps_latitude,
            "gps_longitude": gps_longitude,
            "gps_speed": (speed_mps * METERS_PER_SECOND_TO_KILOMETERS_PER_HOUR),
            "t": elapsed_time,
            "x_true": x_true,
            "y_true": y_true,
            "x_obs": x_obs,
            "y_obs": y_obs,
            "theta": np.array(simulator.theta_all),
            "omega": np.array(simulator.omega_all),
            "radius": np.array(simulator.radius_all),
            "v": speed_mps,
            "sigma": simulator.sigma,
            "simulation_running": np.array(simulator.motor_state_all),
        }
    )

    return trajectory_df


def save_trajectory_data(df, filename):
    """
    Save simulated trajectory data as a CSV file.

    If only a filename is given, the file is saved in data/simulated.
    If a full path is given, the file is saved at that location.

    Parameters
    ----------
    df : pd.DataFrame
        Simulated trajectory data.
    filename : str or pathlib.Path
        Output CSV filename or full output path.

    Returns
    -------
    output_path : pathlib.Path
        Path to the saved CSV file.
    """
    output_path = Path(filename)

    # If only a filename is given, save it in the default data directory.
    if not output_path.is_absolute():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DATA_DIR / output_path
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    export_data = df.copy()
    if "time" in export_data.columns:
        export_data["time"] = export_data["time"].map(_format_utc_timestamp)

    for coordinate_column in ("gps_latitude", "gps_longitude"):
        if coordinate_column in export_data.columns:
            export_data[coordinate_column] = export_data[coordinate_column].map(
                lambda value: f"{value:.8f}"
            )

    # Keep fixed timestamp and GPS precision while rounding other floats.
    export_data.to_csv(output_path, index=False, float_format="%.3f")

    print(f"Saved simulated data to: {output_path}")

    return output_path
