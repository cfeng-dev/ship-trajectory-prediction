"""DataFrame creation and CSV export for simulated trajectory data."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.simulation.coordinates import (
    local_to_gps_coordinates,
)
from ship_trajectory_prediction.simulation.core import add_observation_noise

METERS_PER_SECOND_TO_KILOMETERS_PER_HOUR = 3.6
DATA_DIR = project_path("data/simulated")


@dataclass(frozen=True)
class TrajectorySaveResult:
    """Result details for a saved simulation run."""

    output_path: Path
    run_id: int
    appended: bool
    continued: bool


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


def _prepare_export_data(df, run_id):
    """Add a run ID and apply stable CSV formatting."""
    export_data = df.copy()

    if "run_id" in export_data.columns:
        export_data["run_id"] = run_id
    else:
        column_position = 1 if "time" in export_data.columns else 0
        export_data.insert(column_position, "run_id", run_id)

    if "time" in export_data.columns:
        export_data["time"] = export_data["time"].map(_format_utc_timestamp)

    for coordinate_column in ("gps_latitude", "gps_longitude"):
        if coordinate_column in export_data.columns:
            export_data[coordinate_column] = export_data[coordinate_column].map(
                lambda value: f"{value:.8f}"
            )

    return export_data


def _get_next_run_id(output_path, expected_columns):
    """Validate an existing CSV and return its next run ID."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        return 0, False

    try:
        existing_columns = pd.read_csv(output_path, nrows=0).columns.tolist()
    except pd.errors.EmptyDataError:
        return 0, False

    if existing_columns == expected_columns:
        existing_run_ids = pd.read_csv(output_path, usecols=["run_id"])["run_id"]
        if existing_run_ids.empty:
            return 0, True

        numeric_run_ids = pd.to_numeric(existing_run_ids, errors="coerce")
        if (
            numeric_run_ids.isna().any()
            or (numeric_run_ids < 0).any()
            or (numeric_run_ids % 1 != 0).any()
        ):
            raise ValueError("Existing CSV contains invalid run_id values.")

        return int(numeric_run_ids.max()) + 1, True

    legacy_columns = [column for column in expected_columns if column != "run_id"]
    if existing_columns == legacy_columns:
        legacy_data = pd.read_csv(output_path)
        migrated_data = _prepare_export_data(legacy_data, run_id=0)
        migrated_data.to_csv(output_path, index=False, float_format="%.3f")
        next_run_id = 1 if not legacy_data.empty else 0
        return next_run_id, True

    raise ValueError(
        "Existing CSV columns do not match the current simulation export schema. "
        "Choose a new filename to avoid mixing incompatible data."
    )


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


def save_trajectory_data(df, filename, existing_run_id=None):
    """
    Save or append simulated trajectory data as a CSV file.

    If only a filename is given, the file is saved in data/simulated.
    A new file starts with ``run_id = 0``. If a compatible file already
    exists, the trajectory is appended using the next available run ID.
    New samples from an already saved simulation session can be appended to
    their existing run by passing ``existing_run_id``.

    Parameters
    ----------
    df : pd.DataFrame
        Simulated trajectory data.
    filename : str or pathlib.Path
        Output CSV filename or full output path.
    existing_run_id : int or None, optional
        Existing run to continue. The DataFrame must contain only samples that
        have not been saved previously.

    Returns
    -------
    result : TrajectorySaveResult
        Saved path, assigned run ID, and how data was appended.
    """
    output_path = Path(filename)

    # If only a filename is given, save it in the default data directory.
    if not output_path.is_absolute():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DATA_DIR / output_path
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_columns = _prepare_export_data(df, run_id=0).columns.tolist()
    next_run_id, append_to_existing_file = _get_next_run_id(
        output_path,
        expected_columns,
    )

    continued_run = existing_run_id is not None
    if continued_run:
        if not append_to_existing_file:
            raise ValueError(
                "Cannot continue a run because the existing CSV file is missing "
                "or empty."
            )

        existing_run_ids = pd.read_csv(output_path, usecols=["run_id"])["run_id"]
        numeric_run_ids = pd.to_numeric(existing_run_ids, errors="coerce")
        if existing_run_id not in numeric_run_ids.to_numpy():
            raise ValueError(
                f"Run ID {existing_run_id} does not exist in the selected CSV file."
            )
        run_id = existing_run_id
    else:
        run_id = next_run_id

    export_data = _prepare_export_data(df, run_id=run_id)

    # Keep fixed timestamp and GPS precision while rounding other floats.
    export_data.to_csv(
        output_path,
        mode="a" if append_to_existing_file else "w",
        header=not append_to_existing_file,
        index=False,
        float_format="%.3f",
    )

    print(f"Saved simulated run {run_id} to: {output_path}")

    return TrajectorySaveResult(
        output_path=output_path,
        run_id=run_id,
        appended=append_to_existing_file,
        continued=continued_run,
    )
