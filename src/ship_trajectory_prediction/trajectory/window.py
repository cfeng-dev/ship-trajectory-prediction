"""Shared preparation of observed and held-out trajectory windows."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ship_trajectory_prediction.coordinates import gps_to_local_coordinates

# Default unit expected for values in the CSV column ``gps_speed``.
DEFAULT_GPS_SPEED_UNIT = "km/h"
KILOMETERS_PER_HOUR_TO_METERS_PER_SECOND = 1 / 3.6


@dataclass(frozen=True)
class TrajectoryWindowData:
    """Model-neutral trajectory values for inference and held-out evaluation."""

    timestamps: pd.DatetimeIndex
    time_seconds: np.ndarray
    x_meters: np.ndarray
    y_meters: np.ndarray
    gps_speed_mps: np.ndarray
    observation_count: int

    @property
    def prediction_count(self):
        """Return the number of held-out future observations."""
        return len(self.time_seconds) - self.observation_count

    @property
    def observed_slice(self):
        """Return the slice selecting observations used for inference."""
        return slice(0, self.observation_count)

    @property
    def prediction_slice(self):
        """Return the slice selecting held-out observations."""
        return slice(self.observation_count, None)


def prepare_trajectory_window(
    data,
    observation_count=20,
    prediction_count=10,
    *,
    start_index=0,
    gps_speed_unit=DEFAULT_GPS_SPEED_UNIT,
) -> TrajectoryWindowData:
    """Prepare one model-neutral trajectory window in SI units.

    Position and time values are validated centrally. GPS speeds are interpreted
    according to ``gps_speed_unit`` and stored in meters per second, while
    non-numeric and negative values are retained as ``NaN`` and negative values
    respectively. Individual models apply their own speed requirements.
    """
    _validate_window_arguments(
        observation_count,
        prediction_count,
        start_index,
        gps_speed_unit,
    )

    required_columns = {
        "time",
        "run_id",
        "gps_latitude",
        "gps_longitude",
        "gps_speed",
    }
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if data["run_id"].nunique(dropna=False) != 1:
        raise ValueError("Trajectory data must contain exactly one run_id.")

    prepared_data = data.copy()
    prepared_data["time"] = pd.to_datetime(
        prepared_data["time"],
        utc=True,
        format="mixed",
    )
    prepared_data = prepared_data.sort_values("time").reset_index(drop=True)

    window_size = observation_count + prediction_count
    stop_index = start_index + window_size
    if stop_index > len(prepared_data):
        raise ValueError(
            "Not enough trajectory rows for the requested observation and "
            "prediction window."
        )

    window_data = prepared_data.iloc[start_index:stop_index]
    timestamps = pd.DatetimeIndex(window_data["time"])
    time_seconds = (timestamps - timestamps[0]).total_seconds().to_numpy(dtype=float)
    if np.any(np.diff(time_seconds) <= 0):
        raise ValueError("Trajectory timestamps must be strictly increasing.")

    longitude = pd.to_numeric(
        window_data["gps_longitude"],
        errors="coerce",
    ).to_numpy(dtype=float)
    latitude = pd.to_numeric(
        window_data["gps_latitude"],
        errors="coerce",
    ).to_numpy(dtype=float)
    x_meters, y_meters = gps_to_local_coordinates(
        longitude,
        latitude,
        unit="m",
    )

    gps_speed_mps = pd.to_numeric(
        window_data["gps_speed"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if gps_speed_unit == "km/h":
        gps_speed_mps *= KILOMETERS_PER_HOUR_TO_METERS_PER_SECOND

    return TrajectoryWindowData(
        timestamps=timestamps,
        time_seconds=time_seconds,
        x_meters=x_meters,
        y_meters=y_meters,
        gps_speed_mps=gps_speed_mps,
        observation_count=observation_count,
    )


def estimate_initial_heading(x_meters, y_meters):
    """Estimate an initial heading from the first moving positions."""
    x_meters, y_meters = _validate_position_arrays(x_meters, y_meters)

    endpoint = min(4, len(x_meters) - 1)
    while endpoint > 0:
        delta_x = x_meters[endpoint] - x_meters[0]
        delta_y = y_meters[endpoint] - y_meters[0]
        if np.hypot(delta_x, delta_y) > 1e-8:
            return float(np.arctan2(delta_y, delta_x))
        endpoint -= 1
    raise ValueError("Observed positions do not contain enough movement for heading.")


def estimate_turn_direction(x_meters, y_meters):
    """Infer clockwise or counterclockwise motion from observed positions."""
    x_meters, y_meters = _validate_position_arrays(x_meters, y_meters)
    delta_x = np.diff(x_meters)
    delta_y = np.diff(y_meters)
    moving = np.hypot(delta_x, delta_y) > 1e-8
    headings = np.unwrap(np.arctan2(delta_y[moving], delta_x[moving]))
    if len(headings) < 2:
        raise ValueError("Observed positions do not contain enough movement for turn.")

    total_heading_change = float(headings[-1] - headings[0])
    return -1 if total_heading_change < 0 else 1


def estimate_positive_speed_median(speed_mps):
    """Return the median of finite positive speed values in meters per second."""
    speed_mps = np.asarray(speed_mps, dtype=float)
    if speed_mps.ndim != 1:
        raise ValueError("speed_mps must be one-dimensional.")
    valid_speeds = speed_mps[np.isfinite(speed_mps) & (speed_mps > 0)]
    if len(valid_speeds) == 0:
        raise ValueError("Observed gps_speed must contain positive finite values.")
    return float(np.median(valid_speeds))


def _validate_position_arrays(x_meters, y_meters):
    """Return matching finite one-dimensional position arrays."""
    x_meters = np.asarray(x_meters, dtype=float)
    y_meters = np.asarray(y_meters, dtype=float)
    if x_meters.ndim != 1 or y_meters.ndim != 1 or x_meters.shape != y_meters.shape:
        raise ValueError(
            "x_meters and y_meters must be matching one-dimensional arrays."
        )
    if not np.all(np.isfinite(x_meters)) or not np.all(np.isfinite(y_meters)):
        raise ValueError("x_meters and y_meters must contain finite values.")
    return x_meters, y_meters


def _validate_window_arguments(
    observation_count,
    prediction_count,
    start_index,
    gps_speed_unit,
):
    """Validate shared trajectory-window configuration."""
    integer_arguments = {
        "observation_count": (observation_count, 3),
        "prediction_count": (prediction_count, 1),
        "start_index": (start_index, 0),
    }
    for name, (value, minimum) in integer_arguments.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"{name} must be an integer greater than or equal to {minimum}."
            )

    if gps_speed_unit not in {"km/h", "m/s"}:
        raise ValueError("gps_speed_unit must be 'km/h' or 'm/s'.")
