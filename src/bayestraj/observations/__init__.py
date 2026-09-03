"""Recorded ship observations, preprocessing, and exploratory plotting."""

from bayestraj.observations.coordinates import (
    calculate_gps_distances,
    calculate_signed_curvature_from_gps,
    calculate_signed_turn_rate_from_gps,
    calculate_speed_from_gps,
    gps_to_local_coordinates,
    local_to_gps_coordinates,
)
from bayestraj.observations.io import (
    read_ship_data,
    resample_trajectory_data,
)
from bayestraj.observations.paths import DATA_ROOT, data_path
from bayestraj.observations.position import (
    PositionObservations,
    resolve_position_observations,
    simulate_position_observations,
)
from bayestraj.observations.window import (
    DEFAULT_GPS_SPEED_UNIT,
    DEFAULT_MAX_TIME_GAP_SECONDS,
    TrajectoryWindowData,
    prepare_trajectory_window,
)

__all__ = [
    "DEFAULT_GPS_SPEED_UNIT",
    "DEFAULT_MAX_TIME_GAP_SECONDS",
    "DATA_ROOT",
    "PositionObservations",
    "TrajectoryWindowData",
    "calculate_gps_distances",
    "calculate_signed_curvature_from_gps",
    "calculate_signed_turn_rate_from_gps",
    "calculate_speed_from_gps",
    "data_path",
    "gps_to_local_coordinates",
    "local_to_gps_coordinates",
    "prepare_trajectory_window",
    "read_ship_data",
    "resolve_position_observations",
    "resample_trajectory_data",
    "simulate_position_observations",
]
