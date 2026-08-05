"""Shared coordinate and movement calculations."""

import numpy as np

EARTH_RADIUS_KM = 6371.0088
METERS_PER_KILOMETER = 1000
SECONDS_PER_HOUR = 3600


def gps_to_local_coordinates(longitude, latitude, unit="m"):
    """Convert GPS coordinates to local east/north coordinates.

    The first GPS position defines the local origin. The equirectangular
    approximation is appropriate for the relatively short trajectories used
    in this project:

    ``x = R * (longitude - longitude_0) * cos(mean_latitude)``
    ``y = R * (latitude - latitude_0)``

    Parameters
    ----------
    longitude, latitude : array-like
        GPS coordinates in degrees.
    unit : {"m", "km"}, optional
        Output distance unit.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Local x (east) and y (north) coordinates.
    """
    if unit not in {"m", "km"}:
        raise ValueError("unit must be 'm' or 'km'.")

    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    _validate_coordinate_arrays(longitude, latitude)

    longitude_radians = np.radians(longitude)
    latitude_radians = np.radians(latitude)
    reference_longitude = longitude_radians[0]
    reference_latitude = latitude_radians[0]
    mean_latitude = (latitude_radians + reference_latitude) / 2

    x_coordinates = (
        EARTH_RADIUS_KM
        * (longitude_radians - reference_longitude)
        * np.cos(mean_latitude)
    )
    y_coordinates = EARTH_RADIUS_KM * (latitude_radians - reference_latitude)

    if unit == "m":
        x_coordinates *= METERS_PER_KILOMETER
        y_coordinates *= METERS_PER_KILOMETER

    return x_coordinates, y_coordinates


def local_to_gps_coordinates(
    x_coordinates,
    y_coordinates,
    reference_longitude,
    reference_latitude,
    unit="m",
):
    """Convert local east/north coordinates to GPS coordinates.

    The reference GPS position defines the local origin ``(0, 0)``. This is
    the inverse of the equirectangular approximation used by
    :func:`gps_to_local_coordinates` and is suitable for the short distances
    used in this project.

    Parameters
    ----------
    x_coordinates, y_coordinates : array-like
        Local east and north coordinates.
    reference_longitude, reference_latitude : float
        GPS coordinates of the local origin in degrees.
    unit : {"m", "km"}, optional
        Unit of the local coordinates.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Longitude and latitude in degrees.
    """
    if unit not in {"m", "km"}:
        raise ValueError("unit must be 'm' or 'km'.")

    x_coordinates = np.asarray(x_coordinates, dtype=float)
    y_coordinates = np.asarray(y_coordinates, dtype=float)
    _validate_coordinate_arrays(x_coordinates, y_coordinates)

    if not np.isfinite(reference_longitude) or not np.isfinite(reference_latitude):
        raise ValueError("reference GPS coordinates must be finite.")
    if not -90 < reference_latitude < 90:
        raise ValueError("reference_latitude must be between -90 and 90 degrees.")

    if unit == "m":
        x_coordinates = x_coordinates / METERS_PER_KILOMETER
        y_coordinates = y_coordinates / METERS_PER_KILOMETER

    reference_longitude_radians = np.radians(reference_longitude)
    reference_latitude_radians = np.radians(reference_latitude)
    latitude_radians = reference_latitude_radians + (y_coordinates / EARTH_RADIUS_KM)
    mean_latitude = (latitude_radians + reference_latitude_radians) / 2
    longitude_radians = reference_longitude_radians + (
        x_coordinates / (EARTH_RADIUS_KM * np.cos(mean_latitude))
    )

    return np.degrees(longitude_radians), np.degrees(latitude_radians)


def calculate_gps_distances(longitude, latitude):
    """Calculate great-circle distances between consecutive GPS positions.

    The haversine formula is used and distances are returned in meters.
    """
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    _validate_coordinate_arrays(longitude, latitude)

    longitude_radians = np.radians(longitude)
    latitude_radians = np.radians(latitude)
    delta_longitude = np.diff(longitude_radians)
    delta_latitude = np.diff(latitude_radians)
    haversine_value = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(latitude_radians[:-1])
        * np.cos(latitude_radians[1:])
        * np.sin(delta_longitude / 2) ** 2
    )
    # Floating-point rounding can otherwise move the value just outside [0, 1].
    haversine_value = np.clip(haversine_value, 0, 1)
    angular_distance = 2 * np.arctan2(
        np.sqrt(haversine_value),
        np.sqrt(1 - haversine_value),
    )

    return EARTH_RADIUS_KM * METERS_PER_KILOMETER * angular_distance


def calculate_speed_from_gps(data, unit="km/h"):
    """Calculate ship speed from GPS positions and timestamps.

    Actual timestamp differences are used, so missing or irregular samples are
    handled without assuming the usual 10-second measurement interval. The
    first value and values following a non-positive interval are ``NaN``.
    """
    if unit not in {"m/s", "km/h"}:
        raise ValueError("unit must be 'm/s' or 'km/h'.")

    distance_meters = calculate_gps_distances(
        data["gps_longitude"].to_numpy(),
        data["gps_latitude"].to_numpy(),
    )
    time_seconds = data["time"].diff().dt.total_seconds().to_numpy()[1:]
    speed_mps = np.full(len(data), np.nan)
    speed_mps[1:] = np.divide(
        distance_meters,
        time_seconds,
        out=np.full_like(distance_meters, np.nan),
        where=time_seconds > 0,
    )

    if unit == "km/h":
        return speed_mps * SECONDS_PER_HOUR / METERS_PER_KILOMETER

    return speed_mps


def calculate_signed_curvature_from_gps(
    data,
    *,
    min_displacement_m=1.0,
    max_time_gap_s=15.0,
):
    """Calculate discrete signed trajectory curvature from GPS positions.

    Curvature is evaluated at every interior position from the wrapped course
    change between the adjacent GPS segments divided by their mean length.
    Positive values represent counterclockwise turns and negative values
    clockwise turns. The first, last, and invalid samples are returned as
    ``NaN`` because curvature is not defined there.

    Parameters
    ----------
    data : pandas.DataFrame
        Trajectory data containing timestamps, longitude, and latitude.
    min_displacement_m : float, optional
        Minimum length required for each adjacent GPS segment. This suppresses
        unstable course estimates during standstill or GPS jitter.
    max_time_gap_s : float or None, optional
        Largest accepted duration of either adjacent segment. ``None`` applies
        no upper time-gap limit.

    Returns
    -------
    numpy.ndarray
        Signed curvature in inverse meters, aligned with the input rows.
    """
    if not np.isfinite(min_displacement_m) or min_displacement_m <= 0:
        raise ValueError("min_displacement_m must be a positive finite number.")
    if max_time_gap_s is not None and (
        not np.isfinite(max_time_gap_s) or max_time_gap_s <= 0
    ):
        raise ValueError("max_time_gap_s must be a positive finite number or None.")

    sample_count = len(data)
    curvature = np.full(sample_count, np.nan)
    if sample_count < 3:
        return curvature

    x_coordinates, y_coordinates = gps_to_local_coordinates(
        data["gps_longitude"].to_numpy(),
        data["gps_latitude"].to_numpy(),
        unit="m",
    )
    delta_x = np.diff(x_coordinates)
    delta_y = np.diff(y_coordinates)
    segment_length = np.hypot(delta_x, delta_y)
    segment_duration = data["time"].diff().dt.total_seconds().to_numpy()[1:]
    course = np.arctan2(delta_y, delta_x)
    course_change = (np.diff(course) + np.pi) % (2 * np.pi) - np.pi
    mean_segment_length = 0.5 * (segment_length[:-1] + segment_length[1:])

    valid = (
        np.isfinite(course_change)
        & np.isfinite(mean_segment_length)
        & (segment_length[:-1] >= min_displacement_m)
        & (segment_length[1:] >= min_displacement_m)
        & np.isfinite(segment_duration[:-1])
        & np.isfinite(segment_duration[1:])
        & (segment_duration[:-1] > 0)
        & (segment_duration[1:] > 0)
    )
    if max_time_gap_s is not None:
        valid &= (segment_duration[:-1] <= max_time_gap_s) & (
            segment_duration[1:] <= max_time_gap_s
        )

    interior_curvature = np.full(sample_count - 2, np.nan)
    interior_curvature[valid] = course_change[valid] / mean_segment_length[valid]
    curvature[1:-1] = interior_curvature
    return curvature


def calculate_signed_turn_rate_from_gps(
    data,
    *,
    min_displacement_m=1.0,
    max_time_gap_s=15.0,
):
    """Calculate GPS-derived signed course rate in radians per second.

    The rate is evaluated at every interior position from the wrapped course
    change between adjacent GPS segments divided by the elapsed time between
    their midpoints. Positive values represent counterclockwise turns and
    negative values clockwise turns. The first, last, and invalid samples are
    returned as ``NaN``.

    Parameters
    ----------
    data : pandas.DataFrame
        Trajectory data containing timestamps, longitude, and latitude.
    min_displacement_m : float, optional
        Minimum length required for each adjacent GPS segment. This suppresses
        unstable course estimates during standstill or GPS jitter.
    max_time_gap_s : float or None, optional
        Largest accepted duration of either adjacent segment. ``None`` applies
        no upper time-gap limit.

    Returns
    -------
    numpy.ndarray
        Signed turn rate in radians per second, aligned with the input rows.
    """
    if not np.isfinite(min_displacement_m) or min_displacement_m <= 0:
        raise ValueError("min_displacement_m must be a positive finite number.")
    if max_time_gap_s is not None and (
        not np.isfinite(max_time_gap_s) or max_time_gap_s <= 0
    ):
        raise ValueError("max_time_gap_s must be a positive finite number or None.")

    sample_count = len(data)
    turn_rate = np.full(sample_count, np.nan)
    if sample_count < 3:
        return turn_rate

    x_coordinates, y_coordinates = gps_to_local_coordinates(
        data["gps_longitude"].to_numpy(),
        data["gps_latitude"].to_numpy(),
        unit="m",
    )
    delta_x = np.diff(x_coordinates)
    delta_y = np.diff(y_coordinates)
    segment_length = np.hypot(delta_x, delta_y)
    segment_duration = data["time"].diff().dt.total_seconds().to_numpy()[1:]
    course = np.arctan2(delta_y, delta_x)
    course_change = (np.diff(course) + np.pi) % (2 * np.pi) - np.pi
    midpoint_time_difference = 0.5 * (segment_duration[:-1] + segment_duration[1:])

    valid = (
        np.isfinite(course_change)
        & np.isfinite(midpoint_time_difference)
        & (midpoint_time_difference > 0)
        & (segment_length[:-1] >= min_displacement_m)
        & (segment_length[1:] >= min_displacement_m)
        & np.isfinite(segment_duration[:-1])
        & np.isfinite(segment_duration[1:])
        & (segment_duration[:-1] > 0)
        & (segment_duration[1:] > 0)
    )
    if max_time_gap_s is not None:
        valid &= (segment_duration[:-1] <= max_time_gap_s) & (
            segment_duration[1:] <= max_time_gap_s
        )

    interior_turn_rate = np.full(sample_count - 2, np.nan)
    interior_turn_rate[valid] = course_change[valid] / midpoint_time_difference[valid]
    turn_rate[1:-1] = interior_turn_rate
    return turn_rate


def _validate_coordinate_arrays(longitude, latitude):
    """Validate paired, non-empty one-dimensional coordinate arrays."""
    if longitude.ndim != 1 or latitude.ndim != 1:
        raise ValueError("longitude and latitude must be one-dimensional.")
    if len(longitude) != len(latitude):
        raise ValueError("longitude and latitude must have the same length.")
    if len(longitude) == 0:
        raise ValueError("longitude and latitude must not be empty.")
    if not np.all(np.isfinite(longitude)) or not np.all(np.isfinite(latitude)):
        raise ValueError("longitude and latitude must contain finite values.")
