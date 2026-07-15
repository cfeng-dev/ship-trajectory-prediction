"""Coordinate conversions used by the ship trajectory simulator.

The implementation is intentionally local so simulation remains independent
from the trajectory package.
"""

import numpy as np

EARTH_RADIUS_KM = 6371.0088
METERS_PER_KILOMETER = 1000


def local_to_gps_coordinates(
    x_coordinates,
    y_coordinates,
    reference_longitude,
    reference_latitude,
    unit="m",
):
    """Convert local east/north coordinates to GPS coordinates.

    The reference GPS position defines the local origin ``(0, 0)``. The
    equirectangular approximation is suitable for the short distances used by
    the simulator.

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


def _validate_coordinate_arrays(x_coordinates, y_coordinates):
    """Validate paired, non-empty one-dimensional coordinate arrays."""
    if x_coordinates.ndim != 1 or y_coordinates.ndim != 1:
        raise ValueError("x_coordinates and y_coordinates must be one-dimensional.")
    if len(x_coordinates) != len(y_coordinates):
        raise ValueError("x_coordinates and y_coordinates must have the same length.")
    if len(x_coordinates) == 0:
        raise ValueError("x_coordinates and y_coordinates must not be empty.")
    if not np.all(np.isfinite(x_coordinates)) or not np.all(np.isfinite(y_coordinates)):
        raise ValueError("x_coordinates and y_coordinates must contain finite values.")
