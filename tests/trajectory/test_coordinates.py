"""Tests for GPS coordinate, distance, and speed calculations."""

import numpy as np
import pandas as pd
import pytest

from ship_trajectory_prediction.trajectory.coordinates import (
    calculate_gps_distances,
    calculate_speed_from_gps,
    gps_to_local_coordinates,
)


def test_first_local_coordinate_is_origin():
    """Local coordinates should start at the first GPS position."""
    longitude = [8.3122, 8.3132]
    latitude = [47.0515, 47.0525]

    x_coordinates, y_coordinates = gps_to_local_coordinates(
        longitude,
        latitude,
        unit="m",
    )

    assert x_coordinates[0] == pytest.approx(0.0)
    assert y_coordinates[0] == pytest.approx(0.0)
    assert x_coordinates[1] > 0
    assert y_coordinates[1] > 0


def test_meter_and_kilometer_coordinates_are_consistent():
    """Meter coordinates should be 1000 times kilometer coordinates."""
    longitude = [8.3122, 8.3132]
    latitude = [47.0515, 47.0525]

    x_meters, y_meters = gps_to_local_coordinates(longitude, latitude, unit="m")
    x_kilometers, y_kilometers = gps_to_local_coordinates(
        longitude,
        latitude,
        unit="km",
    )

    np.testing.assert_allclose(x_meters, x_kilometers * 1000)
    np.testing.assert_allclose(y_meters, y_kilometers * 1000)


def test_identical_positions_have_zero_distance():
    """Two identical GPS positions should have no distance between them."""
    distances = calculate_gps_distances(
        longitude=[8.3122, 8.3122],
        latitude=[47.0515, 47.0515],
    )

    np.testing.assert_allclose(distances, [0.0])


def test_one_latitude_degree_has_expected_distance_at_equator():
    """One latitude degree at the equator should be about 111.2 km."""
    distances = calculate_gps_distances(
        longitude=[0.0, 0.0],
        latitude=[0.0, 1.0],
    )

    assert distances[0] == pytest.approx(111_195, rel=1e-3)


def test_speed_uses_actual_timestamp_difference():
    """Speed should equal traveled distance divided by the real time interval."""
    data = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:20Z"]),
            "gps_longitude": [0.0, 0.0],
            # About 20 meters north at the equator.
            "gps_latitude": [0.0, 20 / 111_195],
        }
    )

    speed_mps = calculate_speed_from_gps(data, unit="m/s")

    assert np.isnan(speed_mps[0])
    assert speed_mps[1] == pytest.approx(1.0, rel=1e-3)


def test_non_positive_time_interval_produces_nan_speed():
    """A zero or negative time interval should not produce a finite speed."""
    data = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"]),
            "gps_longitude": [8.0, 8.001],
            "gps_latitude": [47.0, 47.001],
        }
    )

    speed = calculate_speed_from_gps(data)

    assert np.isnan(speed[1])


@pytest.mark.parametrize("unit", ["miles", "knots", "degrees"])
def test_invalid_coordinate_unit_is_rejected(unit):
    """Unsupported coordinate units should raise a clear error."""
    with pytest.raises(ValueError, match="unit must be"):
        gps_to_local_coordinates([8.0], [47.0], unit=unit)
