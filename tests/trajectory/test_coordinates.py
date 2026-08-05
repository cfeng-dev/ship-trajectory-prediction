"""Tests for GPS coordinate, distance, and speed calculations."""

import numpy as np
import pandas as pd
import pytest

from ship_trajectory_prediction.coordinates import (
    calculate_gps_distances,
    calculate_signed_curvature_from_gps,
    calculate_signed_turn_rate_from_gps,
    calculate_speed_from_gps,
    gps_to_local_coordinates,
    local_to_gps_coordinates,
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


def test_local_coordinates_convert_back_to_gps():
    """Converting GPS coordinates to local meters and back should be reversible."""
    longitude = np.array([8.3122, 8.3132])
    latitude = np.array([47.0515, 47.0525])

    x_coordinates, y_coordinates = gps_to_local_coordinates(
        longitude,
        latitude,
        unit="m",
    )
    converted_longitude, converted_latitude = local_to_gps_coordinates(
        x_coordinates,
        y_coordinates,
        reference_longitude=longitude[0],
        reference_latitude=latitude[0],
        unit="m",
    )

    np.testing.assert_allclose(converted_longitude, longitude)
    np.testing.assert_allclose(converted_latitude, latitude)


def test_local_origin_maps_to_reference_gps_position():
    """The local origin should represent the configured GPS reference point."""
    longitude, latitude = local_to_gps_coordinates(
        [0.0],
        [0.0],
        reference_longitude=8.3122,
        reference_latitude=47.0515,
    )

    assert longitude[0] == pytest.approx(8.3122)
    assert latitude[0] == pytest.approx(47.0515)


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


def test_straight_trajectory_has_zero_signed_curvature():
    """A straight sequence of equal segments should have zero curvature."""
    data = _trajectory_data_from_local_coordinates([0, 10, 20], [0, 0, 0])

    curvature = calculate_signed_curvature_from_gps(data)

    assert np.isnan(curvature[0])
    assert curvature[1] == pytest.approx(0.0, abs=1e-10)
    assert np.isnan(curvature[2])


@pytest.mark.parametrize(("final_y", "expected_sign"), [(10, 1), (-10, -1)])
def test_signed_curvature_preserves_turn_direction(final_y, expected_sign):
    """Left and right quarter turns should have opposite curvature signs."""
    data = _trajectory_data_from_local_coordinates(
        [0, 10, 10],
        [0, 0, final_y],
    )

    curvature = calculate_signed_curvature_from_gps(data)

    assert np.sign(curvature[1]) == expected_sign
    assert abs(curvature[1]) == pytest.approx(np.pi / 20, rel=1e-5)


def test_curvature_ignores_short_gps_segments():
    """Small displacements should not create unstable curvature estimates."""
    data = _trajectory_data_from_local_coordinates([0, 1, 1], [0, 0, 1])

    curvature = calculate_signed_curvature_from_gps(
        data,
        min_displacement_m=2.0,
    )

    assert np.all(np.isnan(curvature))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"min_displacement_m": 0.0}, "min_displacement_m"),
        ({"max_time_gap_s": 0.0}, "max_time_gap_s"),
    ],
)
def test_curvature_rejects_invalid_filter_settings(options, message):
    """Curvature filters should require positive finite values."""
    data = _trajectory_data_from_local_coordinates([0, 10, 20], [0, 0, 0])

    with pytest.raises(ValueError, match=message):
        calculate_signed_curvature_from_gps(data, **options)


@pytest.mark.parametrize(("final_y", "expected_sign"), [(10, 1), (-10, -1)])
def test_signed_turn_rate_uses_course_change_over_actual_time(final_y, expected_sign):
    """A quarter turn should use adjacent segment midpoint times."""
    data = _trajectory_data_from_local_coordinates(
        [0, 10, 10],
        [0, 0, final_y],
    )
    data["time"] = pd.to_datetime(
        ["2026-01-01T00:00:00Z", "2026-01-01T00:00:10Z", "2026-01-01T00:00:30Z"]
    )

    turn_rate = calculate_signed_turn_rate_from_gps(data, max_time_gap_s=30.0)

    assert np.isnan(turn_rate[0])
    assert np.sign(turn_rate[1]) == expected_sign
    assert abs(turn_rate[1]) == pytest.approx(np.pi / 30, rel=1e-5)
    assert np.isnan(turn_rate[2])


def test_turn_rate_ignores_short_gps_segments():
    """Small displacements should not create unstable turn-rate estimates."""
    data = _trajectory_data_from_local_coordinates([0, 1, 1], [0, 0, 1])

    turn_rate = calculate_signed_turn_rate_from_gps(
        data,
        min_displacement_m=2.0,
    )

    assert np.all(np.isnan(turn_rate))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"min_displacement_m": 0.0}, "min_displacement_m"),
        ({"max_time_gap_s": 0.0}, "max_time_gap_s"),
    ],
)
def test_turn_rate_rejects_invalid_filter_settings(options, message):
    """Turn-rate filters should require positive finite values."""
    data = _trajectory_data_from_local_coordinates([0, 10, 20], [0, 0, 0])

    with pytest.raises(ValueError, match=message):
        calculate_signed_turn_rate_from_gps(data, **options)


@pytest.mark.parametrize("unit", ["miles", "knots", "degrees"])
def test_invalid_coordinate_unit_is_rejected(unit):
    """Unsupported coordinate units should raise a clear error."""
    with pytest.raises(ValueError, match="unit must be"):
        gps_to_local_coordinates([8.0], [47.0], unit=unit)


def _trajectory_data_from_local_coordinates(x_coordinates, y_coordinates):
    """Create timestamped GPS test data from local meter coordinates."""
    longitude, latitude = local_to_gps_coordinates(
        x_coordinates,
        y_coordinates,
        reference_longitude=8.3122,
        reference_latitude=47.0515,
        unit="m",
    )
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=len(longitude), freq="10s"),
            "gps_longitude": longitude,
            "gps_latitude": latitude,
        }
    )
