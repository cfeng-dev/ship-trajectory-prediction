"""Tests for coordinate conversions used by the simulator."""

import numpy as np
import pytest

from ship_trajectory_prediction.coordinates import (
    local_to_gps_coordinates,
)


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


def test_meter_and_kilometer_inputs_produce_same_gps_position():
    """Equivalent distances in meters and kilometers should produce equal GPS."""
    gps_from_meters = local_to_gps_coordinates(
        [1000.0],
        [500.0],
        reference_longitude=8.3122,
        reference_latitude=47.0515,
        unit="m",
    )
    gps_from_kilometers = local_to_gps_coordinates(
        [1.0],
        [0.5],
        reference_longitude=8.3122,
        reference_latitude=47.0515,
        unit="km",
    )

    np.testing.assert_allclose(gps_from_meters, gps_from_kilometers)


def test_coordinate_conversion_rejects_mismatched_arrays():
    """Local x and y arrays should contain the same number of values."""
    with pytest.raises(ValueError, match="same length"):
        local_to_gps_coordinates(
            [0.0, 1.0],
            [0.0],
            reference_longitude=8.3122,
            reference_latitude=47.0515,
        )
