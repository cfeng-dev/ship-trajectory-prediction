"""Trajectory Predictor owns the coordinate conversion used by its display."""

import numpy as np
from trajectory_predictor.coordinates import local_to_gps_coordinates


def test_local_origin_maps_to_its_gps_reference():
    longitude, latitude = local_to_gps_coordinates(
        np.array([0.0]),
        np.array([0.0]),
        reference_longitude=8.312259928385417,
        reference_latitude=47.05150553385417,
    )

    np.testing.assert_allclose(longitude, [8.312259928385417])
    np.testing.assert_allclose(latitude, [47.05150553385417])
