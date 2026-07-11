"""Basic tests for the deterministic simulation core."""

import numpy as np

from ship_trajectory_prediction.simulation.core import (
    simulate_straight_trajectory,
)


def test_straight_trajectory_moves_in_heading_direction():
    time = np.array([0.0, 1.0, 2.0])

    x_position, y_position = simulate_straight_trajectory(
        time,
        x0=0.0,
        y0=0.0,
        v=2.0,
        theta=0.0,
    )

    np.testing.assert_allclose(x_position, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(y_position, [0.0, 0.0, 0.0], atol=1e-12)
