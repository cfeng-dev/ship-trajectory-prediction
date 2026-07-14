"""Basic tests for the deterministic simulation core."""

import numpy as np

from ship_trajectory_prediction.simulation.core import (
    ShipSimulator,
    simulate_straight_trajectory,
)


def test_straight_trajectory_moves_in_heading_direction():
    """A zero heading should move the ship only along the positive x-axis."""
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


def test_simulator_stores_speed_for_each_step():
    """Stored speed should reflect speed changes during a simulation."""
    simulator = ShipSimulator(v=2.0)

    simulator.step(omega=0.0, motor_running=True)
    simulator.v = 3.0
    simulator.step(omega=0.0, motor_running=True)

    assert simulator.v_all == [2.0, 3.0]
