"""Vectorized CTRV state dynamics shared by online inference methods."""

from __future__ import annotations

import numpy as np

SPEED_STATE_LOWER_MPS = 0.0
PROCESS_REFERENCE_INTERVAL_SECONDS = 10.0

STATE_X_INDEX = 0
STATE_Y_INDEX = 1
STATE_SPEED_INDEX = 2
STATE_HEADING_INDEX = 3
STATE_TURN_RATE_INDEX = 4
STATE_COUNT = 5


def displacements(
    speed: np.ndarray,
    heading: np.ndarray,
    turn_rate: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable vectorized CTRV x/y displacements."""
    half_turn = 0.5 * turn_rate * dt
    distance = speed * dt * np.sinc(half_turn / np.pi)
    midpoint_heading = heading + half_turn
    return (
        distance * np.cos(midpoint_heading),
        distance * np.sin(midpoint_heading),
    )


def transition_states(states: np.ndarray, dt: float) -> np.ndarray:
    """Advance position and heading while retaining speed and turn rate."""
    transitioned = states.copy()
    displacement_x, displacement_y = displacements(
        states[:, STATE_SPEED_INDEX],
        states[:, STATE_HEADING_INDEX],
        states[:, STATE_TURN_RATE_INDEX],
        dt,
    )
    transitioned[:, STATE_X_INDEX] += displacement_x
    transitioned[:, STATE_Y_INDEX] += displacement_y
    transitioned[:, STATE_HEADING_INDEX] = wrap_angles(
        states[:, STATE_HEADING_INDEX] + states[:, STATE_TURN_RATE_INDEX] * dt
    )
    return transitioned


def transition_jacobians(states: np.ndarray, dt: float) -> np.ndarray:
    """Return one local CTRV transition Jacobian for every state mean."""
    particle_count = states.shape[0]
    jacobians = np.broadcast_to(
        np.eye(STATE_COUNT),
        (particle_count, STATE_COUNT, STATE_COUNT),
    ).copy()
    speed = states[:, STATE_SPEED_INDEX]
    heading = states[:, STATE_HEADING_INDEX]
    turn_rate = states[:, STATE_TURN_RATE_INDEX]
    half_turn = 0.5 * turn_rate * dt
    distance_per_speed = dt * np.sinc(half_turn / np.pi)
    midpoint_heading = heading + half_turn
    cosine = np.cos(midpoint_heading)
    sine = np.sin(midpoint_heading)
    displacement_x = speed * distance_per_speed * cosine
    displacement_y = speed * distance_per_speed * sine
    jacobians[:, STATE_X_INDEX, STATE_SPEED_INDEX] = distance_per_speed * cosine
    jacobians[:, STATE_Y_INDEX, STATE_SPEED_INDEX] = distance_per_speed * sine
    jacobians[:, STATE_X_INDEX, STATE_HEADING_INDEX] = -displacement_y
    jacobians[:, STATE_Y_INDEX, STATE_HEADING_INDEX] = displacement_x

    difference_step = 1e-6
    displacement_x_plus, displacement_y_plus = displacements(
        speed,
        heading,
        turn_rate + difference_step,
        dt,
    )
    displacement_x_minus, displacement_y_minus = displacements(
        speed,
        heading,
        turn_rate - difference_step,
        dt,
    )
    jacobians[:, STATE_X_INDEX, STATE_TURN_RATE_INDEX] = (
        displacement_x_plus - displacement_x_minus
    ) / (2.0 * difference_step)
    jacobians[:, STATE_Y_INDEX, STATE_TURN_RATE_INDEX] = (
        displacement_y_plus - displacement_y_minus
    ) / (2.0 * difference_step)
    jacobians[:, STATE_HEADING_INDEX, STATE_TURN_RATE_INDEX] = dt
    return jacobians


def normalize_states(
    states: np.ndarray,
    covariances: np.ndarray | None = None,
) -> None:
    """Reflect negative speeds at zero without changing vessel heading."""
    negative_speed = states[:, STATE_SPEED_INDEX] < 0.0
    states[negative_speed, STATE_SPEED_INDEX] *= -1.0
    if covariances is not None and np.any(negative_speed):
        covariances[negative_speed, STATE_SPEED_INDEX, :] *= -1.0
        covariances[negative_speed, :, STATE_SPEED_INDEX] *= -1.0
    states[:, STATE_SPEED_INDEX] = np.maximum(
        states[:, STATE_SPEED_INDEX],
        SPEED_STATE_LOWER_MPS,
    )
    states[:, STATE_HEADING_INDEX] = wrap_angles(states[:, STATE_HEADING_INDEX])


def wrap_angles(values: np.ndarray) -> np.ndarray:
    """Wrap radians to the closed-open interval [-pi, pi)."""
    return (values + np.pi) % (2.0 * np.pi) - np.pi
