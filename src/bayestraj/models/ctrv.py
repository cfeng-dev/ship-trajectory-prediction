"""Constant-turn-rate-and-velocity (CTRV) state and motion dynamics."""

from dataclasses import dataclass
from math import cos, isfinite, sin

import numpy as np

SMALL_TURN_ANGLE = 1e-4
SPEED_STATE_LOWER_MPS = 0.0
PROCESS_REFERENCE_INTERVAL_SECONDS = 10.0

STATE_X_INDEX = 0
STATE_Y_INDEX = 1
STATE_SPEED_INDEX = 2
STATE_HEADING_INDEX = 3
STATE_TURN_RATE_INDEX = 4
STATE_COUNT = 5


@dataclass(frozen=True, slots=True)
class CTRVState:
    """State of a deterministic planar CTRV motion model.

    Positions are expressed in meters, speed in meters per second, heading in
    radians, and signed turn rate in radians per second. Positive turn rates
    turn counterclockwise and negative turn rates clockwise.
    """

    x: float
    y: float
    speed: float
    heading: float
    turn_rate: float

    def __post_init__(self) -> None:
        """Normalize numeric values and reject non-physical states."""
        for field_name in ("x", "y", "speed", "heading", "turn_rate"):
            value = _finite_float(getattr(self, field_name), name=field_name)
            object.__setattr__(self, field_name, value)

        if self.speed < 0:
            raise ValueError("speed must be non-negative.")


def ctrv_step(state: CTRVState, dt: float) -> CTRVState:
    """Advance one deterministic CTRV state by ``dt`` seconds.

    For a non-zero turn rate ``omega``, the exact position transition is

    ``x_next = x + speed / omega * (sin(heading_next) - sin(heading))``
    ``y_next = y + speed / omega * (cos(heading) - cos(heading_next))``

    where ``heading_next = heading + omega * dt``. Near zero turn angle, an
    algebraically equivalent midpoint/sinc form with a Taylor approximation is
    used to avoid cancellation and division by a very small turn rate. Heading
    remains unwrapped so repeated propagation is continuous.
    """
    if not isinstance(state, CTRVState):
        raise TypeError("state must be a CTRVState instance.")
    dt = _positive_finite_float(dt, name="dt")

    turn_angle = state.turn_rate * dt
    next_heading = state.heading + turn_angle

    if abs(turn_angle) < SMALL_TURN_ANGLE:
        half_turn = 0.5 * turn_angle
        distance = state.speed * dt * _stable_sinc(half_turn)
        midpoint_heading = state.heading + half_turn
        next_x = state.x + distance * cos(midpoint_heading)
        next_y = state.y + distance * sin(midpoint_heading)
    else:
        radius = state.speed / state.turn_rate
        next_x = state.x + radius * (sin(next_heading) - sin(state.heading))
        next_y = state.y + radius * (cos(state.heading) - cos(next_heading))

    return CTRVState(
        x=next_x,
        y=next_y,
        speed=state.speed,
        heading=next_heading,
        turn_rate=state.turn_rate,
    )


def predict_ctrv(
    initial_state: CTRVState,
    dt: float,
    steps: int,
) -> tuple[CTRVState, ...]:
    """Return ``steps`` future CTRV states at a constant time interval.

    The returned tuple contains future states only; the input state is not
    included and is never modified.
    """
    if not isinstance(initial_state, CTRVState):
        raise TypeError("initial_state must be a CTRVState instance.")
    dt = _positive_finite_float(dt, name="dt")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be an integer greater than or equal to 1.")

    states = []
    current_state = initial_state
    for _ in range(steps):
        current_state = ctrv_step(current_state, dt)
        states.append(current_state)
    return tuple(states)


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


def _stable_sinc(value: float) -> float:
    """Return a fourth-order approximation of sin(value) / value."""
    value_squared = value * value
    return 1 - value_squared / 6 + value_squared * value_squared / 120


def _positive_finite_float(value: float, *, name: str) -> float:
    """Return one positive finite float or raise a clear error."""
    try:
        value = _finite_float(value, name=name)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive finite number.") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def _finite_float(value: float, *, name: str) -> float:
    """Return one finite float or raise a clear error."""
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a finite number.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return value
