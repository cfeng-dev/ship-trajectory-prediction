"""Minimal CTRV transition owned by the standalone simulation package."""

from dataclasses import dataclass
from math import cos, isfinite, sin

SMALL_TURN_ANGLE = 1e-4


@dataclass(frozen=True, slots=True)
class CTRVState:
    """Planar position, speed, heading, and signed turn rate."""

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


def as_ctrv_state(state) -> CTRVState:
    """Copy any state-shaped object into the simulation-owned state type."""
    if isinstance(state, CTRVState):
        return state

    try:
        return CTRVState(
            x=state.x,
            y=state.y,
            speed=state.speed,
            heading=state.heading,
            turn_rate=state.turn_rate,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("state must provide a valid CTRV state.") from error


def ctrv_step(state: CTRVState, dt: float) -> CTRVState:
    """Advance one deterministic CTRV state by ``dt`` seconds."""
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
