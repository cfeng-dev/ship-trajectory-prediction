"""Tests for the deterministic CTRV motion model."""

from math import cos, pi, sin

import pytest

from ship_trajectory_prediction.models.ctrv import (
    SMALL_TURN_ANGLE,
    CTRVState,
    ctrv_step,
    predict_ctrv,
)


def test_straight_motion_with_zero_turn_rate():
    """Zero turn rate should produce constant-heading straight motion."""
    state = CTRVState(
        x=2.0,
        y=-3.0,
        speed=4.0,
        heading=pi / 6,
        turn_rate=0.0,
    )

    result = ctrv_step(state, dt=2.5)

    distance = state.speed * 2.5
    assert result.x == pytest.approx(state.x + distance * cos(state.heading))
    assert result.y == pytest.approx(state.y + distance * sin(state.heading))
    assert result.speed == state.speed
    assert result.heading == state.heading
    assert result.turn_rate == state.turn_rate


def test_uniform_quarter_circle_uses_exact_ctrv_equations():
    """A quarter turn should reach the analytically known circle position."""
    state = CTRVState(
        x=0.0,
        y=0.0,
        speed=2.0,
        heading=0.0,
        turn_rate=0.5,
    )

    result = ctrv_step(state, dt=pi)

    radius = state.speed / state.turn_rate
    assert result.x == pytest.approx(radius)
    assert result.y == pytest.approx(radius)
    assert result.heading == pytest.approx(pi / 2)
    assert result.speed == state.speed
    assert result.turn_rate == state.turn_rate


def test_signed_turn_rate_controls_turn_direction():
    """Positive rates should turn left and negative rates right."""
    left = ctrv_step(
        CTRVState(0.0, 0.0, speed=4.0, heading=0.0, turn_rate=0.2),
        dt=1.0,
    )
    right = ctrv_step(
        CTRVState(0.0, 0.0, speed=4.0, heading=0.0, turn_rate=-0.2),
        dt=1.0,
    )

    assert left.x == pytest.approx(right.x)
    assert left.y == pytest.approx(-right.y)
    assert left.y > 0
    assert right.y < 0
    assert left.heading > 0
    assert right.heading < 0


def test_transition_is_continuous_near_zero_turn_rate():
    """The stable approximation should meet the exact curved transition."""
    dt = 1.0
    turn_rate_below = SMALL_TURN_ANGLE * (1 - 1e-6) / dt
    turn_rate_above = SMALL_TURN_ANGLE * (1 + 1e-6) / dt
    common_values = {
        "x": 1.0,
        "y": -2.0,
        "speed": 5.0,
        "heading": 0.7,
    }

    straight = ctrv_step(CTRVState(**common_values, turn_rate=0.0), dt)
    almost_straight = ctrv_step(
        CTRVState(**common_values, turn_rate=1e-12),
        dt,
    )
    below_boundary = ctrv_step(
        CTRVState(**common_values, turn_rate=turn_rate_below),
        dt,
    )
    above_boundary = ctrv_step(
        CTRVState(**common_values, turn_rate=turn_rate_above),
        dt,
    )

    assert almost_straight.x == pytest.approx(straight.x, abs=1e-9)
    assert almost_straight.y == pytest.approx(straight.y, abs=1e-9)
    assert above_boundary.x == pytest.approx(below_boundary.x, abs=1e-8)
    assert above_boundary.y == pytest.approx(below_boundary.y, abs=1e-8)


def test_multi_step_prediction_matches_repeated_single_steps():
    """Multi-step propagation should be repeated single-step propagation."""
    initial_state = CTRVState(
        x=10.0,
        y=-4.0,
        speed=3.5,
        heading=-0.3,
        turn_rate=0.08,
    )

    prediction = predict_ctrv(initial_state, dt=0.5, steps=6)
    repeated_steps = []
    current_state = initial_state
    for _ in range(6):
        current_state = ctrv_step(current_state, dt=0.5)
        repeated_steps.append(current_state)

    assert prediction == tuple(repeated_steps)


def test_propagation_does_not_modify_input_state():
    """Every propagation call should create new immutable state objects."""
    initial_state = CTRVState(
        x=1.0,
        y=2.0,
        speed=3.0,
        heading=0.4,
        turn_rate=-0.1,
    )
    original_values = (
        initial_state.x,
        initial_state.y,
        initial_state.speed,
        initial_state.heading,
        initial_state.turn_rate,
    )

    next_state = ctrv_step(initial_state, dt=1.0)
    prediction = predict_ctrv(initial_state, dt=1.0, steps=3)

    assert (
        initial_state.x,
        initial_state.y,
        initial_state.speed,
        initial_state.heading,
        initial_state.turn_rate,
    ) == original_values
    assert next_state is not initial_state
    assert all(state is not initial_state for state in prediction)


@pytest.mark.parametrize("dt", [0.0, -1.0, float("inf"), float("nan"), True])
def test_invalid_time_steps_are_rejected(dt):
    """Time steps must be positive finite numbers."""
    state = CTRVState(0.0, 0.0, 1.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="dt must be a positive finite number"):
        ctrv_step(state, dt)


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_invalid_prediction_counts_are_rejected(steps):
    """Multi-step prediction requires a positive integer count."""
    state = CTRVState(0.0, 0.0, 1.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="steps must be an integer"):
        predict_ctrv(state, dt=1.0, steps=steps)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("x", float("inf"), "x must be a finite number"),
        ("heading", float("nan"), "heading must be a finite number"),
        ("speed", -0.1, "speed must be non-negative"),
    ],
)
def test_invalid_state_values_are_rejected(field, value, message):
    """State components should be finite and speed should be non-negative."""
    values = {
        "x": 0.0,
        "y": 0.0,
        "speed": 1.0,
        "heading": 0.0,
        "turn_rate": 0.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        CTRVState(**values)
