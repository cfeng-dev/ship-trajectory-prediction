"""Tests for simulation GUI control semantics without creating a window."""

import subprocess
import sys
from textwrap import dedent
from unittest.mock import Mock

import numpy as np
import pytest

from ship_trajectory_prediction.simulation.gui import ShipTrajectoryGUI


def test_gui_module_import_does_not_require_tkinter():
    """Headless environments should still import the testable GUI class."""
    script = dedent(
        """
        import builtins

        original_import = builtins.__import__

        def import_without_tkinter(
            name,
            globals=None,
            locals=None,
            fromlist=(),
            level=0,
        ):
            if name == "tkinter" or name.startswith("tkinter."):
                raise ModuleNotFoundError("Tkinter intentionally unavailable")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = import_without_tkinter

        from ship_trajectory_prediction.simulation.gui import ShipTrajectoryGUI

        assert ShipTrajectoryGUI.__name__ == "ShipTrajectoryGUI"
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


class SliderStub:
    """Minimal slider replacement for testing GUI control methods."""

    def __init__(self, value):
        self.value = value

    def get(self):
        """Return the current slider value."""
        return self.value

    def set(self, value):
        """Store a new slider value."""
        self.value = value


def create_gui_stub(steering=0):
    """Create an uninitialized GUI containing only steering dependencies."""
    gui = ShipTrajectoryGUI.__new__(ShipTrajectoryGUI)
    gui.steering_slider = SliderStub(steering)
    gui.keyboard_steering_step = 1
    gui.min_steering_deg_per_second = -45
    gui.max_steering_deg_per_second = 45
    gui.update_status = Mock()
    return gui


@pytest.mark.parametrize(
    ("steering_degrees", "expected_sign"),
    [(14, 1), (-14, -1), (0, 0)],
)
def test_steering_value_uses_mathematical_turn_rate_sign(
    steering_degrees,
    expected_sign,
):
    """Positive steering should turn left and negative steering right."""
    gui = create_gui_stub(steering_degrees)

    omega = gui.get_omega_from_steering()

    assert omega == pytest.approx(np.deg2rad(steering_degrees))
    assert np.sign(omega) == expected_sign


def test_left_keyboard_control_increases_steering_rate():
    """The left arrow should move the steering rate toward positive values."""
    gui = create_gui_stub()

    gui.steer_left()

    assert gui.steering_slider.get() == 1
    gui.update_status.assert_called_once_with()


def test_right_keyboard_control_decreases_steering_rate():
    """The right arrow should move the steering rate toward negative values."""
    gui = create_gui_stub()

    gui.steer_right()

    assert gui.steering_slider.get() == -1
    gui.update_status.assert_called_once_with()
