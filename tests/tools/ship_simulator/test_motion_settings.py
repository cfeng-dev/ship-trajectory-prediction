"""Motion settings apply atomically and simulator presentation stays local."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from ship_simulator import config as simulator_config
from ship_simulator.config import GUIConfig
from ship_simulator.core import ShipSimulator


def _values(**changes):
    values = dict(
        initial_speed="2.4",
        min_speed="0.0",
        max_speed="4.0",
        speed_resolution="0.2",
        max_steering_deg_per_second="6.0",
        steering_resolution="0.2",
    )
    return values | changes


def test_motion_parameters_derive_symmetric_turn_limits_and_keyboard_steps():
    parsed = simulator_config.parse_motion_parameters(_values())
    assert parsed["initial_speed"] == 2.4
    assert parsed["min_steering_deg_per_second"] == -6.0
    assert parsed["max_steering_deg_per_second"] == 6.0
    assert parsed["keyboard_speed_step"] == 0.2
    assert parsed["keyboard_steering_step"] == 0.2


@pytest.mark.parametrize(
    "changes",
    [
        {"initial_speed": "nan"},
        {"max_speed": "inf"},
        {"min_speed": "-1"},
        {"min_speed": "4"},
        {"initial_speed": "5"},
        {"speed_resolution": "0"},
        {"speed_resolution": "0.3"},
        {"max_steering_deg_per_second": "-2"},
        {"steering_resolution": "0.7"},
        {"steering_resolution": "0"},
    ],
)
def test_invalid_motion_parameters_are_rejected(changes):
    with pytest.raises(ValueError):
        simulator_config.parse_motion_parameters(_values(**changes))


class _Slider:
    def __init__(self, value):
        self.value, self.options = value, {}

    def set(self, value):
        self.value = value

    def get(self):
        return self.value

    def configure(self, **options):
        self.options.update(options)


def _gui():
    config = GUIConfig()
    return SimpleNamespace(
        **vars(config),
        config=config,
        simulator=ShipSimulator(v=config.initial_speed),
        speed_slider=_Slider(config.initial_speed),
        steering_slider=_Slider(0.0),
        simulation_started=False,
        update_status=Mock(),
        update_plot=Mock(),
    )


def _timed_gui(monkeypatch, factor=1):
    from ship_simulator import gui as simulator_gui

    clock = [0.0]
    monkeypatch.setattr(simulator_gui, "perf_counter", lambda: clock[0])
    gui = _gui()
    gui.simulation_time_factor = factor
    gui.simulation_time_accumulator = 0.0
    gui.last_simulation_tick = None
    gui.simulation_running = False
    gui.simulation_start_time = None
    gui.saved_run_states = {}
    gui.root = Mock()
    gui.update_simulation_button = Mock()
    gui.get_speed_from_slider = gui.speed_slider.get
    gui.get_omega_from_steering = lambda: 0.13
    gui.advance_simulation = lambda: simulator_gui.ShipTrajectoryGUI.advance_simulation(
        gui
    )
    gui.simulation_loop = lambda: simulator_gui.ShipTrajectoryGUI.simulation_loop(gui)
    return gui, clock, simulator_gui.ShipTrajectoryGUI


@pytest.mark.parametrize("factor", [1, 5, 10, 20])
def test_time_factor_preserves_motion_and_ground_truth_csv(monkeypatch, factor):
    from ship_simulator.io import create_simulation_dataframe

    gui, clock, gui_type = _timed_gui(monkeypatch, factor)
    gui_type.start_or_continue_simulation(gui)
    for elapsed in [0.027, 0.101, 0.221, 0.5, 1.003, 2.05, 3.05]:
        clock[0] = elapsed
        gui.simulation_loop()

    expected = ShipSimulator(v=gui.initial_speed, dt=gui.simulation_dt)
    for _ in range(61 * factor):
        expected.step(0.13, True)
    assert gui.simulator.current_time == pytest.approx(3.05 * factor)
    np.testing.assert_array_equal(gui.simulator.x_all, expected.x_all)
    np.testing.assert_array_equal(gui.simulator.y_all, expected.y_all)
    data = create_simulation_dataframe(gui.simulator)
    pd.testing.assert_frame_equal(data, create_simulation_dataframe(expected))
    assert len(data) >= 2
    np.testing.assert_allclose(
        np.diff(data["time"]), gui.csv_export_interval_seconds, atol=1e-10
    )
    np.testing.assert_array_equal(data["ground_truth_x"], data["x"])
    np.testing.assert_array_equal(data["ground_truth_y"], data["y"])
    gui.root.after.assert_called_with(gui.update_interval_ms, gui.simulation_loop)


def test_time_factor_excludes_pause_time_and_reset_clears_pending_time(monkeypatch):
    gui, clock, gui_type = _timed_gui(monkeypatch, 10)
    gui_type.start_or_continue_simulation(gui)
    clock[0] = 0.253
    gui_type.pause_simulation(gui)
    assert gui.simulator.current_time == pytest.approx(2.5)
    clock[0] = 100.0
    gui.simulation_loop()
    assert gui.simulator.current_time == pytest.approx(2.5)
    clock[0] = 120.0
    gui_type.start_or_continue_simulation(gui)
    clock[0] = 120.2
    gui.simulation_loop()
    assert gui.simulator.current_time == pytest.approx(4.5)
    gui_type.reset(gui)
    assert gui.last_simulation_tick is None
    assert gui.simulation_time_accumulator == 0.0
    assert gui.simulation_time_factor == 10
    assert not gui.simulator.has_data()


def test_changing_time_factor_accounts_for_time_at_previous_factor(monkeypatch):
    gui, clock, gui_type = _timed_gui(monkeypatch)
    gui.simulation_time_factor_var = Mock()
    gui.simulation_time_factor_var.get.return_value = "20×"
    gui_type.start_or_continue_simulation(gui)
    clock[0] = 0.2
    gui_type.change_simulation_time_factor(gui)
    assert gui.simulator.current_time == pytest.approx(0.2)
    assert gui.simulation_time_factor == gui.config.simulation_time_factor == 20
    clock[0] = 0.3
    gui.simulation_loop()
    assert gui.simulator.current_time == pytest.approx(2.2)


def test_apply_updates_sliders_simulator_and_persistent_reset_values(monkeypatch):
    pytest.importorskip("tkinter")
    from ship_simulator import dialogs

    gui = _gui()
    assert dialogs.apply_motion_parameters(gui, _values())
    assert gui.simulator.v == gui.initial_speed == gui.config.initial_speed == 2.4
    assert gui.speed_slider.get() == 2.4
    assert gui.speed_slider.options == dict(from_=0.0, to=4.0, resolution=0.2)
    assert gui.steering_slider.options == dict(from_=6.0, to=-6.0, resolution=0.2)
    assert gui.keyboard_speed_step == gui.keyboard_steering_step == 0.2
    gui.update_status.assert_called_once()
    gui.update_plot.assert_called_once()


@pytest.mark.parametrize("running", [False, True])
def test_rejected_apply_preserves_all_motion_state(monkeypatch, running):
    pytest.importorskip("tkinter")
    from ship_simulator import dialogs

    monkeypatch.setattr(dialogs.messagebox, "showerror", Mock())
    monkeypatch.setattr(dialogs.messagebox, "showwarning", Mock())
    gui = _gui()
    gui.simulation_started = running
    before = vars(gui.config).copy()
    assert not dialogs.apply_motion_parameters(gui, _values(initial_speed="100"))
    assert vars(gui.config) == before
    assert (
        gui.initial_speed
        == gui.simulator.v
        == gui.speed_slider.get()
        == before["initial_speed"]
    )
    assert gui.speed_slider.options == gui.steering_slider.options == {}
    gui.update_status.assert_not_called()
    gui.update_plot.assert_not_called()


def test_simulator_menu_has_local_units_and_motion_settings_only(monkeypatch):
    pytest.importorskip("tkinter")
    from ship_simulator import view

    monkeypatch.setattr(view.tk, "Menu", lambda *args, **kwargs: Mock())
    monkeypatch.setattr(view.tk, "StringVar", lambda **kwargs: Mock())
    gui = Mock()
    gui.coordinate_display_mode = "local"
    gui.root.tk.call.return_value = "x11"
    view.create_menu_bar(gui)
    assert [
        call.kwargs["value"] for call in gui.view_menu.add_radiobutton.call_args_list
    ] == ["local", "km"]
    assert (
        gui.settings_menu.add_command.call_args.kwargs["command"]
        == gui.show_motion_parameters_dialog
    )
    assert (
        gui.settings_menu.add_command.call_args.kwargs["label"]
        == "Movement Parameters..."
    )


def test_local_status_and_plot_switch_units_without_a_gps_origin():
    pytest.importorskip("tkinter")
    from matplotlib.figure import Figure
    from ship_simulator import plotting, view

    gui = _gui()
    gui.simulator.x_current, gui.simulator.y_current = 1250.0, -250.0
    gui.get_omega_from_steering = lambda: 0.0
    gui.get_speed_from_slider = gui.speed_slider.get
    gui.get_simulation_state = lambda: "stopped"
    for name in ("simulation", "position", "heading", "omega", "speed", "time"):
        setattr(gui, f"{name}_value_label", Mock())
    gui.ax = Figure().subplots()
    gui.coordinate_display_mode = "km"
    view.update_status_display(gui)
    plotting.configure_coordinate_display(gui)
    assert (
        gui.position_value_label.config.call_args.kwargs["text"]
        == "x = 1.250 km\ny = -0.250 km"
    )
    assert gui.ax.get_xlabel() == "x [km]"
    assert gui.ax.xaxis.get_major_formatter()(1000, None) == "1"
    gui.coordinate_display_mode = "local"
    view.update_status_display(gui)
    plotting.configure_coordinate_display(gui)
    assert (
        gui.position_value_label.config.call_args.kwargs["text"]
        == "x = 1250.00 m\ny = -250.00 m"
    )
    assert gui.ax.get_xlabel() == "x [m]"


def test_reset_unlocks_motion_settings_and_keeps_applied_initial_speed():
    pytest.importorskip("tkinter")
    from ship_simulator import dialogs
    from ship_simulator.gui import ShipTrajectoryGUI

    gui = _gui()
    assert dialogs.apply_motion_parameters(gui, _values())
    gui.simulation_started = gui.simulation_running = True
    gui.simulation_button = Mock()
    gui.settings_menu = Mock()
    gui.motion_parameters_menu_index = 0
    gui.get_simulation_state = lambda: (
        "running" if gui.simulation_running else "stopped"
    )
    gui.get_speed_from_slider = gui.speed_slider.get
    gui.update_simulation_button = lambda: ShipTrajectoryGUI.update_simulation_button(
        gui
    )
    gui.saved_run_states = {"run.csv": 1}
    gui.last_saved_csv_path = "run.csv"
    gui.update_simulation_button()
    assert gui.settings_menu.entryconfigure.call_args.kwargs["state"] == "disabled"
    ShipTrajectoryGUI.reset(gui)
    assert gui.settings_menu.entryconfigure.call_args.kwargs["state"] == "normal"
    assert gui.simulator.v == gui.speed_slider.get() == gui.initial_speed == 2.4
    assert not gui.simulation_started
    assert not gui.saved_run_states
    assert gui.last_saved_csv_path is None


def test_speed_slider_preserves_configured_precision_and_keyboard_steps():
    tk = pytest.importorskip("tkinter")
    from ship_simulator import controls, dialogs
    from ship_simulator.gui import ShipTrajectoryGUI

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display is unavailable.")
    root.withdraw()
    try:
        gui = _gui()
        controls.create_speed_section(gui, root)
        assert dialogs.apply_motion_parameters(
            gui,
            _values(initial_speed="12.34", max_speed="20", speed_resolution="0.01"),
        )
        assert gui.speed_slider.get() == pytest.approx(12.34)
        ShipTrajectoryGUI.increase_speed(gui)
        assert gui.speed_slider.get() == pytest.approx(12.35)
    finally:
        root.destroy()


@pytest.mark.parametrize("button", ["Apply", "Cancel"])
def test_motion_dialog_uses_local_draft_and_closes_on_apply_or_cancel(button):
    tk = pytest.importorskip("tkinter")
    from ship_simulator import dialogs

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display is unavailable.")
    root.withdraw()
    gui = _gui()
    gui.root = root
    gui.apply_motion_parameters = lambda values, parent=None: (
        dialogs.apply_motion_parameters(gui, values, parent)
    )
    try:
        dialog = dialogs.show_motion_parameters_dialog(gui)
        form = dialog.winfo_children()[0]
        entry = next(
            child for child in form.winfo_children() if isinstance(child, tk.Entry)
        )
        entry.delete(0, tk.END)
        entry.insert(0, "2.5")
        assert gui.initial_speed == 2.2
        buttons = [
            child
            for frame in form.winfo_children()
            for child in frame.winfo_children()
            if isinstance(child, tk.Button)
        ]
        next(child for child in buttons if child.cget("text") == button).invoke()
        assert not dialog.winfo_exists()
        assert gui.initial_speed == (2.5 if button == "Apply" else 2.2)
    finally:
        root.destroy()
