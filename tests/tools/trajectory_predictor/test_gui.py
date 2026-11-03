"""Non-visible Tk integration checks for the embedded trajectory predictor."""

from threading import Event
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")
ttk = pytest.importorskip("tkinter.ttk")

from ship_simulator.controls import STATUS_ROW_LABELS  # noqa: E402
from trajectory_predictor import controls as predictor_controls  # noqa: E402
from trajectory_predictor.controls import (  # noqa: E402
    CSV_BROWSE_BUTTON_WIDTH,
    CSV_STATE_ROW_KEYS,
    ScrollableForm,
    SettingsPanel,
    format_elapsed_time,
    format_reference_position,
    split_analysis_data_fields,
)
from trajectory_predictor.dashboard import (  # noqa: E402
    PARAMETER_NAMES,
    PosteriorDashboardTrajectory,
    PosteriorDashboardUpdate,
)
from trajectory_predictor.gui import TrajectoryPredictor  # noqa: E402
from trajectory_predictor.session import PosteriorAnalysisWorker  # noqa: E402
from trajectory_predictor.settings import (  # noqa: E402
    DATA_OPTION_FIELDS,
    MAIN_DATA_FIELDS,
)
from trajectory_predictor.view import (  # noqa: E402
    DISABLED_INPUT_BACKGROUND,
    INPUT_BACKGROUND,
    TEXT_COLOR,
    centered_window_position,
    input_style_settings,
)


def pump_until(root, predicate):
    deadline = monotonic() + 8
    while monotonic() < deadline:
        root.update()
        if predicate():
            return
        Event().wait(0.01)
    pytest.fail("Tk condition was not reached")


@pytest.mark.parametrize(
    ("screen_width", "screen_height", "window_width", "window_height", "expected"),
    (
        (1920, 1080, 1400, 700, (260, 190)),
        (1280, 900, 1100, 720, (90, 90)),
    ),
)
def test_main_window_position_matches_ship_simulator_centering(
    screen_width, screen_height, window_width, window_height, expected
):
    """The explorer uses the simulator's direct screen-centering formula."""
    assert (
        centered_window_position(
            screen_width, screen_height, window_width, window_height
        )
        == expected
    )


def test_main_window_position_supports_small_upward_offset():
    """The explorer can sit slightly above the mathematical screen center."""
    assert centered_window_position(1920, 1080, 1400, 700, vertical_offset=40) == (
        260,
        150,
    )


def test_form_scrollbar_uses_the_native_tk_widget(monkeypatch):
    """The predictor sidebar follows the Ship Simulator scrollbar implementation."""
    from trajectory_predictor import controls

    calls = []

    class Scrollbar:
        def __init__(self, parent, **options):
            calls.append((parent, options))

    canvas = SimpleNamespace(yview=object())
    monkeypatch.setattr(controls.tk, "Scrollbar", Scrollbar)

    controls.create_form_scrollbar("parent", canvas)

    assert calls == [("parent", {"orient": "vertical", "command": canvas.yview})]


def test_scrollbar_prioritizes_interaction_before_moving_the_canvas(monkeypatch):
    from trajectory_predictor import controls

    events, options = [], {}
    canvas = SimpleNamespace(yview=lambda *args: events.append(args))
    monkeypatch.setattr(
        controls.tk, "Scrollbar", lambda _parent, **kwargs: options.update(kwargs)
    )
    controls.create_form_scrollbar(
        "parent", canvas, on_scroll=lambda: events.append("scroll")
    )
    options["command"]("moveto", "0.4")
    assert events == ["scroll", ("moveto", "0.4")]


@pytest.mark.parametrize(
    ("event", "movement"),
    [
        (SimpleNamespace(delta=-120, num=None), (1, "units")),
        (SimpleNamespace(delta=0, num=4), (-1, "units")),
        (SimpleNamespace(delta=0, num=5), (1, "units")),
        (SimpleNamespace(delta=0, num=None), None),
    ],
)
def test_mouse_wheel_reports_scroll_activity_before_canvas_changes(event, movement):
    events, bindings = [], {}
    canvas = SimpleNamespace(
        yview=lambda: (0.0, 0.5),
        yview_scroll=lambda *args: events.append(args),
        bind=lambda name, callback: bindings.update({name: callback}),
        winfo_children=lambda: [],
    )
    form = ScrollableForm.__new__(ScrollableForm)
    form.canvas = form.body = canvas
    form.on_scroll = lambda: events.append("scroll")
    form.bind_mouse_wheel()
    bindings["<MouseWheel>"](event)
    assert events == ([] if movement is None else ["scroll", movement])


def test_csv_browse_button_uses_a_two_character_width():
    assert CSV_BROWSE_BUTTON_WIDTH == 2


def test_scrollable_form_uses_precise_touchpad_scroll_on_tk_9():
    """Mac trackpads move the sidebar through Tk's TouchpadScroll event."""

    class Widget:
        def __init__(self, children=()):
            self.children = children
            self.bindings = {}

        def bind(self, event_name, callback):
            self.bindings[event_name] = callback

        def winfo_children(self):
            return self.children

    class Canvas(Widget):
        def __init__(self):
            super().__init__()
            self.tk = SimpleNamespace(call=lambda *_args: (0, -50))
            self.positions = []

        def bbox(self, _tag):
            return (0, 0, 200, 1000)

        def yview(self):
            return (0.2, 0.6)

        def yview_moveto(self, position):
            self.positions.append(position)

    canvas = Canvas()
    form = ScrollableForm.__new__(ScrollableForm)
    form.canvas = canvas
    form.body = Widget()
    activity = []
    form.on_scroll = lambda: activity.append("scroll")

    form.bind_mouse_wheel()
    result = form.body.bindings["<TouchpadScroll>"](SimpleNamespace(delta=-1))

    assert result == "break"
    assert canvas.positions == [pytest.approx(0.25)]
    assert activity == ["scroll"]


def test_input_styles_keep_fields_light_in_all_widget_states():
    """The light application palette must not inherit macOS dark input fields."""
    settings = input_style_settings()

    for style_name in ("Predictor.TEntry", "Predictor.TCombobox"):
        assert settings[style_name]["configure"]["fieldbackground"] == INPUT_BACKGROUND
        assert settings[style_name]["map"]["fieldbackground"] == [
            ("disabled", DISABLED_INPUT_BACKGROUND),
            ("readonly", INPUT_BACKGROUND),
        ]
    assert settings["Predictor.TEntry"]["configure"]["insertcolor"] == TEXT_COLOR
    assert settings["Predictor.TEntry"]["configure"]["insertwidth"] == 2
    assert settings["Predictor.TButton"]["configure"]["background"] == (
        INPUT_BACKGROUND
    )


@pytest.mark.parametrize(
    ("windowing_system", "uses_clam"),
    (("aqua", True), ("win32", False)),
)
def test_input_styles_use_clam_only_to_override_macos_aqua(
    monkeypatch, windowing_system, uses_clam
):
    """Only Aqua needs a non-native theme to keep fields light."""
    from trajectory_predictor import view

    calls = []

    class Style:
        def __init__(self, root):
            calls.append(("create", root))

        def theme_use(self, theme_name):
            calls.append(("theme_use", theme_name))

        def configure(self, style_name, **settings):
            calls.append(("configure", style_name, settings))

        def map(self, style_name, **settings):
            calls.append(("map", style_name, settings))

    monkeypatch.setattr(view.ttk, "Style", Style)
    root = SimpleNamespace(
        tk=SimpleNamespace(call=lambda *_: windowing_system),
    )
    view.configure_input_styles(root)

    assert (("theme_use", "clam") in calls) is uses_clam


def test_plot_toolbar_uses_the_application_light_palette():
    """The embedded Matplotlib controls must not inherit system dark colours."""
    from trajectory_predictor.view import (
        PLOT_BACKGROUND,
        configure_plot_toolbar,
    )

    class Widget:
        def __init__(self, widget_class="Label"):
            self.options = {}
            self.widget_class = widget_class

        def configure(self, **options):
            self.options.update(options)

        def winfo_class(self):
            return self.widget_class

    class Toolbar(Widget):
        def __init__(self):
            super().__init__()
            self.button = Widget("Button")
            self.toggle_button = Widget("Checkbutton")
            self.label = Widget()
            self.spacer = Widget("Frame")
            self._buttons = {"Home": self.button, "Pan": self.toggle_button}
            self.recoloured_buttons = []

        def winfo_children(self):
            return (self.button, self.toggle_button, self.label, self.spacer)

        def _set_image_for_button(self, button):
            self.recoloured_buttons.append(button)

    toolbar = Toolbar()
    configure_plot_toolbar(toolbar)

    assert toolbar.options["bg"] == PLOT_BACKGROUND
    assert toolbar.button.options["bg"] == INPUT_BACKGROUND
    assert "selectcolor" not in toolbar.button.options
    assert toolbar.toggle_button.options["selectcolor"] == "#b8d8e8"
    assert toolbar.label.options["bg"] == PLOT_BACKGROUND
    assert toolbar.recoloured_buttons == [toolbar.button, toolbar.toggle_button]


def test_reference_position_uses_ship_simulator_style_two_line_value():
    """The CSV status keeps x and y vertically aligned under Position."""
    assert format_reference_position(11.0, -3.0) == "x = 11.00 m\ny = -3.00 m"
    assert format_reference_position(1.5, -2.0, coordinate_display_mode="km") == (
        "x = 1.50 km\ny = -2.00 km"
    )


def test_status_panels_share_motion_state_row_order():
    """Position, speed, heading, and turn rate appear in the same order."""
    assert CSV_STATE_ROW_KEYS == (
        "position",
        "speed",
        "heading",
        "turn_rate",
        "time",
    )
    assert STATUS_ROW_LABELS[1:5] == (
        "Position:",
        "Speed:",
        "Heading:",
        "Turn rate:",
    )
    assert "Turn Radius:" not in STATUS_ROW_LABELS
    assert format_reference_position(8.31, 47.05, coordinate_display_mode="gps") == (
        "x = 8.31 m\ny = 47.05 m"
    )


def test_analysis_controls_lock_until_the_active_analysis_is_cancelled(monkeypatch):
    """A running analysis keeps its settings immutable until cancellation."""
    from trajectory_predictor import gui

    settings = SimpleNamespace(analysis=object())
    worker_starts, worker_requests, locked_states = [], [], []
    app = TrajectoryPredictor.__new__(TrajectoryPredictor)
    app._closing = False
    app._analysis_active = False
    app._settings = None
    app._error = None
    app._computing = None
    app.plot_view = None
    app.controls = SimpleNamespace(
        values=lambda: {},
        show_reference_state=lambda _state: None,
        show_analysis_metrics=lambda _metrics: None,
        show_posterior_medians=lambda _medians: None,
        show_radar_association=lambda _diagnostics: None,
        set_analysis_active=lambda active: locked_states.append(active),
    )

    app.placeholder = SimpleNamespace(
        configure=lambda **_options: None, pack=lambda **_options: None
    )
    app.status = SimpleNamespace(set=lambda _text: None)
    app.worker = SimpleNamespace(
        start=lambda analysis: worker_starts.append(analysis),
        request=lambda count: worker_requests.append(count),
    )
    monkeypatch.setattr(gui, "parse_settings", lambda _values: settings)

    app.start_analysis()
    app.cancel_analysis()

    assert worker_starts == [settings.analysis]
    assert worker_requests == [0]
    assert locked_states == [True, False]
    assert not app._analysis_active


@pytest.mark.parametrize(
    ("seconds", "expected"),
    ((20.0, "20.0 s"), (80.0, "1 min 20 s"), (3_735.0, "1 h 02 min 15 s")),
)
def test_reference_time_uses_adaptive_elapsed_format(seconds, expected):
    assert format_elapsed_time(seconds) == expected


def test_settings_panel_displays_reference_csv_state(root):
    """The sidebar presents reference values and the displayed forecast metrics."""
    panel = SettingsPanel(root, lambda: None, lambda: None)

    panel.show_reference_state((11.0, -3.0, 2.0, 2.0, 2.0, "m", 12.5))
    panel.show_analysis_metrics(
        SimpleNamespace(
            ade_m=2.5,
            fde_m=4.0,
            rmse_m=3.5,
            energy_score_m=3.25,
            negative_log_predictive_density=12.5,
            joint_coverage_50_count=3,
            joint_coverage_90_count=7,
            joint_coverage_total=8,
            inference_time_seconds=0.125,
        )
    )
    panel.show_posterior_medians(
        {
            "current_speed": 2.0,
            "current_heading": 3.0,
            "current_turn_rate": None,
            "position_observation_noise": 4.0,
            "speed_process_noise": 5.0,
            "turn_rate_process_noise": 6.0,
        }
    )
    panel.show_radar_association(
        SimpleNamespace(
            status="accepted",
            detection_id=7,
            nis=1.25,
            measurement_covariance=np.array([[4.0, 0.5], [0.5, 9.0]]),
            cycle_counter=42,
        )
    )
    posterior_median_labels = {
        child.cget("text")
        for child in panel.posterior_medians_section.winfo_children()
        if child.winfo_class() == "Label"
    }

    assert panel.posterior_state_values["position"].get() == "x = 11.00 m\ny = -3.00 m"
    assert panel.posterior_state_values["speed"].get() == "2.00 m/s"
    assert panel.posterior_state_values["heading"].get() == "2.00°"
    assert panel.posterior_state_values["turn_rate"].get() == "2.00°/s"
    assert panel.posterior_state_values["time"].get() == "12.5 s"
    assert panel.analysis_metric_values["forecast_ade"].get() == "2.50 m"
    assert panel.analysis_metric_values["forecast_fde"].get() == "4.00 m"
    assert panel.analysis_metric_values["forecast_rmse"].get() == "3.50 m"
    assert panel.analysis_metric_values["energy_score"].get() == "3.25 m"
    assert (
        panel.analysis_metric_values["negative_log_predictive_density"].get() == "12.50"
    )
    assert panel.analysis_metric_values["joint_coverage_50"].get() == "37.5% (3/8)"
    assert panel.analysis_metric_values["joint_coverage_90"].get() == "87.5% (7/8)"
    assert panel.analysis_metric_values["inference_time"].get() == "0.125 s"
    assert panel.posterior_median_values["current_speed"].get() == "2.00 m/s"
    assert panel.posterior_median_values["current_heading"].get() == "3.00°"
    assert panel.posterior_median_values["current_turn_rate"].get() == "—"
    assert panel.posterior_median_values["position_observation_noise"].get() == "4.00 m"
    assert panel.posterior_median_values["speed_process_noise"].get() == "5.00 m/s"
    assert panel.posterior_median_values["turn_rate_process_noise"].get() == "6.00°/s"
    assert panel.radar_association_values["status"].get() == "Associated"
    assert panel.radar_association_values["cycle"].get() == "42"
    assert panel.radar_association_values["detection_id"].get() == "7"
    assert panel.radar_association_values["nis"].get() == "1.25"
    assert panel.radar_association_values["cov_xx"].get() == "4.000"
    assert panel.radar_association_values["cov_xy"].get() == "0.500"
    assert panel.radar_association_values["cov_yy"].get() == "9.000"
    assert "Speed-process noise:" in posterior_median_labels
    assert "Turn-rate-process noise:" in posterior_median_labels


def test_settings_panel_does_not_create_online_predictive_diagnostics(root):
    panel = SettingsPanel(root, lambda: None)
    root.update_idletasks()

    assert not hasattr(panel, "online_predictive_diagnostics_section")


def test_posterior_median_process_noise_labels_are_compact():
    assert (
        predictor_controls.POSTERIOR_MEDIAN_LABELS_BY_KEY["speed_process_noise"]
        == "Speed-process noise"
    )
    assert (
        predictor_controls.POSTERIOR_MEDIAN_LABELS_BY_KEY["turn_rate_process_noise"]
        == "Turn-rate-process noise"
    )


def test_help_documents_analysis_setup_options():
    from trajectory_predictor.help import POSTERIOR_HELP_SECTIONS

    sections = dict(POSTERIOR_HELP_SECTIONS)

    assert [control for control, _description in sections["Analysis setup"]] == [
        "Start index",
        "Observation interval",
        "Maximum observations",
        "Additional position noise",
        "Position-noise seed",
        "Forecast steps",
        "Future trajectories",
        "Inference seed",
        "Playback interval",
    ]


def test_help_explains_analysis_metrics():
    from trajectory_predictor.help import POSTERIOR_HELP_SECTIONS

    sections = dict(POSTERIOR_HELP_SECTIONS)

    assert dict(sections["Evaluation metrics"]) == {
        "Forecast ADE (median)": "Mean 2D error of the current median forecast against available future reference positions, in meters; lower is better.",
        "Forecast FDE (median)": "Position error of the median at the final forecast step in meters; lower is better.",
        "Forecast RMSE (median)": "Root mean squared 2D position error of the median across all forecast steps in meters; lower is better.",
        "Forecast Energy Score": "Joint 2D probabilistic forecast error; lower is better.",
        "NLPD": "Mean negative log predictive density of future reference positions across the forecast horizon; unavailable without a forecast or future reference positions.",
        "Joint 50% coverage": "Cumulative share of evaluated future positions inside per-step 2D 50% covariance ellipses, shown as percentage and count; not whole-path coverage.",
        "Joint 90% coverage": "Cumulative share of evaluated future positions inside per-step 2D 90% covariance ellipses, shown as percentage and count; not whole-path coverage.",
        "Inference time": "Wall-clock time for the current update, including forecast generation but excluding GUI drawing; not forecast accuracy.",
    }


@pytest.mark.parametrize("tool", ["bayesian", "imm"])
def test_help_explains_both_forecast_summaries_and_fixed_metric_basis(tool):
    from imm_predictor.dialogs import IMM_HELP_SECTIONS
    from trajectory_predictor.help import POSTERIOR_HELP_SECTIONS

    sections = dict(
        POSTERIOR_HELP_SECTIONS if tool == "bayesian" else IMM_HELP_SECTIONS
    )
    display = dict(
        sections["Display" if tool == "bayesian" else "Plot display and uncertainty"]
    )
    assert "on by default" in display["Forecast (median)"]
    assert "off by default" in display["Forecast (mean)"]
    assert "together" in display["Forecast (mean)"]
    for label in ("ADE", "FDE", "RMSE"):
        assert f"Forecast {label} (median)" in dict(sections["Evaluation metrics"])


@pytest.mark.parametrize(
    ("package", "title"),
    [
        ("trajectory_predictor", "Bayesian CTRV — Ship Trajectory Predictor"),
        ("imm_predictor", "IMM — Ship Trajectory Predictor"),
    ],
)
def test_main_window_title_identifies_the_ship_trajectory_predictor(
    monkeypatch, package, title
):
    import importlib

    view = importlib.import_module(f"{package}.view")

    class Window:
        def title(self, value):
            self.title_value = value

        def winfo_screenwidth(self):
            return 1920

        def winfo_screenheight(self):
            return 1080

        def geometry(self, _value):
            pass

        def minsize(self, _width, _height):
            pass

        def resizable(self, _width, _height):
            pass

        def bind(self, _event, _callback, **_kwargs):
            pass

        def configure(self, **_kwargs):
            pass

        def columnconfigure(self, _column, **_kwargs):
            pass

        def rowconfigure(self, _row, **_kwargs):
            pass

        grid_columnconfigure = columnconfigure
        grid_rowconfigure = rowconfigure

    window = Window()
    monkeypatch.setattr(view, "configure_input_styles", lambda _root: None)
    view.configure_window(window)

    assert window.title_value == title


@pytest.mark.parametrize("platform", ["win32", "linux"])
@pytest.mark.parametrize("package", ["trajectory_predictor", "imm_predictor"])
def test_window_keeps_maximize_and_locks_only_its_own_windows_frame(
    monkeypatch, platform, package
):
    import importlib
    import sys
    from unittest.mock import Mock

    view = importlib.import_module(f"{package}.view")

    root = Mock()
    root.winfo_screenwidth.return_value = 1920
    root.winfo_screenheight.return_value = 1080
    remove_frame = Mock()
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(view, "configure_input_styles", lambda _root: None)
    monkeypatch.setattr(
        view, "_remove_windows_resize_frame", remove_frame, raising=False
    )
    view.configure_window(root)
    root.resizable.assert_called_once_with(True, True)
    remove_frame.assert_not_called()  # Wait for the native wrapper to be mapped.
    if platform != "win32":
        root.bind.assert_not_called()
        return
    (event_name, on_map), options = root.bind.call_args
    assert event_name == "<Map>" and options == {"add": "+"}
    on_map(SimpleNamespace(widget=object()))
    remove_frame.assert_not_called()
    on_map(SimpleNamespace(widget=root))
    remove_frame.assert_called_once_with(root)


@pytest.fixture
def native_window_api(monkeypatch):
    import ctypes
    from unittest.mock import Mock

    native = SimpleNamespace(
        GetParent=Mock(return_value=0x100000001),
        GetWindowLongPtrW=Mock(return_value=0x00CF0000),
        SetWindowLongPtrW=Mock(return_value=0x00CF0000),
        GetWindowLongW=Mock(return_value=0x00CF0000),
        SetWindowLongW=Mock(return_value=0x00CF0000),
        SetWindowPos=Mock(return_value=1),
    )
    monkeypatch.setattr(
        ctypes, "WinDLL", lambda *_args, **_kwargs: native, raising=False
    )
    monkeypatch.setattr(ctypes, "set_last_error", lambda _value: None, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    monkeypatch.setattr(
        ctypes,
        "WinError",
        lambda code: OSError(code, "native frame update failed"),
        raising=False,
    )
    suffix = "LongPtrW" if ctypes.sizeof(ctypes.c_void_p) == 8 else "LongW"
    return (
        native,
        getattr(native, f"GetWindow{suffix}"),
        getattr(native, f"SetWindow{suffix}"),
    )


def test_windows_frame_lock_preserves_titlebar_buttons_and_window_position(
    native_window_api,
):
    from unittest.mock import Mock

    from trajectory_predictor import view

    native, get_style, set_style = native_window_api
    root = SimpleNamespace(winfo_id=Mock(return_value=42))
    view._remove_windows_resize_frame(root)
    native.GetParent.assert_called_once_with(42)
    get_style.assert_called_once_with(0x100000001, -16)
    set_style.assert_called_once_with(0x100000001, -16, 0x00CB0000)
    style = set_style.call_args.args[2]
    assert style & 0x00010000  # WS_MAXIMIZEBOX
    assert style & 0x00020000  # WS_MINIMIZEBOX
    assert not style & 0x00040000  # WS_THICKFRAME
    native.SetWindowPos.assert_called_once_with(0x100000001, None, 0, 0, 0, 0, 0x0037)
    get_style.return_value = style
    view._remove_windows_resize_frame(root)
    assert set_style.call_count == native.SetWindowPos.call_count == 1


@pytest.mark.parametrize("failure", ["read", "write", "refresh"])
def test_windows_frame_lock_reports_native_failures(native_window_api, failure):
    from trajectory_predictor import view

    native, get_style, set_style = native_window_api
    {"read": get_style, "write": set_style, "refresh": native.SetWindowPos}[
        failure
    ].return_value = 0
    with pytest.raises(OSError, match="native frame update failed"):
        view._remove_windows_resize_frame(SimpleNamespace(winfo_id=lambda: 42))


def test_windows_frame_lock_with_hidden_native_window():
    import ctypes
    import sys
    from ctypes import wintypes

    from trajectory_predictor import view

    if sys.platform != "win32":
        pytest.skip("Requires native Windows window styles.")
    native = ctypes.WinDLL("user32", use_last_error=True)
    native.CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        wintypes.LPVOID,
    ]
    native.CreateWindowExW.restype = wintypes.HWND
    native.DestroyWindow.argtypes = [wintypes.HWND]
    native.DestroyWindow.restype = wintypes.BOOL
    native.IsWindowVisible.argtypes = [wintypes.HWND]
    native.IsWindowVisible.restype = wintypes.BOOL
    native.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    native.GetWindowRect.restype = wintypes.BOOL
    suffix = "LongPtrW" if ctypes.sizeof(ctypes.c_void_p) == 8 else "LongW"
    get_style = getattr(native, f"GetWindow{suffix}")
    get_style.argtypes = [wintypes.HWND, ctypes.c_int]
    get_style.restype = ctypes.c_ssize_t

    # Use predefined classes and omit WS_VISIBLE; no Tcl interpreter is needed.
    frame = native.CreateWindowExW(
        0, "STATIC", "Resize test", 0x00CF0000, 20, 20, 400, 300, None, None, None, None
    )
    if not frame:
        pytest.skip(f"Native Windows window unavailable: {ctypes.get_last_error()}")
    try:
        client = native.CreateWindowExW(
            0, "STATIC", "", 0x40000000, 0, 0, 100, 100, frame, None, None, None
        )
        assert client, ctypes.WinError(ctypes.get_last_error())
        before = wintypes.RECT()
        after = wintypes.RECT()
        assert native.GetWindowRect(frame, ctypes.byref(before))
        style = get_style(frame, -16)
        assert style & 0x00040000 and style & 0x00010000
        view._remove_windows_resize_frame(SimpleNamespace(winfo_id=lambda: client))
        assert get_style(frame, -16) == style & ~0x00040000
        assert native.GetWindowRect(frame, ctypes.byref(after))
        assert tuple(
            getattr(before, field) for field in ("left", "top", "right", "bottom")
        ) == tuple(
            getattr(after, field) for field in ("left", "top", "right", "bottom")
        )
        assert not native.IsWindowVisible(frame)
    finally:
        assert native.DestroyWindow(frame)


def test_imm_route_legend_identifies_the_filtered_track():
    from imm_predictor.player import IMMPlayerConfig, IMMPlayerStage
    from imm_predictor.plot_view import IMMPlotView
    from matplotlib.figure import Figure

    stage = IMMPlayerStage(
        observation_count=2,
        cycle_counter=2,
        observation_time_seconds=1.0,
        detection_id=1,
        nis=0.5,
        measurement_covariance=np.eye(2),
        update_time_seconds=0.01,
        tracked_times_seconds=np.array([0.0, 1.0]),
        tracked_positions=np.array([[10.0, 20.0], [11.0, 22.0]]),
        forecast=None,
    )
    view = IMMPlotView.__new__(IMMPlotView)
    view.figure = Figure()
    view.route_axis = view.figure.subplots()
    view.store = SimpleNamespace(config=IMMPlayerConfig(), reference=None)
    view.current_state = SimpleNamespace(stage=stage)
    view._coordinate_display_mode = "m"
    view._route_has_been_drawn = False
    view.show_reference_var = SimpleNamespace(get=lambda: False)
    view.follow_ship_var = SimpleNamespace(get=lambda: False)

    view._draw_route()

    assert [text.get_text() for text in view.route_axis.get_legend().get_texts()] == [
        "IMM track through N",
        "Current position",
    ]
    track = view.route_axis.lines[0]
    np.testing.assert_array_equal(track.get_xdata(), [10.0, 11.0])
    np.testing.assert_array_equal(track.get_ydata(), [20.0, 22.0])


def _imm_motion_stage(*, zero_variance=False):
    from imm_predictor.player import IMMPlayerStage, IMMStatePosterior

    scales = [1.0, 1.0, np.deg2rad(10.0), np.deg2rad(0.5), 1.0]
    return IMMPlayerStage(
        observation_count=2,
        cycle_counter=2,
        observation_time_seconds=1.0,
        detection_id=1,
        nis=0.5,
        measurement_covariance=np.eye(2),
        update_time_seconds=0.01,
        tracked_times_seconds=np.array([0.0, 1.0]),
        tracked_positions=np.array([[10.0, 20.0], [11.0, 22.0]]),
        forecast=None,
        posterior=IMMStatePosterior(
            [11.0, 22.0, np.deg2rad(30.0), np.deg2rad(2.0), 4.0],
            np.zeros((5, 5)) if zero_variance else np.diag(np.square(scales)),
        ),
    )


def _headless_imm_panels(stage, *, figure_size=(10, 6), settings_visible=True):
    from imm_predictor.player import IMMPlayerConfig
    from imm_predictor.plot_view import IMMPlotView, _IMMPredictorLayout
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    view = IMMPlotView.__new__(IMMPlotView)
    view.figure = Figure(figsize=figure_size)
    view.canvas = FigureCanvasAgg(view.figure)
    view.store = SimpleNamespace(
        config=IMMPlayerConfig(prediction_count=0),
        reference=None,
        maximum_observation_count=5,
    )
    view.current_state = None if stage is None else SimpleNamespace(stage=stage)
    view._coordinate_display_mode = "m"
    view._panel_group = "motion"
    view._route_has_been_drawn = False
    view._destroyed = view._failed = False
    view._playing = view._suspended = False
    view._on_playback_change = None
    view._after_id = None
    view.show_reference_var = SimpleNamespace(get=lambda: False)
    view.follow_ship_var = SimpleNamespace(get=lambda: False)
    view.settings_visible_var = _Value(settings_visible)
    view._build_axes()
    view._build_controls()
    view.figure.set_layout_engine(_IMMPredictorLayout(view))
    return view


@pytest.mark.parametrize("coordinate_mode", ["m", "km"])
@pytest.mark.parametrize("forecast_enabled", [False, True])
def test_imm_forecast_switches_redraw_cached_mean_and_median(
    coordinate_mode, forecast_enabled
):
    from dataclasses import replace

    from imm_predictor.gui import IMMPredictorApp

    means, medians = np.array([[14, 25], [15, 26]]), np.array([[12, 23], [13, 24]])
    stage = replace(
        _imm_motion_stage(),
        forecast=SimpleNamespace(
            position_means=means,
            time_seconds=np.array([2.0, 3.0]),
            position_covariances=np.repeat(np.eye(2)[None], 2, axis=0),
            mode_probabilities=np.tile([0.7, 0.2, 0.1], (2, 1)),
        )
        if forecast_enabled
        else None,
        forecast_median_positions=medians if forecast_enabled else None,
    )
    view = _headless_imm_panels(stage)
    view.show_reference_var = _Value(False)
    view._coordinate_display_mode = coordinate_mode
    view.store.config = replace(
        view.store.config,
        prediction_region_style="none",
        prediction_count=2 if forecast_enabled else 0,
    )
    view._draw()
    assert "Forecast (mean)" not in [line.get_label() for line in view.route_axis.lines]
    app = IMMPredictorApp.__new__(IMMPredictorApp)
    app._closing = False
    app._analysis_active = True
    app.config = view.store.config
    app.store, app.plot_view = view.store, view
    for show_median, show_mean in [
        (True, False),
        (False, True),
        (True, True),
        (False, False),
    ]:
        assert app._apply_plot_settings(
            replace(
                app.config,
                show_forecast_median=show_median,
                show_forecast_mean=show_mean,
            )
        )
        lines = {line.get_label(): line for line in view.route_axis.lines}
        assert ("Forecast (median)" in lines) == (forecast_enabled and show_median)
        assert ("Forecast (mean)" in lines) == (forecast_enabled and show_mean)
        scale = 0.001 if coordinate_mode == "km" else 1
        if forecast_enabled and show_median:
            np.testing.assert_allclose(
                lines["Forecast (median)"].get_xydata()[1:], medians * scale
            )
        if forecast_enabled and show_mean:
            np.testing.assert_allclose(
                lines["Forecast (mean)"].get_xydata()[1:], means * scale
            )
        if forecast_enabled and show_mean and show_median:
            assert (
                lines["Forecast (mean)"].get_linestyle()
                != lines["Forecast (median)"].get_linestyle()
            )
        assert view.current_state.stage is stage


def test_imm_motion_panels_draw_fused_gaussian_marginals_in_display_units():
    view = _headless_imm_panels(_imm_motion_stage())
    view._draw()

    for axis, mean, sigma in zip(
        view.motion_axes, [4.0, 30.0, 2.0], [1, 10, 0.5], strict=True
    ):
        line = axis.lines[0]
        x, density = line.get_data()
        peak = np.argmax(density)
        assert x[peak] == pytest.approx(mean)
        assert density[peak] == pytest.approx(1 / (sigma * np.sqrt(2 * np.pi)))
        assert np.trapezoid(density, x) == pytest.approx(1.0, abs=1e-4)
        assert axis.get_ylabel() == "Density"
        assert "Gaussian approximation" in axis.get_legend_handles_labels()[1]
    assert [axis.get_title() for axis in view.motion_axes] == [
        "Speed posterior",
        "Heading posterior",
        "Turn-rate posterior",
    ]


def _imm_mixture_stage(*, weights=(0.2, 0.7, 0.1), point_mass=False, headings=None):
    from dataclasses import replace

    from imm_predictor.player import IMMStateMixture, IMMStatePosterior

    components = []
    for index, (speed, sigma) in enumerate(
        zip([1, 5, 9], [0.2, 0.4, 0.8], strict=True)
    ):
        mean = [
            11,
            22,
            np.deg2rad(30 if headings is None else headings[index]),
            np.deg2rad(2 * index),
            speed,
        ]
        covariance = np.diag([1, 1, np.deg2rad(2) ** 2, np.deg2rad(0.5) ** 2, sigma**2])
        if point_mass and index == 0:
            covariance[3, 3] = 0.0
        components.append(IMMStatePosterior(mean, covariance))
    return replace(
        _imm_motion_stage(),
        posterior_mixture=IMMStateMixture(weights, tuple(components)),
    )


def test_imm_motion_panels_preserve_weighted_mixture_peaks():
    view = _headless_imm_panels(_imm_mixture_stage())
    view._draw()
    axis = view.motion_axes[0]
    x, density = axis.lines[0].get_data()
    for speed, expected in zip(
        [1, 5, 9], [0.39894228, 0.69814918, 0.04986779], strict=True
    ):
        assert density[np.argmin(abs(x - speed))] == pytest.approx(expected, abs=1e-6)
    assert np.trapezoid(density, x) == pytest.approx(1, abs=1e-4)
    assert np.trapezoid(x * density, x) == pytest.approx(4.6, abs=1e-3)
    assert density[np.argmin(abs(x - 1))] > density[np.argmin(abs(x - 3))]
    assert "Posterior density" in axis.get_legend_handles_labels()[1]


@pytest.mark.parametrize("mixture_backend", [False, True])
def test_imm_motion_panels_share_one_density_legend(mixture_backend):
    from dataclasses import replace

    stage = _imm_mixture_stage() if mixture_backend else _imm_motion_stage()
    view = _headless_imm_panels(stage)
    view._draw()
    curves = [axis.lines[0].get_xydata().copy() for axis in view.motion_axes]
    expected_label = (
        "Posterior density" if mixture_backend else "Gaussian approximation"
    )
    for show_legend in (True, False, True):
        view.store.config = replace(view.store.config, show_legend=show_legend)
        view._draw()
        legend = view.motion_axes[0].get_legend()
        if show_legend:
            assert [text.get_text() for text in legend.get_texts()] == [expected_label]
        else:
            assert legend is None
        assert all(axis.get_legend() is None for axis in view.motion_axes[1:])
        assert view.current_state.stage is stage
        for axis, curve in zip(view.motion_axes, curves, strict=True):
            np.testing.assert_array_equal(axis.lines[0].get_xydata(), curve)


def test_imm_motion_panels_keep_cv_turn_rate_mass_separate_from_density():
    view = _headless_imm_panels(
        _imm_mixture_stage(weights=(0.6, 0.3, 0.1), point_mass=True)
    )
    view._draw()
    axis = view.motion_axes[2]
    x, density = axis.lines[0].get_data()
    assert np.trapezoid(density, x) == pytest.approx(0.4, abs=1e-4)
    assert axis.get_ylabel() == "Density"
    assert "Point mass at 0: 60.0%" in axis.get_legend_handles_labels()[1]
    assert axis.lines[1].get_xdata() == pytest.approx([0, 0])
    assert axis.get_legend() is None
    assert "Point mass at 0: 60.0%" in [text.get_text() for text in axis.texts]


def test_imm_motion_panels_refresh_the_model_weights_without_adding_axes():
    view = _headless_imm_panels(_imm_mixture_stage())
    view._draw()
    x, density = view.motion_axes[0].lines[0].get_data()
    original = density[np.argmin(abs(x - 1))]
    axes_count = len(view.figure.axes)
    view.current_state = SimpleNamespace(
        stage=_imm_mixture_stage(weights=(0.8, 0.1, 0.1))
    )
    view._draw()
    x, density = view.motion_axes[0].lines[0].get_data()
    assert density[np.argmin(abs(x - 1))] == pytest.approx(4 * original)
    assert len(view.figure.axes) == axes_count


def test_imm_heading_mixture_uses_one_local_branch_at_the_angle_boundary():
    view = _headless_imm_panels(
        _imm_mixture_stage(weights=(0.5, 0.5, 0), headings=(179, -179, 0))
    )
    view._draw()
    axis = view.motion_axes[1]
    left, right = axis.get_xlim()
    assert right - left < 25
    x, density = axis.lines[0].get_data()
    assert abs(abs(x[np.argmax(density)]) - 180) < 1
    assert np.trapezoid(density, x) == pytest.approx(1, abs=1e-4)


@pytest.mark.parametrize("sigma", [1e-6, 1e-12])
def test_imm_mixture_does_not_bridge_separated_narrow_peaks(sigma):
    from dataclasses import replace

    from imm_predictor.player import IMMStateMixture, IMMStatePosterior

    components = tuple(
        IMMStatePosterior([0, 0, 0, 0, mean], np.diag([1, 1, 1, 1, sigma**2]))
        for mean in (0, 1, 2)
    )
    stage = replace(
        _imm_motion_stage(),
        posterior_mixture=IMMStateMixture([0.5, 0.5, 0], components),
    )
    view = _headless_imm_panels(stage)
    view._draw()
    x, density = view.motion_axes[0].lines[0].get_data()
    assert np.interp(0.5, x, density) < 1e-6
    assert np.trapezoid(density, x) == pytest.approx(1, abs=1e-4)


def test_imm_point_mass_weights_remain_visible_without_a_legend():
    from dataclasses import replace

    view = _headless_imm_panels(
        _imm_mixture_stage(weights=(0.6, 0.3, 0.1), point_mass=True)
    )
    view.store.config = replace(view.store.config, show_legend=False)
    view._draw()
    axis = view.motion_axes[2]
    assert axis.get_legend() is None
    assert any("60.0%" in text.get_text() for text in axis.texts)


def test_imm_fused_only_fallback_remains_identified_without_a_legend():
    from dataclasses import replace

    view = _headless_imm_panels(_imm_motion_stage())
    view.store.config = replace(view.store.config, show_legend=False)
    view._draw()
    for axis in view.motion_axes:
        assert axis.get_legend() is None
        assert "Gaussian approximation" in [text.get_text() for text in axis.texts]


@pytest.mark.parametrize("show_legend", [False, True])
def test_imm_zero_variance_fused_fallback_is_identified(show_legend):
    from dataclasses import replace

    view = _headless_imm_panels(_imm_motion_stage(zero_variance=True))
    view.store.config = replace(view.store.config, show_legend=show_legend)
    view._draw()
    for axis in view.motion_axes:
        assert any("Fused-state fallback" in text.get_text() for text in axis.texts)


def test_imm_pure_cv_turn_rate_uses_a_probability_stem_without_invented_density():
    view = _headless_imm_panels(_imm_mixture_stage(weights=(1, 0, 0), point_mass=True))
    view._draw()
    axis = view.motion_axes[2]
    assert axis.get_ylabel() == "Probability mass"
    x, probability = axis.lines[0].get_data()
    np.testing.assert_array_equal(x, [0, 0])
    np.testing.assert_array_equal(probability, [0, 1])
    assert len(axis.lines) == 1


@pytest.mark.parametrize("coordinate_mode", ["m", "km"])
def test_imm_sidebar_uses_mixture_medians_and_keeps_reference_values(coordinate_mode):
    from imm_predictor.controls import IMMControls
    from imm_predictor.player import IMMForecastMetrics, _posterior_medians

    panel = IMMControls.__new__(IMMControls)
    panel._values = {
        name: _Value("old")
        for name in (
            "speed",
            "heading",
            "turn_rate",
            "reference_position",
            "reference_speed",
            "reference_heading",
            "reference_turn_rate",
            "reference_time",
            "ade",
            "fde",
            "rmse",
            "energy_score",
            "negative_log_predictive_density",
            "coverage_50",
            "coverage_90",
            "update_time",
            "association_status",
            "detection_id",
            "nis",
            "cov_xx",
            "cov_xy",
            "cov_yy",
            "cycle",
        )
    }
    stage = _imm_mixture_stage(weights=(0.6, 0.3, 0.1), point_mass=True)
    medians = _posterior_medians(stage)
    state = SimpleNamespace(
        stage=stage,
        posterior_medians=medians,
        metrics=IMMForecastMetrics(None, None, None, None, None, 0, 0, 0),
        reference_state=SimpleNamespace(
            position_meters=np.array([1000, -2000]),
            speed_mps=3.0,
            heading_degrees=-4.0,
            turn_rate_degrees_per_second=0.5,
            time_seconds=12,
            speed_is_derived=False,
            heading_is_derived=False,
            turn_rate_is_derived=True,
        ),
    )
    panel.show_display_state(state, coordinate_display_mode=coordinate_mode)
    assert panel._values["speed"].get() == "1.19 m/s"
    assert panel._values["heading"].get() == "30.00 °"
    assert panel._values["turn_rate"].get() == "0.00 °/s"
    assert panel._values["reference_turn_rate"].get() == "0.50 °/s (derived)"
    assert panel._values["reference_position"].get() == (
        "x = 1000.00 m\ny = -2000.00 m"
        if coordinate_mode == "m"
        else "x = 1.00 km\ny = -2.00 km"
    )
    state.posterior_medians = None
    panel.show_display_state(state, coordinate_display_mode=coordinate_mode)
    assert all(
        panel._values[name].get() == "-" for name in ("speed", "heading", "turn_rate")
    )


def test_imm_sidebar_replaces_track_state_with_three_posterior_medians(root):
    from imm_predictor.controls import IMMControls

    panel = IMMControls(
        root,
        on_open_file=lambda _path: None,
        on_cancel=lambda: None,
        on_start=lambda: None,
    )
    widgets = list(panel.winfo_children())
    sections = {}
    while widgets:
        widget = widgets.pop()
        if isinstance(widget, tk.LabelFrame):
            sections[widget.cget("text")] = widget
        widgets.extend(widget.winfo_children())
    assert "IMM track state" not in sections
    labels = [
        widget.cget("text")
        for widget in sections["Posterior medians"].winfo_children()
        if isinstance(widget, tk.Label) and not widget.cget("textvariable")
    ]
    assert labels == ["Speed:", "Heading:", "Turn rate:"]
    assert "position" not in panel._values
    panel.clear_display_state()
    assert all(value.get() == "-" for value in panel._values.values())


@pytest.mark.parametrize("figure_size", [(6.5, 6), (7, 6), (10, 6), (14, 7.5)])
def test_imm_panel_switch_preserves_stage_route_and_readable_layout(figure_size):
    stage = _imm_motion_stage()
    view = _headless_imm_panels(stage, figure_size=figure_size)
    view._draw()
    limits = view.route_axis.get_xlim(), view.route_axis.get_ylim()
    assert all(axis.get_visible() for axis in view.motion_axes)
    assert not view.mode_axis.get_visible()
    renderer = view.canvas.get_renderer()
    for label in view.group_selector.labels:
        assert label.get_window_extent(renderer).x1 <= view.figure.bbox.x1
    slider_box = view.slider.ax.get_window_extent(renderer)
    selector_box = view.group_selector.ax.get_window_extent(renderer)
    assert slider_box.x1 < selector_box.x0
    for upper, lower in zip(view.motion_axes[:-1], view.motion_axes[1:], strict=True):
        assert upper.get_tightbbox(renderer).y0 > lower.get_tightbbox(renderer).y1

    view.group_selector.set_active(1)
    assert view.group_selector.value_selected == "Forecast uncertainty"
    assert all(not axis.get_visible() for axis in view.motion_axes)
    assert view.mode_axis.get_visible() and view.uncertainty_axis.get_visible()
    assert view.mode_axis.get_title() == "Forecast mode probabilities"
    assert view.current_state.stage is stage
    np.testing.assert_allclose(
        (view.route_axis.get_xlim(), view.route_axis.get_ylim()), limits
    )
    view.group_selector.set_active(0)
    assert all(axis.get_visible() for axis in view.motion_axes)
    assert view.motion_axes[0].lines[0].get_xdata()[200] == pytest.approx(4.0)
    view.disable_navigation()
    assert not view.group_selector.active


@pytest.mark.parametrize("selected_group", [0, 1])
def test_imm_hiding_settings_shows_both_live_panel_groups_and_restores_selection(
    selected_group,
):
    from dataclasses import replace

    from imm_predictor.player import IMMStatePosterior, _forecast_uncertainty

    stage = replace(
        _imm_motion_stage(),
        forecast=SimpleNamespace(
            position_means=np.array([[12.0, 23.0], [13.0, 24.0]]),
            time_seconds=np.array([2.0, 3.0]),
            position_covariances=np.repeat(np.eye(2)[None], 2, axis=0),
            mode_probabilities=np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]]),
        ),
    )
    view = _headless_imm_panels(stage, figure_size=(16, 8))
    view.store.config = replace(
        view.store.config, prediction_count=2, prediction_region_style="none"
    )
    view.current_state = SimpleNamespace(
        stage=stage, uncertainty=_forecast_uncertainty(stage)
    )
    view._on_settings_visibility_change = view.set_settings_visible
    view._draw()
    view.group_selector.set_active(selected_group)
    previous_selection = view.group_selector.value_selected
    limits = view.route_axis.get_xlim(), view.route_axis.get_ylim()
    compact_slider_width = view.slider.ax.get_position().width

    view.settings_checkbox.set_active(0, state=False)

    assert view.settings_visible_var.get() is False
    assert all(
        axis.get_visible()
        for axis in (*view.motion_axes, view.mode_axis, view.uncertainty_axis)
    )
    assert not view.group_selector.ax.get_visible()
    assert not view.group_selector.active
    assert not view.group_selector._buttons.get_visible()
    assert all(not label.get_visible() for label in view.group_selector.labels)
    assert view.slider.ax.get_position().width > compact_slider_width
    assert view.route_axis.get_position().x1 < view.motion_axes[0].get_position().x0
    assert view.motion_axes[0].get_position().x1 < view.mode_axis.get_position().x0
    assert view.current_state.stage is stage
    np.testing.assert_allclose(
        (view.route_axis.get_xlim(), view.route_axis.get_ylim()), limits
    )
    renderer = view.canvas.get_renderer()
    assert (
        view.motion_axes[0].get_tightbbox(renderer).x1
        < view.mode_axis.get_tightbbox(renderer).x0
    )

    next_stage = replace(
        stage,
        observation_count=3,
        posterior=IMMStatePosterior([11, 22, 0.3, 0.01, 6.0], np.eye(5)),
        forecast=SimpleNamespace(
            **(
                vars(stage.forecast)
                | {"mode_probabilities": np.array([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3]])}
            )
        ),
    )
    view.current_state = SimpleNamespace(
        stage=next_stage, uncertainty=_forecast_uncertainty(next_stage)
    )
    view._draw()
    x, density = view.motion_axes[0].lines[0].get_data()
    assert x[np.argmax(density)] == pytest.approx(6.0)
    np.testing.assert_array_equal(view.mode_axis.lines[0].get_ydata(), [0.4, 0.3])

    view.settings_checkbox.set_active(0, state=True)

    assert view.group_selector.value_selected == previous_selection
    assert view.group_selector.active
    assert view.group_selector.ax.get_visible()
    assert all(axis.get_visible() == (selected_group == 0) for axis in view.motion_axes)
    assert view.mode_axis.get_visible() == (selected_group == 1)
    assert view.uncertainty_axis.get_visible() == (selected_group == 1)
    assert view.current_state.stage is next_stage
    view.disable_navigation()
    view.set_settings_visible(False)
    view.set_settings_visible(True)
    assert not view.group_selector.active


@pytest.mark.parametrize("initial_stage", [None, "posterior"])
def test_imm_initially_hidden_settings_keep_both_groups_visible_after_resize(
    initial_stage,
):
    stage = None if initial_stage is None else _imm_motion_stage()
    view = _headless_imm_panels(stage, figure_size=(14, 8), settings_visible=False)
    view._draw()
    view.figure.set_size_inches(18, 9)
    view.canvas.draw()

    assert all(
        axis.get_visible()
        for axis in (*view.motion_axes, view.mode_axis, view.uncertainty_axis)
    )
    assert not view.group_selector.ax.get_visible()
    assert not view.group_selector.active
    assert view.mode_axis.get_title() == "Forecast mode probabilities"
    assert view.uncertainty_axis.get_title() == "Forecast uncertainty"
    assert "Forecast disabled (0 steps)" in [
        text.get_text() for text in view.mode_axis.texts
    ]
    assert view.motion_axes[0].get_title() == "Speed posterior"
    renderer = view.canvas.get_renderer()
    for upper, lower in zip(view.motion_axes[:-1], view.motion_axes[1:], strict=True):
        assert upper.get_tightbbox(renderer).y0 > lower.get_tightbbox(renderer).y1
    assert (
        view.mode_axis.get_tightbbox(renderer).y0
        > view.uncertainty_axis.get_tightbbox(renderer).y1
    )


def test_imm_motion_panels_show_no_invented_prior_at_n_zero():
    view = _headless_imm_panels(None)
    view._draw()
    for axis in view.motion_axes:
        assert not axis.lines
        assert "Available after\nN = 1" in [text.get_text() for text in axis.texts]


def test_imm_motion_panels_preserve_zero_variance_as_point_masses():
    view = _headless_imm_panels(_imm_motion_stage(zero_variance=True))
    view._draw()
    for axis, mean in zip(view.motion_axes, [4.0, 30.0, 2.0], strict=True):
        assert axis.get_ylabel() == "Probability mass"
        assert axis.lines[0].get_xdata() == pytest.approx([mean, mean])
        assert "Point mass" in axis.get_legend_handles_labels()[1]


@pytest.fixture
def root():
    try:
        window = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk display unavailable: {error}")
    window.geometry("1000x700")
    window.withdraw()
    errors = []
    window.report_callback_exception = lambda *args: errors.append(args)
    yield window
    try:
        window.destroy()
    except tk.TclError:
        pass
    assert not errors


@pytest.mark.parametrize(
    ("content_height", "screen_height", "expected_height"),
    (
        (540, 1080, 540),
        (180, 1080, 320),
        (900, 800, 680),
    ),
)
def test_compact_dialog_height_tracks_content_and_screen_limit(
    content_height,
    screen_height,
    expected_height,
):
    """Short editors stay compact while longer forms remain scrollable."""
    from trajectory_predictor.dialogs import dialog_height_for_content

    assert dialog_height_for_content(content_height, screen_height) == expected_height


@pytest.mark.parametrize(
    ("group", "expected_width"),
    (
        ("smc", 470),
        ("vi", 470),
        ("mcmc", 470),
        ("priors", 560),
        ("data", 560),
    ),
)
def test_settings_dialog_width_matches_the_edited_group(group, expected_width):
    """Small inference forms do not use the wider data-editor layout."""
    from trajectory_predictor.dialogs import dialog_width_for_group

    assert dialog_width_for_group(group, screen_width=1920) == expected_width


def test_settings_dialog_focuses_without_grabbing_the_main_window():
    """The main window close control remains available while a dialog is open."""
    from trajectory_predictor.dialogs import SettingsDialog

    calls = []
    dialog = SimpleNamespace(
        grab_set=lambda: calls.append("grab"),
        focus_set=lambda: calls.append("focus"),
    )

    SettingsDialog._on_map(dialog, SimpleNamespace(widget=dialog))

    assert calls == ["focus"]


def test_settings_dialog_is_positioned_before_it_is_shown(root):
    """A settings dialog must not visibly move from Tk's default position."""
    from trajectory_predictor.dialogs import SettingsDialog

    calls = []

    class ProbeDialog(SettingsDialog):
        def withdraw(self):
            calls.append("withdraw")
            return super().withdraw()

        def update_idletasks(self):
            calls.append("layout")
            return super().update_idletasks()

        def geometry(self, value):
            calls.append("geometry")
            return super().geometry(value)

        def deiconify(self):
            calls.append("deiconify")
            return super().deiconify()

    panel = SettingsPanel(root, lambda: None)
    dialog = ProbeDialog(root, panel, "priors")
    try:
        assert calls.index("withdraw") < calls.index("layout")
        assert calls.index("layout") < calls.index("geometry")
        assert calls.index("geometry") < calls.index("deiconify")
    finally:
        dialog.cancel()


def test_plot_display_window_is_positioned_before_it_is_shown(root):
    """The plot-display window must not visibly move from Tk's default position."""
    from trajectory_predictor.dialogs import PlotDisplayWindow

    calls = []

    class ProbeWindow(PlotDisplayWindow):
        def withdraw(self):
            calls.append("withdraw")
            return super().withdraw()

        def update_idletasks(self):
            calls.append("layout")
            return super().update_idletasks()

        def geometry(self, value):
            calls.append("geometry")
            return super().geometry(value)

        def deiconify(self):
            calls.append("deiconify")
            return super().deiconify()

    panel = SettingsPanel(root, lambda: None)
    window = ProbeWindow(root, panel)
    try:
        assert calls.index("withdraw") < calls.index("layout")
        assert calls.index("layout") < calls.index("geometry")
        assert calls.index("geometry") < calls.index("deiconify")
    finally:
        window.close()


def test_gui_remains_responsive_and_replaces_analysis(root, tmp_path, monkeypatch):
    entered, release = Event(), Event()
    prepares = []

    def prepare(settings):
        prepares.append(settings)
        coordinates = np.arange(4.0)
        trajectory = PosteriorDashboardTrajectory(*([coordinates] * 4))

        def load(count):
            entered.set()
            assert release.wait(8)
            return PosteriorDashboardUpdate(
                count, {name: np.linspace(1, 2, 20) for name in PARAMETER_NAMES}
            )

        return trajectory, 4, 1, load

    worker = PosteriorAnalysisWorker(prepare)
    app = TrajectoryPredictor(root, worker=worker)
    assert not hasattr(app, "status_label")
    messages = []
    monkeypatch.setattr(
        "trajectory_predictor.gui.messagebox.showerror",
        lambda *args, **kwargs: messages.append(args),
    )
    data_file = tmp_path / "route.csv"
    data_file.touch()
    app.controls.variables["data"]["data_file"].set(str(data_file))
    try:
        app.start_analysis()
        pump_until(root, lambda: app.plot_view is not None)
        first_view = app.plot_view
        navigator = first_view.navigator
        assert type(navigator._playback_timer).__name__ == "TimerTk"
        navigator.toggle_playback(None)
        navigator.advance_playback()
        pump_until(root, entered.is_set)
        heartbeat = []
        root.after(0, lambda: heartbeat.append(True))
        pump_until(root, lambda: bool(heartbeat))
        assert navigator.observation_count == 0
        app.handle_space(SimpleNamespace(widget=root))
        assert not navigator.is_playing
        release.set()
        pump_until(root, lambda: bool(navigator._updates_by_count))
        assert navigator.observation_count == 0
        entry = tk.Entry(root)
        assert app.handle_space(SimpleNamespace(widget=entry)) is None
        assert not navigator.is_playing
        assert (
            app.handle_space(SimpleNamespace(widget=first_view.canvas.get_tk_widget()))
            is None
        )
        app.handle_space(SimpleNamespace(widget=root))
        assert navigator.is_playing
        navigator.advance_playback()
        navigator.pause_playback()
        assert navigator.observation_count == 1
        original_limits = navigator.trajectory_axis.get_xlim()
        first_view.toolbar.push_current()
        navigator.trajectory_axis.set_xlim(0.5, 1.5)
        app.reset_plot_view()
        assert navigator.trajectory_axis.get_xlim() == original_limits
        assert navigator.observation_count == 1
        app.toggle_settings()
        assert not app.settings_visible
        app.toggle_settings()
        assert app.settings_visible
        # Invalid edits must leave the currently displayed analysis intact.
        app.controls.variables["data"]["start_index"].set("bad")
        app.start_analysis()
        assert app.plot_view is first_view
        assert messages
        app.controls.variables["data"]["start_index"].set("0")
        app.controls.variables["data"]["inference_method"].set("smc")
        app.start_analysis()
        pump_until(root, lambda: app.plot_view is not None)
        assert app.plot_view is not first_view
        assert app.plot_view.navigator.observation_count == 0
        assert not app.plot_view.navigator._updates_by_count
        assert prepares[-1].experiment.inference_method == "smc"
    finally:
        release.set()
        app.close()
        worker.join(5)
        root.update()
    assert not worker.is_alive


def test_worker_error_is_shown_in_an_analysis_failed_dialog(root, monkeypatch):
    app = TrajectoryPredictor(root)
    messages = []
    monkeypatch.setattr(
        "trajectory_predictor.gui.messagebox.showerror",
        lambda *args, **kwargs: messages.append((args, kwargs)),
    )
    try:
        app._show_error("ValueError: Run ID 999 was not found.")

        assert messages == [
            (
                ("Analysis failed", "ValueError: Run ID 999 was not found."),
                {"parent": root},
            )
        ]
    finally:
        app.close()
        app.worker.join(5)
        root.update()


def test_settings_panel_scrolls_analysis_and_data_together(root):
    panel = SettingsPanel(root, lambda: None)
    panel.pack(fill="both", expand=True)
    root.update_idletasks()
    action_container = panel.apply_button.master
    assert panel.settings_form.master is panel
    assert action_container.master is panel.settings_form.body
    assert panel.data_section.master is panel.settings_form.body
    assert action_container.winfo_width() == panel.data_section.winfo_width()
    packed_sections = panel.settings_form.body.pack_slaves()
    assert packed_sections.index(action_container) < packed_sections.index(
        panel.data_section
    )
    assert packed_sections[-1] is panel.data_section


def test_analysis_section_contains_method_and_start_button(root):
    panel = SettingsPanel(root, lambda: None)
    panel.pack(fill="both", expand=True)
    root.update_idletasks()

    assert panel.analysis_fields.master is panel.analysis_section
    assert panel.analysis_section.pack_slaves() == [
        panel.apply_button,
        panel.analysis_fields,
    ]
    assert panel.apply_button.cget("text") == "Start analysis"
    method_field = next(
        child
        for child in panel.analysis_fields.winfo_children()
        if child.winfo_class() == "TCombobox"
    )
    assert method_field.cget("textvariable") == str(
        panel.variables["data"]["inference_method"]
    )
    assert method_field.grid_info()["column"] == "0"


def test_inference_method_is_grouped_with_analysis_fields():
    analysis, data = split_analysis_data_fields(
        {"data_file": "route.csv", "inference_method": "smc"}
    )

    assert analysis == {"inference_method": "smc"}
    assert data == {"data_file": "route.csv"}


def test_sidebar_keeps_only_data_selection_and_moves_analysis_inputs_to_settings():
    """The default sidebar exposes CSV selection, not inference input settings."""
    assert MAIN_DATA_FIELDS == (
        "data_file",
        "inference_method",
    )
    assert DATA_OPTION_FIELDS[:5] == (
        "start_index",
        "observation_interval_seconds",
        "maximum_observation_count",
        "position_noise_std_m",
        "position_noise_seed",
    )
    assert "coordinate_display_mode" not in DATA_OPTION_FIELDS


def test_start_index_label_and_field_share_vertical_center(root):
    from trajectory_predictor.dialogs import SettingsDialog

    panel = SettingsPanel(root, lambda: None)
    dialog = SettingsDialog(root, panel, "data")
    try:
        dialog.update_idletasks()
        body = dialog._form.body

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        label = next(
            child
            for child in descendants(body)
            if child.winfo_class() == "Label"
            and child.cget("text") == "Start index (from 0)"
        )
        field = next(
            child
            for child in descendants(body)
            if child.winfo_class() == "TEntry"
            and child.cget("textvariable") == str(dialog.variables["start_index"])
        )
        label_center = label.winfo_rooty() + label.winfo_height() / 2
        field_center = field.winfo_rooty() + field.winfo_height() / 2
        assert label_center == pytest.approx(field_center, abs=1.0)
    finally:
        dialog.cancel()


def test_data_file_browse_button_matches_windows_width_and_entry_height(root):
    """The file chooser remains a regular small button across platforms."""
    from trajectory_predictor.dialogs import SettingsDialog

    panel = SettingsPanel(root, lambda: None)
    dialog = SettingsDialog(root, panel, "data")
    try:
        dialog.update_idletasks()

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        browse_button = next(
            child
            for child in descendants(dialog._form.body)
            if child.winfo_class() == "TButton" and child.cget("text") == "…"
        )
        data_file_entry = next(
            child
            for child in descendants(dialog._form.body)
            if child.winfo_class() == "TEntry"
            and child.cget("textvariable") == str(dialog.variables["data_file"])
        )
        assert 35 <= browse_button.winfo_reqwidth() <= 60
        assert browse_button.cget("style") == "Predictor.TButton"
        assert browse_button.winfo_height() == data_file_entry.winfo_height()
    finally:
        dialog.cancel()


def test_data_dialog_contains_only_data_and_replay_options(root):
    from trajectory_predictor.dialogs import SettingsDialog

    panel = SettingsPanel(root, lambda: None)
    dialog = SettingsDialog(root, panel, "data")
    try:
        section_labels = {
            section.cget("text")
            for section in dialog._form.body.winfo_children()
            if section.winfo_class() == "Labelframe"
        }
        assert section_labels == {"Analysis setup"}
    finally:
        dialog.cancel()


def test_priors_dialog_explains_the_configured_distribution_families(root):
    from trajectory_predictor.dialogs import SettingsDialog, prior_label_width

    panel = SettingsPanel(root, lambda: None)
    dialog = SettingsDialog(root, panel, "priors")
    try:
        assert dialog._header is None
        section_labels = [
            section.cget("text")
            for section in dialog._form.body.winfo_children()
            if section.winfo_class() == "Labelframe"
        ]

        assert section_labels == [
            "Speed — half-normal distribution",
            "Turn rate — normal distribution around 0",
            "Observation noise — exponential distribution",
            "Speed process noise — exponential distribution",
            "Turn-rate process noise — exponential distribution",
        ]
        assert any(
            child.cget("text") == "Heading — uniform distribution (−180° to 180°)"
            for child in dialog._form.body.winfo_children()
            if child.winfo_class() == "Label"
        )
        assert all(
            "bold" in section.cget("font").lower()
            for section in dialog._form.body.winfo_children()
            if section.winfo_class() == "Labelframe"
        )
        dialog.update_idletasks()
        entry_x_positions = {
            child.winfo_x()
            for section in dialog._form.body.winfo_children()
            if section.winfo_class() == "Labelframe"
            for child in section.winfo_children()
            if child.winfo_class() == "TEntry"
        }
        assert len(entry_x_positions) == 1
        label_widths = {
            child.cget("width")
            for section in dialog._form.body.winfo_children()
            if section.winfo_class() == "Labelframe"
            for child in section.winfo_children()
            if child.winfo_class() == "Label"
        }
        assert label_widths == {prior_label_width()}
        assert all(
            child.cget("anchor") == "w"
            for section in dialog._form.body.winfo_children()
            if section.winfo_class() == "Labelframe"
            for child in section.winfo_children()
            if child.winfo_class() == "Label"
        )
    finally:
        dialog.cancel()


def test_plot_dialog_contains_display_options(root):
    from trajectory_predictor.dialogs import SettingsDialog

    panel = SettingsPanel(root, lambda: None)
    dialog = SettingsDialog(root, panel, "plot")
    try:
        assert (
            dialog.variables["prediction_region_style"].get()
            == "Horizon density region"
        )
        labels = {
            child.cget("text")
            for child in dialog._form.body.winfo_children()
            if child.winfo_class() == "Checkbutton"
        }
        assert labels == {
            "Show legend",
            "Show reference trajectory",
            "Show observations / Bayesian track through N",
            "Show current position",
            "Show possible future trajectories",
            "Show forecast (median)",
            "Show 50% posterior-predictive region",
            "Show 90% posterior-predictive region",
        }
    finally:
        dialog.cancel()


def test_plot_display_initial_values_translate_prediction_style():
    from trajectory_predictor.dialogs import _plot_display_initial_values

    values = _plot_display_initial_values(
        {
            "prediction_region_style": "funnel",
            "follow_ship_view_span_m": 600.0,
        }
    )

    assert values == {
        "prediction_region_style": "Horizon density region",
        "follow_ship_view_span_m": 600.0,
    }


def test_plot_display_window_closes_from_its_cancel_button(root):
    from trajectory_predictor.dialogs import PlotDisplayWindow
    from trajectory_predictor.settings import DISPLAY_OPTION_FIELDS

    panel = SimpleNamespace(
        variables={
            "data": {
                key: tk.BooleanVar(root, value=True) for key in DISPLAY_OPTION_FIELDS
            }
            | {
                "prediction_region_style": tk.StringVar(root, value="funnel"),
                "follow_ship_view_span_m": tk.StringVar(root, value="600"),
            }
        }
    )
    closed = []
    window = PlotDisplayWindow(root, panel, on_close=lambda: closed.append(True))
    try:
        assert (
            window.variables["prediction_region_style"].get()
            == "Horizon density region"
        )
        prediction_region_selector = next(
            child
            for frame in window.winfo_children()
            for child in frame.winfo_children()
            if isinstance(child, ttk.Combobox)
        )
        assert int(prediction_region_selector.cget("width")) == 30
        button_labels = {
            child.cget("text")
            for frame in window.winfo_children()
            for child in frame.winfo_children()
            if isinstance(child, tk.Button)
        }
        assert button_labels == {"Apply", "Cancel"}
        cancel_button = next(
            child
            for frame in window.winfo_children()
            for child in frame.winfo_children()
            if isinstance(child, tk.Button) and child.cget("text") == "Cancel"
        )
        cancel_button.invoke()

        assert closed == [True]
        assert not window.winfo_exists()
    finally:
        if window.winfo_exists():
            window.close()


def test_invalid_follow_ship_view_span_shows_an_error_instead_of_raising(monkeypatch):
    from trajectory_predictor.gui import TrajectoryPredictor
    from trajectory_predictor.settings import DISPLAY_OPTION_FIELDS

    def reject_display_options(_options):
        raise ValueError("follow_ship_view_span_m must be a finite positive number.")

    errors = []
    plot_view = SimpleNamespace(set_display_options=reject_display_options)
    app = TrajectoryPredictor.__new__(TrajectoryPredictor)
    app.root = object()
    app._closing = False
    app.plot_view = plot_view
    app.controls = SimpleNamespace(
        variables={
            "data": {
                key: SimpleNamespace(get=lambda: True) for key in DISPLAY_OPTION_FIELDS
            }
            | {
                "prediction_region_style": SimpleNamespace(
                    get=lambda: "Covariance ellipses"
                ),
                "follow_ship_view_span_m": SimpleNamespace(get=lambda: "nan"),
            }
        }
    )
    monkeypatch.setattr(
        "trajectory_predictor.gui.messagebox.showerror",
        lambda *args, **_kwargs: errors.append(args),
    )

    app._update_display_options()

    assert errors == [
        ("Plot display", "follow_ship_view_span_m must be a finite positive number.")
    ]


def test_plot_display_apply_commits_the_local_draft_and_closes():
    from trajectory_predictor.dialogs import PlotDisplayWindow
    from trajectory_predictor.settings import DISPLAY_OPTION_FIELDS

    draft_variables = {
        key: _Value(key != "show_legend") for key in DISPLAY_OPTION_FIELDS
    } | {
        "prediction_region_style": _Value("Per-step density contours"),
        "follow_ship_view_span_m": _Value("250"),
    }
    shared_variables = {key: _Value(True) for key in DISPLAY_OPTION_FIELDS} | {
        "connect_sample_trajectories": _Value(False),
        "prediction_region_style": _Value("Covariance ellipses"),
        "follow_ship_view_span_m": _Value("600"),
    }
    window = PlotDisplayWindow.__new__(PlotDisplayWindow)
    window.variables = draft_variables
    window.panel = SimpleNamespace(variables={"data": shared_variables})
    closed = []
    window.close = lambda: closed.append(True)

    assert shared_variables["connect_sample_trajectories"].get() is False
    window.apply()

    assert shared_variables["connect_sample_trajectories"].get() is True
    assert shared_variables["show_legend"].get() is False
    assert (
        shared_variables["prediction_region_style"].get() == "Per-step density contours"
    )
    assert shared_variables["follow_ship_view_span_m"].get() == 250.0
    assert closed == [True]


def test_dialog_cancel_and_apply_do_not_start_an_analysis(root, monkeypatch):
    # Keep the parent withdrawn: no visible test window or actual inference.
    from trajectory_predictor.dialogs import SettingsDialog

    panel = SettingsPanel(root, lambda: pytest.fail("Unexpected analysis start"))
    original = panel.values()
    dialog = SettingsDialog(root, panel, "priors")
    dialog.variables["speed_prior_upper_mps"].set("30")
    dialog.cancel()
    assert panel.values() == original
    dialog = SettingsDialog(root, panel, "smc")
    errors = []
    monkeypatch.setattr(
        "trajectory_predictor.dialogs.messagebox.showerror",
        lambda *args, **kwargs: errors.append(args),
    )
    dialog.variables["particle_count"].set("0")
    dialog.apply()
    assert dialog.winfo_exists()
    assert panel.values() == original
    assert errors
    dialog.variables["particle_count"].set("64")
    dialog.apply()
    assert not dialog.winfo_exists()
    assert panel.variables["smc"]["particle_count"].get() == "64"
    assert panel.values()["sequential_vi"] == original["sequential_vi"]


def test_data_settings_dialog_keeps_an_empty_observation_interval_after_apply(
    monkeypatch,
):
    from trajectory_predictor.dialogs import SettingsDialog

    dialog = SettingsDialog.__new__(SettingsDialog)
    dialog.group = "data"
    dialog.variables = {"observation_interval_seconds": _Value("")}
    saved_interval = _Value("10")
    dialog.panel = SimpleNamespace(
        variables={"data": {"observation_interval_seconds": saved_interval}}
    )
    closed = []
    dialog.cancel = lambda: closed.append(True)
    monkeypatch.setattr(
        "trajectory_predictor.dialogs.validate_dialog_values",
        lambda _group, _values: {"observation_interval_seconds": None},
    )

    dialog.apply()

    assert saved_interval.get() == ""
    assert closed == [True]


class _MenuRecorder:
    """Record our menu registrations without needing a Tcl display."""

    def __init__(self, _parent, **_options):
        self.entries = []
        self.destroyed = False

    def destroy(self):
        self.destroyed = True

    def add_command(self, **options):
        self.entries.append(options)

    add_cascade = add_command
    add_checkbutton = add_command
    add_radiobutton = add_command

    def add_separator(self):
        pass

    def entry(self, label):
        return next(entry for entry in self.entries if entry["label"] == label)

    def entryconfigure(self, index, **options):
        entry = self.entries[index] if isinstance(index, int) else self.entry(index)
        entry.update(options)

    def index(self, label):
        if label == "end":
            return len(self.entries) - 1
        return self.entries.index(self.entry(label))


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


@pytest.fixture
def imm_interaction_app(monkeypatch):
    from unittest.mock import Mock

    from imm_predictor import view
    from imm_predictor.gui import IMMPredictorApp
    from imm_predictor.player import IMMPlayerConfig

    callbacks, cancelled = {}, []
    tokens = []

    def schedule(delay, callback):
        token = len(tokens)
        tokens.append(token)
        callbacks[token] = (delay, callback)
        return token

    def cancel(token):
        cancelled.append(token)
        callbacks.pop(token, None)

    app = IMMPredictorApp.__new__(IMMPredictorApp)
    app.root = Mock(after=schedule, after_cancel=cancel)
    app.root.callbacks, app.root.cancelled = callbacks, cancelled
    app._closing = app._handling_error = app._resizing = app._scrolling = False
    app._resize_after_id = app._scroll_after_id = None
    app._last_window_size = (800, 600)
    app._menu_state = None
    app._error = app._settings_dialog = app._plot_settings_dialog = None
    app._help_window = None
    app._analysis_active = True
    app.config = IMMPlayerConfig()
    app.controls, app.placeholder = Mock(), Mock()
    app.settings_visible_var = _Value(True)
    app.coordinate_display_mode_var = _Value("m")
    plot = app.plot_view = _headless_imm_panels(None)
    plot._playing = plot._suspended = False
    plot._on_playback_change = app._update_menu_state
    plot._on_stage_changed = Mock()
    plot._on_error = lambda error: pytest.fail(str(error))
    plot._draw = Mock()
    plot.after, plot.after_cancel = schedule, cancel
    plot.store.maximum_observation_count = 3
    plot.store.display_state_for = Mock(
        side_effect=lambda count: SimpleNamespace(
            stage=SimpleNamespace(observation_count=count)
        )
    )
    app.store = plot.store
    monkeypatch.setattr(view.tk, "Menu", _MenuRecorder)
    view.create_menu_bar(app)
    app._update_menu_state()
    return app


def _assert_imm_menu_state(app, *, locked, active=True):
    for index, entry in enumerate(app._menu_bar.entries):
        disabled = locked or (active and index == app._settings_menu_index)
        assert entry["state"] == ("disabled" if disabled else "normal")
    assert app._file_menu.entries[app._open_csv_menu_index]["state"] == (
        "disabled" if active else "normal"
    )


def test_imm_start_wires_resize_scroll_and_playback_notifications(monkeypatch):
    from unittest.mock import Mock

    from imm_predictor import gui

    root, controls, plot = Mock(), Mock(), Mock()
    root.winfo_screenwidth.return_value = 1920
    root.winfo_screenheight.return_value = 1080
    plot.is_playing = False
    controls_factory = Mock(return_value=controls)
    plot_factory = Mock(return_value=plot)
    monkeypatch.setattr(gui.view, "configure_window", Mock())
    monkeypatch.setattr(gui.tk, "Menu", _MenuRecorder)
    monkeypatch.setattr(gui.tk, "BooleanVar", lambda _root, *, value: _Value(value))
    monkeypatch.setattr(gui.tk, "StringVar", lambda _root, *, value: _Value(value))
    monkeypatch.setattr(gui.tk, "Frame", Mock())
    monkeypatch.setattr(gui.tk, "Label", Mock())
    monkeypatch.setattr(gui, "IMMControls", controls_factory)
    monkeypatch.setattr(gui, "IMMPlotView", plot_factory)
    monkeypatch.setattr(gui, "IMMPlaybackStore", Mock())
    app = gui.IMMPredictorApp(root)

    assert controls_factory.call_args.kwargs["on_scroll"] == app._on_sidebar_scroll
    assert any(
        args == ("<Configure>", app._on_window_resize) and kwargs == {"add": "+"}
        for args, kwargs in root.bind.call_args_list
    )
    _assert_imm_menu_state(app, locked=False, active=False)
    root.update_idletasks.side_effect = lambda: _assert_imm_menu_state(app, locked=True)
    app.start_analysis()
    assert plot_factory.call_args.kwargs["on_playback_change"] == app._update_menu_state
    _assert_imm_menu_state(app, locked=False)
    controls.set_analysis_active.assert_called_with(True)


def test_imm_menus_follow_playback_pause_completion_and_cancel(imm_interaction_app):
    from unittest.mock import Mock

    app = imm_interaction_app
    plot = app.plot_view
    _assert_imm_menu_state(app, locked=False)
    plot.toggle_playback()
    _assert_imm_menu_state(app, locked=True)
    plot._advance_playback()
    assert plot._observation_count() == 1
    plot.handle_key("space")
    _assert_imm_menu_state(app, locked=False)
    plot.toggle_playback()
    plot._advance_playback()
    plot._advance_playback()
    assert plot._observation_count() == 3
    assert not plot.is_playing
    assert plot._after_id is None
    _assert_imm_menu_state(app, locked=False)
    plot.destroy = Mock()
    app.reload_session()
    _assert_imm_menu_state(app, locked=False, active=False)


def test_imm_playback_error_unlocks_menus_and_stops_steps(
    imm_interaction_app, monkeypatch
):
    from unittest.mock import Mock

    from imm_predictor import gui

    app = imm_interaction_app
    plot = app.plot_view
    plot._on_error = lambda error: app._report_callback_exception(
        type(error), error, error.__traceback__
    )
    plot.store.display_state_for.side_effect = ValueError("invalid radar input")
    show_error = Mock()
    monkeypatch.setattr(gui.messagebox, "showerror", show_error)
    plot.toggle_playback()
    plot._advance_playback()
    assert not plot.is_playing
    assert plot._after_id is None
    _assert_imm_menu_state(app, locked=False, active=False)
    show_error.assert_called_once()
    plot._advance_playback()
    plot.store.display_state_for.assert_called_once_with(1)


@pytest.mark.parametrize("manual_pause", [False, True])
def test_imm_scroll_defers_steps_and_preserves_playback_intent(
    imm_interaction_app, manual_pause
):
    from imm_predictor.gui import SIDEBAR_SCROLL_DELAY_MS

    app = imm_interaction_app
    plot = app.plot_view
    plot.toggle_playback()
    _, late_step = app.root.callbacks[plot._after_id]
    app._on_sidebar_scroll()
    first_resume = app._scroll_after_id
    app._on_sidebar_scroll()
    assert first_resume in app.root.cancelled
    assert plot._after_id is None
    late_step()
    assert not plot.show_observation_count(2)
    plot.store.display_state_for.assert_not_called()
    plot._draw.assert_not_called()
    assert plot.is_playing
    _assert_imm_menu_state(app, locked=True)
    if manual_pause:
        plot.handle_key("space")
    delay, resume = app.root.callbacks[app._scroll_after_id]
    assert delay == SIDEBAR_SCROLL_DELAY_MS
    resume()
    assert not app._scrolling
    assert plot.is_playing is not manual_pause
    _assert_imm_menu_state(app, locked=not manual_pause)
    if manual_pause:
        assert plot._after_id is None
        plot._advance_playback()
        plot.store.display_state_for.assert_not_called()
    else:
        assert plot._after_id is not None
        plot._advance_playback()
        plot.store.display_state_for.assert_called_once_with(1)


@pytest.mark.parametrize("first_to_finish", ["scroll", "resize"])
def test_imm_resize_permanently_pauses_even_during_scroll(
    imm_interaction_app, first_to_finish
):
    app = imm_interaction_app
    plot = app.plot_view
    plot.toggle_playback()
    app._on_sidebar_scroll()
    app._on_window_resize(SimpleNamespace(widget=object(), width=1100, height=700))
    app._on_window_resize(SimpleNamespace(widget=app.root, width=800, height=600))
    assert not app._resizing
    app._on_window_resize(SimpleNamespace(widget=app.root, width=1100, height=700))
    first_timer = app._resize_after_id
    app._on_window_resize(SimpleNamespace(widget=app.root, width=1200, height=800))
    assert first_timer in app.root.cancelled
    assert not plot.is_playing
    _assert_imm_menu_state(app, locked=True)
    finishers = (
        (app._finish_sidebar_scroll, app._finish_window_resize)
        if first_to_finish == "scroll"
        else (app._finish_window_resize, app._finish_sidebar_scroll)
    )
    finishers[0]()
    assert plot._suspended
    finishers[1]()
    assert not plot._suspended
    assert not plot.is_playing
    plot._advance_playback()
    plot.store.display_state_for.assert_not_called()
    _assert_imm_menu_state(app, locked=False)
    plot.toggle_playback()
    plot._advance_playback()
    plot.store.display_state_for.assert_called_once_with(1)


def test_imm_close_cancels_interactions_without_resuming(imm_interaction_app):
    from unittest.mock import Mock

    app = imm_interaction_app
    plot = app.plot_view
    plot.toggle_playback()
    app._on_sidebar_scroll()
    app._on_window_resize(SimpleNamespace(widget=app.root, width=1100, height=700))
    tokens = (app._scroll_after_id, app._resize_after_id)
    callbacks = [app.root.callbacks[token][1] for token in tokens]
    plot.destroy = Mock(side_effect=plot.disable_navigation)
    app.close()
    assert all(token in app.root.cancelled for token in tokens)
    for callback in callbacks:
        callback()
    assert not plot.is_playing
    assert plot._after_id is None
    plot.store.display_state_for.assert_not_called()
    app.root.destroy.assert_called_once_with()


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (SimpleNamespace(delta=-120), (1, "units")),
        (SimpleNamespace(num=4), (-1, "units")),
        (SimpleNamespace(delta=0), None),
    ],
)
def test_imm_wheel_prioritizes_scroll_before_moving_the_canvas(event, expected):
    from unittest.mock import Mock

    from imm_predictor.controls import IMMControls

    calls = []
    panel = IMMControls.__new__(IMMControls)
    panel.on_scroll = lambda: calls.append("activity")
    panel._canvas = Mock()
    panel._canvas.yview.return_value = (0.1, 0.6)
    panel._canvas.yview_scroll.side_effect = lambda *args: calls.append(args)
    assert panel._on_mouse_wheel(event) == "break"
    assert calls == ([] if expected is None else ["activity", expected])


def test_imm_touchpad_prioritizes_scroll_before_moving_the_canvas():
    from unittest.mock import Mock

    from imm_predictor.controls import IMMControls

    calls = []
    panel = IMMControls.__new__(IMMControls)
    panel.on_scroll = lambda: calls.append("activity")
    panel._canvas = Mock()
    panel._canvas.tk.call.return_value = (0, -12)
    panel._canvas.yview.return_value = (0.1, 0.6)
    panel._canvas.bbox.return_value = (0, 0, 100, 1000)
    panel._canvas.yview_moveto.side_effect = lambda value: calls.append(value)
    assert panel._on_touchpad_scroll(SimpleNamespace(delta=1)) == "break"
    assert calls == ["activity", pytest.approx(0.112)]


def test_imm_scrollbar_uses_the_shared_interaction_callback(monkeypatch):
    from unittest.mock import Mock

    from imm_predictor import controls

    panel = controls.IMMControls.__new__(controls.IMMControls)
    panel.on_scroll = Mock()
    panel._add_analysis_section = panel._add_data_section = Mock()
    panel._add_value_section = Mock()
    panel._bind_mouse_wheel = Mock()
    canvas, frame, scrollbar = Mock(), Mock(), Mock()
    factory = Mock(return_value=scrollbar)
    monkeypatch.setattr(controls.tk, "Canvas", Mock(return_value=canvas))
    monkeypatch.setattr(controls.tk, "Frame", Mock(return_value=frame))
    monkeypatch.setattr(controls.tk, "Label", Mock())
    monkeypatch.setattr(controls, "create_form_scrollbar", factory)
    panel._build_scrollable_layout()
    assert factory.call_args.args == (panel, canvas)
    factory.call_args.kwargs["on_scroll"]()
    panel.on_scroll.assert_called_once_with()


def test_failure_disables_real_matplotlib_widgets_without_changing_selection():
    from unittest.mock import Mock

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
    from trajectory_predictor.plot_view import PosteriorPlotView

    figure = Figure()
    FigureCanvasAgg(figure)
    radio = RadioButtons(
        figure.add_axes([0.1, 0.1, 0.2, 0.2]), ["Motion", "Noise"], active=1
    )
    checks = CheckButtons(figure.add_axes([0.1, 0.4, 0.2, 0.2]), ["Settings"], [True])
    widgets = [
        Button(figure.add_axes([0.4, 0.1, 0.2, 0.1]), "Start"),
        Slider(figure.add_axes([0.4, 0.4, 0.4, 0.1]), "N", 0, 10, valinit=5),
        radio,
        checks,
    ]
    navigator = SimpleNamespace(
        pause_playback=Mock(),
        disconnect=Mock(),
        playback_button=widgets[0],
        slider=widgets[1],
        group_selector=radio,
        settings_checkbox=checks,
    )
    view = SimpleNamespace(navigator=navigator)
    PosteriorPlotView.disable_navigation(view)
    PosteriorPlotView.disable_navigation(view)
    assert all(not widget.active for widget in widgets)
    assert radio.value_selected == "Noise"
    assert checks.get_status() == [True]
    assert widgets[1].val == 5


@pytest.fixture
def resizing_app():
    from unittest.mock import Mock

    callbacks, tokens, cancelled = {}, [], []

    def schedule(delay, callback):
        token = f"timer-{len(tokens)}"
        tokens.append(token)
        callbacks[token] = (delay, callback)
        return token

    def cancel(token):
        cancelled.append(token)
        callbacks.pop(token, None)

    app = TrajectoryPredictor.__new__(TrajectoryPredictor)
    app.root = SimpleNamespace(after=schedule, after_cancel=cancel)
    app.root.callbacks, app.root.cancelled = callbacks, cancelled
    app._closing = app._closed = app._resizing = False
    app._scrolling = False
    app._scroll_after_id = None
    app._resize_after_id = None
    app._last_window_size = (800, 600)
    app._analysis_active = True
    app._error = None
    app._computing = None
    app._settings = SimpleNamespace(
        analysis=SimpleNamespace(experiment=SimpleNamespace(inference_method="smc"))
    )
    app.worker = Mock()
    app.worker.drain.return_value = [object()]
    app.plot_view = Mock()
    app.controls = Mock()
    app.status = Mock()
    app._settings_dialog = app._help_window = app._plot_display_window = None
    app._handle_event = Mock()
    return app


def test_resize_defers_updates_and_pauses_playback_until_manual_start(resizing_app):
    from unittest.mock import call

    app = resizing_app
    navigator = app.plot_view.navigator
    app._on_window_resize(SimpleNamespace(widget=app.root, width=1200, height=800))
    first_timer = app._resize_after_id
    app._on_window_resize(SimpleNamespace(widget=app.root, width=1100, height=750))
    assert first_timer in app.root.cancelled
    assert app._resize_after_id != first_timer
    assert app.worker.set_suspended.call_args_list == [call(True)]
    assert navigator.set_suspended.call_args_list == [call(True)]
    navigator.pause_playback.assert_called_once_with()
    app._poll()
    app.worker.drain.assert_not_called()
    app._handle_event.assert_not_called()
    delay, resume = app.root.callbacks[app._resize_after_id]
    assert delay == 200
    resume()
    assert not app._resizing
    assert app.worker.set_suspended.call_args_list == [call(True), call(False)]
    assert navigator.set_suspended.call_args_list == [call(True), call(False)]
    navigator.toggle_playback.assert_not_called()
    app.plot_view = None
    app._poll()
    app.worker.drain.assert_called_once()
    app._handle_event.assert_called_once()


def test_resize_does_not_advance_cached_results_until_playback_is_started(resizing_app):
    from matplotlib.figure import Figure
    from trajectory_predictor.dashboard import (
        create_sequential_posterior_dashboard_figure,
    )

    from bayestraj.models.bayesian_ctrv import BayesianCTRVPriors

    app = resizing_app
    coordinates = np.arange(4.0)
    _, navigator = create_sequential_posterior_dashboard_figure(
        PosteriorDashboardTrajectory(*([coordinates] * 4)),
        BayesianCTRVPriors(),
        maximum_observation_count=4,
        figure=Figure(),
        request_update=app.worker.request,
    )
    app.plot_view = SimpleNamespace(navigator=navigator)
    try:
        navigator.toggle_playback(None)
        navigator.advance_playback()
        assert navigator.is_playing and navigator.is_waiting
        app._on_window_resize(SimpleNamespace(widget=app.root, width=1200, height=800))
        app.worker.request.assert_called_with(0)
        navigator.accept_update(
            PosteriorDashboardUpdate(
                1, {name: np.array([0.5, 1.0, 1.5]) for name in PARAMETER_NAMES}
            )
        )
        app._finish_window_resize()
        navigator.advance_playback()  # A late timer must not restart playback.
        assert not navigator.is_playing
        assert not navigator.is_waiting
        assert navigator.observation_count == 0
        app.worker.request.assert_called_with(0)
        navigator.toggle_playback(None)
        navigator.advance_playback()
        assert navigator.is_playing
        assert navigator.observation_count == 1
        app.worker.request.assert_called_with(1)
    finally:
        navigator.disconnect()


def test_resize_ignores_child_layout_events_and_window_moves(resizing_app):
    app = resizing_app
    app._on_window_resize(SimpleNamespace(widget=object(), width=1200, height=800))
    app._on_window_resize(SimpleNamespace(widget=app.root, width=800, height=600))
    assert not app.root.callbacks
    app.worker.set_suspended.assert_not_called()


def test_closing_during_resize_cancels_resume_without_restarting_worker(resizing_app):
    from unittest.mock import call

    app = resizing_app
    app._on_window_resize(SimpleNamespace(widget=app.root, width=1200, height=800))
    token = app._resize_after_id
    resume = app.root.callbacks[token][1]
    app.close()
    assert token in app.root.cancelled
    resume()
    assert app.worker.set_suspended.call_args_list == [call(True)]
    app.worker.close.assert_called_once()


def test_sidebar_scroll_defers_rendering_until_the_last_scroll_event(resizing_app):
    from unittest.mock import call

    app = resizing_app
    app._on_sidebar_scroll()
    first_timer = app._scroll_after_id
    app._on_sidebar_scroll()
    assert first_timer in app.root.cancelled
    app._poll()
    app.worker.drain.assert_not_called()
    app.plot_view.navigator.pause_playback.assert_not_called()
    delay, finish = app.root.callbacks[app._scroll_after_id]
    assert 100 <= delay <= 250
    finish()
    assert not app._scrolling
    assert app.worker.set_suspended.call_args_list == [call(True), call(False)]
    assert app.plot_view.navigator.set_suspended.call_args_list == [
        call(True),
        call(False),
    ]
    app.plot_view = None
    app._poll()
    app.worker.drain.assert_called_once()


@pytest.mark.parametrize("manual_pause", [False, True])
def test_sidebar_scroll_retains_playback_intent_and_pending_results(
    resizing_app, manual_pause
):
    app = resizing_app
    navigator = _attach_menu_test_navigator(app)
    try:
        navigator.toggle_playback(None)
        navigator.advance_playback()
        app._on_sidebar_scroll()
        navigator.accept_update(
            PosteriorDashboardUpdate(
                1, {name: np.array([0.5, 1.0, 1.5]) for name in PARAMETER_NAMES}
            )
        )
        assert navigator.observation_count == 0
        if manual_pause:
            navigator.pause_playback()
        app._finish_sidebar_scroll()
        assert navigator.is_playing is not manual_pause
        assert navigator.observation_count == (0 if manual_pause else 1)
        assert 1 in navigator._updates_by_count
    finally:
        navigator.disconnect()


@pytest.mark.parametrize("first_to_finish", ["scroll", "resize"])
def test_sidebar_scroll_and_resize_wait_until_both_interactions_finish(
    resizing_app, first_to_finish
):
    app = resizing_app
    app._on_sidebar_scroll()
    app._on_window_resize(SimpleNamespace(widget=app.root, width=1200, height=800))
    first, last = (
        (app._finish_sidebar_scroll, app._finish_window_resize)
        if first_to_finish == "scroll"
        else (app._finish_window_resize, app._finish_sidebar_scroll)
    )
    first()
    assert all(call.args == (True,) for call in app.worker.set_suspended.call_args_list)
    last()
    app.worker.set_suspended.assert_called_with(False)


def test_close_cancels_pending_scroll_resume_without_restarting_worker(resizing_app):
    app = resizing_app
    app._on_sidebar_scroll()
    token = app._scroll_after_id
    finish = app.root.callbacks[token][1]
    app.close()
    assert token in app.root.cancelled
    finish()
    app.worker.set_suspended.assert_called_once_with(True)


def test_poll_reports_callback_errors_and_keeps_scheduling():
    from unittest.mock import Mock

    app = TrajectoryPredictor.__new__(TrajectoryPredictor)
    app._closing = app._closed = False
    app._error = None
    app.root = Mock()
    app.worker = Mock()
    app.worker.drain.return_value = [object()]
    app._handle_event = Mock(side_effect=ValueError("render failed"))
    app._report_callback_exception = Mock()
    app.plot_view = None
    app._poll()
    assert app._report_callback_exception.call_args.args[0] is ValueError
    app.root.after.assert_called_once_with(75, app._poll)


@pytest.mark.parametrize("failure_stage", ["view", "dialog_query"])
def test_predictor_close_still_closes_worker_after_view_cleanup_fails(failure_stage):
    from unittest.mock import Mock

    app = TrajectoryPredictor.__new__(TrajectoryPredictor)
    app._closing = False
    app.controls = Mock()
    app.worker = Mock()
    app._settings_dialog = None
    app._help_window = app._plot_display_window = None
    app.plot_view = Mock()
    if failure_stage == "view":
        app.plot_view.destroy.side_effect = RuntimeError("cleanup failed")
    else:
        app._help_window = Mock()
        app._help_window.winfo_exists.side_effect = tk.TclError("window gone")
    app.status = Mock()
    app.close()
    app.close()
    app.worker.close.assert_called_once_with()
    assert app._closing


def test_reference_turn_rate_display_marks_only_derived_values():
    panel = SettingsPanel.__new__(SettingsPanel)
    panel.posterior_state_values = {key: _Value("") for key in CSV_STATE_ROW_KEYS}
    state = (11.0, -3.0, 2.0, 2.0, 4.0, "m", 12.5)
    panel.show_reference_state(
        state, turn_rate_is_derived=True, speed_is_derived=True, heading_is_derived=True
    )
    assert panel.posterior_state_values["turn_rate"].get() == "4.00°/s (derived)"
    assert panel.posterior_state_values["speed"].get() == "2.00 m/s (derived)"
    assert panel.posterior_state_values["heading"].get() == "2.00° (derived)"
    panel.show_reference_state(state)
    assert panel.posterior_state_values["turn_rate"].get() == "4.00°/s"
    panel.show_reference_state(
        (11.0, -3.0, 2.0, 2.0, np.nan, "m", 12.5), turn_rate_is_derived=True
    )
    assert panel.posterior_state_values["turn_rate"].get() == "—"


def test_toggle_settings_switches_posterior_display_mode():
    app = TrajectoryPredictor.__new__(TrajectoryPredictor)
    app.settings_visible = True
    app.settings_visible_var = _Value(True)
    app.controls = SimpleNamespace(
        grid=lambda: None,
        grid_remove=lambda: None,
    )
    selector_visibility = []
    app.plot_view = SimpleNamespace(
        set_settings_visible=lambda visible: selector_visibility.append(visible)
    )

    app.toggle_settings()
    app.toggle_settings()

    assert selector_visibility == [False, True]
    assert app.settings_visible_var.get() is True


def test_set_settings_visible_synchronizes_sidebar_menu_and_dashboard():
    app = TrajectoryPredictor.__new__(TrajectoryPredictor)
    app.settings_visible = True
    app.settings_visible_var = _Value(True)
    sidebar_changes = []
    app.controls = SimpleNamespace(
        grid=lambda: sidebar_changes.append("shown"),
        grid_remove=lambda: sidebar_changes.append("hidden"),
    )
    dashboard_changes = []
    app.plot_view = SimpleNamespace(
        set_settings_visible=lambda visible: dashboard_changes.append(visible)
    )

    app.set_settings_visible(False)
    app.set_settings_visible(True)

    assert sidebar_changes == ["hidden", "shown"]
    assert dashboard_changes == [False, True]
    assert app.settings_visible is True
    assert app.settings_visible_var.get() is True


@pytest.fixture
def playback_menu_app(monkeypatch):
    from unittest.mock import Mock

    from trajectory_predictor import view

    app = TrajectoryPredictor.__new__(TrajectoryPredictor)
    app.root = Mock()
    app.language, app.language_var = "en", _Value("en")
    app.settings_visible_var = _Value(True)
    app.controls = Mock(variables={"data": {"coordinate_display_mode": _Value("m")}})
    app.worker = Mock()
    app.status = _Value("Ready.")
    app.placeholder = Mock()
    app._closing = app._closed = app._resizing = False
    app._analysis_active = True
    app._last_window_size = (800, 600)
    app._resize_after_id = None
    app._error = app._computing = None
    app._settings = SimpleNamespace(
        analysis=SimpleNamespace(experiment=SimpleNamespace(inference_method="smc"))
    )
    app.plot_view = None
    monkeypatch.setattr(view.tk, "Menu", _MenuRecorder)
    monkeypatch.setattr("trajectory_predictor.gui.localize_widgets", Mock())
    monkeypatch.setattr("trajectory_predictor.gui.messagebox.showerror", Mock())
    view.create_menu_bar(app)
    return app


def _attach_menu_test_navigator(app, *, maximum=2):
    from matplotlib.figure import Figure
    from trajectory_predictor.dashboard import (
        create_sequential_posterior_dashboard_figure,
    )

    from bayestraj.models.bayesian_ctrv import BayesianCTRVPriors

    coordinates = np.arange(4.0)
    _, navigator = create_sequential_posterior_dashboard_figure(
        PosteriorDashboardTrajectory(*([coordinates] * 4)),
        BayesianCTRVPriors(),
        maximum_observation_count=maximum,
        figure=Figure(),
        request_update=app._request_update,
        on_playback_change=app._update_menu_state,
    )
    app.plot_view = SimpleNamespace(
        navigator=navigator,
        set_language=navigator.set_language,
        disable_navigation=navigator.disconnect,
        destroy=navigator.disconnect,
    )
    return navigator


def _assert_header_menu_state(app, *, locked, analysis_active=True):
    for index, entry in enumerate(app._menu_bar.entries):
        disabled = locked or (analysis_active and index == app._settings_menu_index)
        assert entry["state"] == ("disabled" if disabled else "normal"), entry["label"]


def test_header_menus_follow_loading_playback_pause_completion_and_cancel(
    playback_menu_app,
):
    from matplotlib.backend_bases import KeyEvent

    app = playback_menu_app
    app._update_menu_state()
    _assert_header_menu_state(app, locked=True)
    navigator = _attach_menu_test_navigator(app)
    try:
        app._update_menu_state()
        _assert_header_menu_state(app, locked=False)
        navigator.toggle_playback(None)
        _assert_header_menu_state(app, locked=True)
        navigator.advance_playback()
        app.language_var.set("de")  # Tk sets the radio variable before its callback.
        app.set_language("de")
        assert app.language == app.language_var.get() == "en"
        navigator.handle_key_press(
            KeyEvent("key_press_event", navigator.figure.canvas, key=" ")
        )
        _assert_header_menu_state(app, locked=False)
        app.set_language("de")
        assert app.language == "de"
        navigator.toggle_playback(None)
        _assert_header_menu_state(app, locked=True)
        app._on_window_resize(SimpleNamespace(widget=app.root, width=1200, height=800))
        assert not navigator.is_playing
        _assert_header_menu_state(app, locked=True)
        app._finish_window_resize()
        _assert_header_menu_state(app, locked=False)
        navigator.toggle_playback(None)
        _assert_header_menu_state(app, locked=True)
        for count in (1, 2):
            navigator.advance_playback()
            navigator.accept_update(
                PosteriorDashboardUpdate(
                    count, {name: np.array([0.5, 1.0, 1.5]) for name in PARAMETER_NAMES}
                )
            )
        assert not navigator.is_playing
        _assert_header_menu_state(app, locked=False)
        app.cancel_analysis()
        _assert_header_menu_state(app, locked=False, analysis_active=False)
    finally:
        navigator.disconnect()


@pytest.mark.parametrize("result", ["update", "error"])
def test_header_menus_wait_for_manual_step_and_unlock_after_its_result(
    playback_menu_app, result
):
    from matplotlib.backend_bases import KeyEvent
    from trajectory_predictor.session import AnalysisEvent

    app = playback_menu_app
    navigator = _attach_menu_test_navigator(app)
    try:
        app._update_menu_state()
        navigator.handle_key_press(
            KeyEvent("key_press_event", navigator.figure.canvas, key="right")
        )
        assert not navigator.is_playing and navigator.is_waiting
        _assert_header_menu_state(app, locked=True)
        payload = (
            PosteriorDashboardUpdate(
                1, {name: np.array([0.5, 1.0, 1.5]) for name in PARAMETER_NAMES}
            )
            if result == "update"
            else "inference failed"
        )
        app.worker.drain.return_value = [AnalysisEvent(0, result, payload)]
        app._poll()
        assert app._error == (None if result == "update" else payload)
        _assert_header_menu_state(app, locked=False, analysis_active=result == "update")
    finally:
        navigator.disconnect()


def test_menu_routes_settings_and_uses_graceful_close(monkeypatch):
    # Only native window creation is replaced. Real controller methods handle
    # visibility, method selection, and closing with an open settings editor.
    from trajectory_predictor import view

    root_options, button_options, closed = {}, {}, []
    dialogs = []
    plot_windows = []

    class Dialog:
        def __init__(self, _root, _panel, group):
            self.group = group
            self.exists = True
            dialogs.append(self)

        def winfo_exists(self):
            return self.exists

        def cancel(self):
            self.exists = False

    app = TrajectoryPredictor.__new__(TrajectoryPredictor)
    app.root = SimpleNamespace(configure=lambda **options: root_options.update(options))
    app.worker = SimpleNamespace(close=lambda: closed.append(True))
    app.controls = SimpleNamespace(
        variables={
            "data": {
                "coordinate_display_mode": _Value("m"),
                "inference_method": _Value("smc"),
                "show_legend": _Value(True),
                "show_reference_trajectory": _Value(True),
                "show_observed_trajectory": _Value(True),
                "show_current_position": _Value(True),
                "show_sample_trajectories": _Value(True),
                "show_median_forecast": _Value(True),
                "show_prediction_region_50": _Value(True),
                "show_prediction_region_90": _Value(True),
            }
        },
        apply_button=SimpleNamespace(
            configure=lambda **options: button_options.update(options)
        ),
        cancel_button=SimpleNamespace(
            configure=lambda **options: button_options.update(options)
        ),
        grid_remove=lambda: None,
        grid=lambda: None,
    )
    app.settings_visible = True
    app.settings_visible_var = _Value(True)
    app.status = _Value("")
    app.language = "en"
    app.language_var = _Value("en")
    app._closing = False
    app._analysis_active = False
    app._settings_dialog = None
    app.show_plot_display = lambda: plot_windows.append(True)
    app.plot_view = None
    monkeypatch.setattr(view.tk, "Menu", _MenuRecorder)
    monkeypatch.setattr("trajectory_predictor.gui.SettingsDialog", Dialog)
    translated = []
    monkeypatch.setattr(
        "trajectory_predictor.gui.localize_widgets",
        lambda _root, language: translated.append(language),
    )
    view.create_menu_bar(app)
    menu = root_options["menu"]
    view.set_analysis_menu_enabled(app, False)
    assert menu.entries[app._settings_menu_index]["state"] == "disabled"
    assert menu.entry("Language")["state"] == "normal"
    assert app._file_menu.entries[app._open_csv_menu_index]["state"] == "disabled"
    view.set_analysis_menu_enabled(app, True)
    assert menu.entries[app._settings_menu_index]["state"] == "normal"
    assert app._file_menu.entries[app._open_csv_menu_index]["state"] == "normal"
    assert [entry["label"] for entry in menu.entries] == [
        "File",
        "View",
        "Settings",
        "Language",
        "Help",
    ]
    assert [entry["label"] for entry in menu.entry("Help")["menu"].entries] == [
        "Show Help"
    ]
    assert [entry["label"] for entry in menu.entry("Settings")["menu"].entries] == [
        "Priors…",
        "Inference parameters…",
        "Analysis setup…",
    ]
    language_menu = menu.entry("Language")["menu"]
    assert [entry["label"] for entry in language_menu.entries] == ["English", "Deutsch"]
    app._analysis_active = True
    original_settings = app._settings = object()
    plot_languages = []
    app.plot_view = SimpleNamespace(set_language=plot_languages.append)
    language_menu.entry("Deutsch")["command"]()
    german_menu = root_options["menu"]
    assert german_menu is menu
    assert not menu.destroyed
    assert not language_menu.destroyed
    assert [entry["label"] for entry in german_menu.entries] == [
        "Datei",
        "Ansicht",
        "Einstellungen",
        "Sprache",
        "Hilfe",
    ]
    assert app.language_var.get() == "de"
    assert app._settings is original_settings
    assert app._analysis_active
    assert app.controls.variables["data"]["inference_method"].get() == "smc"
    assert german_menu.entries[app._settings_menu_index]["state"] == "disabled"
    assert german_menu.entry("Sprache")["state"] == "normal"
    german_menu.entry("Sprache")["menu"].entry("English")["command"]()
    assert translated == plot_languages == ["de", "en"]
    assert root_options["menu"] is menu
    assert not menu.destroyed
    app._analysis_active = False
    view.set_analysis_menu_enabled(app, True)
    app.plot_view = None
    menu = root_options["menu"]
    view_menu = menu.entry("View")["menu"]
    assert [entry["label"] for entry in view_menu.entries] == [
        "Show settings",
        "Coordinate display",
        "Plot display…",
    ]
    coordinate_menu = view_menu.entry("Coordinate display")["menu"]
    assert [entry["label"] for entry in coordinate_menu.entries] == [
        "Local [m]",
        "Local [km]",
    ]
    coordinate_menu.entry("Local [km]")["command"]()
    assert app.controls.variables["data"]["coordinate_display_mode"].get() == "km"
    view_menu.entry("Plot display…")["command"]()
    assert plot_windows == [True]
    menu.entry("View")["menu"].entry("Show settings")["command"]()
    assert not app.settings_visible
    assert app.settings_visible_var.get() is False
    menu.entry("Settings")["menu"].entry("Inference parameters…")["command"]()
    assert dialogs[0].group == "smc"
    menu.entry("File")["menu"].entry("Exit")["command"]()
    assert app._closing
    assert not dialogs[0].exists
    assert closed == [True]
    assert button_options["state"] == "disabled"


def test_language_switch_retains_native_tk_window_and_menu(root):
    """Exercise Tcl menu callbacks as well as the display-independent contract."""
    worker = SimpleNamespace(close=lambda: None)
    app = TrajectoryPredictor(root, worker=worker)
    errors = []
    root.report_callback_exception = lambda *error: errors.append(error)
    menu = app._menu_bar
    state = root.state()
    callbacks = {
        source: (entry_menu, index)
        for entry_menu, index, source in app._menu_label_entries
        if source in {"Deutsch", "English"}
    }
    try:
        for selection, language in (("Deutsch", "de"), ("English", "en")) * 3:
            entry_menu, index = callbacks[selection]
            entry_menu.invoke(index)
            root.update_idletasks()
            assert not errors
            assert root.winfo_exists()
            assert root.state() == state
            assert app._menu_bar is menu
            assert menu.winfo_exists()
            assert entry_menu.winfo_exists()
            assert app.language == language
            assert not app._closing
    finally:
        app.close()
        if app._poll_id is not None:
            root.after_cancel(app._poll_id)


def test_help_window_is_reused_without_blocking_the_main_window(monkeypatch):
    """Help uses one non-modal child window instead of a message box."""
    opened = []

    class HelpWindow:
        def __init__(self, _root, *, on_close):
            self.on_close = on_close
            self.exists = True
            self.lifted = False
            self.focused = False
            opened.append(self)

        def winfo_exists(self):
            return self.exists

        def lift(self):
            self.lifted = True

        def focus_set(self):
            self.focused = True

    app = TrajectoryPredictor.__new__(TrajectoryPredictor)
    app.root = object()
    app._closing = False
    app._help_window = None
    monkeypatch.setattr("trajectory_predictor.gui.PosteriorHelpWindow", HelpWindow)

    app.show_help()
    app.show_help()

    assert len(opened) == 1
    assert opened[0].lifted
    assert opened[0].focused
    opened[0].on_close()
    assert app._help_window is None


def test_help_descriptions_use_one_shared_left_column_width(monkeypatch):
    """Every help section starts descriptions at the same horizontal position."""
    from trajectory_predictor import help as predictor_help

    options = []

    class Label:
        def __init__(self, _parent, **kwargs):
            options.append(kwargs)

        def grid(self, **_kwargs):
            pass

    monkeypatch.setattr(predictor_help.tk, "Label", Label)

    predictor_help.add_help_description_rows(object(), (("Space", "Play"),))

    assert options[0]["width"] == predictor_help.HELP_DESCRIPTION_COLUMN_WIDTH
    assert options[1]["wraplength"] == predictor_help.HELP_DESCRIPTION_WRAP_LENGTH
    assert predictor_help.HELP_DESCRIPTION_WRAP_LENGTH == 300


def test_help_window_uses_ship_simulator_content_height_and_top_left_position():
    """Help stays tall enough for reading and opens at the top-left screen corner."""
    from trajectory_predictor.help import (
        help_content_height_for_screen,
        help_window_position,
    )

    assert help_content_height_for_screen(900, 1080) == 702
    assert help_content_height_for_screen(400, 1080) == 400
    assert help_window_position() == (0, 0)
