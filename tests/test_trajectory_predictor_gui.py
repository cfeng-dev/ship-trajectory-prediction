"""Non-visible Tk integration checks for the embedded trajectory predictor."""

from threading import Event
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from ship_simulator.controls import STATUS_ROW_LABELS  # noqa: E402
from trajectory_predictor.controls import (  # noqa: E402
    CSV_BROWSE_BUTTON_WIDTH,
    CSV_STATE_ROW_KEYS,
    SettingsPanel,
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


def test_csv_browse_button_uses_a_two_character_width():
    assert CSV_BROWSE_BUTTON_WIDTH == 2


def test_input_styles_keep_fields_light_in_all_widget_states():
    """The light application palette must not inherit macOS dark input fields."""
    settings = input_style_settings()

    for style_name in ("Predictor.TEntry", "Predictor.TCombobox"):
        assert settings[style_name]["configure"]["fieldbackground"] == INPUT_BACKGROUND
        assert settings[style_name]["map"]["fieldbackground"] == [
            ("disabled", DISABLED_INPUT_BACKGROUND),
            ("readonly", INPUT_BACKGROUND),
        ]
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
        "lon = 8.3100°\nlat = 47.0500°"
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


def test_settings_panel_displays_reference_csv_state(root):
    """The sidebar presents reference values and the displayed forecast metrics."""
    panel = SettingsPanel(root, lambda: None, lambda: None)

    panel.show_reference_state((11.0, -3.0, 2.0, 2.0, 2.0, "m", 12.5))
    panel.show_analysis_metrics(
        SimpleNamespace(
            ade_m=2.5,
            fde_m=4.0,
            joint_coverage_count=7,
            joint_coverage_total=8,
            inference_time_seconds=0.125,
        )
    )

    assert panel.posterior_state_values["position"].get() == "x = 11.00 m\ny = -3.00 m"
    assert panel.posterior_state_values["speed"].get() == "2.00 m/s"
    assert panel.posterior_state_values["heading"].get() == "2.00°"
    assert panel.posterior_state_values["turn_rate"].get() == "2.00°/s"
    assert panel.posterior_state_values["time"].get() == "12.5 s"
    assert panel.analysis_metric_values["forecast_ade"].get() == "2.50 m"
    assert panel.analysis_metric_values["forecast_fde"].get() == "4.00 m"
    assert panel.analysis_metric_values["joint_coverage"].get() == "87.5% (7/8)"
    assert panel.analysis_metric_values["inference_time"].get() == "0.125 s"


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


def test_main_window_title_identifies_the_ship_trajectory_predictor(monkeypatch):
    from trajectory_predictor import view

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

        def configure(self, **_kwargs):
            pass

        def columnconfigure(self, _column, **_kwargs):
            pass

        def rowconfigure(self, _row, **_kwargs):
            pass

    window = Window()
    monkeypatch.setattr(view, "configure_input_styles", lambda _root: None)
    view.configure_window(window)

    assert window.title_value == "Bayesian CTRV — Ship Trajectory Predictor"


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
        ("rbpf", 470),
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
        {"data_file": "route.csv", "inference_method": "rbpf", "run_id": "102"}
    )

    assert analysis == {"inference_method": "rbpf"}
    assert data == {"data_file": "route.csv", "run_id": "102"}


def test_sidebar_keeps_only_data_selection_and_moves_analysis_inputs_to_settings():
    """The default sidebar exposes CSV selection, not inference input settings."""
    assert MAIN_DATA_FIELDS == (
        "data_file",
        "run_id",
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


def test_dialog_labels_and_fields_share_vertical_center(root):
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
            if child.winfo_class() == "Label" and child.cget("text") == "Run ID"
        )
        field = next(
            child
            for child in descendants(body)
            if child.winfo_class() == "TEntry"
            and child.cget("textvariable") == str(dialog.variables["run_id"])
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
        labels = {
            child.cget("text")
            for child in dialog._form.body.winfo_children()
            if child.winfo_class() == "Checkbutton"
        }
        assert labels == {
            "Show legend",
            "Show recorded trajectory",
            "Show observations through N",
            "Show current position",
            "Show possible future trajectories",
            "Show forecast (median)",
            "Show 50% posterior-predictive region",
            "Show 90% posterior-predictive region",
        }
    finally:
        dialog.cancel()


def test_plot_display_window_closes_from_its_ok_button(root):
    from trajectory_predictor.dialogs import PlotDisplayWindow
    from trajectory_predictor.settings import DISPLAY_OPTION_FIELDS

    panel = SimpleNamespace(
        variables={
            "data": {
                key: tk.BooleanVar(root, value=True) for key in DISPLAY_OPTION_FIELDS
            }
            | {"follow_ship_view_span_m": tk.StringVar(root, value="600")}
        }
    )
    closed = []
    window = PlotDisplayWindow(root, panel, on_close=lambda: closed.append(True))
    try:
        ok_button = next(
            child
            for frame in window.winfo_children()
            for child in frame.winfo_children()
            if isinstance(child, tk.Button) and child.cget("text") == "OK"
        )
        ok_button.invoke()

        assert closed == [True]
        assert not window.winfo_exists()
    finally:
        if window.winfo_exists():
            window.close()


def test_dialog_cancel_and_apply_do_not_start_an_analysis(root, monkeypatch):
    # Keep the parent withdrawn: no visible test window or actual inference.
    from trajectory_predictor.dialogs import SettingsDialog

    panel = SettingsPanel(root, lambda: pytest.fail("Unexpected analysis start"))
    original = panel.values()
    dialog = SettingsDialog(root, panel, "priors")
    dialog.variables["speed_prior_upper_mps"].set("30")
    dialog.cancel()
    assert panel.values() == original
    dialog = SettingsDialog(root, panel, "rbpf")
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
    assert panel.variables["rbpf"]["particle_count"].get() == "64"
    assert panel.values()["smc"] == original["smc"]


class _MenuRecorder:
    """Record our menu registrations without needing a Tcl display."""

    def __init__(self, _parent, **_options):
        self.entries = []

    def add_command(self, **options):
        self.entries.append(options)

    add_cascade = add_command
    add_checkbutton = add_command
    add_radiobutton = add_command

    def add_separator(self):
        pass

    def entry(self, label):
        return next(entry for entry in self.entries if entry["label"] == label)

    def entryconfigure(self, label, **options):
        entry = self.entries[label] if isinstance(label, int) else self.entry(label)
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
    app._closing = False
    app._analysis_active = False
    app._settings_dialog = None
    app.show_plot_display = lambda: plot_windows.append(True)
    app.plot_view = None
    monkeypatch.setattr(view.tk, "Menu", _MenuRecorder)
    monkeypatch.setattr("trajectory_predictor.gui.SettingsDialog", Dialog)
    view.create_menu_bar(app)
    menu = root_options["menu"]
    view.set_analysis_menu_enabled(app, False)
    assert menu.entries[app._settings_menu_index]["state"] == "disabled"
    assert app._file_menu.entries[app._open_csv_menu_index]["state"] == "disabled"
    view.set_analysis_menu_enabled(app, True)
    assert menu.entries[app._settings_menu_index]["state"] == "normal"
    assert app._file_menu.entries[app._open_csv_menu_index]["state"] == "normal"
    assert [entry["label"] for entry in menu.entries] == [
        "File",
        "View",
        "Settings",
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
        "GPS [°]",
    ]
    coordinate_menu.entry("GPS [°]")["command"]()
    assert app.controls.variables["data"]["coordinate_display_mode"].get() == "gps"
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
