"""Non-visible Tk integration checks for the embedded posterior explorer."""

from threading import Event
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from posterior_explorer.controls import SettingsPanel  # noqa: E402
from posterior_explorer.gui import PosteriorExplorer  # noqa: E402

from bayestraj.validation.bayesian_ctrv_posterior_dashboard import (  # noqa: E402
    PARAMETER_NAMES,
    PosteriorDashboardTrajectory,
    PosteriorDashboardUpdate,
)
from bayestraj.validation.posterior_session import PosteriorAnalysisWorker  # noqa: E402


def pump_until(root, predicate):
    deadline = monotonic() + 8
    while monotonic() < deadline:
        root.update()
        if predicate():
            return
        Event().wait(0.01)
    pytest.fail("Tk condition was not reached")


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
    from posterior_explorer.dialogs import dialog_height_for_content

    assert dialog_height_for_content(content_height, screen_height) == expected_height


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
    app = PosteriorExplorer(root, worker=worker)
    assert not hasattr(app, "status_label")
    messages = []
    monkeypatch.setattr(
        "posterior_explorer.gui.messagebox.showerror",
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


def test_analysis_section_contains_only_new_analysis_button(root):
    panel = SettingsPanel(root, lambda: None)
    panel.pack(fill="both", expand=True)
    root.update_idletasks()

    assert panel.apply_button.master.pack_slaves() == [panel.apply_button]


def test_dialog_labels_and_fields_share_vertical_center(root):
    from posterior_explorer.dialogs import SettingsDialog

    panel = SettingsPanel(root, lambda: None)
    dialog = SettingsDialog(root, panel, "data")
    try:
        dialog.update_idletasks()
        body = dialog._form.body
        label = next(
            child
            for child in body.winfo_children()
            if child.winfo_class() == "Label" and child.cget("text") == "Run-ID"
        )
        field = next(
            child
            for child in body.winfo_children()
            if child.winfo_class() == "TEntry"
            and child.cget("textvariable") == str(dialog.variables["run_id"])
        )
        label_center = label.winfo_rooty() + label.winfo_height() / 2
        field_center = field.winfo_rooty() + field.winfo_height() / 2
        assert label_center == pytest.approx(field_center, abs=1.0)
    finally:
        dialog.cancel()


def test_dialog_cancel_and_apply_do_not_start_an_analysis(root, monkeypatch):
    # Keep the parent withdrawn: no visible test window or actual inference.
    from posterior_explorer.dialogs import SettingsDialog

    panel = SettingsPanel(root, lambda: pytest.fail("Unexpected analysis start"))
    original = panel.values()
    dialog = SettingsDialog(root, panel, "priors")
    dialog.variables["speed_prior_upper_mps"].set("30")
    dialog.cancel()
    assert panel.values() == original
    dialog = SettingsDialog(root, panel, "rbpf")
    errors = []
    monkeypatch.setattr(
        "posterior_explorer.dialogs.messagebox.showerror",
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

    def add_separator(self):
        pass

    def entry(self, label):
        return next(entry for entry in self.entries if entry["label"] == label)


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def test_menu_routes_settings_and_uses_graceful_close(monkeypatch):
    # Only native window creation is replaced. Real controller methods handle
    # visibility, method selection, and closing with an open settings editor.
    from posterior_explorer import view

    root_options, button_options, closed = {}, {}, []
    dialogs = []

    class Dialog:
        def __init__(self, _root, _panel, group):
            self.group = group
            self.exists = True
            dialogs.append(self)

        def winfo_exists(self):
            return self.exists

        def cancel(self):
            self.exists = False

    app = PosteriorExplorer.__new__(PosteriorExplorer)
    app.root = SimpleNamespace(configure=lambda **options: root_options.update(options))
    app.worker = SimpleNamespace(close=lambda: closed.append(True))
    app.controls = SimpleNamespace(
        variables={"data": {"inference_method": _Value("smc")}},
        apply_button=SimpleNamespace(
            configure=lambda **options: button_options.update(options)
        ),
        grid_remove=lambda: None,
        grid=lambda: None,
    )
    app.settings_visible = True
    app.settings_visible_var = _Value(True)
    app.status = _Value("")
    app._closing = False
    app._settings_dialog = None
    app.plot_view = None
    monkeypatch.setattr(view.tk, "Menu", _MenuRecorder)
    monkeypatch.setattr("posterior_explorer.gui.SettingsDialog", Dialog)
    view.create_menu_bar(app)
    menu = root_options["menu"]
    assert [entry["label"] for entry in menu.entries] == [
        "File",
        "View",
        "Settings",
        "Help",
    ]
    menu.entry("View")["menu"].entry("Einstellungen anzeigen")["command"]()
    assert not app.settings_visible
    assert app.settings_visible_var.get() is False
    menu.entry("Settings")["menu"].entry("Inferenzparameter…")["command"]()
    assert dialogs[0].group == "smc"
    menu.entry("File")["menu"].entry("Schließen")["command"]()
    assert app._closing
    assert not dialogs[0].exists
    assert closed == [True]
    assert button_options["state"] == "disabled"
