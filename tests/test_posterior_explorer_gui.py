"""Non-visible Tk integration checks for the embedded posterior explorer."""

from threading import Event
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

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
    window.withdraw()
    errors = []
    window.report_callback_exception = lambda *args: errors.append(args)
    yield window
    try:
        window.destroy()
    except tk.TclError:
        pass
    assert not errors


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
