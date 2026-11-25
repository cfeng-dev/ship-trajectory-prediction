"""CSV-only exploration preserves forecast cohorts and spatial meaning."""

import importlib
import os
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


def _module():
    return importlib.import_module("bayestraj.validation.evaluation_viewer")


def _write_csvs(directory, *, student=True, route=True):
    directory.mkdir(exist_ok=True)
    for method, offset in (("imm", 1.0), ("smc", 0.0)):
        table = pd.DataFrame(
            {
                "window_index": [10, 10, 20, 20] if method == "imm" else [0, 0, 1, 1],
                "forecast_origin_seconds": [100.0, 100.0, 105.0, 105.0],
                "horizon_step": [1, 2, 1, 2],
                "horizon_seconds": [2.5, 5.0, 2.5, 5.0],
                "prediction_count": 2,
                "x_actual": [10.0, 15.0, 20.0, 25.0],
                "y_actual": [0.0, 0.0, 5.0, 10.0],
                "point_estimate": "mean",
                "position_error_m": np.array([3.0, 4.0, 5.0, 12.0]) + offset,
                "gaussian_energy_score_m": [2.0, 4.0, 6.0, 10.0],
                "gaussian_negative_log_predictive_density": [-1.0, 3.0, 5.0, 7.0],
                "fit_runtime_seconds": [0.01, 0.01, 0.02, 0.02],
                "inference_method": method,
                "model_variant": "imm" if method == "imm" else "bayesian",
                "joint_covered_50": [True, False, False, True],
                "joint_covered_90": [True, True, False, True],
            }
        )
        table["x_mean"] = table["x_actual"] + table["position_error_m"]
        table["y_mean"] = table["y_actual"]
        table["x_median"] = table["x_mean"] + 1.0
        table["y_median"] = table["y_actual"]
        if student:
            table["student_t_negative_log_predictive_density"] = [0.5, 1.5, 2.5, 3.5]
        if route:
            for name in ("actual", "mean", "median"):
                for coordinate in ("x", "y"):
                    table[f"{coordinate}_{name}_route"] = table[f"{coordinate}_{name}"]
            table["forecast_origin_x_route"] = [5.0, 5.0, 15.0, 15.0]
            table["forecast_origin_y_route"] = [0.0, 0.0, 0.0, 0.0]
        table.to_csv(directory / f"{method}.csv", index=False)
    return directory


def test_saved_forecasts_share_physical_origins_and_reuse_existing_metrics(tmp_path):
    data = _module().load_evaluation_directory(_write_csvs(tmp_path))
    np.testing.assert_array_equal(data.origins, [100.0, 105.0])
    np.testing.assert_array_equal(data.elapsed, [0.0, 5.0])
    smc = data.models["Bayesian CTRV (SMC)"]
    np.testing.assert_allclose(smc.metrics["ade_m"], [3.5, 8.5])
    np.testing.assert_allclose(smc.metrics["fde_m"], [4.0, 12.0])
    np.testing.assert_allclose(smc.metrics["rmse_m"], [np.sqrt(12.5), np.sqrt(84.5)])
    np.testing.assert_allclose(
        smc.metrics["mean_negative_log_predictive_density"], [1, 6]
    )
    np.testing.assert_allclose(
        smc.metrics["mean_student_t_negative_log_predictive_density"], [1, 3]
    )
    np.testing.assert_allclose(smc.forecasts[0], [[13, 0], [19, 0]])
    np.testing.assert_allclose(smc.origin_positions, [[5, 0], [15, 0]])
    assert data.nearest_index(4.0) == 1
    assert data.nearest_index(-10.0) == 0


@pytest.mark.parametrize("bad_score", [None, np.nan, np.inf])
def test_student_t_is_only_offered_for_complete_paired_scores(tmp_path, bad_score):
    directory = _write_csvs(tmp_path)
    path = directory / "imm.csv"
    table = pd.read_csv(path)
    if bad_score is None:
        table.drop(columns="student_t_negative_log_predictive_density", inplace=True)
    else:
        table.loc[0, "student_t_negative_log_predictive_density"] = bad_score
    table.to_csv(path, index=False)
    data = _module().load_evaluation_directory(directory)
    assert (
        "mean_student_t_negative_log_predictive_density" not in data.available_metrics
    )
    assert "mean_negative_log_predictive_density" in data.available_metrics


def test_conflicting_reference_positions_at_the_same_target_time_are_rejected(tmp_path):
    directory = _write_csvs(tmp_path)
    for path in directory.glob("*.csv"):
        table = pd.read_csv(path)
        table.loc[2:, "forecast_origin_seconds"] = 102.5
        table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="reference.*target time"):
        _module().load_evaluation_directory(directory)


@pytest.mark.parametrize(
    "column", ["x_actual_route", "x_mean_route", "forecast_origin_x_route"]
)
def test_nonfinite_route_geometry_is_rejected(tmp_path, column):
    directory = _write_csvs(tmp_path)
    path = directory / "imm.csv"
    table = pd.read_csv(path)
    table.loc[0, column] = np.nan
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="coordinate|geometry|finite"):
        _module().load_evaluation_directory(directory)


def test_legacy_forecasts_use_saved_coordinates_without_inventing_origins(tmp_path):
    data = _module().load_evaluation_directory(_write_csvs(tmp_path, route=False))
    assert data.models["IMM"].origin_positions is None
    np.testing.assert_allclose(data.models["IMM"].forecasts[0], [[14, 0], [20, 0]])


def test_mismatched_horizons_are_rejected_by_the_shared_reader(tmp_path):
    directory = _write_csvs(tmp_path)
    path = directory / "imm.csv"
    table = pd.read_csv(path)
    table.loc[table.horizon_step == 2, "horizon_seconds"] = 6.0
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="horizon_seconds"):
        _module().load_evaluation_directory(directory)


def test_route_forecasts_must_represent_the_saved_position_errors(tmp_path):
    directory = _write_csvs(tmp_path)
    path = directory / "imm.csv"
    table = pd.read_csv(path)
    table["x_mean_route"] += 100
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="route.*error"):
        _module().load_evaluation_directory(directory)


def test_unmatched_origins_do_not_enter_the_metrics_or_reference(tmp_path):
    directory = _write_csvs(tmp_path)
    path = directory / "smc.csv"
    table = pd.read_csv(path)
    extra = table.iloc[:2].copy()
    extra["window_index"] = 50
    extra["forecast_origin_seconds"] = 90.0
    pd.concat([extra, table]).to_csv(path, index=False)
    with pytest.warns(UserWarning, match="unmatched origins"):
        data = _module().load_evaluation_directory(directory)
    np.testing.assert_array_equal(data.origins, [100, 105])
    np.testing.assert_array_equal(data.reference_times, [102.5, 105, 107.5, 110])


def test_plot_uses_one_metric_per_axis_and_matching_colors_and_time(tmp_path):
    data = _module().load_evaluation_directory(_write_csvs(tmp_path))
    plotting = importlib.import_module("evaluation_viewer.plot_view")
    settings = importlib.import_module("evaluation_viewer.settings")
    plot = plotting.EvaluationPlot()
    options = settings.ViewerOptions(follow_metric_time=False)
    plot.draw(data, 1, options, tuple(data.models))
    plot.figure.canvas.draw()
    for axis, metric in zip(
        plot.metric_axes, ("ade_m", "mean_negative_log_predictive_density"), strict=True
    ):
        lines = {line.get_label(): line for line in axis.lines}
        np.testing.assert_allclose(
            lines["IMM"].get_ydata(), data.models["IMM"].metrics[metric]
        )
        np.testing.assert_allclose(lines["_current_time"].get_xdata(), [5, 5])
        assert lines["IMM"].get_color() == data.models["IMM"].color
    assert plot.trajectory_axis.get_aspect() == 1.0
    plot.draw(
        data,
        0,
        replace(
            options, top_metric="fde_m", follow_metric_time=True, metric_window_width=3
        ),
        ("IMM",),
    )
    assert plot.metric_axes[0].get_xlim() == (0, 3)
    assert plot.metric_axes[0].get_ylabel() == "FDE [m]"
    assert "Bayesian CTRV (SMC)" not in [
        line.get_label() for line in plot.metric_axes[0].lines
    ]
    plt.close(plot.figure)


@pytest.mark.parametrize("prediction_count", [1, 2])
def test_trajectory_lines_start_with_saved_future_points_and_reference_is_separate(
    tmp_path, prediction_count
):
    directory = _write_csvs(tmp_path)
    for path in directory.glob("*.csv"):
        table = pd.read_csv(path)
        table = table[table["horizon_step"] <= prediction_count].copy()
        table["prediction_count"] = prediction_count
        table["forecast_origin_x_route"] = 1000.0 if path.stem == "imm" else 2000.0
        table.to_csv(path, index=False)
    data = _module().load_evaluation_directory(directory)
    plotting = importlib.import_module("evaluation_viewer.plot_view")
    options = importlib.import_module("evaluation_viewer.settings").ViewerOptions()
    plot = plotting.EvaluationPlot()
    try:
        plot.draw(data, 1, options, tuple(data.models))
        lines = {line.get_label(): line for line in plot.trajectory_axis.lines}
        expected = {
            "IMM": [[26.0, 5.0], [38.0, 10.0]],
            "Bayesian CTRV (SMC)": [[25.0, 5.0], [37.0, 10.0]],
        }
        for label, points in expected.items():
            np.testing.assert_allclose(
                lines[label].get_xydata(), points[:prediction_count]
            )
        markers = {
            artist.get_label(): artist for artist in plot.trajectory_axis.collections
        }
        assert not set(data.models).intersection(markers)
        if prediction_count == 2:
            assert set(markers) == {"Reference at forecast origin"}
            np.testing.assert_allclose(
                markers["Reference at forecast origin"].get_offsets(), [[15.0, 0.0]]
            )
            np.testing.assert_allclose(plot.trajectory_axis.get_xlim(), [-35.0, 65.0])
        else:
            assert not markers  # The CSV does not contain the reference at this origin.
    finally:
        plt.close(plot.figure)


def test_cli_help_does_not_import_tk_or_start_inference(monkeypatch, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "tkinter", None)
    cli = importlib.import_module("evaluation_viewer.cli")
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    assert "--input-dir" in capsys.readouterr().out


def test_single_origin_plots_and_metric_legend_leaves_room_for_all_models(tmp_path):
    directory = _write_csvs(tmp_path)
    for method in ("vi", "mcmc"):
        table = pd.read_csv(directory / "smc.csv")
        table["inference_method"] = method
        table.to_csv(directory / f"{method}.csv", index=False)
    for path in directory.glob("*.csv"):
        pd.read_csv(path).iloc[:2].to_csv(path, index=False)
    data = _module().load_evaluation_directory(directory)
    plotting = importlib.import_module("evaluation_viewer.plot_view")
    options = importlib.import_module("evaluation_viewer.settings").ViewerOptions(
        follow_metric_time=False
    )
    plot = plotting.EvaluationPlot()
    plot.draw(data, 0, options, tuple(data.models))
    plot.figure.canvas.draw()
    assert plot.metric_axes[0].get_xlim() == (0, 1)
    renderer = plot.figure.canvas.get_renderer()
    for axis in plot.metric_axes:
        legend_bottom = axis.get_legend().get_window_extent(renderer).y0
        for line in axis.lines:
            if line.get_label().startswith("_"):
                continue
            points = axis.transData.transform(
                np.column_stack((line.get_xdata(), line.get_ydata()))
            )
            assert points[:, 1].max() < legend_bottom - 5
    plt.close(plot.figure)


@pytest.mark.parametrize(
    "key,value",
    [
        ("metric_window_width", 0),
        ("metric_window_width", np.nan),
        ("ship_view_span", -1),
        ("ship_view_span", np.inf),
    ],
)
def test_invalid_display_spans_are_rejected(key, value):
    options = importlib.import_module("evaluation_viewer.settings").ViewerOptions
    with pytest.raises(ValueError, match="greater than zero"):
        options(**{key: value})


class _Clock:
    def __init__(self):
        self.pending = {}
        self.counter = 0

    def after(self, delay, callback):
        self.counter += 1
        self.pending[self.counter] = callback
        return self.counter

    def after_cancel(self, handle):
        self.pending.pop(handle)

    def tick(self):
        handle = next(iter(self.pending))
        self.pending.pop(handle)()


def test_playback_stops_at_last_origin_and_seeking_cancels_pending_timer():
    clock = _Clock()
    frames = []
    playback = importlib.import_module("evaluation_viewer.session").Playback(
        clock, 3, frames.append
    )
    playback.start()
    assert len(clock.pending) == 1
    clock.tick()
    assert playback.index == 1
    playback.seek(0)
    assert not playback.playing
    assert not clock.pending
    playback.start()
    clock.tick()
    clock.tick()
    assert frames[-1] == 2
    assert not playback.playing
    assert not clock.pending
    playback.start()
    assert playback.index == 0
    playback.close()
    assert not clock.pending


def test_predictor_style_controls_seek_without_feedback_during_playback_updates():
    plotting = importlib.import_module("evaluation_viewer.plot_view")
    changes = []
    bar = plotting.PlaybackBar(
        on_seek=lambda value: changes.append(("seek", value)),
        on_play=lambda: changes.append(("play", True)),
        on_follow=lambda value: changes.append(("follow", value)),
        on_settings=lambda value: changes.append(("settings", value)),
    )
    try:
        assert not bar.slider.active
        bar.set_count(3)
        bar.show_index(2, elapsed_seconds=5.0)
        assert changes == []
        assert "3/3" in bar.slider.valtext.get_text()
        assert "5.0 s" in bar.slider.valtext.get_text()
        bar.slider.set_val(1)
        assert changes == [("seek", 1)]
        bar.follow_checkbox.set_active(0)
        assert changes[-1] == ("follow", False)
        bar.settings_checkbox.set_active(0)
        assert changes[-1] == ("settings", False)
        bar.set_settings_visible(True)
        bar.set_follow_ship(True)
        assert changes[-1] == ("settings", False)
        bar.set_language("de")
        bar.set_playing(True)
        assert bar.playback_button.label.get_text() == "Pause"
        assert bar.settings_checkbox.labels[0].get_text() == "Einstellungen anzeigen"
        bar.figure.canvas.draw()
    finally:
        plt.close(bar.figure)


def test_predictor_style_playback_button_and_single_origin_slider():
    from matplotlib.backend_bases import MouseEvent

    plotting = importlib.import_module("evaluation_viewer.plot_view")
    played = []
    bar = plotting.PlaybackBar(
        on_seek=lambda value: None,
        on_play=lambda: played.append(True),
        on_follow=lambda value: None,
        on_settings=lambda value: None,
    )
    try:
        bar.set_count(1)
        bar.show_index(0, elapsed_seconds=0.0)
        assert not bar.slider.active
        assert bar.slider.valtext.get_text().startswith("1/1")
        bar.figure.canvas.draw()
        x, y = bar.playback_button.ax.transAxes.transform((0.5, 0.5))
        for event in ("button_press_event", "button_release_event"):
            bar.figure.canvas.callbacks.process(
                event, MouseEvent(event, bar.figure.canvas, x, y, button=1)
            )
        assert played == [True]
    finally:
        plt.close(bar.figure)


def test_metric_click_and_slider_seek_the_same_cached_origin(tmp_path):
    pytest.importorskip("tkinter")
    from types import SimpleNamespace

    gui = importlib.import_module("evaluation_viewer.gui")
    app = gui.EvaluationViewer.__new__(gui.EvaluationViewer)
    app.data = _module().load_evaluation_directory(_write_csvs(tmp_path))
    app.plot = SimpleNamespace(metric_axes=(object(), object()))
    app.toolbar = SimpleNamespace(mode="")
    frames = []
    clock = _Clock()
    app.playback = importlib.import_module("evaluation_viewer.session").Playback(
        clock, 2, frames.append
    )
    app.playback.start()
    app._plot_click(
        SimpleNamespace(inaxes=app.plot.metric_axes[0], xdata=4.5, button=1)
    )
    assert frames[-1] == 1
    assert not clock.pending
    app._slider("0.0")
    assert frames[-1] == 0
    app.toolbar.mode = "pan/zoom"
    app._plot_click(
        SimpleNamespace(inaxes=app.plot.metric_axes[0], xdata=5.0, button=1)
    )
    assert frames[-1] == 0


def test_language_switch_preserves_unapplied_model_metric_and_window_edits():
    pytest.importorskip("tkinter")
    from types import SimpleNamespace

    gui = importlib.import_module("evaluation_viewer.gui")
    options = importlib.import_module("evaluation_viewer.settings").ViewerOptions

    class Variable:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    app = gui.EvaluationViewer.__new__(gui.EvaluationViewer)
    app.options = options()
    app.pause = lambda: None
    app.playback = None
    app._build_menu = lambda: None
    app.navigation = SimpleNamespace(set_language=lambda language: None)
    app.plot = SimpleNamespace(clear=lambda language: None)
    app.canvas = SimpleNamespace(draw_idle=lambda: None)

    def rebuild():
        app.model_variables = {"IMM": Variable(True)}
        app.metric_choices = {app.tr("ADE"): "ade_m", app.tr("FDE"): "fde_m"}
        app.metric_variables = {"top_metric": Variable("ADE")}
        app.display_variables = {"metric_window_width": Variable("60.0")}

    app._build_sidebar = rebuild
    rebuild()
    app.model_variables["IMM"].set(False)
    app.metric_variables["top_metric"].set("FDE")
    app.display_variables["metric_window_width"].set("120")
    app.set_language("de")
    assert app.model_variables["IMM"].get() is False
    assert app.metric_variables["top_metric"].get() == "FDE"
    assert app.display_variables["metric_window_width"].get() == "120"


@pytest.mark.parametrize(
    "widget_class", ["Checkbutton", "TCheckbutton", "TButton", "Radiobutton", "TScale"]
)
def test_keyboard_shortcuts_do_not_intercept_sidebar_controls(widget_class):
    pytest.importorskip("tkinter")
    from types import SimpleNamespace

    gui = importlib.import_module("evaluation_viewer.gui")
    app = gui.EvaluationViewer.__new__(gui.EvaluationViewer)
    actions = []
    app.toggle_playback = lambda: actions.append("play")
    event = SimpleNamespace(
        widget=SimpleNamespace(winfo_class=lambda: widget_class), keysym="space"
    )
    app._keyboard(event)
    assert not actions


def test_real_tk_viewer_loads_and_applies_settings_when_a_display_is_available(
    tmp_path,
):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk display unavailable: {error}")
    root.withdraw()
    gui = importlib.import_module("evaluation_viewer.gui")
    app = gui.EvaluationViewer(root, input_dir=_write_csvs(tmp_path))
    try:
        app.load_directory()
        root.update()
        assert app.data is not None
        app.metric_variables["top_metric"].set("FDE")
        app.apply_options()
        assert app.options.top_metric == "fde_m"
        app.step(1)
        assert app.playback.index == 1
        app.set_language("de")
        assert app.plot.metric_axes[0].get_ylabel() == "FDE [m]"
    finally:
        app.close()


def test_stored_results_preview_when_explicitly_requested(tmp_path):
    directory = os.environ.get("EVALUATION_VIEWER_INPUT_DIR")
    if not directory:
        pytest.skip(
            "Set EVALUATION_VIEWER_INPUT_DIR to validate existing saved results."
        )
    data = _module().load_evaluation_directory(directory)
    plotting = importlib.import_module("evaluation_viewer.plot_view")
    options = importlib.import_module("evaluation_viewer.settings").ViewerOptions()
    plot = plotting.EvaluationPlot()
    try:
        for index in (0, len(data.origins) - 1, data.nearest_index(900)):
            plot.draw(data, index, options, tuple(data.models))
            plot.figure.canvas.draw()
        output = tmp_path / "evaluation_viewer_preview.png"
        plot.figure.savefig(output, dpi=130)
        assert output.stat().st_size > 1000
        print(f"Preview: {output}")
    finally:
        plt.close(plot.figure)
