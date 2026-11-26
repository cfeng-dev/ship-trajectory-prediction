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


def _write_observations(path):
    pd.DataFrame(
        {
            "time": [95.0, 100.0, 102.5, 105.0, 110.0],
            "x": [1.0, 5.5, 10.5, 15.5, 900.0],
            "y": [1.0, 0.5, 0.5, 1.5, 900.0],
            "ground_truth_x": -1000.0,
            "ground_truth_y": -1000.0,
        }
    ).to_csv(path, index=False)
    return path


def test_observation_history_keeps_source_time_and_excludes_future_samples(tmp_path):
    directory = _write_csvs(tmp_path / "forecasts")
    observations = _write_observations(tmp_path / "observations.csv")
    data = _module().load_evaluation_directory(directory, observations_csv=observations)
    np.testing.assert_allclose(
        data.observations.time_seconds, [95, 100, 102.5, 105, 110]
    )
    plotting = importlib.import_module("evaluation_viewer.plot_view")
    options = importlib.import_module("evaluation_viewer.settings").ViewerOptions()
    plot = plotting.EvaluationPlot()
    try:
        plot.draw(data, 1, options, tuple(data.models))
        lines = {line.get_label(): line for line in plot.trajectory_axis.lines}
        np.testing.assert_allclose(
            lines["Observations through N"].get_xydata(),
            [[1, 1], [5.5, 0.5], [10.5, 0.5], [15.5, 1.5]],
        )
        np.testing.assert_allclose(
            lines["Current position"].get_xydata(), [[15.5, 1.5]]
        )
        assert lines["Observations through N"].get_color() == "#24557A"
        assert lines["Current position"].get_color() == "#D97706"
        assert not plot.trajectory_axis.collections
        np.testing.assert_allclose(plot.trajectory_axis.get_xlim(), [-34.5, 65.5])
        np.testing.assert_allclose(
            lines["Bayesian CTRV (SMC)"].get_xydata(), [[25, 5], [37, 10]]
        )
        np.testing.assert_allclose(lines["IMM"].get_xydata(), [[26, 5], [38, 10]])
        imm_links = [
            line
            for line in plot.trajectory_axis.lines
            if line.get_linestyle() == "--"
            and line.get_color() == data.models["IMM"].color
        ]
        assert len(imm_links) == 1
        np.testing.assert_allclose(imm_links[0].get_xydata(), [[15.5, 1.5], [26, 5]])
        links = [
            line
            for line in plot.trajectory_axis.lines
            if line.get_linestyle() == "--"
            and line.get_color() == data.models["Bayesian CTRV (SMC)"].color
        ]
        np.testing.assert_allclose(links[0].get_xydata(), [[15.5, 1.5], [25, 5]])
        assert "Connection to current position" not in [
            entry.get_text() for entry in plot.trajectory_axis.get_legend().get_texts()
        ]
        plot.draw(data, 0, options, tuple(data.models))
        history = next(
            line
            for line in plot.trajectory_axis.lines
            if line.get_label() == "Observations through N"
        )
        np.testing.assert_allclose(history.get_xydata(), [[1, 1], [5.5, 0.5]])
        plot.draw(data, 1, options, ("Bayesian CTRV (SMC)",))
        lines = {line.get_label(): line for line in plot.trajectory_axis.lines}
        np.testing.assert_allclose(
            lines["Current position"].get_xydata(), [[15.5, 1.5]]
        )
    finally:
        plt.close(plot.figure)


@pytest.mark.parametrize(
    "problem", ["duplicate", "nonfinite", "time_offset", "missing_column"]
)
def test_invalid_observation_files_are_rejected_without_replacing_them_with_reference(
    tmp_path, problem
):
    directory = _write_csvs(tmp_path / "forecasts")
    path = _write_observations(tmp_path / "observations.csv")
    table = pd.read_csv(path)
    if problem == "duplicate":
        table.loc[1, "time"] = table.loc[0, "time"]
    elif problem == "nonfinite":
        table.loc[1, "x"] = np.nan
    elif problem == "time_offset":
        table["time"] -= 95
    else:
        table.drop(columns="x", inplace=True)
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Observation"):
        _module().load_evaluation_directory(directory, observations_csv=path)


def test_missing_observation_at_origin_is_not_interpolated_or_called_current(tmp_path):
    directory = _write_csvs(tmp_path / "forecasts")
    path = _write_observations(tmp_path / "observations.csv")
    table = pd.read_csv(path)
    table[table["time"] != 105].to_csv(path, index=False)
    data = _module().load_evaluation_directory(directory, observations_csv=path)
    plotting = importlib.import_module("evaluation_viewer.plot_view")
    options = importlib.import_module("evaluation_viewer.settings").ViewerOptions()
    plot = plotting.EvaluationPlot()
    try:
        plot.draw(data, 1, options, tuple(data.models))
        lines = {line.get_label(): line for line in plot.trajectory_axis.lines}
        assert "Current position" not in lines
        np.testing.assert_allclose(
            lines["Latest observation"].get_xydata(), [[10.5, 0.5]]
        )
        assert "Connection to current position" not in [
            entry.get_text() for entry in plot.trajectory_axis.get_legend().get_texts()
        ]
        assert not any(
            line.get_linestyle() == "--"
            and line.get_color() in {model.color for model in data.models.values()}
            for line in plot.trajectory_axis.lines
        )
    finally:
        plt.close(plot.figure)


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
    options = settings.ViewerOptions()
    assert not options.follow_metric_time
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
        for label in data.models:
            assert lines[label].get_marker() in ("None", "", " ")
        assert axis.get_xlim() == (0, 5)
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
def test_all_forecast_lines_keep_only_future_points_and_share_the_current_marker(
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
        imm_links = [
            line
            for line in plot.trajectory_axis.lines
            if line.get_linestyle() == "--"
            and line.get_color() == data.models["IMM"].color
        ]
        assert len(imm_links) == 1
        np.testing.assert_allclose(
            imm_links[0].get_xydata(), [[1000.0, 0.0], [26.0, 5.0]]
        )
        markers = {
            artist.get_label(): artist for artist in plot.trajectory_axis.collections
        }
        assert not set(data.models).intersection(markers)
        assert set(markers) == {"IMM track at forecast origin"}
        np.testing.assert_allclose(
            markers["IMM track at forecast origin"].get_offsets(), [[1000.0, 0.0]]
        )
        connections = [
            line
            for line in plot.trajectory_axis.lines
            if line.get_linestyle() == "--"
            and line.get_color() == data.models["Bayesian CTRV (SMC)"].color
        ]
        assert len(connections) == 1
        np.testing.assert_allclose(
            connections[0].get_xydata(), [[1000.0, 0.0], [25.0, 5.0]]
        )
        np.testing.assert_allclose(
            data.models["Bayesian CTRV (SMC)"].forecasts[1],
            [[25.0, 5.0], [37.0, 10.0]][:prediction_count],
        )
        assert "Connection to IMM track" not in [
            entry.get_text() for entry in plot.trajectory_axis.get_legend().get_texts()
        ]
        np.testing.assert_allclose(plot.trajectory_axis.get_xlim(), [950.0, 1050.0])
    finally:
        plt.close(plot.figure)


@pytest.mark.parametrize(
    "route,selected", [(True, ("Bayesian CTRV (SMC)",)), (False, ("IMM",))]
)
def test_missing_or_hidden_imm_origins_do_not_create_a_reference_anchor(
    tmp_path, route, selected
):
    data = _module().load_evaluation_directory(_write_csvs(tmp_path, route=route))
    plotting = importlib.import_module("evaluation_viewer.plot_view")
    options = importlib.import_module("evaluation_viewer.settings").ViewerOptions()
    plot = plotting.EvaluationPlot()
    try:
        plot.draw(data, 1, options, selected)
        assert not plot.trajectory_axis.collections
        line = next(
            line
            for line in plot.trajectory_axis.lines
            if line.get_label() == selected[0]
        )
        expected = (
            [[26.0, 5.0], [38.0, 10.0]]
            if selected[0] == "IMM"
            else [[25.0, 5.0], [37.0, 10.0]]
        )
        np.testing.assert_allclose(line.get_xydata(), expected)
        assert not any(
            line.get_linestyle() == "--"
            and line.get_color() == data.models[selected[0]].color
            for line in plot.trajectory_axis.lines
        )
    finally:
        plt.close(plot.figure)


@pytest.mark.parametrize("different_origins", [False, True])
def test_connections_require_one_unambiguous_selected_imm_origin(
    tmp_path, different_origins
):
    directory = _write_csvs(tmp_path)
    extra = pd.read_csv(directory / "imm.csv")
    if different_origins:
        extra["forecast_origin_x_route"] += 10
    extra.to_csv(directory / "imm_other.csv", index=False)
    data = _module().load_evaluation_directory(directory)
    plotting = importlib.import_module("evaluation_viewer.plot_view")
    options = importlib.import_module("evaluation_viewer.settings").ViewerOptions()
    plot = plotting.EvaluationPlot()
    try:
        plot.draw(data, 1, options, tuple(data.models))
        connections = [
            line
            for line in plot.trajectory_axis.lines
            if line.get_linestyle() == "--"
            and line.get_color() == data.models["Bayesian CTRV (SMC)"].color
        ]
        assert len(connections) == (0 if different_origins else 1)
        if connections:
            np.testing.assert_allclose(
                connections[0].get_xydata(), [[15.0, 0.0], [25.0, 5.0]]
            )
    finally:
        plt.close(plot.figure)


def test_cli_help_does_not_import_tk_or_start_inference(monkeypatch, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "tkinter", None)
    cli = importlib.import_module("evaluation_viewer.cli")
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--input-dir" in output
    assert "--observations-csv" in output


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

    def update_idletasks(self):
        pass

    def destroy(self):
        self.destroyed = True


class _LoadControl:
    def __init__(self):
        self.properties = {}
        self.visible = False

    def configure(self, **kwargs):
        self.properties.update(kwargs)

    def bind(self, *args):
        pass

    def grid(self, **kwargs):
        self.visible = True

    def grid_remove(self):
        self.visible = False

    def tkraise(self):
        pass


def _prepare_loading_controls(app):
    app._loading = False
    app._load_worker = None
    app._load_poll_id = None
    app._closed = False
    app._load_inputs = [_LoadControl()]
    app.load_button = _LoadControl()
    app.placeholder = _LoadControl()


def _finish_loading(app):
    app._load_worker.thread.join(timeout=5)
    assert not app._load_worker.thread.is_alive()
    app.root.after_cancel(app._load_poll_id)
    app._load_poll_id = None
    app._poll_load()


@pytest.fixture
def loading_viewer(monkeypatch, tmp_path):
    pytest.importorskip("tkinter")
    from types import SimpleNamespace

    gui = importlib.import_module("evaluation_viewer.gui")
    app = gui.EvaluationViewer.__new__(gui.EvaluationViewer)
    app.root = _Clock()
    app.options = gui.ViewerOptions()
    app.playback = app.navigation = app.data = None
    app.directory = SimpleNamespace(get=lambda: str(tmp_path))
    app.observations_csv = SimpleNamespace(get=lambda: "")
    app.status = SimpleNamespace(set=lambda value: setattr(app, "status_text", value))
    _prepare_loading_controls(app)
    jobs = []

    class PendingLoad:
        def __init__(self, *args, **kwargs):
            self.messages = []
            self.cancelled = False
            jobs.append(self)

        def take_messages(self):
            messages, self.messages = self.messages, []
            return messages

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(gui, "EvaluationLoader", PendingLoad, raising=False)
    return app, jobs


def test_loading_feedback_and_button_follow_predictor_states(loading_viewer):
    app, jobs = loading_viewer
    app.load_directory(show_error=False)
    assert app.data is None
    assert app.placeholder.visible
    assert app.status_text == "Preparing analysis …"
    assert app.load_button.properties["text"] == "Cancel analysis"
    assert app.load_button.properties["bg"] == "#f7d9d9"
    assert all(item.properties["state"] == "disabled" for item in app._load_inputs)
    app.load_directory()
    assert len(jobs) == 1
    jobs[0].messages.append(("progress", "Preparing metrics …"))
    app.root.tick()
    assert app.placeholder.properties["text"] == "Preparing metrics …"
    assert app.status_text == "Preparing metrics …"
    assert len(app.root.pending) == 1
    jobs[0].messages.append(("error", ValueError("Invalid CSV")))
    app._show_load_error = False
    app.root.tick()
    assert not app._loading
    assert app.status_text == "Invalid CSV"
    assert app.load_button.properties["text"] == "Start analysis"
    assert app.load_button.properties["bg"] == "#ffffff"
    assert all(item.properties["state"] == "normal" for item in app._load_inputs)
    assert not app.root.pending


def test_cancelled_load_cannot_replace_data_and_can_be_restarted(loading_viewer):
    app, jobs = loading_viewer
    previous = app.data = object()
    app._set_status = lambda: None
    app.load_directory(show_error=False)
    app.cancel_loading()
    assert jobs[0].cancelled
    assert app.data is previous
    assert not app.root.pending
    assert not app.placeholder.visible
    jobs[0].messages.append(("ready", object()))
    app.load_directory()
    received = []
    app._apply_loaded_data = received.append
    new_data = object()
    jobs[1].messages.append(("ready", new_data))
    app.root.tick()
    assert received == [new_data]
    assert not app._loading
    assert not app.root.pending


def test_closing_viewer_cancels_loading_and_its_poll_timer(loading_viewer):
    app, jobs = loading_viewer
    app.load_directory(show_error=False)
    app.close()
    assert jobs[0].cancelled
    assert not app.root.pending
    assert app.root.destroyed


def test_background_evaluation_load_reports_progress_and_returns_saved_data(tmp_path):
    session = importlib.import_module("evaluation_viewer.session")
    job = session.EvaluationLoader(_write_csvs(tmp_path))
    job.thread.join(timeout=5)
    assert not job.thread.is_alive()
    messages = job.take_messages()
    assert messages[0] == ("progress", "Preparing analysis …")
    assert messages[-1][0] == "ready"
    assert len(messages[-1][1].models) == 2
    assert any(
        kind == "progress" and "metrics" in value.lower() for kind, value in messages
    )


def test_background_evaluation_load_reports_file_errors(tmp_path):
    session = importlib.import_module("evaluation_viewer.session")
    job = session.EvaluationLoader(tmp_path / "missing")
    job.thread.join(timeout=5)
    assert not job.thread.is_alive()
    messages = job.take_messages()
    assert messages[-1][0] == "error"
    assert isinstance(messages[-1][1], (OSError, ValueError))
    assert not any(kind == "ready" for kind, _ in messages)


def test_cancelling_background_load_discards_data_before_preparing_metrics(
    tmp_path, monkeypatch
):
    from threading import Event

    session = importlib.import_module("evaluation_viewer.session")
    data_module = _module()
    started, resume = Event(), Event()

    def read(*args, **kwargs):
        started.set()
        assert resume.wait(timeout=5)
        return {"unused": pd.DataFrame()}

    monkeypatch.setattr(data_module.comparison, "read_model_comparison_directory", read)
    job = session.EvaluationLoader(tmp_path)
    try:
        assert started.wait(timeout=5)
        job.cancel()
    finally:
        resume.set()
        job.thread.join(timeout=5)
    assert not job.thread.is_alive()
    assert not any(kind in {"ready", "error"} for kind, _ in job.take_messages())


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


def test_viewer_loads_selected_observations_and_retains_data_on_file_error(tmp_path):
    pytest.importorskip("tkinter")
    from types import SimpleNamespace

    gui = importlib.import_module("evaluation_viewer.gui")
    directory = _write_csvs(tmp_path / "forecasts")
    observations = _write_observations(tmp_path / "observations.csv")
    app = gui.EvaluationViewer.__new__(gui.EvaluationViewer)
    app.root = _Clock()
    app.options = importlib.import_module("evaluation_viewer.settings").ViewerOptions()
    app.directory = SimpleNamespace(get=lambda: str(directory))
    app.observations_csv = SimpleNamespace(get=lambda: str(observations))
    app.pause = lambda: None
    app.playback = None
    app.plot = object()
    app._build_sidebar = lambda: None
    counts, frames, errors = [], [], []
    app.navigation = SimpleNamespace(set_count=counts.append)
    app.toolbar = SimpleNamespace(update=lambda: None)
    app._frame = frames.append
    app.status = SimpleNamespace(
        set=errors.append, get=lambda: errors[-1] if errors else ""
    )
    _prepare_loading_controls(app)
    app.load_directory()
    _finish_loading(app)
    loaded = app.data
    np.testing.assert_allclose(loaded.observations.positions[1], [5.5, 0.5])
    assert counts == [2]
    assert frames == [0]
    app.observations_csv = SimpleNamespace(get=lambda: str(tmp_path / "missing.csv"))
    app.load_directory(show_error=False)
    _finish_loading(app)
    assert app.data is loaded
    assert frames == [0]
    assert errors


def test_viewer_opens_without_loading_or_creating_plots_until_requested(
    tmp_path, monkeypatch
):
    pytest.importorskip("tkinter")
    from types import SimpleNamespace

    gui = importlib.import_module("evaluation_viewer.gui")
    directory = _write_csvs(tmp_path / "forecasts")
    observations = _write_observations(tmp_path / "observations.csv")

    class Widget:
        def __init__(self, *args, value=None, **kwargs):
            self.value = value
            self.properties = kwargs

        def configure(self, **kwargs):
            self.properties.update(kwargs)

        def bind(self, *args):
            pass

        def tkraise(self):
            pass

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

        def grid(self, *args, **kwargs):
            pass

        def grid_remove(self):
            pass

        def columnconfigure(self, *args, **kwargs):
            pass

        def rowconfigure(self, *args, **kwargs):
            pass

    class Root(_Clock):
        def title(self, *args):
            pass

        def bind(self, *args, **kwargs):
            pass

        def protocol(self, *args):
            pass

    def forbidden(*args, **kwargs):
        pytest.fail("A plot canvas was created before clicking Start analysis.")

    def build_sidebar(app):
        app._load_inputs = []
        app.load_button = Widget()
        app.model_variables = {}
        app.metric_choices = {app.tr(value): key for key, value in gui.METRICS.items()}
        app.metric_variables = {
            key: Widget(value=app.tr(gui.METRICS[getattr(app.options, key)]))
            for key in ("top_metric", "bottom_metric")
        }

    for name in ("Frame", "Label", "StringVar", "BooleanVar"):
        monkeypatch.setattr(gui.tk, name, Widget)
    monkeypatch.setattr(gui.view, "configure_window", lambda root: None)
    monkeypatch.setattr(gui.view, "PredictorFigureCanvas", forbidden)
    monkeypatch.setattr(gui.EvaluationViewer, "_build_sidebar", build_sidebar)
    monkeypatch.setattr(gui.EvaluationViewer, "_build_menu", lambda self: None)
    original_read = pd.read_csv
    reads = []
    allowed = False

    def read(*args, **kwargs):
        assert allowed, "CSV data was read before the explicit load action."
        reads.append(args[0])
        return original_read(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read)
    root = Root()
    app = gui.EvaluationViewer(root, input_dir=directory, observations_csv=observations)
    assert app.data is None
    assert app.plot is None
    assert not reads
    assert not root.pending
    assert app._apply_plot_options({"follow_ship": False, "metric_window_width": "120"})
    app._models_selected()
    app.set_language("de")
    app.settings_visible.set(False)
    app.toggle_settings()
    assert not app.options.follow_ship
    assert app.options.metric_window_width == 120.0
    assert app.placeholder.properties["text"] == app.tr(
        "Choose the input paths, then click Start analysis."
    )
    assert app.plot is None
    assert not reads
    assert not root.pending
    created, frames = [], []

    def create_plots():
        created.append(True)
        app.plot = object()
        app.navigation = SimpleNamespace(set_count=lambda count: None)
        app.toolbar = SimpleNamespace(update=lambda: None)

    app._create_plot_ui = create_plots
    app._frame = frames.append
    allowed = True
    app.directory.set(str(tmp_path / "missing"))
    app.load_directory(show_error=False)
    _finish_loading(app)
    assert app.data is None
    assert app.plot is None
    assert not created
    app.directory.set(str(directory))
    app.load_directory()
    _finish_loading(app)
    assert reads
    assert app.data.observations is not None
    assert created == [True]
    assert frames == [0]


def test_file_selection_only_changes_paths_until_the_load_action(tmp_path, monkeypatch):
    pytest.importorskip("tkinter")

    gui = importlib.import_module("evaluation_viewer.gui")
    app = gui.EvaluationViewer.__new__(gui.EvaluationViewer)
    app._loading = False
    app.root = object()
    app.options = importlib.import_module("evaluation_viewer.settings").ViewerOptions()
    app.pause = lambda: None

    class Variable:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    app.directory = Variable(str(tmp_path))
    app.observations_csv = Variable("")
    loads = []
    app.load_directory = lambda: loads.append(True)
    monkeypatch.setattr(
        gui.filedialog, "askdirectory", lambda **kwargs: str(tmp_path / "other")
    )
    monkeypatch.setattr(
        gui.filedialog,
        "askopenfilename",
        lambda **kwargs: str(tmp_path / "observations.csv"),
    )
    app.open_directory()
    app.open_observations()
    assert app.directory.get() == str(tmp_path / "other")
    assert app.observations_csv.get() == str(tmp_path / "observations.csv")
    assert not loads


def test_metric_click_and_slider_seek_the_same_cached_origin(tmp_path):
    pytest.importorskip("tkinter")
    from types import SimpleNamespace

    gui = importlib.import_module("evaluation_viewer.gui")
    app = gui.EvaluationViewer.__new__(gui.EvaluationViewer)
    app._loading = False
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


def test_language_switch_preserves_sidebar_drafts_and_applied_plot_settings():
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
    app._loading = False
    app.options = options(metric_window_width=120.0)
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

    app._build_sidebar = rebuild
    rebuild()
    app.model_variables["IMM"].set(False)
    app.metric_variables["top_metric"].set("FDE")
    app.set_language("de")
    assert app.model_variables["IMM"].get() is False
    assert app.metric_variables["top_metric"].get() == "FDE"
    assert app.options.metric_window_width == 120.0


@pytest.mark.parametrize("key", ["top_metric", "bottom_metric"])
def test_metric_selection_applies_immediately_without_committing_model_drafts(key):
    pytest.importorskip("tkinter")
    from types import SimpleNamespace

    gui = importlib.import_module("evaluation_viewer.gui")
    app = gui.EvaluationViewer.__new__(gui.EvaluationViewer)
    app._loading = False
    app.options = gui.ViewerOptions()
    app.playback = SimpleNamespace(index=1)
    app.navigation = app.toolbar = None
    app.pause = lambda: None
    app.selected_models = ("IMM",)
    app.model_variables = {"IMM": SimpleNamespace(get=lambda: False)}
    app.metric_choices = {"FDE": "fde_m"}
    app.metric_variables = {key: SimpleNamespace(get=lambda: "FDE")}
    frames = []
    app._frame = frames.append
    app._metric_selected(key)
    assert getattr(app.options, key) == "fde_m"
    assert app.selected_models == ("IMM",)
    assert frames == [1]
    app._loading = True
    app.metric_choices = {"FDE": "rmse_m"}
    app._metric_selected(key)
    assert getattr(app.options, key) == "fde_m"
    assert frames == [1]


@pytest.mark.parametrize("playing", [False, True])
@pytest.mark.parametrize("key", ["top_metric", "bottom_metric"])
def test_metric_changes_keep_playback_state_origin_and_single_timer(playing, key):
    pytest.importorskip("tkinter")
    from types import SimpleNamespace

    gui = importlib.import_module("evaluation_viewer.gui")
    app = gui.EvaluationViewer.__new__(gui.EvaluationViewer)
    app._loading = False
    app.options = gui.ViewerOptions()
    app.root = _Clock()
    frames = []
    app._frame = frames.append
    app.playback = importlib.import_module("evaluation_viewer.session").Playback(
        app.root, 4, app._frame
    )
    app.navigation = SimpleNamespace(
        set_playing=lambda value: None, set_follow_ship=lambda value: None
    )
    app.toolbar = SimpleNamespace(update=lambda: None)
    app.metric_choices = {"FDE": "fde_m"}
    app.metric_variables = {key: SimpleNamespace(get=lambda: "FDE")}
    app.playback.seek(1)
    if playing:
        app.playback.start()
    timers = tuple(app.root.pending)
    app._metric_selected(key)
    assert getattr(app.options, key) == "fde_m"
    assert app.playback.index == 1
    assert app.playback.playing is playing
    assert tuple(app.root.pending) == timers
    if playing:
        app.root.tick()
        assert app.playback.index == 2
        assert len(app.root.pending) == 1
    app.playback.close()


def test_model_selection_updates_immediately_and_keeps_at_least_one_model(monkeypatch):
    pytest.importorskip("tkinter")
    from types import SimpleNamespace

    gui = importlib.import_module("evaluation_viewer.gui")
    app = gui.EvaluationViewer.__new__(gui.EvaluationViewer)
    app._loading = False
    app.root = object()
    app.data = object()
    app.options = gui.ViewerOptions()
    app.selected_models = ("IMM", "SMC")
    choices = {"IMM": True, "SMC": False}
    app.model_variables = {
        label: SimpleNamespace(
            get=lambda label=label: choices[label],
            set=lambda value, label=label: choices.__setitem__(label, value),
        )
        for label in choices
    }
    refreshed, errors = [], []
    app._refresh_plots = lambda: refreshed.append(app.selected_models)
    monkeypatch.setattr(
        gui.messagebox, "showerror", lambda *args, **kwargs: errors.append(args)
    )
    app._models_selected()
    assert app.selected_models == ("IMM",)
    assert refreshed == [("IMM",)]
    choices["IMM"] = False
    app._models_selected()
    assert app.selected_models == ("IMM",)
    assert choices == {"IMM": True, "SMC": False}
    assert refreshed == [("IMM",)]
    assert errors


def test_plot_settings_apply_validated_values_without_applying_sidebar_drafts(
    monkeypatch,
):
    pytest.importorskip("tkinter")
    from types import SimpleNamespace

    gui = importlib.import_module("evaluation_viewer.gui")
    app = gui.EvaluationViewer.__new__(gui.EvaluationViewer)
    app._loading = False
    app.root = object()
    app.options = gui.ViewerOptions()
    app.playback = SimpleNamespace(index=1)
    app.pause = lambda: None
    app.selected_models = ("IMM",)
    app.model_variables = {"IMM": SimpleNamespace(get=lambda: False)}
    app.metric_variables = {"top_metric": SimpleNamespace(get=lambda: "FDE")}
    following, frames = [], []
    app.navigation = SimpleNamespace(set_follow_ship=following.append)
    app.toolbar = SimpleNamespace(update=lambda: None)
    app._frame = frames.append
    errors = []
    monkeypatch.setattr(
        gui.messagebox, "showerror", lambda *args, **kwargs: errors.append(args)
    )
    values = {
        "follow_metric_time": True,
        "metric_window_width": "120",
        "follow_ship": False,
        "ship_view_span": "200",
        "show_legend": False,
    }
    assert app._apply_plot_options(values)
    assert app.options.follow_metric_time
    assert app.options.metric_window_width == 120.0
    assert app.options.ship_view_span == 200.0
    assert not app.options.show_legend
    assert following == [False]
    assert frames == [1]
    assert app.selected_models == ("IMM",)
    assert app.options.top_metric == "ade_m"
    previous = app.options
    assert not app._apply_plot_options({**values, "metric_window_width": "0"})
    assert app.options is previous
    assert frames == [1]
    assert errors


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
    app = gui.EvaluationViewer(
        root,
        input_dir=_write_csvs(tmp_path / "forecasts"),
        observations_csv=_write_observations(tmp_path / "observations.csv"),
    )
    try:
        root.update()
        assert app.data is None
        assert app.plot is None
        app.load_directory()
        _finish_loading(app)
        root.update()
        assert app.data is not None
        assert app.data.observations is not None
        choices = [
            child
            for section in app._form.body.winfo_children()
            for child in section.winfo_children()
            if isinstance(child, gui.ttk.Combobox)
        ]
        choices[0].set("FDE")
        choices[0].event_generate("<<ComboboxSelected>>")
        root.update()
        assert app.options.top_metric == "fde_m"
        assert not any(
            isinstance(child, tk.Label)
            and str(child.cget("textvariable")) == str(app.status)
            for child in app._form.body.winfo_children()
        )
        app.step(1)
        assert app.playback.index == 1
        app.set_language("de")
        assert app.plot.metric_axes[0].get_ylabel() == "FDE [m]"
        assert not app.options.follow_metric_time
        previous = app.options
        dialog = app.open_plot_settings()
        root.update()
        sections = [
            section.cget("text")
            for body in dialog.winfo_children()
            for section in body.winfo_children()
            if isinstance(section, tk.LabelFrame)
        ]
        assert sections == ["Metrik-Zeitachse", "Trajektorienansicht"]
        dialog.destroy()
        assert app.options is previous
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
            lines = {line.get_label(): line for line in plot.trajectory_axis.lines}
            for label in data.models:
                assert len(lines[label].get_xydata()) == data.prediction_counts[index]
            plot.figure.canvas.draw()
        output = tmp_path / "evaluation_viewer_preview.png"
        plot.figure.savefig(output, dpi=130)
        assert output.stat().st_size > 1000
        print(f"Preview: {output}")
    finally:
        plt.close(plot.figure)
