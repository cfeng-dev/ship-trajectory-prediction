"""Tests for trajectory plotting and evaluation entrypoints."""

import runpy
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from imm_predictor import player as imm_player

from bayestraj.inference.ctrv_smc import SequentialMonteCarloCTRVConfig
from bayestraj.validation import plotting as validation_plotting
from bayestraj.validation import rolling, runtime_plotting

PROJECT_ROOT = Path(__file__).parents[1]


def test_rolling_plot_uses_exact_forecast_positions_without_shifting_regions(
    monkeypatch,
):
    monkeypatch.setattr(plt, "show", lambda: None)
    arguments = dict(
        forecast_origin_x=100.0,
        forecast_origin_y=200.0,
        x_samples=np.array([[8, 12], [10, 15], [9, 16], [11, 17]]),
        y_samples=np.array([[0, 1], [1, 4], [3, 2], [2, 6]]),
        forecast_time_seconds=np.array([0, 1, 2]),
    )
    figures = []
    axes = []
    try:
        for overrides in ({}, {"forecast_positions": [[2, 0], [0, 5]]}):
            group = validation_plotting.RollingPosteriorPlotData(
                **arguments, **overrides
            )
            figure, axis = validation_plotting.plot_bayesian_rolling_predictions(
                [100, 2, 4],
                [200, 3, 5],
                [group],
                initial_observation_count=1,
                inference_mode="online",
                sample_trajectories_per_forecast=0,
                forecast_label="Forecast",
            )
            figures.append(figure)
            axes.append(axis)
        lines = [
            next(line for line in axis.lines if line.get_label() == "Forecast")
            for axis in axes
        ]
        np.testing.assert_array_equal(lines[0].get_xdata(), [100, 9.5, 15.5])
        np.testing.assert_array_equal(lines[0].get_ydata(), [200, 1.5, 3])
        np.testing.assert_array_equal(lines[1].get_xdata(), [100, 2, 0])
        np.testing.assert_array_equal(lines[1].get_ydata(), [200, 0, 5])
        assert len(axes[0].patches) == len(axes[1].patches) > 0
        for original, overridden in zip(axes[0].patches, axes[1].patches, strict=True):
            np.testing.assert_array_equal(
                original.get_path().vertices, overridden.get_path().vertices
            )
            np.testing.assert_array_equal(
                original.get_patch_transform().get_matrix(),
                overridden.get_patch_transform().get_matrix(),
            )
    finally:
        for figure in figures:
            plt.close(figure)


@pytest.fixture
def imm_median_case():
    # A 75% point mass pins each marginal median, despite the distant second mode.
    mixtures = tuple(
        imm_player.IMMStateMixture(
            [0.75, 0.25, 0.0],
            tuple(
                imm_player.IMMStatePosterior([x, y, 0, 0, 1], np.zeros((5, 5)))
                for x, y in positions
            ),
        )
        for positions in (
            [(2, 0), (34, 12), (99, 99)],
            [(0, 5), (16, 45), (99, 99)],
        )
    )
    return SimpleNamespace(
        module=runpy.run_path(PROJECT_ROOT / "experiments/model_evaluation/imm.py"),
        forecast=SimpleNamespace(
            time_seconds=np.array([1.0, 2.0]),
            position_means=np.array([[10.0, 3.0], [4.0, 15.0]]),
            position_covariances=np.tile(4 * np.eye(2), (2, 1, 1)),
            mode_probabilities=np.tile([0.75, 0.25, 0.0], (2, 1)),
        ),
        reference=SimpleNamespace(
            x_meters=np.array([999.0, 2.0, 4.0]),
            y_meters=np.array([999.0, 3.0, 5.0]),
        ),
        mixtures=mixtures,
    )


def _score_imm_case(case, medians):
    return case.module["_evaluate_forecast"](
        case.forecast,
        forecast_median_positions=medians,
        reference=case.reference,
        specification=rolling.build_online_forecast_specs(
            3, initial_observation_count=1, prediction_count=2, stride=1
        )[0],
        forecast_origin_time=0.0,
        forecast_origin_position=[100.0, 200.0],
        window_runtime_seconds=0.02,
        forecast_runtime_seconds=0.01,
    )


def test_imm_evaluation_uses_median_errors_but_preserves_distribution_scores(
    imm_median_case,
):
    case = imm_median_case
    tracker = SimpleNamespace(forecast_mixtures=lambda times: case.mixtures)
    medians = imm_player._forecast_position_medians(tracker, case.forecast)
    np.testing.assert_array_equal(medians, [[2, 0], [0, 5]])
    table, covariances = _score_imm_case(case, medians)
    mean_table, _ = _score_imm_case(case, case.forecast.position_means)

    np.testing.assert_array_equal(table[["x_median", "y_median"]], medians)
    np.testing.assert_array_equal(table[["x_median_route", "y_median_route"]], medians)
    np.testing.assert_array_equal(table["position_error_m"], [3, 4])
    np.testing.assert_array_equal(mean_table["position_error_m"], [8, 10])
    np.testing.assert_array_equal(covariances, case.forecast.position_covariances)
    changed_columns = {
        "x_median",
        "y_median",
        "x_median_route",
        "y_median_route",
        "position_error_m",
    }
    unchanged = [name for name in table.columns if name not in changed_columns]
    pd.testing.assert_frame_equal(table[unchanged], mean_table[unchanged])
    summary = case.module["_summarize_predictions"](table)
    assert summary.ade_m == 3.5
    assert summary.fde_m == 4.0
    _, diagnostics = runtime_plotting._rolling_window_metric_series(table)
    assert diagnostics["rmse_m"] == pytest.approx([np.sqrt(12.5)])


@pytest.mark.parametrize("medians", [[[1, 2]], [[1, 2], [np.nan, 3]]])
def test_imm_evaluation_rejects_invalid_median_positions(imm_median_case, medians):
    with pytest.raises(ValueError, match="finite|aligned|median"):
        _score_imm_case(imm_median_case, medians)


@pytest.mark.parametrize("show_plot", [False, True])
def test_imm_evaluation_exports_and_plots_the_predictor_medians(
    imm_median_case, tmp_path, monkeypatch, show_plot
):
    case = imm_median_case
    namespace = case.module["main"].__globals__
    source = tmp_path / "radar.csv"
    pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "cycle_counter": [1, 2, 3],
            "sensor_x": 0.0,
            "sensor_y": 0.0,
            "sensor_yaw": 0.0,
            "detection_id": 1,
            "range": [10.0, 11.0, 12.0],
            "azimuth": 0.0,
            "std_range": 0.5,
            "std_azimuth": 0.0,
            "x_relative": [10.0, 11.0, 12.0],
            "y_relative": 0.0,
            "ground_truth_x": case.reference.x_meters,
            "ground_truth_y": case.reference.y_meters,
            "ground_truth_yaw": 0.0,
            "ground_truth_velocity": 1.0,
        }
    ).to_csv(source, index=False)

    class Tracker:
        estimated_position = np.array([100.0, 200.0])
        last_observation_time_seconds = 0.0

        @classmethod
        def initialize(cls, cycle, *, initial_detection_id):
            return cls()

        def forecast(self, times, *, seed):
            np.testing.assert_array_equal(times, [1, 2])
            return case.forecast

        def forecast_mixtures(self, times):
            np.testing.assert_array_equal(times, [1, 2])
            return case.mixtures

    output = tmp_path / "metrics.csv"
    monkeypatch.setitem(namespace, "_load_optional_imm_tracker", lambda: Tracker)
    monkeypatch.setitem(namespace, "DATA_FILE", source)
    monkeypatch.setitem(namespace, "OUTPUT_CSV", output)
    monkeypatch.setitem(namespace, "SHOW_ROLLING_DIAGNOSTICS_PLOT", False)
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    arguments = ["--observations", "1", "--predictions", "2", "--stride", "2"]
    if not show_plot:
        arguments.append("--no-plot")
    try:
        table, summary = case.module["main"](arguments)
        assert summary.ade_m == 3.5
        assert summary.fde_m == 4.0
        np.testing.assert_array_equal(table["position_error_m"], [3, 4])
        exported = pd.read_csv(output)
        np.testing.assert_array_equal(
            exported[["x_median", "y_median"]], [[2, 0], [0, 5]]
        )
        assert len(shown) == int(show_plot)
        if show_plot:
            forecast_line = next(
                line
                for line in shown[0].axes[0].lines
                if line.get_label() == "IMM-Prognosemediane"
            )
            np.testing.assert_array_equal(forecast_line.get_xdata(), [100, 2, 0])
            np.testing.assert_array_equal(forecast_line.get_ydata(), [200, 0, 5])
    finally:
        for figure in shown:
            plt.close(figure)


def test_imm_evaluation_loader_matches_predictor_medians_without_changing_filter():
    if not imm_player.IMM_PYTHON.is_file():
        pytest.skip("Optional local IMM implementation is unavailable")
    backend = runpy.run_path(str(imm_player.IMM_PYTHON))
    module = runpy.run_path(PROJECT_ROOT / "experiments/model_evaluation/imm.py")
    state = backend["_IMMState"](
        cv_state=np.array([10.0, 3.0, 20.0, 4.0]),
        cv_covariance=np.zeros((4, 4)),
        ctrv_state=np.array([10.0, 20.0, np.arctan2(4, 3), 0.2, 5.0]),
        ctrv_covariance=np.zeros((5, 5)),
        ca_state=np.array([10.0, 3.0, 0.0, 20.0, 4.0, 0.0]),
        ca_covariance=np.zeros((6, 6)),
        mode_probabilities=np.array([0.75, 0.2, 0.05]),
    )
    arguments = dict(
        _last_observation_time_seconds=0.0,
        _sensor_position=np.array([100.0, -50.0]),
    )
    original = backend["BR24IMMTracker"](_state=deepcopy(state), **arguments)
    evaluator = module["_load_optional_imm_tracker"]()(
        _state=deepcopy(state), **arguments
    )
    predictor = imm_player._load_optional_imm_tracker()(
        _state=deepcopy(state), **arguments
    )
    times = [0.4, 2.7]
    expected_forecast = original.forecast(times)
    forecast = evaluator.forecast(times)
    expected_medians = imm_player._forecast_position_medians(
        predictor, predictor.forecast(times)
    )
    actual_medians = imm_player._forecast_position_medians(evaluator, forecast)
    assert np.max(np.abs(expected_medians - expected_forecast.position_means)) > 0.01
    np.testing.assert_array_equal(actual_medians, expected_medians)
    for name in ("position_means", "position_covariances", "mode_probabilities"):
        np.testing.assert_array_equal(
            getattr(forecast, name), getattr(expected_forecast, name)
        )
        np.testing.assert_array_equal(
            getattr(evaluator.forecast(times), name), getattr(expected_forecast, name)
        )
    np.testing.assert_array_equal(
        evaluator.estimated_position, original.estimated_position
    )


@pytest.mark.parametrize(
    ("arguments", "diagnostics_enabled", "expected_plots"),
    [
        ([], True, 6),
        (["--stride", "2"], True, 6),
        (["--no-plot"], True, 0),
        ([], False, 0),
    ],
)
def test_imm_evaluation_entrypoint_adds_metric_plots_without_changing_export(
    tmp_path, monkeypatch, arguments, diagnostics_enabled, expected_plots
):
    module = runpy.run_path(PROJECT_ROOT / "experiments/model_evaluation/imm.py")
    predictions = pd.DataFrame(
        {
            "window_index": [0, 0, 1, 1],
            "forecast_origin_seconds": [100.0, 100.0, 102.5, 102.5],
            "horizon_step": [1, 2, 1, 2],
            "position_error_m": [3.0, 4.0, 5.0, 12.0],
            "log_predictive_density": [-1.0, -3.0, -5.0, -7.0],
            "energy_score_m": [2.0, 4.0, 6.0, 10.0],
            "fit_runtime_seconds": [0.01, 0.01, 0.02, 0.02],
            "inference_method": ["imm"] * 4,
        }
    )
    original = predictions.copy(deep=True)
    summary = object()
    output_csv = tmp_path / "results" / "imm.csv"
    namespace = module["main"].__globals__

    def evaluate(options):
        assert options.show_plot == ("--no-plot" not in arguments)
        assert options.stride == (2 if "--stride" in arguments else 1)
        return predictions, summary

    monkeypatch.setitem(namespace, "run_imm_evaluation", evaluate)
    monkeypatch.setitem(namespace, "OUTPUT_CSV", output_csv)
    monkeypatch.setitem(namespace, "SHOW_ROLLING_DIAGNOSTICS_PLOT", diagnostics_enabled)
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    try:
        result, returned_summary = module["main"](arguments)
        assert result is predictions and returned_summary is summary
        pd.testing.assert_frame_equal(result, original)
        pd.testing.assert_frame_equal(pd.read_csv(output_csv), original)
        assert len(shown) == expected_plots
        for figure in shown:
            assert figure._suptitle.get_text() == "IMM"
            np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [0, 2.5])
    finally:
        for figure in shown:
            plt.close(figure)


@pytest.mark.parametrize(
    ("script_name", "title", "expected_x_coordinates"),
    (
        (
            "plot_shiptech_trajectory.py",
            "Shiptech-Trajektorie mit Fahrtrichtung",
            [0.0, 3.0, 6.0],
        ),
        (
            "plot_br24_imm_tracked_trajectory.py",
            "IMM-getrackte BR24-Zieltrajektorie mit Fahrtrichtung",
            [0.0, 3.0, 6.0],
        ),
    ),
)
def test_processed_trajectory_plot_script_loads_its_configured_csv(
    tmp_path,
    monkeypatch,
    script_name,
    title,
    expected_x_coordinates,
):
    """Each entrypoint plots an interchangeable processed position CSV."""
    input_csv = tmp_path / "processed_trajectory.csv"
    pd.DataFrame(
        {
            "time": [0.0, 10.0, 20.0],
            "x": [0.0, 3.0, 6.0],
            "y": [0.0, 4.0, 8.0],
        }
    ).to_csv(input_csv, index=False)

    module = runpy.run_path(
        PROJECT_ROOT / "experiments" / "data_exploration" / script_name
    )
    monkeypatch.setitem(module["main"].__globals__, "INPUT_CSV", input_csv)
    monkeypatch.setattr(module["plotting"].plt, "show", lambda: None)
    try:
        module["main"]()

        trajectory_axis = plt.gcf().axes[0]
        assert trajectory_axis.get_title() == title
        np.testing.assert_allclose(
            trajectory_axis.lines[0].get_xdata(),
            expected_x_coordinates,
        )
    finally:
        plt.close("all")


def test_br24_reference_plot_uses_raw_points_within_radar_window(tmp_path, monkeypatch):
    reference_csv = tmp_path / "reference.csv"
    sensor_csv = tmp_path / "sensor.csv"
    pd.DataFrame(
        {
            "time": [1.0, 5.0, 7.0, 10.0, 12.0],
            "x": [0.0, 3.0, 6.0, 9.0, 12.0],
            "y": [0.0, 4.0, 8.0, 12.0, 16.0],
        }
    ).to_csv(reference_csv, index=False)
    pd.DataFrame({"Time": [10.0, 5.0]}).to_csv(sensor_csv, index=False)
    script = (
        PROJECT_ROOT / "experiments/data_exploration/plot_br24_reference_trajectory.py"
    )
    module = runpy.run_path(script)
    monkeypatch.setitem(module["main"].__globals__, "INPUT_CSV", reference_csv)
    monkeypatch.setitem(module["main"].__globals__, "SENSOR_CSV", sensor_csv)
    monkeypatch.setattr(module["plotting"].plt, "show", lambda: None)

    try:
        module["main"]()
        line = plt.gcf().axes[0].lines[0]
        np.testing.assert_allclose(line.get_xdata(), [3.0, 6.0, 9.0])
        np.testing.assert_allclose(line.get_ydata(), [4.0, 8.0, 12.0])
    finally:
        plt.close("all")


def test_br24_bayesian_plot_tracks_every_cycle_without_reference_input(
    tmp_path, monkeypatch
):
    """The displayed track is online SMC output, including missed cycles."""
    script = (
        PROJECT_ROOT
        / "experiments/data_exploration/plot_br24_bayesian_ctrv_tracked_trajectory.py"
    )
    assert script.exists()
    module = runpy.run_path(script)
    input_csv = tmp_path / "radar.csv"
    radar = pd.DataFrame(
        {
            "time": [10.0, 10.0, 11.5, 14.0],
            "cycle_counter": [4, 4, 5, 6],
            "sensor_x": [100.0] * 4,
            "sensor_y": [-20.0] * 4,
            "sensor_yaw": [0.0] * 4,
            "detection_id": [10, 9, 10, np.nan],
            "range": [200.0, 10.0, 10.0, np.nan],
            "azimuth": [0.0, 0.0, 0.0, np.nan],
            "std_range": [0.5, 0.5, 0.5, np.nan],
            "std_azimuth": [0.0, 0.0, 0.0, np.nan],
            "x_relative": [200.0, 10.0, 10.0, np.nan],
            "y_relative": [0.0, 0.0, 0.0, np.nan],
            "ground_truth_x": [9999.0] * 4,
            "ground_truth_y": [-9999.0] * 4,
            "ground_truth_yaw": [0.0] * 4,
            "ground_truth_velocity": [0.0] * 4,
        }
    )
    radar.to_csv(input_csv, index=False)
    globals_ = module["main"].__globals__
    monkeypatch.setitem(globals_, "INPUT_CSV", input_csv)
    monkeypatch.setitem(
        globals_,
        "SMC_CONFIG",
        SequentialMonteCarloCTRVConfig(particle_count=128, posterior_draw_count=16),
    )
    monkeypatch.setattr(module["plotting"].plt, "show", lambda: None)
    plotted_data = []
    plot_trajectory = module["plotting"].plot_processed_trajectory

    def record_and_plot(data, **kwargs):
        plotted_data.append(data.copy())
        return plot_trajectory(data, **kwargs)

    monkeypatch.setattr(
        module["plotting"], "plot_processed_trajectory", record_and_plot
    )
    try:
        module["main"]()
        axis = plt.gcf().axes[0]
        first_x = np.asarray(axis.lines[0].get_xdata()).copy()
        first_y = np.asarray(axis.lines[0].get_ydata()).copy()
        assert len(first_x) == 3
        assert abs(first_x[0] - 110.0) < 2.0
        assert abs(first_y[0] + 20.0) < 2.0
        assert plotted_data[0]["time"].tolist() == [10.0, 11.5, 14.0]
        assert not np.allclose(first_x[1:], 110.0)
        assert (
            axis.get_title()
            == "Bayesian-CTRV-getrackte BR24-Zieltrajektorie mit Fahrtrichtung"
        )
        plt.close("all")

        radar["ground_truth_x"] = -1e6
        radar["ground_truth_y"] = 1e6
        radar["ground_truth_yaw"] = 2.0
        radar["ground_truth_velocity"] = 1000.0
        radar.to_csv(input_csv, index=False)
        module["main"]()
        axis = plt.gcf().axes[0]
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), first_x)
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), first_y)
    finally:
        plt.close("all")
