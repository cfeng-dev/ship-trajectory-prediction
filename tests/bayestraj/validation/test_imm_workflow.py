"""IMM evaluation configuration belongs in the shared validation workflow."""

import importlib
import runpy
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import bayestraj.validation.cli as cli


def test_imm_main_delegates_header_settings_and_return_value(tmp_path, monkeypatch):
    workflow = importlib.import_module("bayestraj.validation.imm_workflow")
    script = Path(__file__).resolve().parents[3] / "scripts/model_evaluation/imm.py"
    module = runpy.run_path(script)
    calls = []
    result = (object(), object())

    def run(**kwargs):
        calls.append(kwargs)
        return result

    def unexpected_workflow(**kwargs):
        pytest.fail("The entrypoint must delegate orchestration to run_evaluation.")

    monkeypatch.setattr(workflow, "run_evaluation", run)
    monkeypatch.setattr(workflow, "run_imm_evaluation", unexpected_workflow)
    monkeypatch.setitem(module["main"].__globals__, "OUTPUT_CSV", tmp_path / "imm.csv")
    arguments = ["--max-windows", "1"]

    assert module["main"](arguments) is result

    assert calls == [
        dict(
            data_file=module["DATA_FILE"],
            imm_python=module["IMM_PYTHON"],
            output_csv=tmp_path / "imm.csv",
            prediction_count=module["PREDICTION_COUNT"],
            max_windows=module["MAX_WINDOWS"],
            show_rolling_diagnostics_plot=module["SHOW_ROLLING_DIAGNOSTICS_PLOT"],
            description=module["__doc__"],
            argv=arguments,
        )
    ]
    assert not (tmp_path / "imm.csv").exists()


@pytest.mark.parametrize(
    ("arguments", "expected_observations"),
    [([], 2), (["--observations", "5"], 5)],
)
def test_imm_header_uses_internal_defaults_and_keeps_diagnostics(
    tmp_path, monkeypatch, arguments, expected_observations
):
    script = Path(__file__).resolve().parents[3] / "scripts/model_evaluation/imm.py"
    module = runpy.run_path(script)
    settings = module["main"].__globals__
    calls = []
    plots = []
    table = pd.DataFrame(
        {
            "time": pd.to_datetime([49078.5], unit="s", utc=True),
            "forecast_origin_time": pd.to_datetime([49075.8], unit="s", utc=True),
            "forecast_origin_seconds": [49075.8],
            "horizon_seconds": [2.7],
            "fit_runtime_seconds": [0.002],
            "position_error_m": [1.0],
        }
    )
    original = table.copy(deep=True)

    def evaluate(**kwargs):
        calls.append(kwargs)
        return table, None

    def diagnostics(predictions):
        assert (tmp_path / "imm.csv").is_file()
        plots.append(predictions)

    monkeypatch.setattr(module["evaluation"], "run_imm_evaluation", evaluate)
    monkeypatch.setattr(
        module["evaluation"].runtime_plotting,
        "plot_imm_rolling_diagnostics",
        diagnostics,
    )
    monkeypatch.setitem(settings, "OUTPUT_CSV", tmp_path / "imm.csv")

    module["main"](arguments)

    assert not {
        "INITIAL_OBSERVATION_COUNT",
        "STRIDE",
        "CREDIBLE_INTERVAL",
        "GAUSSIAN_PLOT_DRAWS",
        "PLOT_SEED",
        "SHOW_ROLLING_PLOT",
    }.intersection(module)
    assert module["SHOW_ROLLING_DIAGNOSTICS_PLOT"] is True
    assert calls[0]["options"].stride == 1
    assert calls[0]["options"].observation_count == expected_observations
    assert set(calls[0]) == {"data_file", "imm_python", "options"}
    assert len(plots) == 1 and plots[0] is table
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "imm.csv"),
        table.drop(columns=["time", "forecast_origin_time"]),
    )
    pd.testing.assert_frame_equal(table, original)


def test_imm_associated_evaluation_uses_covariances_and_evaluation_only_reference(
    tmp_path, monkeypatch
):
    workflow = importlib.import_module("bayestraj.validation.imm_workflow")
    path = tmp_path / "associated.csv"
    data = pd.DataFrame(
        {
            "time": [49073.2, 49075.8, 49081.0, 49084.1],
            "cycle_counter": [2, 3, 5, 6],
            "detection_id": [1, 1, 2, 2],
            "x": [10.0, 12.0, 16.0, 18.0],
            "y": [-4.0] * 4,
            "cov_xx": [2.0] * 4,
            "cov_xy": [0.25] * 4,
            "cov_yy": [1.0] * 4,
            "ground_truth_x": [20.0, 22.0, 26.0, 28.0],
            "ground_truth_y": [0.0] * 4,
        }
    )
    measurements = []

    class Tracker:
        @classmethod
        def initialize_position(cls, time_seconds, position, measurement_covariance):
            instance = cls()
            instance.update_position(time_seconds, position, measurement_covariance)
            return instance

        def update_position(self, time_seconds, position, measurement_covariance):
            measurements.append(
                (
                    time_seconds,
                    np.asarray(position).copy(),
                    measurement_covariance.copy(),
                )
            )
            self.last_observation_time_seconds = time_seconds
            self.estimated_position = np.asarray(position)

        def process_cycle(self, cycle):
            pytest.fail("Associated inputs must not run radar association")

        def forecast(self, times, *, seed):
            return SimpleNamespace(
                time_seconds=np.asarray(times),
                position_means=np.tile(self.estimated_position, (len(times), 1)),
                position_covariances=np.tile(np.eye(2), (len(times), 1, 1)),
                mode_probabilities=np.tile([1.0, 0.0, 0.0], (len(times), 1)),
            )

    monkeypatch.setattr(workflow, "_load_optional_imm_tracker", lambda _: Tracker)
    output_path = tmp_path / "imm.csv"

    def evaluate():
        data.to_csv(path, index=False)
        return workflow.run_evaluation(
            data_file=path,
            imm_python=tmp_path / "backend.py",
            output_csv=output_path,
            prediction_count=2,
            max_windows=2,
            show_rolling_diagnostics_plot=False,
            argv=["--observations", "2", "--no-plot"],
        )[0]

    before = evaluate()
    assert "forecast_origin_time" not in before
    data["ground_truth_x"] += 1000.0
    after = evaluate()
    assert [item[0] for item in measurements[:3]] == [49073.2, 49075.8, 49081.0]
    np.testing.assert_array_equal(
        [item[1] for item in measurements[:3]],
        [[10.0, -4.0], [12.0, -4.0], [16.0, -4.0]],
    )
    np.testing.assert_array_equal(measurements[0][2], [[2.0, 0.25], [0.25, 1.0]])
    np.testing.assert_array_equal(before["x_actual"], [26.0, 28.0, 28.0])
    np.testing.assert_array_equal(
        before[["x_median", "y_median"]], after[["x_median", "y_median"]]
    )
    np.testing.assert_allclose(after["x_actual"] - before["x_actual"], 1000.0)
    exported = pd.read_csv(output_path)
    start = exported.columns.get_loc("gaussian_negative_log_predictive_density")
    assert tuple(exported.columns[start : start + 4]) == (
        "gaussian_negative_log_predictive_density",
        "student_t_negative_log_predictive_density",
        "gaussian_energy_score_m",
        "student_t_energy_score_m",
    )
    for column in (
        "student_t_energy_score_m",
        "student_t_negative_log_predictive_density",
    ):
        assert (after[column] > before[column]).all()
        np.testing.assert_allclose(exported[column], after[column])


def _parse(arguments):
    return cli.parse_imm_evaluation_arguments(
        description="IMM evaluation test",
        observation_count=7,
        prediction_count=3,
        stride=2,
        max_windows=4,
        show_plot=True,
        argv=arguments,
    )


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], (7, 3, 2, 4, True)),
        (
            [
                "--observations",
                "5",
                "--predictions",
                "2",
                "--stride",
                "1",
                "--max-windows",
                "6",
                "--no-plot",
            ],
            (5, 2, 1, 6, False),
        ),
    ],
)
def test_shared_imm_cli_preserves_defaults_and_overrides(arguments, expected):
    options = _parse(arguments)
    assert (
        options.observation_count,
        options.prediction_count,
        options.stride,
        options.max_windows,
        options.show_plot,
    ) == expected


@pytest.mark.parametrize(
    "flag", ["--observations", "--predictions", "--stride", "--max-windows"]
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_shared_imm_cli_rejects_nonpositive_counts(flag, value):
    with pytest.raises(ValueError, match="positive integer"):
        _parse([flag, value])


@pytest.mark.parametrize("use_defaults", [False, True])
def test_shared_imm_workflow_preserves_scoring_and_seed_without_trajectory_plots(
    tmp_path, monkeypatch, use_defaults
):
    workflow = importlib.import_module("bayestraj.validation.imm_workflow")
    clock = [0.0]
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: clock[0])
    original_metrics = workflow.metrics.gaussian_position_metrics
    original_medians = workflow.imm_adapter.forecast_position_medians

    def timed_metrics(*args, **kwargs):
        clock[0] += 10.0
        return original_metrics(*args, **kwargs)

    def timed_medians(*args, **kwargs):
        clock[0] += 1.0
        return original_medians(*args, **kwargs)

    monkeypatch.setattr(workflow.metrics, "gaussian_position_metrics", timed_metrics)
    monkeypatch.setattr(
        workflow.imm_adapter, "forecast_position_medians", timed_medians
    )
    options = cli.IMMEvaluationOptions(1, 2, 2, 1, True)
    data_file, backend = tmp_path / "radar.csv", tmp_path / "backend.py"
    cycles = tuple(
        SimpleNamespace(
            time_seconds=float(index),
            cycle_counter=index + 1,
            detections=(SimpleNamespace(detection_id=1),),
        )
        for index in range(3)
    )
    reference = SimpleNamespace(
        cycle_counters=np.array([1, 2, 3]),
        time_seconds=np.array([0.0, 1.0, 2.0]),
        x_meters=np.zeros(3),
        y_meters=np.zeros(3),
    )

    class Tracker:
        estimated_position = np.zeros(2)
        last_observation_time_seconds = 0.0

        @classmethod
        def initialize(cls, cycle, *, initial_detection_id):
            return cls()

        def forecast(self, times, *, seed):
            clock[0] += 3.0
            assert seed == (42 if use_defaults else 137)
            np.testing.assert_array_equal(times, [1, 2])
            return SimpleNamespace(
                time_seconds=np.asarray(times),
                position_means=np.zeros((2, 2)),
                position_covariances=np.tile(np.eye(2), (2, 1, 1)),
                mode_probabilities=np.tile([1, 0, 0], (2, 1)),
            )

    def load(path):
        assert path == backend
        return Tracker

    def read_input(path):
        assert path == data_file
        return cycles

    def read_reference(path):
        assert path == data_file
        return reference

    def unexpected_plot(*args, **kwargs):
        pytest.fail("The IMM workflow must not create trajectory figures.")

    monkeypatch.setattr(workflow, "_load_optional_imm_tracker", load)
    monkeypatch.setattr(workflow.br24, "read_br24_tracking_input", read_input)
    monkeypatch.setattr(workflow.br24, "read_br24_tracking_reference", read_reference)
    monkeypatch.setattr(plt, "subplots", unexpected_plot)
    overrides = {} if use_defaults else {"credible_interval": 0.5, "forecast_seed": 137}
    table, summary = workflow.run_imm_evaluation(
        data_file=data_file,
        imm_python=backend,
        options=options,
        **overrides,
    )
    assert summary.window_count == 1
    assert table["fit_runtime_seconds"].eq(3.0).all()
    assert "window_runtime_seconds" not in table
    assert summary.mean_fit_runtime_seconds == 3.0
    assert summary.total_computation_time_seconds == 3.0
    # Standard normal central 90% by default, or the explicitly requested 50%.
    quantile = 1.6448536269514722 if use_defaults else 0.6744897501960817
    np.testing.assert_allclose(table["x_lower"], -quantile)
    np.testing.assert_allclose(table["x_upper"], quantile)
    assert list(tmp_path.iterdir()) == []
