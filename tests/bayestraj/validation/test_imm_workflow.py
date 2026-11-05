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
    table = pd.DataFrame({"position_error_m": [1.0]})

    def evaluate(**kwargs):
        calls.append(kwargs)
        return table, None

    def diagnostics(predictions):
        assert (tmp_path / "imm.csv").is_file()
        plots.append(predictions)

    monkeypatch.setattr(module["workflow"], "run_imm_evaluation", evaluate)
    monkeypatch.setattr(
        module["runtime_plotting"], "plot_imm_rolling_diagnostics", diagnostics
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
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "imm.csv"), table)


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
    # Standard normal central 90% by default, or the explicitly requested 50%.
    quantile = 1.6448536269514722 if use_defaults else 0.6744897501960817
    np.testing.assert_allclose(table["x_lower"], -quantile)
    np.testing.assert_allclose(table["x_upper"], quantile)
    assert list(tmp_path.iterdir()) == []
