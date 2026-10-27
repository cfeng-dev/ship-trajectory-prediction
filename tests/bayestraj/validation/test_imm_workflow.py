"""IMM evaluation configuration belongs in the shared validation workflow."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

import bayestraj.validation.cli as cli


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


def test_shared_imm_workflow_uses_explicit_paths_seed_and_plot_settings(
    tmp_path, monkeypatch
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
            assert seed == 137
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

    captured = []

    def plot(route_x, route_y, groups, **kwargs):
        captured.extend(groups)

    monkeypatch.setattr(workflow, "_load_optional_imm_tracker", load)
    monkeypatch.setattr(workflow.br24, "read_br24_tracking_input", read_input)
    monkeypatch.setattr(workflow.br24, "read_br24_tracking_reference", read_reference)
    monkeypatch.setattr(
        workflow.validation_plotting, "plot_bayesian_rolling_predictions", plot
    )
    table, summary = workflow.run_imm_evaluation(
        data_file=data_file,
        imm_python=backend,
        options=options,
        credible_interval=0.5,
        gaussian_plot_draws=7,
        plot_seed=137,
    )
    assert summary.window_count == 1
    assert len(captured) == 1
    assert captured[0].x_samples.shape == captured[0].y_samples.shape == (7, 2)
    np.testing.assert_array_equal(captured[0].forecast_positions, [[0, 0], [0, 0]])
    # Standard normal central 50% quantiles; no accidental fallback to 90% defaults.
    np.testing.assert_allclose(table["x_lower"], -0.6744897501960817)
    np.testing.assert_allclose(table["x_upper"], 0.6744897501960817)
    assert list(tmp_path.iterdir()) == []
