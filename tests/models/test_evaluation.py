"""Tests for shared posterior trajectory evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from ship_trajectory_prediction.models.evaluation import (
    PositionEvaluation,
    evaluate_position_predictions,
    format_position_evaluation,
    print_position_evaluation,
)


def test_evaluate_position_predictions_calculates_shared_metrics():
    """ADE, FDE, horizons, and radial coverage should use held-out positions."""
    window = FakeWindow()
    fit = FakeFit(
        x_prediction=np.array(
            [
                [1.0, 3.0],
                [2.0, 4.0],
                [3.0, 5.0],
            ]
        ),
        y_prediction=np.array(
            [
                [-1.0, -1.0],
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        ),
    )

    evaluation = evaluate_position_predictions(fit, window)

    assert isinstance(evaluation, PositionEvaluation)
    assert evaluation.errors_m == pytest.approx([1.0, 2.0])
    assert evaluation.ade_m == pytest.approx(1.5)
    assert evaluation.fde_m == pytest.approx(2.0)
    assert evaluation.radial_coverage == pytest.approx(0.5)
    assert evaluation.prediction_table["horizon_seconds"].tolist() == [10.0, 20.0]
    assert evaluation.prediction_table["radial_covered"].tolist() == [True, False]
    assert evaluation.mean_prediction_radius_m == pytest.approx(np.sqrt(2))
    assert evaluation.mean_marginal_interval_width_m > 0


@pytest.mark.parametrize("credible_interval", [0, 1, -0.1, 1.1, np.nan, "bad"])
def test_evaluate_position_predictions_rejects_invalid_interval(
    credible_interval,
):
    """Credible interval probabilities must lie strictly between zero and one."""
    with pytest.raises(ValueError, match="credible_interval"):
        evaluate_position_predictions(
            create_valid_fit(),
            FakeWindow(),
            credible_interval=credible_interval,
        )


def test_evaluate_position_predictions_rejects_mismatched_sample_shapes():
    """Both coordinates must provide one draw for every prediction horizon."""
    fit = FakeFit(
        x_prediction=np.ones((3, 2)),
        y_prediction=np.ones((3, 1)),
    )

    with pytest.raises(ValueError, match="unexpected shape"):
        evaluate_position_predictions(fit, FakeWindow())


def test_evaluate_position_predictions_rejects_non_finite_draws():
    """Non-finite posterior draws cannot produce meaningful metrics."""
    fit = create_valid_fit()
    fit.variables["x_prediction"][0, 0] = np.nan

    with pytest.raises(ValueError, match="finite draws"):
        evaluate_position_predictions(fit, FakeWindow())


def test_evaluate_position_predictions_requires_cmdstan_style_fit():
    """The evaluator should fail clearly for unsupported fit objects."""
    with pytest.raises(TypeError, match="CmdStan-style"):
        evaluate_position_predictions(object(), FakeWindow())


@pytest.mark.parametrize(
    "prediction_times",
    [
        [0.0, 10.0, 30.0, 20.0],
        [0.0, 10.0, 20.0, 20.0],
        [0.0, 20.0, 10.0, 30.0],
        [0.0, 10.0, np.nan, 30.0],
    ],
)
def test_evaluate_position_predictions_rejects_invalid_horizons(
    prediction_times,
):
    """FDE requires finite positive horizons in chronological order."""
    window = FakeWindow()
    window.time_seconds = np.asarray(prediction_times, dtype=float)

    with pytest.raises(ValueError, match="Prediction horizons"):
        evaluate_position_predictions(create_valid_fit(), window)


def test_format_and_print_position_evaluation(capsys):
    """Console helpers should report identical shared accuracy metrics."""
    evaluation = evaluate_position_predictions(create_valid_fit(), FakeWindow())

    report = format_position_evaluation(evaluation)
    print_position_evaluation(evaluation)
    captured = capsys.readouterr().out

    assert "ADE" in report
    assert "FDE" in report
    assert "Radial 90% coverage" in report
    assert "Per-horizon accuracy" in report
    assert report in captured


def test_format_position_evaluation_aligns_metric_separators():
    """Every metric separator should use the longest label width."""
    evaluation = evaluate_position_predictions(create_valid_fit(), FakeWindow())

    report = format_position_evaluation(evaluation)
    metric_lines = report.splitlines()[1:6]

    assert len({line.index(":") for line in metric_lines}) == 1


def test_format_position_evaluation_rejects_invalid_input():
    """Formatting should require a completed position evaluation."""
    with pytest.raises(TypeError, match="PositionEvaluation"):
        format_position_evaluation(object())


def create_valid_fit():
    """Return finite posterior draws for both held-out horizons."""
    return FakeFit(
        x_prediction=np.array(
            [
                [1.0, 3.0],
                [2.0, 4.0],
                [3.0, 5.0],
            ]
        ),
        y_prediction=np.array(
            [
                [-1.0, -1.0],
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        ),
    )


class FakeWindow:
    """Minimal shared trajectory-window interface for metric tests."""

    def __init__(self):
        self.timestamps = pd.date_range(
            "2026-01-10 08:00:00",
            periods=4,
            freq="10s",
            tz="UTC",
        )
        self.time_seconds = np.array([0.0, 10.0, 20.0, 30.0])
        self.x_meters = np.array([0.0, 1.0, 3.0, 6.0])
        self.y_meters = np.zeros(4)
        self.observation_count = 2

    @property
    def prediction_count(self):
        """Return the number of held-out positions."""
        return 2

    @property
    def prediction_slice(self):
        """Return the slice selecting held-out positions."""
        return slice(self.observation_count, None)


class FakeFit:
    """Minimal CmdStan-style fit object for posterior draw tests."""

    def __init__(self, **variables):
        self.variables = variables

    def stan_variable(self, name):
        """Return one stored posterior variable."""
        return self.variables[name]
