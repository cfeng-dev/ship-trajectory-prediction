"""Tests for probabilistic position-evaluation metrics."""

from types import SimpleNamespace

import numpy as np
import pytest

import bayestraj.validation.metrics as metrics


class _FakeFit:
    def __init__(self, variables):
        self._variables = variables

    def stan_variable(self, variable_name, **_kwargs):
        return self._variables[variable_name]


def test_public_joint_log_predictive_density_matches_isotropic_normal() -> None:
    values = metrics.joint_log_predictive_density(
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        np.array([[0.0, -1.0], [0.0, -1.0]]),
        np.array([1.0, 1.0]),
    )

    assert values.shape == (2,)
    assert np.all(np.isfinite(values))
    assert values == pytest.approx([-np.log(2 * np.pi)] * 2)


@pytest.mark.parametrize("sigma", [1.0, 0.1])
def test_position_evaluation_reports_joint_negative_log_predictive_density(sigma):
    window = SimpleNamespace(
        prediction_count=1,
        prediction_slice=slice(1, 2),
        observation_count=1,
        x_meters=np.asarray([0.0, 0.0]),
        y_meters=np.asarray([0.0, 0.0]),
        time_seconds=np.asarray([0.0, 10.0]),
        timestamps=np.asarray(["origin", "target"]),
    )
    fit = _FakeFit(
        {
            "x_prediction": np.asarray([[0.0], [0.0]]),
            "y_prediction": np.asarray([[0.0], [0.0]]),
            "sigma_position_observation": np.asarray([sigma, sigma]),
        }
    )

    evaluation = metrics.evaluate_position_predictions(fit, window)

    expected = np.log(2 * np.pi * sigma**2)
    assert evaluation.total_negative_log_predictive_density == pytest.approx(expected)
    assert evaluation.mean_negative_log_predictive_density == pytest.approx(expected)
    assert evaluation.energy_score_m == pytest.approx(0.0)
    assert evaluation.joint_coverage_50 == pytest.approx(1.0)
    assert evaluation.joint_coverage_90 == pytest.approx(1.0)
    assert evaluation.prediction_table[
        "negative_log_predictive_density"
    ].tolist() == pytest.approx([expected])
    assert "log_predictive_density" not in evaluation.prediction_table
    assert evaluation.prediction_table["joint_covered_50"].tolist() == [True]
    assert evaluation.prediction_table["joint_covered_90"].tolist() == [True]

    report_lines = metrics.format_position_evaluation(
        evaluation,
        computation_time_seconds=0.1,
    ).splitlines()
    assert any("Mean negative log predictive density" in line for line in report_lines)
    computation_time_line = next(
        index
        for index, line in enumerate(report_lines)
        if line.startswith("Computation time")
    )
    interval_width_line = next(
        index
        for index, line in enumerate(report_lines)
        if line.startswith("Mean marginal interval width")
    )
    assert computation_time_line == interval_width_line + 1
