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


def test_latent_gaussian_density_uses_mean_full_covariance_and_paired_draws():
    # Mean=(1,0), covariance=[[4,4/3],[4/3,4/3]], median=(0,0).
    values = metrics.latent_position_log_predictive_density(
        [2.0, 12.0],
        [1.0, 21.0],
        [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [4.0, 14.0]],
        [[-1.0, 19.0], [1.0, 21.0], [-1.0, 19.0], [1.0, 21.0]],
    )
    expected = -np.log(2 * np.pi) - 0.5 * np.log(32 / 9) - 0.375
    np.testing.assert_allclose(values, [expected, expected])


def test_gaussian_evaluation_uses_full_covariance_for_regions_and_intervals():
    from scipy.stats import chi2, norm

    covariance = np.array([[[4.0, 1.0], [1.0, 9.0]]])
    target = np.linalg.cholesky(covariance[0]) @ [1.0, 1.0]
    values = metrics.gaussian_position_metrics([target], [[0.0, 0.0]], covariance)
    assert values["squared_mahalanobis_distance"][0] == pytest.approx(2.0)
    assert not values["joint_covered_50"][0]
    assert values["joint_covered_90"][0]
    assert values["prediction_radius_m"][0] == pytest.approx(
        np.sqrt(chi2.ppf(0.9, 2)) * 35.0**0.25
    )
    assert values["mean_marginal_interval_width_m"][0] == pytest.approx(
        norm.ppf(0.95) * 5.0
    )
    assert values["x_lower"][0] == pytest.approx(-norm.ppf(0.95) * 2.0)
    assert values["y_upper"][0] == pytest.approx(norm.ppf(0.95) * 3.0)
    assert values["evaluation_definition"] == metrics.EVALUATION_DEFINITION


def test_gaussian_energy_matches_centered_isotropic_analytic_score():
    values = metrics.gaussian_position_metrics([[0.0, 0.0]], [[0.0, 0.0]], [np.eye(2)])
    expected = np.sqrt(np.pi / 2.0) * (1 - 1 / np.sqrt(2.0))
    assert values["energy_score_m"][0] == pytest.approx(expected, rel=1e-10)


def test_gaussian_energy_handles_nearly_one_dimensional_forecasts():
    values = metrics.gaussian_position_metrics(
        [[0.0, 0.0]], [[0.0, 0.0]], [np.diag([1.0, 1e-12])]
    )
    expected = np.sqrt(2.0 / np.pi) * (1 - 1 / np.sqrt(2.0))
    assert values["energy_score_m"][0] == pytest.approx(expected, abs=1e-7)


def test_gaussian_scores_are_rotation_invariant_and_scale_in_metres():
    target = np.array([[3.0, -2.0]])
    mean = np.array([[0.5, 0.3]])
    covariance = np.array([[[4.0, 1.3], [1.3, 9.0]]])
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    baseline = metrics.gaussian_position_metrics(target, mean, covariance)
    rotated = metrics.gaussian_position_metrics(
        target @ rotation.T,
        mean @ rotation.T,
        rotation[None] @ covariance @ rotation.T[None],
    )
    scaled = metrics.gaussian_position_metrics(target * 5, mean * 5, covariance * 25)
    np.testing.assert_allclose(
        rotated["energy_score_m"], baseline["energy_score_m"], rtol=1e-12
    )
    np.testing.assert_allclose(scaled["energy_score_m"], baseline["energy_score_m"] * 5)
    np.testing.assert_allclose(
        scaled["prediction_radius_m"], baseline["prediction_radius_m"] * 5
    )
    np.testing.assert_allclose(
        rotated["squared_mahalanobis_distance"],
        baseline["squared_mahalanobis_distance"],
    )


@pytest.mark.parametrize("problem", ["shape", "nan", "negative_variance"])
def test_gaussian_position_density_rejects_invalid_distributions(problem):
    covariance = np.eye(2)[None, :, :]
    if problem == "shape":
        covariance = np.eye(3)[None, :, :]
    elif problem == "nan":
        covariance[0, 0, 0] = np.nan
    else:
        covariance[0, 0, 0] = -1.0
    with pytest.raises(ValueError):
        metrics.gaussian_position_log_predictive_density(
            [[0.0, 0.0]], [[0.0, 0.0]], covariance
        )


def test_gaussian_position_density_regularizes_degenerate_covariance_consistently():
    from imm_predictor import player

    from bayestraj.validation import imm_workflow

    expected = -np.log(2 * np.pi * 1e-12)
    observed = metrics.gaussian_position_log_predictive_density(
        [[0.0, 0.0]], [[0.0, 0.0]], np.zeros((1, 2, 2))
    )
    assert observed[0] == pytest.approx(expected)
    assert imm_workflow._gaussian_log_density(
        [0, 0], [0, 0], np.zeros((2, 2))
    ) == pytest.approx(expected)
    assert player._gaussian_log_density(
        [0, 0], [0, 0], np.zeros((2, 2))
    ) == pytest.approx(expected)


def test_imm_visual_covariances_use_the_shared_relative_floor():
    from imm_predictor import player

    from bayestraj.validation import imm_workflow

    covariance = np.array([np.diag([1e6, 0.0])])
    for regularize in (
        metrics.regularize_position_covariances,
        imm_workflow._regularize_covariances,
        player._regularize_covariances,
    ):
        assert regularize(covariance)[0, 1, 1] == pytest.approx(1e-6)


@pytest.mark.parametrize("noise", [None, 0.1, 100.0])
def test_evaluators_and_both_predictors_share_latent_gaussian_nlpd(noise):
    from bayesian_predictor import dashboard
    from imm_predictor import player

    from bayestraj.validation import imm_workflow

    positions = np.array([[[0.0, -1.0]], [[0.0, 1.0]], [[0.0, -1.0]], [[4.0, 1.0]]])
    window = SimpleNamespace(
        prediction_count=1,
        prediction_slice=slice(2, 3),
        observation_count=2,
        x_meters=np.array([0.0, 0.0, 2.0]),
        y_meters=np.array([0.0, 0.0, 1.0]),
        time_seconds=np.array([0.0, 10.0, 20.0]),
        timestamps=np.array(["a", "b", "c"]),
    )
    variables = {"x_prediction": positions[:, :, 0], "y_prediction": positions[:, :, 1]}
    if noise is not None:
        variables["sigma_position_observation"] = np.full(4, noise)
    bayesian = metrics.evaluate_position_predictions(_FakeFit(variables), window)
    trajectory = dashboard.PosteriorDashboardTrajectory(
        window.x_meters, window.y_meters, np.zeros(3), np.zeros(3)
    )
    update = dashboard.PosteriorDashboardUpdate(
        observation_count=2,
        samples_by_parameter={name: np.ones(4) for name in dashboard.PARAMETER_NAMES},
        forecast=dashboard.PosteriorDashboardForecast(
            time_offsets_seconds=[10.0],
            median_positions=[[0.0, 0.0]],
            sample_positions=positions[:1],
            posterior_positions=positions,
            position_observation_noise_samples=None
            if noise is None
            else np.full(4, noise),
        ),
    )
    gui = dashboard.analysis_metrics_at(trajectory, {2: update}, 2)
    forecast = SimpleNamespace(
        time_seconds=np.array([20.0]),
        position_means=np.array([[1.0, 0.0]]),
        position_covariances=np.array([[[4.0, 4 / 3], [4 / 3, 4 / 3]]]),
        mode_probabilities=np.array([[1.0, 0.0, 0.0]]),
    )
    reference = SimpleNamespace(x_meters=window.x_meters, y_meters=window.y_meters)
    stage = SimpleNamespace(
        observation_count=2, forecast=forecast, forecast_median_positions=None
    )
    imm_gui = player._forecast_metrics(stage, reference, coverage_stages=(stage,))
    imm, _ = imm_workflow._evaluate_forecast(
        forecast,
        credible_interval=0.9,
        forecast_median_positions=forecast.position_means,
        reference=reference,
        specification=SimpleNamespace(
            prediction_count=1,
            forecast_start_index=2,
            observation_count=2,
            window_index=0,
            start_index=0,
        ),
        forecast_origin_time=10.0,
        forecast_origin_position=np.zeros(2),
        forecast_runtime_seconds=0.0,
    )
    expected = np.log(2 * np.pi) + 0.5 * np.log(32 / 9) + 0.375
    assert bayesian.mean_negative_log_predictive_density == pytest.approx(expected)
    assert gui.negative_log_predictive_density == pytest.approx(expected)
    assert imm_gui.negative_log_predictive_density == pytest.approx(expected)
    assert imm.negative_log_predictive_density.iloc[0] == pytest.approx(expected)
    assert bayesian.energy_score_m == pytest.approx(imm.energy_score_m.iloc[0])
    assert gui.energy_score_m == pytest.approx(imm.energy_score_m.iloc[0])
    assert imm_gui.energy_score_m == pytest.approx(imm.energy_score_m.iloc[0])
    for name in (
        "x_lower",
        "x_upper",
        "y_lower",
        "y_upper",
        "prediction_radius_m",
        "mean_marginal_interval_width_m",
        "squared_mahalanobis_distance",
        "squared_mahalanobis_radius",
        "joint_covered_50",
        "joint_covered_90",
    ):
        np.testing.assert_allclose(bayesian.prediction_table[name], imm[name])
    for probability in (50, 90):
        expected_count = int(imm[f"joint_covered_{probability}"].sum())
        assert getattr(gui, f"joint_coverage_{probability}_count") == expected_count
        assert getattr(imm_gui, f"joint_coverage_{probability}_count") == expected_count
    assert (
        bayesian.prediction_table.evaluation_definition.iloc[0]
        == metrics.EVALUATION_DEFINITION
    )
    assert imm.evaluation_definition.iloc[0] == metrics.EVALUATION_DEFINITION
    assert bayesian.prediction_table.nlpd_definition.iloc[0] == "gaussian_latent_2d_v1"
    assert imm.nlpd_definition.iloc[0] == "gaussian_latent_2d_v1"


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

    expected = np.log(2 * np.pi * 1e-12)
    assert evaluation.total_negative_log_predictive_density == pytest.approx(expected)
    assert evaluation.mean_negative_log_predictive_density == pytest.approx(expected)
    expected_energy = np.sqrt(np.pi / 2.0) * (1 - 1 / np.sqrt(2)) * 1e-6
    assert evaluation.energy_score_m == pytest.approx(expected_energy, abs=2e-9)
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
