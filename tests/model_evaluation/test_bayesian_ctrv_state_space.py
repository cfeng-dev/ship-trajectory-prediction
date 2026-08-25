"""Tests for the Bayesian CTRV state-space rolling evaluation."""

import numpy as np
import pandas as pd
import pytest

import experiments.model_evaluation.bayesian_ctrv_state_space as experiment
import ship_trajectory_prediction.validation.bayesian_ctrv_state_space_workflow as workflow
import ship_trajectory_prediction.validation.rolling as rolling
from ship_trajectory_prediction.models.bayesian_ctrv_state_space import (
    BayesianCTRVPriors,
)
from ship_trajectory_prediction.models.deterministic_ctrv import CTRVState
from ship_trajectory_prediction.observations import prepare_trajectory_window
from ship_trajectory_prediction.simulation.synthetic_ctrv import (
    SyntheticCTRVNoise,
    simulate_synthetic_ctrv_data,
)


def _create_route():
    """Return one deterministic route for overlapping rolling windows."""
    return simulate_synthetic_ctrv_data(
        count=10,
        dt_seconds=5.0,
        initial_state=CTRVState(
            x=0.0,
            y=0.0,
            speed=3.0,
            heading=0.2,
            turn_rate=0.01,
        ),
        noise=SyntheticCTRVNoise(
            sigma_position_gps=0.0,
            sigma_speed_gps=0.0,
            sigma_position_process=0.0,
            sigma_speed_process=0.0,
            sigma_turn_rate_process=0.0,
        ),
        seed=7,
    )


def test_rolling_experiment_adds_reproducible_five_meter_position_noise():
    """The rolling default should match the single-fit 5 m noise scenario."""
    assert experiment.EXPERIMENT.additional_position_noise_std_m == 5.0
    assert experiment.EXPERIMENT.position_noise_seed == 2026

    first_x, first_y = workflow._simulate_route_position_noise(
        10,
        additional_noise_std_m=(experiment.EXPERIMENT.additional_position_noise_std_m),
        seed=experiment.EXPERIMENT.position_noise_seed,
    )
    second_x, second_y = workflow._simulate_route_position_noise(
        10,
        additional_noise_std_m=(experiment.EXPERIMENT.additional_position_noise_std_m),
        seed=experiment.EXPERIMENT.position_noise_seed,
    )

    np.testing.assert_array_equal(first_x, second_x)
    np.testing.assert_array_equal(first_y, second_y)
    assert np.any(first_x != 0.0)
    assert np.any(first_y != 0.0)


def test_summary_aligns_all_metric_separators(capsys):
    """Every aggregate metric should use the same separator column."""
    per_horizon_table = pd.DataFrame(
        {
            "horizon_step": [1],
            "forecast_count": [114],
            "mean_horizon_seconds": [10.0],
            "ade_m": [10.213],
            "median_error_m": [6.797],
            "radial_coverage": [0.86],
            "mean_prediction_radius_m": [18.853],
            "mean_marginal_interval_width_m": [28.861],
        }
    )
    summary = rolling.RollingPositionSummary(
        inference_method="vi",
        window_count=114,
        forecast_count=341,
        ade_m=20.61,
        fde_m=30.20,
        mean_window_runtime_seconds=6.12,
        median_window_runtime_seconds=5.87,
        total_computation_time_seconds=697.8,
        radial_coverage=0.783,
        mean_prediction_radius_m=27.36,
        mean_marginal_interval_width_m=42.36,
        vi_convergence_rate=0.026,
        mcmc_diagnostics_pass_rate=None,
        per_horizon_table=per_horizon_table,
    )

    workflow._print_summary(summary, credible_interval=0.9)

    output = capsys.readouterr().out
    metric_lines = [line for line in output.splitlines() if " : " in line]
    assert len(metric_lines) == 12
    assert len({line.index(":") for line in metric_lines}) == 1
    assert "Mean window runtime" in output
    assert "Total computation time" in output
    assert "11.63 min" in output


def test_per_horizon_summary_uses_compact_terminal_columns():
    """Printed horizon metrics should fit into a narrow terminal."""
    table = pd.DataFrame(
        {
            "horizon_step": [1, 2, 3],
            "forecast_count": [114, 114, 113],
            "mean_horizon_seconds": [10.0, 20.0, 30.0],
            "ade_m": [10.213, 18.980, 29.031],
            "median_error_m": [6.797, 11.342, 17.732],
            "radial_coverage": [0.860, 0.798, 0.779],
            "mean_prediction_radius_m": [18.853, 27.939, 37.367],
            "mean_marginal_interval_width_m": [28.861, 42.985, 57.911],
        }
    )

    output = rolling.format_per_horizon_table(table)

    lines = output.splitlines()
    assert "horizon_step" not in output
    assert "Horizon[s]" in lines[0]
    assert "Coverage" in lines[0]
    assert "86.0%" in lines[1]
    assert max(map(len, lines)) <= 80


def test_main_calls_fully_bayesian_workflow_without_model_variant(monkeypatch):
    """The Bayesian entry point should call its dedicated rolling workflow."""
    captured = {}
    expected_result = object()
    monkeypatch.setattr(
        experiment.workflow,
        "_run_bayesian_ctrv_evaluation",
        lambda **kwargs: captured.update(kwargs) or expected_result,
    )

    result = experiment.main([])

    assert result is expected_result
    assert not hasattr(experiment, "MODEL_VARIANT")
    assert captured["model_name"] == "bayesian"
    assert captured["model_label"] == "Fully Bayesian CTRV"
    assert captured["fit_model"] is workflow.bayesian_model.fit_bayesian_ctrv_model
    assert captured["data_file"] == experiment.DATA_FILE
    assert captured["experiment"] is not experiment.EXPERIMENT
    assert captured["experiment"] == experiment.EXPERIMENT
    assert captured["priors"] is not experiment.PRIORS
    assert captured["vi_config"] is experiment.VI_CONFIG
    assert captured["mcmc_config"] is experiment.MCMC_CONFIG
    assert captured["experiment"].inference_method == (
        experiment.EXPERIMENT.inference_method
    )
    assert captured["vi_algorithm"] == experiment.VI_CONFIG["algorithm"]


def test_cli_options_are_applied_inside_the_src_workflow(monkeypatch):
    """Rolling CLI overrides should become independent workflow configs."""
    captured = {}
    monkeypatch.setattr(
        experiment.workflow,
        "_run_bayesian_ctrv_evaluation",
        lambda **kwargs: captured.update(kwargs),
    )

    experiment.main(
        [
            "--window-mode",
            "expanding",
            "--observations",
            "17",
            "--predictions",
            "4",
            "--stride",
            "2",
            "--inference",
            "mcmc",
            "--vi-algorithm",
            "fullrank",
            "--turn-rate-prior-scale",
            "0.003",
            "--seed",
            "99",
            "--position-noise-std-m",
            "1.25",
            "--position-noise-seed",
            "7",
            "--require-converged",
            "--max-windows",
            "5",
            "--plot-each-window",
        ]
    )

    configured = captured["experiment"]
    assert configured.window_mode == "expanding"
    assert configured.observation_count == 17
    assert configured.prediction_count == 4
    assert configured.stride == 2
    assert configured.inference_method == "mcmc"
    assert configured.inference_seed == 99
    assert configured.additional_position_noise_std_m == 1.25
    assert configured.position_noise_seed == 7
    assert captured["priors"].turn_rate_state_prior_scale == pytest.approx(0.003)
    assert captured["vi_algorithm"] == "fullrank"
    assert captured["require_converged"] is True
    assert captured["max_windows"] == 5
    assert captured["plot_each_window"] is True


def test_overlapping_windows_reuse_the_same_route_position_noise():
    """One physical observation must retain its perturbation across windows."""
    route = _create_route()
    route_noise_x, route_noise_y = workflow._simulate_route_position_noise(
        len(route),
        additional_noise_std_m=2.0,
        seed=2026,
    )
    first_window = prepare_trajectory_window(
        route,
        observation_count=5,
        prediction_count=2,
        start_index=0,
    )
    second_window = prepare_trajectory_window(
        route,
        observation_count=5,
        prediction_count=2,
        start_index=2,
    )

    first_observations = workflow._build_window_position_observations(
        first_window,
        route_start_index=0,
        route_noise_x=route_noise_x,
        route_noise_y=route_noise_y,
        additional_noise_std_m=2.0,
        noise_seed=2026,
    )
    second_observations = workflow._build_window_position_observations(
        second_window,
        route_start_index=2,
        route_noise_x=route_noise_x,
        route_noise_y=route_noise_y,
        additional_noise_std_m=2.0,
        noise_seed=2026,
    )

    first_x_noise = (
        first_observations.x_meters - first_window.x_meters[first_window.observed_slice]
    )
    first_y_noise = (
        first_observations.y_meters - first_window.y_meters[first_window.observed_slice]
    )
    second_x_noise = (
        second_observations.x_meters
        - second_window.x_meters[second_window.observed_slice]
    )
    second_y_noise = (
        second_observations.y_meters
        - second_window.y_meters[second_window.observed_slice]
    )

    np.testing.assert_allclose(first_x_noise[2:], second_x_noise[:3])
    np.testing.assert_allclose(first_y_noise[2:], second_y_noise[:3])
    np.testing.assert_allclose(first_x_noise, route_noise_x[:5])
    np.testing.assert_allclose(second_y_noise, route_noise_y[2:7])
    assert first_observations.additional_noise_std_m == 2.0
    assert first_observations.observation_noise_std_m == 2.0
    assert first_observations.noise_seed == 2026


def test_zero_position_noise_keeps_the_recorded_route_unchanged():
    """Disabling the option should preserve the original observations exactly."""
    route = _create_route()
    window = prepare_trajectory_window(
        route,
        observation_count=5,
        prediction_count=2,
    )
    route_noise_x, route_noise_y = workflow._simulate_route_position_noise(
        len(route),
        additional_noise_std_m=0.0,
        seed=2026,
    )
    observations = workflow._build_window_position_observations(
        window,
        route_start_index=0,
        route_noise_x=route_noise_x,
        route_noise_y=route_noise_y,
        additional_noise_std_m=0.0,
        noise_seed=2026,
    )

    np.testing.assert_array_equal(
        observations.x_meters,
        window.x_meters[window.observed_slice],
    )
    np.testing.assert_array_equal(
        observations.y_meters,
        window.y_meters[window.observed_slice],
    )
    assert observations.observation_noise_std_m == 2.0


class FakeFit:
    """Provide CmdStanVB-like scalar noise draws for retry tests."""

    variational_sample = object()

    def __init__(self, *, sigma_speed_process):
        self.variables = {
            "sigma_position_process": np.array([0.5]),
            "sigma_speed_process": np.asarray(sigma_speed_process, dtype=float),
            "sigma_turn_rate_process": np.array([0.001]),
        }

    def stan_variable(self, name, mean=False):
        """Return one stored fake posterior variable."""
        del mean
        return self.variables[name]


def test_window_diagnostics_report_deterministic_forecast_turn_rate():
    """Rolling diagnostics should report the rate actually used by the forecast."""
    fit = FakeFit(sigma_speed_process=[0.05])
    fit.variables.update(
        {
            "turn_rate_forecast_origin": np.array([0.02, 0.02]),
            "heading_state": np.array([[0.3, 0.4], [0.3, 0.4]]),
            "heading_state_prediction": np.array(
                [[0.5, 0.7], [0.5, 0.7]],
            ),
        }
    )

    diagnostics = workflow._posterior_window_diagnostics(fit)

    assert diagnostics["forecast_origin_turn_rate_rad_s"] == pytest.approx(0.02)
    assert diagnostics["forecast_heading_change_rad"] == pytest.approx(0.3)
    assert "posterior_origin_turn_rate_median_rad_s" not in diagnostics


def test_numerically_exploded_vi_fit_is_retried_with_next_seed(
    monkeypatch,
    capsys,
):
    """A catastrophic VI tail should be discarded instead of breaking the plot."""
    unstable = FakeFit(sigma_speed_process=[0.05, 1e100])
    stable = FakeFit(sigma_speed_process=[0.04, 0.06])
    fits = iter((unstable, stable))
    used_seeds = []

    def fake_fit(*args, seed, **kwargs):
        del args, kwargs
        used_seeds.append(seed)
        return next(fits)

    monkeypatch.setattr(
        workflow.bayesian_model,
        "fit_bayesian_ctrv_model",
        fake_fit,
    )

    fit, seed = workflow._fit_rolling_window(
        object(),
        priors=BayesianCTRVPriors(),
        position_observations=object(),
        inference_method="vi",
        inference_config={},
        initial_seed=62,
    )

    assert fit is stable
    assert seed == 63
    assert used_seeds == [62, 63]
    assert "retrying with seed=63" in capsys.readouterr().out


def test_cmdstan_vi_execution_failure_is_retried_with_next_seed(capsys):
    """A seed-specific CmdStan VI failure should not abort rolling evaluation."""
    stable = FakeFit(sigma_speed_process=[0.04, 0.06])
    used_seeds = []

    def fake_fit(*args, seed, **kwargs):
        del args, kwargs
        used_seeds.append(seed)
        if seed == 62:
            raise RuntimeError("Error during variational inference:")
        return stable

    fit, seed = workflow._fit_rolling_window(
        object(),
        priors=BayesianCTRVPriors(),
        position_observations=object(),
        inference_method="vi",
        inference_config={},
        initial_seed=62,
        fit_model=fake_fit,
    )

    assert fit is stable
    assert seed == 63
    assert used_seeds == [62, 63]
    output = capsys.readouterr().out
    assert "CmdStan VI fit failed" in output
    assert "retrying with seed=63" in output


def test_repeated_cmdstan_vi_execution_failure_fails_clearly():
    """Repeated CmdStan failures should retain a bounded retry policy."""
    used_seeds = []

    def fake_fit(*args, seed, **kwargs):
        del args, kwargs
        used_seeds.append(seed)
        raise RuntimeError("Error during variational inference:")

    with pytest.raises(RuntimeError, match="after 3 attempts; last seed=64"):
        workflow._fit_rolling_window(
            object(),
            priors=BayesianCTRVPriors(),
            position_observations=object(),
            inference_method="vi",
            inference_config={},
            initial_seed=62,
            fit_model=fake_fit,
        )

    assert used_seeds == [62, 63, 64]


def test_unrelated_vi_runtime_error_is_not_retried():
    """Programming and configuration errors must remain immediately visible."""
    used_seeds = []

    def fake_fit(*args, seed, **kwargs):
        del args, kwargs
        used_seeds.append(seed)
        raise RuntimeError("Unexpected configuration error")

    with pytest.raises(RuntimeError, match="Unexpected configuration error"):
        workflow._fit_rolling_window(
            object(),
            priors=BayesianCTRVPriors(),
            position_observations=object(),
            inference_method="vi",
            inference_config={},
            initial_seed=62,
            fit_model=fake_fit,
        )

    assert used_seeds == [62]


def test_repeated_numerical_vi_instability_fails_clearly(monkeypatch):
    """Rolling evaluation must not plot an approximation that remains invalid."""
    unstable = FakeFit(sigma_speed_process=[1e100])
    monkeypatch.setattr(
        workflow.bayesian_model,
        "fit_bayesian_ctrv_model",
        lambda *args, **kwargs: unstable,
    )

    with pytest.raises(RuntimeError, match="remained numerically unstable"):
        workflow._fit_rolling_window(
            object(),
            priors=BayesianCTRVPriors(),
            position_observations=object(),
            inference_method="vi",
            inference_config={},
            initial_seed=62,
        )


def test_mcmc_fit_is_never_subjected_to_vi_retry(monkeypatch):
    """Reference MCMC output should bypass the VI-specific stability policy."""
    fit = object()
    used_seeds = []

    def fake_fit(*args, seed, **kwargs):
        del args, kwargs
        used_seeds.append(seed)
        return fit

    monkeypatch.setattr(
        workflow.bayesian_model,
        "fit_bayesian_ctrv_model",
        fake_fit,
    )

    result, seed = workflow._fit_rolling_window(
        object(),
        priors=BayesianCTRVPriors(),
        position_observations=object(),
        inference_method="mcmc",
        inference_config={},
        initial_seed=62,
    )

    assert result is fit
    assert seed == 62
    assert used_seeds == [62]
