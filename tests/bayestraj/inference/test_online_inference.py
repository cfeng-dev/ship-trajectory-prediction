"""Tests for persistent online particle-filter evaluation state."""

import numpy as np
import pandas as pd
import pytest

import bayestraj.forecasting.bayesian_ctrv as ctrv_forecasting
import bayestraj.forecasting.bayesian_ctrv_workflow as single_ctrv_workflow
import bayestraj.inference.configuration as inference
import bayestraj.inference.ctrv_rbpf as rbpf_model
import bayestraj.inference.ctrv_smc as smc_model
import bayestraj.inference.particle_utils as particle_utils
import bayestraj.models.bayesian_ctrv as ctrv_model
import bayestraj.validation.bayesian_ctrv_workflow as ctrv_workflow
import bayestraj.validation.plotting as plotting
import bayestraj.validation.reporting as reporting
import bayestraj.validation.rolling as rolling


class _FakeOnlineFilter:
    def __init__(self, initial_values):
        self.processed_values = list(initial_values)
        self.processed_observation_count = len(self.processed_values)

    def update_many(self, *value_arrays):
        new_values = np.asarray(value_arrays[-2])
        self.processed_values.extend(new_values.tolist())
        self.processed_observation_count += len(new_values)


@pytest.mark.parametrize("inference_method", ("rbpf", "smc"))
def test_single_window_ctrv_prediction_runs_selected_particle_filter(
    monkeypatch,
    capsys,
    inference_method,
):
    row_count = 8
    trajectory = pd.DataFrame(
        {
            "time": pd.date_range(
                "2026-01-01",
                periods=row_count,
                freq="10s",
                tz="UTC",
            ),
            "run_id": np.full(row_count, 102),
            "gps_latitude": 53.0 + np.arange(row_count) * 0.00001,
            "gps_longitude": 10.0 + np.arange(row_count) * 0.00002,
            "gps_speed": np.full(row_count, 3.6),
        }
    )
    monkeypatch.setattr(
        single_ctrv_workflow.observations_io,
        "read_ship_data",
        lambda *args, **kwargs: trajectory,
    )
    monkeypatch.setattr(
        single_ctrv_workflow.prediction_plotting,
        "plot_prediction",
        lambda *args, **kwargs: None,
    )
    experiment = ctrv_forecasting.ExperimentConfig(
        run_id=102,
        start_index=0,
        observation_count=5,
        prediction_count=3,
        position_noise_std_m=0.0,
        position_noise_seed=2026,
        inference_method=inference_method,
        inference_seed=42,
    )

    result = single_ctrv_workflow.run_bayesian_ctrv_prediction(
        data_file="unused.csv",
        experiment=experiment,
        priors=ctrv_model.BayesianCTRVPriors(),
        vi_config=inference.create_default_vi_config(),
        mcmc_config=inference.create_default_mcmc_config(),
        rbpf_config=rbpf_model.SequentialCTRVFilterConfig(
            particle_count=128,
            posterior_draw_count=16,
        ),
        smc_config=smc_model.SequentialMonteCarloCTRVConfig(
            particle_count=128,
            posterior_draw_count=16,
        ),
        fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
        credible_interval=0.9,
        inference_method=inference_method,
        vi_algorithm="meanfield",
        seed=42,
        position_noise_std_m=0.0,
        position_noise_seed=2026,
        require_converged=False,
        plot_coordinate_mode="m",
        show_time_labels=False,
    )

    fit = result["fit"]
    assert isinstance(fit, particle_utils.SequentialCTRVFit)
    assert fit.stan_variable("x_prediction").shape == (16, 3)
    assert fit.stan_variable("y_prediction").shape == (16, 3)
    assert result["converged"] is None
    assert f"{inference_method.upper()} diagnostics:" in capsys.readouterr().out


def test_online_rolling_plot_label_is_method_neutral():
    assert plotting._bayesian_evaluation_mode_label("online", None) == (
        "Online-Inferenz"
    )


def test_ctrv_online_evaluation_initializes_once_and_updates_only_new_positions(
    monkeypatch,
):
    initialization_count = 0

    def initialize(time_seconds, x_observed, y_observed, **kwargs):
        nonlocal initialization_count
        initialization_count += 1
        assert np.array_equal(time_seconds, x_observed)
        return _FakeOnlineFilter(x_observed)

    monkeypatch.setattr(
        rbpf_model.SequentialBayesianCTRVFilter,
        "initialize",
        staticmethod(initialize),
    )
    values = np.arange(12, dtype=float)
    online_filter = None
    specifications = rolling.build_online_forecast_specs(
        len(values),
        initial_observation_count=5,
        prediction_count=3,
        stride=2,
    )
    for specification in specifications:
        online_filter = ctrv_workflow._advance_online_filter(
            online_filter,
            specification=specification,
            route_time_seconds=values,
            noisy_route_x=values,
            noisy_route_y=-values,
            priors=object(),
            rbpf_config=object(),
            inference_seed=42,
        )

    assert initialization_count == 1
    assert online_filter.processed_values == values[:11].tolist()
    assert len(set(online_filter.processed_values)) == 11


def test_ctrv_online_evaluation_dispatches_to_full_state_smc():
    values = np.arange(8, dtype=float)
    specification = rolling.build_online_forecast_specs(
        len(values),
        initial_observation_count=5,
        prediction_count=3,
        stride=2,
    )[0]

    online_filter = ctrv_workflow._advance_online_filter(
        None,
        specification=specification,
        route_time_seconds=values,
        noisy_route_x=values,
        noisy_route_y=np.zeros_like(values),
        priors=ctrv_model.BayesianCTRVPriors(),
        rbpf_config=object(),
        smc_config=smc_model.SequentialMonteCarloCTRVConfig(
            particle_count=64,
            posterior_draw_count=8,
        ),
        inference_method="smc",
        inference_seed=42,
    )

    assert isinstance(online_filter, smc_model.SequentialMonteCarloCTRVFilter)
    assert (
        online_filter.processed_observation_count == specification.forecast_start_index
    )


def test_ctrv_rbpf_forecast_exposes_shared_posterior_variable_interface():
    fit = rbpf_model.SequentialBayesianCTRVFilter.initialize(
        np.arange(5, dtype=float),
        np.arange(5, dtype=float),
        np.zeros(5),
        priors=ctrv_model.BayesianCTRVPriors(),
        config=rbpf_model.SequentialCTRVFilterConfig(
            particle_count=32,
            posterior_draw_count=8,
        ),
        seed=42,
    ).forecast(np.arange(5, 8, dtype=float), seed=43)

    variable_names = (
        "x_prediction",
        "y_prediction",
        "x_observation_prediction",
        "y_observation_prediction",
    )
    for variable_name in variable_names:
        samples = reporting.posterior_variable_samples(fit, variable_name)
        assert samples.shape == (8, 3)
        assert np.all(np.isfinite(samples))


def test_ctrv_rbpf_exposes_clear_forecast_origin_state_names():
    fit = rbpf_model.SequentialBayesianCTRVFilter.initialize(
        np.arange(5, dtype=float),
        np.arange(5, dtype=float),
        np.zeros(5),
        priors=ctrv_model.BayesianCTRVPriors(),
        config=rbpf_model.SequentialCTRVFilterConfig(
            particle_count=32,
            posterior_draw_count=8,
        ),
        seed=42,
    ).forecast(np.arange(5, 8, dtype=float), seed=43)

    for variable_name in ctrv_model.PARAMETER_NAMES:
        samples = reporting.posterior_variable_samples(fit, variable_name)
        assert samples.shape == (8,)
        assert np.all(np.isfinite(samples))


def test_ctrv_rbpf_samples_current_posterior_without_advancing_filter():
    online_filter = rbpf_model.SequentialBayesianCTRVFilter.initialize(
        np.arange(5, dtype=float),
        np.arange(5, dtype=float),
        np.zeros(5),
        priors=ctrv_model.BayesianCTRVPriors(),
        config=rbpf_model.SequentialCTRVFilterConfig(
            particle_count=32,
            posterior_draw_count=8,
        ),
        seed=42,
    )
    state_before = online_filter.state_means.copy()
    observation_count_before = online_filter.processed_observation_count

    fit = online_filter.sample_current_posterior(seed=43)

    for variable_name in ctrv_model.PARAMETER_NAMES:
        samples = reporting.posterior_variable_samples(fit, variable_name)
        assert samples.shape == (8,)
        assert np.all(np.isfinite(samples))
    assert online_filter.processed_observation_count == observation_count_before
    assert online_filter.state_means == pytest.approx(state_before)


def test_shared_rolling_summary_accepts_online_rbpf_predictions():
    predictions = pd.DataFrame(
        {
            "window_index": [0, 0],
            "horizon_step": [1, 2],
            "horizon_seconds": [1.0, 2.0],
            "position_error_m": [1.0, 2.0],
            "prediction_radius_m": [3.0, 4.0],
            "radial_covered": [True, False],
            "mean_marginal_interval_width_m": [5.0, 6.0],
            "inference_mode": ["online", "online"],
            "inference_method": ["rbpf", "rbpf"],
            "converged": [None, None],
            "mcmc_diagnostics_ok": [None, None],
            "window_runtime_seconds": [0.1, 0.1],
        }
    )

    summary = rolling.summarize_rolling_predictions(predictions)

    assert summary.inference_mode == "online"
    assert summary.inference_method == "rbpf"
    assert summary.vi_convergence_rate is None
    assert summary.mcmc_diagnostics_pass_rate is None


def test_shared_rolling_summary_accepts_online_smc_for_ctrv():
    predictions = pd.DataFrame(
        {
            "window_index": [0, 0],
            "horizon_step": [1, 2],
            "horizon_seconds": [1.0, 2.0],
            "position_error_m": [1.0, 2.0],
            "prediction_radius_m": [3.0, 4.0],
            "radial_covered": [True, False],
            "mean_marginal_interval_width_m": [5.0, 6.0],
            "inference_mode": ["online", "online"],
            "inference_method": ["smc", "smc"],
            "converged": [None, None],
            "mcmc_diagnostics_ok": [None, None],
            "window_runtime_seconds": [0.1, 0.1],
        }
    )

    summary = rolling.summarize_rolling_predictions(
        predictions,
        online_inference_methods=inference.CTRV_ONLINE_INFERENCE_METHODS,
    )

    assert summary.inference_mode == "online"
    assert summary.inference_method == "smc"
    assert summary.vi_convergence_rate is None
    assert summary.mcmc_diagnostics_pass_rate is None


def test_ctrv_summary_prints_joint_coverage_immediately_after_fde(capsys):
    summary = rolling.RollingPositionSummary(
        inference_mode="online",
        inference_method="rbpf",
        window_count=119,
        forecast_count=357,
        ade_m=20.60,
        fde_m=30.79,
        mean_window_runtime_seconds=0.079,
        median_window_runtime_seconds=0.077,
        total_computation_time_seconds=9.350,
        radial_coverage=0.955,
        mean_prediction_radius_m=0.0,
        mean_marginal_interval_width_m=0.0,
        vi_convergence_rate=None,
        mcmc_diagnostics_pass_rate=None,
        per_horizon_table=pd.DataFrame(),
    )

    ctrv_workflow._print_summary(summary, credible_interval=0.9)

    output_lines = capsys.readouterr().out.splitlines()
    fde_line = next(
        index
        for index, line in enumerate(output_lines)
        if line.startswith("Mean maximum-horizon FDE")
    )
    assert output_lines[fde_line + 1].startswith("Joint 2D 90% coverage")
    assert output_lines[fde_line + 2].startswith("Mean window runtime")
