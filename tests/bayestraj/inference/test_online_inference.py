"""Tests for persistent online particle-filter evaluation state."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bayestraj.inference.configuration as inference
import bayestraj.inference.ctrv_sequential_vi as sequential_vi_model
import bayestraj.inference.ctrv_smc as smc_model
import bayestraj.models.bayesian_ctrv as ctrv_model
import bayestraj.validation.bayesian_ctrv_workflow as ctrv_workflow
import bayestraj.validation.plotting as plotting
import bayestraj.validation.rolling as rolling

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_sequential_vi_uses_shared_bayesian_ctrv_evaluation_entrypoint():
    model_evaluation = PROJECT_ROOT / "experiments" / "model_evaluation"

    assert (model_evaluation / "bayesian_ctrv.py").is_file()
    assert not (model_evaluation / "compare_bayesian_ctrv_sequential_vi.py").exists()
    assert "sequential_vi" in inference.CTRV_ROLLING_INFERENCE_METHODS


class _FakeOnlineFilter:
    def __init__(self, initial_values):
        self.processed_values = list(initial_values)
        self.processed_observation_count = len(self.processed_values)

    def update_many(self, *value_arrays):
        new_values = np.asarray(value_arrays[-2])
        self.processed_values.extend(new_values.tolist())
        self.processed_observation_count += len(new_values)


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
        smc_model.SequentialMonteCarloCTRVFilter,
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
            inference_seed=42,
        )

    assert initialization_count == 1
    assert online_filter.processed_values == values[:11].tolist()
    assert len(set(online_filter.processed_values)) == 11


def test_ctrv_sequential_vi_evaluation_initializes_once_and_never_replays(
    monkeypatch,
):
    initialization_count = 0

    def initialize(time_seconds, x_observed, y_observed, **kwargs):
        nonlocal initialization_count
        initialization_count += 1
        assert isinstance(kwargs["config"], sequential_vi_model.SequentialVIConfig)
        return _FakeOnlineFilter(x_observed)

    monkeypatch.setattr(
        sequential_vi_model.SequentialCTRVVI,
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
            priors=ctrv_model.BayesianCTRVPriors(),
            smc_config=object(),
            sequential_vi_config=sequential_vi_model.SequentialVIConfig(
                n_bootstrap=5,
                draws=8,
            ),
            inference_method="sequential_vi",
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


def test_shared_rolling_summary_accepts_online_predictions():
    predictions = pd.DataFrame(
        {
            "window_index": [0, 0],
            "horizon_step": [1, 2],
            "horizon_seconds": [1.0, 2.0],
            "position_error_m": [1.0, 2.0],
            "prediction_radius_m": [3.0, 4.0],
            "radial_covered": [True, False],
            "mean_marginal_interval_width_m": [5.0, 6.0],
            "negative_log_predictive_density": [2.0, 3.0],
            "energy_score_m": [1.0, 2.0],
            "joint_covered_50": [True, False],
            "joint_covered_90": [True, True],
            "inference_mode": ["online", "online"],
            "inference_method": ["smc", "smc"],
            "converged": [None, None],
            "mcmc_diagnostics_ok": [None, None],
            "window_runtime_seconds": [0.1, 0.1],
        }
    )

    summary = rolling.summarize_rolling_predictions(predictions)

    assert summary.inference_mode == "online"
    assert summary.inference_method == "smc"
    assert summary.vi_convergence_rate is None
    assert summary.mcmc_diagnostics_pass_rate is None
    assert summary.total_negative_log_predictive_density == pytest.approx(5.0)
    assert summary.mean_negative_log_predictive_density == pytest.approx(2.5)
    assert summary.per_horizon_table[
        "mean_negative_log_predictive_density"
    ].tolist() == [2.0, 3.0]
    assert summary.mean_energy_score_m == pytest.approx(1.5)
    assert summary.joint_coverage_50 == pytest.approx(0.5)
    assert summary.joint_coverage_90 == pytest.approx(1.0)


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
            "negative_log_predictive_density": [2.0, 3.0],
            "energy_score_m": [1.0, 2.0],
            "joint_covered_50": [True, False],
            "joint_covered_90": [True, True],
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
        inference_method="smc",
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
        total_negative_log_predictive_density=10.0,
        mean_negative_log_predictive_density=2.5,
        mean_energy_score_m=15.0,
        joint_coverage_50=0.5,
        joint_coverage_90=0.95,
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
    assert output_lines[fde_line + 1].startswith("Mean Energy Score")
    assert output_lines[fde_line + 2].startswith(
        "Total negative log predictive density"
    )
    assert output_lines[fde_line + 4].startswith("Joint 2D 50% coverage")
    assert output_lines[fde_line + 5].startswith("Joint 2D 90% coverage")
