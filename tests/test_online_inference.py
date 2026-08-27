"""Tests for persistent online RBPF evaluation state."""

import numpy as np
import pandas as pd
import pytest

import bayestraj.models.bayesian_ctrv as ctrv_model
import bayestraj.models.bayesian_position_model as position_model
import bayestraj.validation.bayesian_ctrv_workflow as ctrv_workflow
import bayestraj.validation.bayesian_position_model_workflow as position_workflow
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
        ctrv_model.SequentialBayesianCTRVFilter,
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


def test_position_online_evaluation_initializes_once_and_updates_only_new_positions(
    monkeypatch,
):
    initialization_count = 0

    def initialize(x_observed, y_observed, **kwargs):
        nonlocal initialization_count
        initialization_count += 1
        return _FakeOnlineFilter(x_observed)

    monkeypatch.setattr(
        position_model.SequentialBayesianPositionFilter,
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
        online_filter = position_workflow._advance_online_filter(
            online_filter,
            specification=specification,
            noisy_route_x=values,
            noisy_route_y=-values,
            priors=object(),
            rbpf_config=object(),
            inference_seed=42,
        )

    assert initialization_count == 1
    assert online_filter.processed_values == values[:11].tolist()
    assert len(set(online_filter.processed_values)) == 11


@pytest.mark.parametrize(
    ("fit", "variable_names", "expected_shape"),
    (
        (
            ctrv_model.SequentialBayesianCTRVFilter.initialize(
                np.arange(5, dtype=float),
                np.arange(5, dtype=float),
                np.zeros(5),
                priors=ctrv_model.BayesianCTRVPriors(),
                config=ctrv_model.SequentialCTRVFilterConfig(
                    particle_count=32,
                    posterior_draw_count=8,
                ),
                seed=42,
            ).forecast(np.arange(5, 8, dtype=float), seed=43),
            (
                "x_prediction",
                "y_prediction",
                "x_observation_prediction",
                "y_observation_prediction",
            ),
            (8, 3),
        ),
        (
            position_model.SequentialBayesianPositionFilter.initialize(
                np.arange(5, dtype=float),
                np.zeros(5),
                priors=position_model.BayesianPositionModelPriors(),
                config=position_model.SequentialPositionFilterConfig(
                    particle_count=32,
                    posterior_draw_count=8,
                ),
                seed=42,
            ).forecast(3, seed=43),
            (
                "x_model_prediction",
                "y_model_prediction",
                "x_observation_prediction",
                "y_observation_prediction",
            ),
            (8, 3),
        ),
    ),
)
def test_rbpf_forecasts_expose_shared_posterior_variable_interface(
    fit,
    variable_names,
    expected_shape,
):
    for variable_name in variable_names:
        samples = reporting.posterior_variable_samples(fit, variable_name)
        assert samples.shape == expected_shape
        assert np.all(np.isfinite(samples))


def test_ctrv_rbpf_exposes_clear_forecast_origin_state_names():
    fit = ctrv_model.SequentialBayesianCTRVFilter.initialize(
        np.arange(5, dtype=float),
        np.arange(5, dtype=float),
        np.zeros(5),
        priors=ctrv_model.BayesianCTRVPriors(),
        config=ctrv_model.SequentialCTRVFilterConfig(
            particle_count=32,
            posterior_draw_count=8,
        ),
        seed=42,
    ).forecast(np.arange(5, 8, dtype=float), seed=43)

    for variable_name in ctrv_model.PARAMETER_NAMES:
        samples = reporting.posterior_variable_samples(fit, variable_name)
        assert samples.shape == (8,)
        assert np.all(np.isfinite(samples))


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
