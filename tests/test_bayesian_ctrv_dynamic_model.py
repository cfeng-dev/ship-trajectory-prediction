"""Focused tests for the shared dynamic Bayesian CTRV model."""

import inspect
import os

import numpy as np
import pandas as pd
import pytest

import bayestraj.models.bayesian_ctrv as ctrv_model
import bayestraj.observations.window as observation_window
import bayestraj.validation.reporting as reporting

RUN_CMDSTAN_INTEGRATION = os.environ.get("RUN_CMDSTAN_INTEGRATION") == "1"
PREDICTION_VARIABLE_NAMES = (
    "x_prediction",
    "y_prediction",
    "x_observation_prediction",
    "y_observation_prediction",
)


def _dynamic_ctrv_window():
    time_seconds = np.arange(8, dtype=float) * 10.0
    speed = 4.0
    heading_initial = 0.3
    turn_rate = 0.01
    x_meters = (
        speed
        / turn_rate
        * (np.sin(heading_initial + turn_rate * time_seconds) - np.sin(heading_initial))
    )
    y_meters = (
        speed
        / turn_rate
        * (
            -np.cos(heading_initial + turn_rate * time_seconds)
            + np.cos(heading_initial)
        )
    )
    return observation_window.TrajectoryWindowData(
        timestamps=pd.date_range(
            "2026-01-01",
            periods=len(time_seconds),
            freq="10s",
            tz="UTC",
        ),
        time_seconds=time_seconds,
        x_meters=x_meters,
        y_meters=y_meters,
        reference_longitude=8.0,
        reference_latitude=47.0,
        gps_speed_mps=np.full(len(time_seconds), np.nan),
        observation_count=5,
    )


def test_batch_stan_data_contains_shared_dynamic_process_inputs():
    priors = ctrv_model.BayesianCTRVPriors()
    stan_data = ctrv_model.build_stan_data(_dynamic_ctrv_window(), priors=priors)

    assert stan_data["sigma_speed_process_prior_rate"] == pytest.approx(
        priors.sigma_speed_process_prior_rate
    )
    assert stan_data["sigma_turn_rate_process_prior_rate"] == pytest.approx(
        priors.sigma_turn_rate_process_prior_rate
    )
    assert stan_data["process_reference_interval_seconds"] == pytest.approx(
        ctrv_model.PROCESS_REFERENCE_INTERVAL_SECONDS
    )
    assert "gps_speed" not in stan_data


def test_batch_and_online_code_use_one_process_reference_interval():
    rbpf_config = ctrv_model.SequentialCTRVFilterConfig()
    update_source = inspect.getsource(ctrv_model.SequentialBayesianCTRVFilter.update)
    forecast_source = inspect.getsource(
        ctrv_model.SequentialBayesianCTRVFilter.forecast
    )

    assert not hasattr(rbpf_config, "process_reference_interval_seconds")
    assert "PROCESS_REFERENCE_INTERVAL_SECONDS" in update_source
    assert "PROCESS_REFERENCE_INTERVAL_SECONDS" in forecast_source


def test_stan_model_contains_dynamic_speed_and_turn_rate_states():
    stan_source = ctrv_model.STAN_FILE.read_text(encoding="utf-8")

    assert "speed_state" in stan_source
    assert "turn_rate_state" in stan_source
    assert "sigma_speed_process" in stan_source
    assert "sigma_turn_rate_process" in stan_source
    assert "process_reference_interval_seconds" in stan_source
    assert "gps_speed" not in stan_source
    assert "sigma_motion_process * process_time_scale" not in stan_source


def test_batch_initialization_covers_dynamic_state_vectors_and_process_scales():
    window = _dynamic_ctrv_window()
    stan_data = ctrv_model.build_stan_data(window)
    initial_values = ctrv_model._default_initial_values(stan_data, seed=42)

    assert initial_values["speed_state"].shape == (window.observation_count,)
    assert initial_values["turn_rate_state"].shape == (window.observation_count,)
    assert np.all(initial_values["speed_state"] >= ctrv_model.SPEED_STATE_LOWER_MPS)
    assert initial_values["sigma_speed_process"] > 0
    assert initial_values["sigma_turn_rate_process"] > 0


def test_zero_motion_state_innovations_recover_constant_ctrv_transition():
    speed = 4.0
    heading = 0.3
    turn_rate = 0.01
    state = np.asarray([[0.0, 0.0, speed, heading, turn_rate]])
    total_time = 0.0

    for time_step in (2.0, 3.0, 5.0):
        state = ctrv_model._ctrv_state_transition(state, time_step)
        total_time += time_step

    expected_x = (
        speed / turn_rate * (np.sin(heading + turn_rate * total_time) - np.sin(heading))
    )
    expected_y = (
        speed
        / turn_rate
        * (-np.cos(heading + turn_rate * total_time) + np.cos(heading))
    )
    assert state[0, 0] == pytest.approx(expected_x)
    assert state[0, 1] == pytest.approx(expected_y)
    assert state[0, 2] == pytest.approx(speed)
    assert state[0, 4] == pytest.approx(turn_rate)


def _assert_dynamic_batch_fit_interface(fit, *, prediction_count):
    for variable_name in (*ctrv_model.PARAMETER_NAMES, *PREDICTION_VARIABLE_NAMES):
        samples = reporting.posterior_variable_samples(fit, variable_name)
        assert np.all(np.isfinite(samples))
    for variable_name in PREDICTION_VARIABLE_NAMES:
        samples = reporting.posterior_variable_samples(fit, variable_name)
        assert samples.ndim == 2
        assert samples.shape[1] == prediction_count


@pytest.mark.skipif(
    not RUN_CMDSTAN_INTEGRATION,
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to run CmdStan VI.",
)
def test_dynamic_batch_ctrv_vi_integration():
    window = _dynamic_ctrv_window()
    fit = ctrv_model.fit_bayesian_ctrv_model(
        window,
        inference_method="vi",
        iter=3_000,
        draws=50,
        require_converged=False,
        seed=42,
    )

    _assert_dynamic_batch_fit_interface(
        fit,
        prediction_count=window.prediction_count,
    )


@pytest.mark.skipif(
    not RUN_CMDSTAN_INTEGRATION,
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to run CmdStan MCMC.",
)
def test_dynamic_batch_ctrv_mcmc_integration():
    window = _dynamic_ctrv_window()
    fit = ctrv_model.fit_bayesian_ctrv_model(
        window,
        inference_method="mcmc",
        chains=1,
        parallel_chains=1,
        iter_warmup=100,
        iter_sampling=50,
        seed=42,
    )

    _assert_dynamic_batch_fit_interface(
        fit,
        prediction_count=window.prediction_count,
    )
