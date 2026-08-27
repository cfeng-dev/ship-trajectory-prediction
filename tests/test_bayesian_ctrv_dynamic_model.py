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
    assert stan_data["speed_state_lower_mps"] == 0.0
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
    assert "sigma_motion_process * process_time_scale" in stan_source


def test_position_process_standard_deviation_uses_sqrt_time_scaling():
    motion_process = 5.0
    reference_interval = ctrv_model.PROCESS_REFERENCE_INTERVAL_SECONDS

    scales = motion_process * np.sqrt(
        np.asarray([0.5, 1.0, 2.0]) * reference_interval / reference_interval
    )

    np.testing.assert_allclose(
        scales,
        motion_process * np.asarray([np.sqrt(0.5), 1.0, np.sqrt(2.0)]),
    )
    np.testing.assert_allclose(
        scales**2 / motion_process**2,
        np.asarray([0.5, 1.0, 2.0]),
    )


@pytest.mark.parametrize(
    ("dt_seconds", "expected_standard_deviation_m"),
    ((2.5, 2.0), (10.0, 4.0), (40.0, 8.0)),
)
def test_irregular_intervals_have_expected_position_process_scale(
    dt_seconds,
    expected_standard_deviation_m,
):
    motion_process = 4.0

    actual = motion_process * np.sqrt(
        dt_seconds / ctrv_model.PROCESS_REFERENCE_INTERVAL_SECONDS
    )

    assert actual == pytest.approx(expected_standard_deviation_m)


def test_stan_scales_history_and_forecast_position_process_noise_only():
    stan_source = ctrv_model.STAN_FILE.read_text(encoding="utf-8")
    history_source, forecast_source = stan_source.split("generated quantities", 1)

    assert (
        "real position_process_scale = sigma_motion_process * process_time_scale;"
        in history_source
    )
    assert "x_true[n] ~ normal(position[1], position_process_scale);" in history_source
    assert "y_true[n] ~ normal(position[2], position_process_scale);" in history_source
    assert (
        "real position_process_scale = sigma_motion_process * process_time_scale;"
        in forecast_source
    )
    assert "expected_position[1], position_process_scale" in forecast_source
    assert "expected_position[2], position_process_scale" in forecast_source
    assert "sigma_position_observation * process_time_scale" not in stan_source
    assert "x_observed ~ normal(x_true, sigma_position_observation);" in stan_source
    assert "y_observed ~ normal(y_true, sigma_position_observation);" in stan_source
    assert "x_prediction[n], sigma_position_observation" in forecast_source
    assert "y_prediction[n], sigma_position_observation" in forecast_source


def test_rbpf_scales_position_process_covariance_and_forecast_draws():
    update_source = inspect.getsource(ctrv_model.SequentialBayesianCTRVFilter.update)
    forecast_source = inspect.getsource(
        ctrv_model.SequentialBayesianCTRVFilter.forecast
    )

    assert "motion_process**2 * process_variance_scale" in update_source
    assert "speed_process**2 * process_variance_scale" in update_source
    assert "turn_rate_process**2 * process_variance_scale" in update_source
    assert (
        "position_process_scale = motion_process[:, None] * process_time_scale"
        in forecast_source
    )
    assert "speed_process * process_time_scale" in forecast_source
    assert "turn_rate_process * process_time_scale" in forecast_source
    assert "observation_noise * process_time_scale" not in forecast_source


def test_stan_uses_folded_normal_speed_transition_and_reflected_forecast():
    stan_source = ctrv_model.STAN_FILE.read_text(encoding="utf-8")

    assert "target += log_sum_exp(" in stan_source
    assert "-speed_state[n] | speed_state[n - 1]" in stan_source
    assert "normal_lccdf" not in stan_source
    assert "fmax(abs(speed_proposal), speed_state_lower_mps)" in stan_source
    assert "heading_previous + pi()" not in stan_source
    assert "heading_for_transition" not in stan_source


@pytest.mark.parametrize(
    ("proposal", "expected_speed"),
    ((4.0, 4.0), (-2.0, 2.0)),
)
def test_rbpf_speed_reflection_preserves_heading(proposal, expected_speed):
    heading = 0.4
    states = np.asarray([[1.0, 2.0, proposal, heading, 0.01]])

    ctrv_model._normalize_ctrv_states(states)

    assert states[0, 2] == pytest.approx(expected_speed)
    assert states[0, 3] == pytest.approx(heading)


def test_rbpf_speed_reflection_transforms_covariance_without_rotating_heading():
    states = np.asarray([[1.0, 2.0, -2.0, 0.4, 0.01]])
    covariance = np.asarray(
        [
            [
                [4.0, 0.2, 0.6, 0.1, 0.0],
                [0.2, 3.0, -0.3, 0.0, 0.1],
                [0.6, -0.3, 2.0, -0.2, 0.4],
                [0.1, 0.0, -0.2, 1.0, 0.2],
                [0.0, 0.1, 0.4, 0.2, 0.5],
            ]
        ]
    )
    reflection = np.diag([1.0, 1.0, -1.0, 1.0, 1.0])
    expected_covariance = reflection @ covariance[0] @ reflection

    ctrv_model._normalize_ctrv_states(states, covariance)

    assert states[0, 2] == pytest.approx(2.0)
    assert states[0, 3] == pytest.approx(0.4)
    np.testing.assert_allclose(covariance[0], expected_covariance)


@pytest.mark.parametrize(
    ("mean", "scale"),
    ((0.0, 1.0), (0.5, 2.0), (4.0, 0.75)),
)
def test_folded_normal_transition_density_integrates_to_one(mean, scale):
    speed = np.linspace(0.0, mean + 12.0 * scale, 200_001)
    normalization = scale * np.sqrt(2.0 * np.pi)
    density = (
        np.exp(-0.5 * ((speed - mean) / scale) ** 2)
        + np.exp(-0.5 * ((-speed - mean) / scale) ** 2)
    ) / normalization

    assert np.trapezoid(density, speed) == pytest.approx(1.0, abs=1e-8)


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
