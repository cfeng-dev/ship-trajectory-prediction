"""Focused tests for the shared dynamic Bayesian CTRV model."""

import inspect
import os
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

import bayestraj.inference.ctrv_cmdstan as batch_inference
import bayestraj.inference.ctrv_rbpf as rbpf
import bayestraj.models.bayesian_ctrv as ctrv_model
import bayestraj.models.ctrv as ctrv_dynamics
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
    rbpf_config = rbpf.SequentialCTRVFilterConfig()
    update_source = inspect.getsource(rbpf.SequentialBayesianCTRVFilter.update)
    forecast_source = inspect.getsource(rbpf.SequentialBayesianCTRVFilter.forecast)

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
    assert "sigma_motion_process" not in stan_source
    assert "sigma_position_process" not in stan_source


def test_ctrv_priors_and_stan_data_have_no_cartesian_process_noise():
    priors = ctrv_model.BayesianCTRVPriors()
    prior_field_names = {field.name for field in fields(priors)}
    stan_data = ctrv_model.build_stan_data(_dynamic_ctrv_window(), priors=priors)

    assert not any("motion_process" in name for name in prior_field_names)
    assert not any("position_process" in name for name in prior_field_names)
    assert not any("motion_process" in name for name in stan_data)
    assert not any("position_process" in name for name in stan_data)
    with pytest.raises(TypeError, match="unexpected keyword"):
        ctrv_model.BayesianCTRVPriors(sigma_motion_process_prior_upper_m=20.0)


def test_noise_parameter_names_contain_only_remaining_ctrv_scales():
    assert ctrv_model.NOISE_PARAMETER_NAMES == (
        "sigma_position_observation",
        "sigma_speed_process",
        "sigma_turn_rate_process",
    )
    assert "sigma_motion_process" not in ctrv_model.PARAMETER_NAMES


def test_stan_positions_are_conditional_transitions_with_no_cartesian_jitter():
    stan_source = ctrv_model.STAN_FILE.read_text(encoding="utf-8")
    history_source, forecast_source = stan_source.split("generated quantities", 1)

    assert "real x_initial;" in history_source
    assert "real y_initial;" in history_source
    assert "vector[N_history] x_state;" in history_source
    assert "vector[N_history] y_state;" in history_source
    assert "x_state[n] = position[1];" in history_source
    assert "y_state[n] = position[2];" in history_source
    assert "x_state[n] ~ normal" not in history_source
    assert "y_state[n] ~ normal" not in history_source
    assert "x_observed ~ normal(x_state, sigma_position_observation);" in history_source
    assert "y_observed ~ normal(y_state, sigma_position_observation);" in history_source
    assert "x_prediction[n] = expected_position[1];" in forecast_source
    assert "y_prediction[n] = expected_position[2];" in forecast_source
    assert "normal_rng(expected_position" not in forecast_source
    assert "sigma_position_observation * process_time_scale" not in stan_source
    assert "x_prediction[n], sigma_position_observation" in forecast_source
    assert "y_prediction[n], sigma_position_observation" in forecast_source


def test_rbpf_propagates_kinematic_process_noise_without_cartesian_q():
    update_source = inspect.getsource(rbpf.SequentialBayesianCTRVFilter.update)
    forecast_source = inspect.getsource(rbpf.SequentialBayesianCTRVFilter.forecast)

    assert "speed_process**2 * process_variance_scale" in update_source
    assert "turn_rate_process**2 * process_variance_scale" in update_source
    assert "motion_process" not in update_source
    assert "motion_process" not in forecast_source
    assert "predicted_covariances[:, _STATE_X_INDEX" not in update_source
    assert "predicted_covariances[:, _STATE_Y_INDEX" not in update_source
    assert "speed_process * process_time_scale" in forecast_source
    assert "turn_rate_process * process_time_scale" in forecast_source
    assert "states[:, :2] +=" not in forecast_source
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

    ctrv_dynamics.normalize_states(states)

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

    ctrv_dynamics.normalize_states(states, covariance)

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
    initial_values = batch_inference._default_initial_values(stan_data, seed=42)

    assert initial_values["speed_state"].shape == (window.observation_count,)
    assert initial_values["turn_rate_state"].shape == (window.observation_count,)
    assert np.isfinite(initial_values["x_initial"])
    assert np.isfinite(initial_values["y_initial"])
    assert "x_true" not in initial_values
    assert "y_true" not in initial_values
    assert "sigma_motion_process" not in initial_values
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
        state = ctrv_dynamics.transition_states(state, time_step)
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


def _controlled_online_filter(
    *,
    observation_noise,
    speed_process,
    turn_rate_process,
    draw_count=2_000,
):
    particle_count = 128
    parameter_values = np.log(
        np.asarray([observation_noise, speed_process, turn_rate_process])
    )
    parameter_particles = np.broadcast_to(
        parameter_values,
        (particle_count, parameter_values.size),
    ).copy()
    state = np.asarray([0.0, 0.0, 4.0, 0.3, 0.01])
    state_means = np.broadcast_to(state, (particle_count, state.size)).copy()
    state_covariances = np.broadcast_to(
        1e-12 * np.eye(state.size),
        (particle_count, state.size, state.size),
    ).copy()
    return rbpf.SequentialBayesianCTRVFilter(
        config=rbpf.SequentialCTRVFilterConfig(
            particle_count=particle_count,
            posterior_draw_count=draw_count,
        ),
        parameter_particles=parameter_particles,
        weights=np.full(particle_count, 1.0 / particle_count),
        state_means=state_means,
        state_covariances=state_covariances,
        generator=np.random.default_rng(42),
        last_observation_time_seconds=0.0,
        processed_observation_count=1,
    )


def test_rbpf_parameter_particles_contain_only_remaining_noise_scales():
    online_filter = _controlled_online_filter(
        observation_noise=1.0,
        speed_process=0.5,
        turn_rate_process=0.01,
        draw_count=32,
    )

    assert online_filter.parameter_particles.shape == (128, 3)
    assert (
        len(
            rbpf._sequential_parameter_values(
                parameter_particles=online_filter.parameter_particles
            )
        )
        == 3
    )


def test_synthetic_ctrv_forecast_tracks_motion_and_remains_probabilistic():
    online_filter = _controlled_online_filter(
        observation_noise=2.0,
        speed_process=0.02,
        turn_rate_process=0.0002,
    )
    future_times = np.asarray([10.0, 20.0, 30.0])
    fit = online_filter.forecast(future_times, seed=43)
    expected_state = np.asarray([[0.0, 0.0, 4.0, 0.3, 0.01]])
    expected_positions = []
    for _ in future_times:
        expected_state = ctrv_dynamics.transition_states(expected_state, 10.0)
        expected_positions.append(expected_state[0, :2].copy())
    expected_positions = np.asarray(expected_positions)
    latent_positions = np.stack(
        (fit.stan_variable("x_prediction"), fit.stan_variable("y_prediction")),
        axis=-1,
    )

    np.testing.assert_allclose(
        np.median(latent_positions, axis=0),
        expected_positions,
        atol=0.5,
    )
    assert np.all(np.std(latent_positions, axis=0) > 0.0)


def test_larger_speed_process_increases_longitudinal_uncertainty():
    future_times = np.asarray([10.0, 20.0, 30.0])
    low_fit = _controlled_online_filter(
        observation_noise=0.1,
        speed_process=1e-9,
        turn_rate_process=1e-9,
    ).forecast(future_times, seed=43)
    high_fit = _controlled_online_filter(
        observation_noise=0.1,
        speed_process=1.0,
        turn_rate_process=1e-9,
    ).forecast(future_times, seed=43)
    heading = 0.3

    def longitudinal_samples(fit):
        return fit.stan_variable("x_prediction")[:, -1] * np.cos(
            heading
        ) + fit.stan_variable("y_prediction")[:, -1] * np.sin(heading)

    assert np.std(longitudinal_samples(high_fit)) > 10.0 * np.std(
        longitudinal_samples(low_fit)
    )


def test_larger_turn_rate_process_increases_lateral_uncertainty():
    future_times = np.asarray([10.0, 20.0, 30.0])
    low_fit = _controlled_online_filter(
        observation_noise=0.1,
        speed_process=1e-9,
        turn_rate_process=1e-9,
    ).forecast(future_times, seed=43)
    high_fit = _controlled_online_filter(
        observation_noise=0.1,
        speed_process=1e-9,
        turn_rate_process=0.03,
    ).forecast(future_times, seed=43)
    heading = 0.3

    def lateral_samples(fit):
        return -fit.stan_variable("x_prediction")[:, -1] * np.sin(
            heading
        ) + fit.stan_variable("y_prediction")[:, -1] * np.cos(heading)

    assert np.std(lateral_samples(high_fit)) > 10.0 * np.std(lateral_samples(low_fit))


def test_observation_noise_widens_only_observation_predictions():
    future_times = np.asarray([10.0])
    low_fit = _controlled_online_filter(
        observation_noise=0.1,
        speed_process=0.1,
        turn_rate_process=0.001,
    ).forecast(future_times, seed=43)
    high_fit = _controlled_online_filter(
        observation_noise=10.0,
        speed_process=0.1,
        turn_rate_process=0.001,
    ).forecast(future_times, seed=43)

    np.testing.assert_allclose(
        low_fit.stan_variable("x_prediction"),
        high_fit.stan_variable("x_prediction"),
    )
    low_residual = low_fit.stan_variable(
        "x_observation_prediction"
    ) - low_fit.stan_variable("x_prediction")
    high_residual = high_fit.stan_variable(
        "x_observation_prediction"
    ) - high_fit.stan_variable("x_prediction")
    assert np.std(high_residual) > 50.0 * np.std(low_residual)


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
    fit = batch_inference.fit_bayesian_ctrv_model(
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
    fit = batch_inference.fit_bayesian_ctrv_model(
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
