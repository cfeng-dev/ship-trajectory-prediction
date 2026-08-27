"""Focused tests for the time-based Bayesian Position Model."""

import inspect
import os

import numpy as np
import pandas as pd
import pytest

import bayestraj.models.bayesian_position_model as position_model
import bayestraj.observations.window as observation_window
import bayestraj.validation.reporting as reporting

RUN_CMDSTAN_INTEGRATION = os.environ.get("RUN_CMDSTAN_INTEGRATION") == "1"
PREDICTION_VARIABLE_NAMES = (
    "x_model_prediction",
    "y_model_prediction",
    "x_observation_prediction",
    "y_observation_prediction",
)


def _irregular_position_window():
    time_seconds = np.asarray([0.0, 5.0, 15.0, 25.0, 45.0, 50.0, 70.0, 80.0])
    timestamps = pd.DatetimeIndex(
        pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(time_seconds, unit="s")
    )
    return observation_window.TrajectoryWindowData(
        timestamps=timestamps,
        time_seconds=time_seconds,
        x_meters=2.0 * time_seconds,
        y_meters=-0.5 * time_seconds,
        reference_longitude=8.0,
        reference_latitude=47.0,
        gps_speed_mps=np.full(time_seconds.size, 999.0),
        observation_count=5,
    )


def test_stan_data_contains_irregular_observation_and_prediction_times():
    window = _irregular_position_window()
    stan_data = position_model.build_stan_data(
        window,
        priors=position_model.BayesianPositionModelPriors(),
    )

    np.testing.assert_array_equal(stan_data["time_observed"], window.time_seconds[:5])
    np.testing.assert_array_equal(stan_data["time_prediction"], window.time_seconds[5:])
    assert stan_data["position_model_reference_interval_seconds"] == pytest.approx(
        position_model.POSITION_MODEL_REFERENCE_INTERVAL_SECONDS
    )
    assert "gps_speed" not in stan_data


@pytest.mark.parametrize(
    ("time_observed", "time_prediction"),
    (
        (np.asarray([0.0, 5.0, 5.0, 20.0, 30.0]), np.asarray([40.0])),
        (np.asarray([0.0, 5.0, 10.0, 20.0, 30.0]), np.asarray([30.0, 40.0])),
        (np.asarray([0.0, 5.0, 10.0, 20.0, 30.0]), np.asarray([40.0, 35.0])),
    ),
)
def test_non_increasing_or_overlapping_model_times_are_rejected(
    time_observed,
    time_prediction,
):
    with pytest.raises(ValueError, match="timestamps"):
        position_model._validate_model_times(time_observed, time_prediction)


def test_rotation_rate_and_motion_scale_use_physical_elapsed_time():
    kappa = np.asarray([np.log(2.0) / 10.0, np.log(2.0) / 10.0])
    omega = np.asarray([0.01, 0.01])
    intervals = np.asarray([5.0, 20.0])

    matrices = position_model._time_scaled_motion_matrices(
        kappa,
        omega,
        intervals,
    )
    transformed_unit_x = np.einsum("nij,j->ni", matrices, np.asarray([1.0, 0.0]))

    np.testing.assert_allclose(
        np.linalg.norm(transformed_unit_x, axis=1),
        np.exp(kappa * intervals),
    )
    np.testing.assert_allclose(
        np.arctan2(transformed_unit_x[:, 1], transformed_unit_x[:, 0]),
        omega * intervals,
    )
    np.testing.assert_allclose(omega * intervals, np.asarray([0.05, 0.20]))


def test_reference_interval_transition_recovers_old_regular_step_structure():
    reference = position_model.POSITION_MODEL_REFERENCE_INTERVAL_SECONDS
    rho_reference = 1.25
    phi_reference = 0.3
    previous_displacement = np.asarray([20.0, -4.0])
    transition = position_model._displacement_transition_matrices(
        np.asarray([np.log(rho_reference) / reference]),
        np.asarray([phi_reference / reference]),
        current_interval_seconds=reference,
        previous_interval_seconds=reference,
    )[0]
    old_rotation_scale = rho_reference * np.asarray(
        [
            [np.cos(phi_reference), -np.sin(phi_reference)],
            [np.sin(phi_reference), np.cos(phi_reference)],
        ]
    )

    np.testing.assert_allclose(transition, old_rotation_scale)
    np.testing.assert_allclose(
        transition @ previous_displacement,
        old_rotation_scale @ previous_displacement,
    )


def test_constant_velocity_rescales_displacement_for_changed_interval():
    transition = position_model._displacement_transition_matrices(
        np.zeros(1),
        np.zeros(1),
        current_interval_seconds=20.0,
        previous_interval_seconds=10.0,
    )[0]

    np.testing.assert_allclose(
        transition @ np.asarray([20.0, 0.0]),
        np.asarray([40.0, 0.0]),
    )


def test_rbpf_transition_uses_correlated_time_scaled_displacement_innovation():
    parameters = np.zeros((2, 4), dtype=float)
    parameters[:, 2] = np.log(np.asarray([1.0, 50.0]))
    parameters[:, 3] = np.log(3.0)

    transitions, process_covariances = position_model._sequential_transition_terms(
        parameters,
        current_interval_seconds=20.0,
        previous_interval_seconds=10.0,
    )

    expected_b = 2.0 * np.eye(2)
    np.testing.assert_allclose(transitions[0, :2, 2:], expected_b)
    np.testing.assert_allclose(transitions[0, 2:, 2:], expected_b)
    expected_q = (
        3.0**2 * 20.0 / position_model.POSITION_MODEL_REFERENCE_INTERVAL_SECONDS
    )
    np.testing.assert_allclose(process_covariances[0, :2, :2], expected_q * np.eye(2))
    np.testing.assert_allclose(process_covariances[0, :2, 2:], expected_q * np.eye(2))
    np.testing.assert_allclose(process_covariances[0, 2:, :2], expected_q * np.eye(2))
    np.testing.assert_allclose(process_covariances[0, 2:, 2:], expected_q * np.eye(2))
    np.testing.assert_allclose(process_covariances[0], process_covariances[1])


def test_motion_residual_standard_deviation_uses_sqrt_time_scaling():
    sigma_reference = 4.0
    reference = position_model.POSITION_MODEL_REFERENCE_INTERVAL_SECONDS
    intervals = np.asarray([5.0, 10.0, 20.0])

    actual = sigma_reference * np.sqrt(intervals / reference)

    np.testing.assert_allclose(
        actual,
        sigma_reference * np.asarray([np.sqrt(0.5), 1.0, np.sqrt(2.0)]),
    )


def test_stan_uses_same_irregular_time_transition_and_unscaled_measurement_noise():
    stan_source = position_model.STAN_FILE.read_text(encoding="utf-8")
    history_source, forecast_source = stan_source.split("generated quantities", 1)

    assert "dt_current / dt_previous" in history_source
    assert "dt / previous_dt" in forecast_source
    assert "log_displacement_scale_rate" in stan_source
    assert "rotation_rate" in stan_source
    assert "sqrt(" in history_source
    assert "dt_current / position_model_reference_interval_seconds" in history_source
    assert "dt / position_model_reference_interval_seconds" in forecast_source
    assert "sigma_position_observation * sqrt" not in stan_source
    assert "x_observed ~ normal(x_true, sigma_position_observation);" in stan_source
    assert "y_observed ~ normal(y_true, sigma_position_observation);" in stan_source


def test_reference_interval_priors_preserve_existing_interpretation():
    priors = position_model.BayesianPositionModelPriors()
    reference = position_model.POSITION_MODEL_REFERENCE_INTERVAL_SECONDS
    normal_quantile = 1.959963984540054

    assert priors.log_displacement_scale_rate_prior_scale * reference == pytest.approx(
        np.log(priors.displacement_scale_prior_factor) / normal_quantile
    )
    assert priors.rotation_rate_prior_scale * reference == pytest.approx(
        np.deg2rad(priors.rotation_angle_prior_abs_upper_deg) / normal_quantile
    )


def test_prior_reference_boundary_has_interpretable_scale_factors_over_time():
    reference = position_model.POSITION_MODEL_REFERENCE_INTERVAL_SECONDS
    kappa_boundary = np.log(2.0) / reference

    scale_factors = np.exp(kappa_boundary * np.asarray([5.0, 10.0, 20.0]))

    np.testing.assert_allclose(scale_factors, np.asarray([np.sqrt(2.0), 2.0, 4.0]))


def test_default_initial_values_use_rate_parameter_names():
    stan_data = position_model.build_stan_data(
        _irregular_position_window(),
        priors=position_model.BayesianPositionModelPriors(),
    )

    initial_values = position_model._default_initial_values(stan_data, seed=42)

    assert "log_displacement_scale_rate" in initial_values
    assert "rotation_rate" in initial_values
    assert "log_displacement_scale" not in initial_values
    assert "rotation_angle" not in initial_values
    assert initial_values["sigma_motion_residual"] > 0.0


def test_online_rbpf_uses_actual_observation_and_future_times():
    window = _irregular_position_window()
    config = position_model.SequentialPositionFilterConfig(
        particle_count=64,
        posterior_draw_count=16,
    )
    online_filter = position_model.SequentialBayesianPositionFilter.initialize(
        window.time_seconds[:5],
        window.x_meters[:5],
        window.y_meters[:5],
        priors=position_model.BayesianPositionModelPriors(),
        config=config,
        seed=42,
    )

    assert online_filter.processed_observation_count == 5
    assert online_filter.last_observation_time_seconds == pytest.approx(45.0)
    assert online_filter.previous_interval_seconds == pytest.approx(20.0)
    fit = online_filter.forecast(window.time_seconds[5:], seed=43)

    for variable_name in (*position_model.PARAMETER_NAMES, *PREDICTION_VARIABLE_NAMES):
        samples = reporting.posterior_variable_samples(fit, variable_name)
        assert np.all(np.isfinite(samples))
    for variable_name in PREDICTION_VARIABLE_NAMES:
        assert fit.stan_variable(variable_name).shape == (16, 3)


def test_online_forecast_changes_when_future_elapsed_time_changes():
    time_seconds = np.asarray([0.0, 5.0, 15.0, 25.0, 45.0])
    online_filter = position_model.SequentialBayesianPositionFilter.initialize(
        time_seconds,
        2.0 * time_seconds,
        np.zeros_like(time_seconds),
        priors=position_model.BayesianPositionModelPriors(),
        config=position_model.SequentialPositionFilterConfig(
            particle_count=64,
            posterior_draw_count=16,
        ),
        seed=42,
    )

    short_gap = online_filter.forecast(np.asarray([50.0]), seed=43)
    long_gap = online_filter.forecast(np.asarray([65.0]), seed=43)

    assert not np.allclose(
        short_gap.stan_variable("x_model_prediction"),
        long_gap.stan_variable("x_model_prediction"),
    )


def test_online_time_validation_rejects_reused_or_count_only_times():
    online_filter = position_model.SequentialBayesianPositionFilter.initialize(
        np.asarray([0.0, 10.0]),
        np.asarray([0.0, 20.0]),
        np.asarray([0.0, 0.0]),
        priors=position_model.BayesianPositionModelPriors(),
        config=position_model.SequentialPositionFilterConfig(
            particle_count=16,
            posterior_draw_count=8,
        ),
        seed=42,
    )

    with pytest.raises(ValueError, match="follow"):
        online_filter.update(10.0, 40.0, 0.0)
    with pytest.raises(ValueError, match="future_time_seconds"):
        online_filter.forecast(3, seed=43)


def test_batch_and_online_sources_share_the_time_dependent_transition_terms():
    stan_source = position_model.STAN_FILE.read_text(encoding="utf-8")
    online_source = inspect.getsource(position_model._displacement_transition_matrices)

    assert "exp(log_displacement_scale_rate * dt)" in stan_source
    assert "rotation_rate * dt" in stan_source
    assert "dt_current / dt_previous" in stan_source
    assert "_time_scaled_motion_matrices" in online_source
    assert "current_interval_seconds / previous_interval_seconds" in online_source


def test_position_model_remains_position_only():
    stan_source = position_model.STAN_FILE.read_text(encoding="utf-8")
    python_source = inspect.getsource(position_model.build_stan_data)

    for sensor_name in ("gps_speed", "shaft_speed", "thruster_speed"):
        assert sensor_name not in stan_source
        assert sensor_name not in python_source


def _assert_batch_fit_interface(fit, *, prediction_count):
    for variable_name in (*position_model.PARAMETER_NAMES, *PREDICTION_VARIABLE_NAMES):
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
def test_time_based_batch_position_vi_integration():
    window = _irregular_position_window()
    fit = position_model.fit_bayesian_position_model(
        window,
        priors=position_model.BayesianPositionModelPriors(),
        inference_method="vi",
        iter=3_000,
        draws=50,
        require_converged=False,
        seed=42,
    )

    _assert_batch_fit_interface(fit, prediction_count=window.prediction_count)


@pytest.mark.skipif(
    not RUN_CMDSTAN_INTEGRATION,
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to run CmdStan MCMC.",
)
def test_time_based_batch_position_mcmc_integration():
    window = _irregular_position_window()
    fit = position_model.fit_bayesian_position_model(
        window,
        priors=position_model.BayesianPositionModelPriors(),
        inference_method="mcmc",
        chains=1,
        parallel_chains=1,
        iter_warmup=100,
        iter_sampling=50,
        seed=42,
    )

    _assert_batch_fit_interface(fit, prediction_count=window.prediction_count)
