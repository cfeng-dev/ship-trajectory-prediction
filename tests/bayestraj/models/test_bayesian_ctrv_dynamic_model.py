"""Focused tests for the shared dynamic Bayesian CTRV model."""

import inspect
import os
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import expon, norm

import bayestraj.inference.configuration as inference
import bayestraj.inference.ctrv_cmdstan as batch_inference
import bayestraj.inference.ctrv_smc as smc
import bayestraj.models.bayesian_ctrv as ctrv_model
import bayestraj.models.ctrv as ctrv_dynamics
import bayestraj.observations.position as position_observations
import bayestraj.observations.window as observation_window
import bayestraj.validation.metrics as metrics
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


def _ten_point_vi_regression_window():
    time_seconds = np.arange(13, dtype=float) * 10.0
    x_meters = np.asarray(
        [
            0.0,
            25.047335981863377,
            56.876702501521876,
            92.71362797807262,
            130.93966573647657,
            170.62997732150617,
            211.55339488615326,
            253.09342616174956,
            294.6335825103842,
            336.4050407392522,
            378.1765851844071,
            420.02529426955994,
            461.79703990155696,
        ]
    )
    y_meters = np.asarray(
        [
            0.0,
            4.524539397369224,
            8.144170915406068,
            10.858894554110531,
            13.573618192814996,
            17.19324971085184,
            19.003065470931247,
            17.19324971085184,
            10.858894554110531,
            2.7147236387044638,
            -6.334355156033984,
            -16.288341830812136,
            -27.147236384922667,
        ]
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
        observation_count=10,
    )


def test_observation_noise_is_inferred_without_a_fixed_floor():
    priors = ctrv_model.BayesianCTRVPriors()
    window = _dynamic_ctrv_window()
    stan_data = ctrv_model.build_stan_data(window, priors=priors)
    assert "sigma_position_observation_floor_m" not in stan_data
    assert not hasattr(priors, "sigma_position_observation_floor_m")
    particle_filter = smc.SequentialMonteCarloCTRVFilter.initialize(
        window.time_seconds[:1],
        window.x_meters[:1],
        window.y_meters[:1],
        priors=priors,
        config=smc.SequentialMonteCarloCTRVConfig(
            particle_count=128, posterior_draw_count=64
        ),
        seed=42,
    )
    fit = particle_filter.sample_current_posterior(seed=42)
    noise = fit.stan_variable("sigma_position_observation")
    assert np.all(noise > 0.0)
    assert np.any(noise < 5.0)
    assert not hasattr(particle_filter, "position_observation_noise_floor_m")


def test_batch_stan_data_contains_shared_dynamic_process_inputs():
    priors = ctrv_model.BayesianCTRVPriors()
    stan_data = ctrv_model.build_stan_data(_dynamic_ctrv_window(), priors=priors)

    assert "sigma_position_observation_floor_m" not in stan_data
    assert stan_data["sigma_speed_process_prior_rate"] == pytest.approx(
        priors.sigma_speed_process_prior_rate
    )
    assert stan_data["sigma_turn_rate_process_prior_rate"] == pytest.approx(
        priors.sigma_turn_rate_process_prior_rate
    )
    assert stan_data["process_reference_interval_seconds"] == pytest.approx(
        ctrv_model.PROCESS_REFERENCE_INTERVAL_SECONDS
    )
    assert "speed_state_lower_mps" not in stan_data
    assert "gps_speed" not in stan_data


@pytest.mark.parametrize("stationary", [False, True])
def test_batch_accepts_two_positions_and_builds_one_transition_initials(stationary):
    window = replace(_dynamic_ctrv_window(), observation_count=2)
    if stationary:
        window = replace(window, x_meters=np.zeros(8), y_meters=np.zeros(8))

    data = ctrv_model.build_stan_data(window)
    initial = batch_inference._default_initial_values(data, seed=42)

    assert data["N_history"] == 2
    np.testing.assert_array_equal(data["x_observed"], window.x_meters[:2])
    np.testing.assert_array_equal(data["time_prediction"], window.time_seconds[2:])
    assert initial["speed_initial"] > 0.0
    assert np.isfinite(initial["turn_rate_initial"])
    assert initial["speed_innovation"].shape == (0,)
    assert initial["turn_rate_innovation"].shape == (0,)
    assert all(np.all(np.isfinite(value)) for value in initial.values())


def _low_speed_br24_regression_case():
    """Accepted fixes at cycles 31/32 that produced a zero initial in chain 3."""
    times = np.array([49149.3, 49152.0, 49154.7])
    window = observation_window.TrajectoryWindowData(
        timestamps=pd.to_datetime(times, unit="s", utc=True),
        time_seconds=times,
        x_meters=np.array([703.8684515975431, 704.06763650503, 0.0]),
        y_meters=np.array([-676.2003122956523, -676.0552542772649, 0.0]),
        reference_longitude=None,
        reference_latitude=None,
        gps_speed_mps=np.full(3, np.nan),
        observation_count=2,
    )
    observations = position_observations.PositionObservations(
        time_seconds=times[:2],
        x_meters=window.x_meters[:2],
        y_meters=window.y_meters[:2],
        position_noise_std_m=0.0,
        noise_seed=0,
        measurement_covariances=np.array(
            [
                [
                    [2.956479715650011, -2.172999101710388],
                    [-2.172999101710388, 3.266479944184643],
                ],
                [
                    [3.3723861710062426, -2.339532025988014],
                    [-2.339532025988014, 2.435718278692386],
                ],
            ]
        ),
    )
    priors = ctrv_model.BayesianCTRVPriors()
    return window, observations, priors


@pytest.mark.parametrize("seed", range(1073266673, 1073266677))
def test_low_speed_br24_initials_stay_strictly_inside_speed_constraint(seed):
    window, observations, priors = _low_speed_br24_regression_case()
    data = ctrv_model.build_stan_data(
        window, position_observations=observations, priors=priors
    )

    initial = batch_inference._default_initial_values(data, seed=seed)
    repeated = batch_inference._default_initial_values(data, seed=seed)

    assert initial["speed_initial"] > 0.0
    assert np.isfinite(np.log(initial["speed_initial"]))
    for name in initial:
        np.testing.assert_array_equal(initial[name], repeated[name])


def test_process_noise_uses_one_second_reference_interval():
    priors = ctrv_model.BayesianCTRVPriors()
    stan_data = ctrv_model.build_stan_data(_dynamic_ctrv_window(), priors=priors)

    assert ctrv_dynamics.PROCESS_REFERENCE_INTERVAL_SECONDS == 1.0
    assert ctrv_model.PROCESS_REFERENCE_INTERVAL_SECONDS == 1.0
    assert smc.ctrv_dynamics.PROCESS_REFERENCE_INTERVAL_SECONDS == 1.0
    assert stan_data["process_reference_interval_seconds"] == 1.0


@pytest.mark.parametrize(
    ("dt_seconds", "expected_scale"),
    ((1.0, 1.0), (10.0, np.sqrt(10.0))),
)
def test_process_time_scale_uses_one_second_reference(dt_seconds, expected_scale):
    assert ctrv_dynamics.process_time_scale(dt_seconds) == pytest.approx(expected_scale)


def test_default_priors_match_ship_independent_configuration():
    priors = ctrv_model.BayesianCTRVPriors()

    assert priors.speed_prior_upper_mps == 10.0
    assert priors.turn_rate_prior_abs_rate_deg_s == 10.0
    assert priors.sigma_position_observation_prior_upper_m == 20.0
    assert priors.sigma_speed_process_prior_upper_mps == 2.0
    assert priors.sigma_turn_rate_process_prior_upper_deg_s == 2.0
    assert {
        priors.speed_prior_tail_probability,
        priors.turn_rate_prior_tail_probability,
        priors.sigma_position_observation_prior_tail_probability,
        priors.sigma_speed_process_prior_tail_probability,
        priors.sigma_turn_rate_process_prior_tail_probability,
    } == {0.05}


def test_default_prior_distribution_parameters_match_tail_statements():
    priors = ctrv_model.BayesianCTRVPriors()
    normal_975_quantile = 1.959963984540054

    assert priors.speed_prior_scale == pytest.approx(10.0 / normal_975_quantile)
    assert priors.turn_rate_prior_scale == pytest.approx(
        np.deg2rad(10.0) / normal_975_quantile
    )
    assert priors.sigma_position_observation_prior_rate == pytest.approx(
        -np.log(0.05) / 20.0
    )
    assert priors.sigma_speed_process_prior_rate == pytest.approx(-np.log(0.05) / 2.0)
    assert priors.sigma_turn_rate_process_prior_rate == pytest.approx(
        -np.log(0.05) / np.deg2rad(2.0)
    )
    assert 1.0 - np.exp(-priors.sigma_speed_process_prior_rate * 2.0) == (
        pytest.approx(0.95)
    )
    assert 1.0 - np.exp(
        -priors.sigma_turn_rate_process_prior_rate * np.deg2rad(2.0)
    ) == pytest.approx(0.95)


def test_default_process_priors_are_slightly_more_permissive_than_old_ten_second_thresholds():
    priors = ctrv_model.BayesianCTRVPriors()
    ten_second_scale = ctrv_dynamics.process_time_scale(10.0)

    assert priors.sigma_speed_process_prior_upper_mps * ten_second_scale == (
        pytest.approx(2.0 * np.sqrt(10.0))
    )
    assert priors.sigma_turn_rate_process_prior_upper_deg_s * ten_second_scale == (
        pytest.approx(2.0 * np.sqrt(10.0))
    )


def test_initial_turn_rate_prior_uses_ten_degrees_per_second():
    priors = ctrv_model.BayesianCTRVPriors()

    assert priors.turn_rate_prior_abs_rate_deg_s == 10.0


def test_turn_rate_prior_uses_a_direct_rate_threshold_in_degrees_per_second():
    priors = ctrv_model.BayesianCTRVPriors()

    assert priors.turn_rate_prior_scale == pytest.approx(
        np.deg2rad(10.0) / 1.959963984540054
    )


def test_batch_and_online_code_use_one_process_time_scale():
    smc_source = inspect.getsource(smc.SequentialMonteCarloCTRVFilter)
    stan_source = ctrv_model.STAN_FILE.read_text(encoding="utf-8")

    assert "ctrv_dynamics.process_time_scale" in smc_source
    assert "sqrt(dt / process_reference_interval_seconds)" in stan_source


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
    assert "x_state[n + 1] = position[1];" in history_source
    assert "y_state[n + 1] = position[2];" in history_source
    assert "x_state[n] ~ normal" not in history_source
    assert "y_state[n] ~ normal" not in history_source
    assert "if (observed_mask[n] == 1)" in history_source
    assert "x_observed[n] ~ normal(x_state[n], sqrt(variance_x));" in history_source
    assert "y_observed[n] ~ normal(" in history_source
    assert "covariance_xy / variance_x" in history_source
    assert "x_prediction[n] = expected_position[1];" in forecast_source
    assert "y_prediction[n] = expected_position[2];" in forecast_source
    assert "normal_rng(expected_position" not in forecast_source
    assert "sigma_position_observation * process_time_scale" not in stan_source
    assert "x_prediction[n], sigma_position_observation" in forecast_source
    assert "y_prediction[n], sigma_position_observation" in forecast_source


def test_stan_history_uses_current_motion_state_for_each_observed_interval():
    stan_source = ctrv_model.STAN_FILE.read_text(encoding="utf-8")
    history_source, forecast_source = stan_source.split("generated quantities", 1)

    assert "vector<lower=0>[N_history - 1] speed_state;" in history_source
    assert "vector[N_history - 1] turn_rate_state;" in history_source
    assert "for (n in 1:(N_history - 1))" in history_source
    assert "speed_state[n]," in history_source
    assert "heading_state[n]," in history_source
    assert "turn_rate_state[n]);" in history_source
    assert "x_state[n + 1] = position[1];" in history_source
    assert "y_state[n + 1] = position[2];" in history_source
    assert "heading_state[n + 1] = wrap_angle(" in history_source
    assert "speed_state[N_history]" not in stan_source
    assert "turn_rate_state[N_history]" not in stan_source
    assert "speed_at_origin = speed_state[N_history - 1];" in forecast_source
    assert "turn_rate_at_origin = turn_rate_state[N_history - 1];" in forecast_source


def test_stan_forecast_propagates_before_evolving_motion_variables():
    forecast_source = ctrv_model.STAN_FILE.read_text(encoding="utf-8").split(
        "generated quantities", 1
    )[1]

    position_index = forecast_source.index("expected_position = ctrv_position(")
    speed_innovation_index = forecast_source.index("speed_proposal = normal_rng(")
    turn_innovation_index = forecast_source.index("turn_rate_previous = normal_rng(")

    assert position_index < speed_innovation_index
    assert position_index < turn_innovation_index


def test_ctrv_step_uses_current_motion_for_nonzero_turn_rate():
    state = ctrv_dynamics.CTRVState(
        x=12.0,
        y=-4.0,
        speed=6.0,
        heading=0.3,
        turn_rate=0.2,
    )
    dt = 2.0

    transitioned = ctrv_dynamics.ctrv_step(state, dt)
    heading_next = state.heading + state.turn_rate * dt

    assert transitioned.x == pytest.approx(
        state.x
        + state.speed / state.turn_rate * (np.sin(heading_next) - np.sin(state.heading))
    )
    assert transitioned.y == pytest.approx(
        state.y
        + state.speed
        / state.turn_rate
        * (-np.cos(heading_next) + np.cos(state.heading))
    )
    assert transitioned.heading == pytest.approx(heading_next)
    assert transitioned.speed == state.speed
    assert transitioned.turn_rate == state.turn_rate


def test_ctrv_step_uses_current_motion_in_straight_line_limit():
    state = ctrv_dynamics.CTRVState(
        x=12.0,
        y=-4.0,
        speed=6.0,
        heading=0.3,
        turn_rate=0.0,
    )
    dt = 2.0

    transitioned = ctrv_dynamics.ctrv_step(state, dt)

    assert transitioned.x == pytest.approx(
        state.x + state.speed * np.cos(state.heading) * dt
    )
    assert transitioned.y == pytest.approx(
        state.y + state.speed * np.sin(state.heading) * dt
    )
    assert transitioned.heading == state.heading


@pytest.mark.skipif(
    not RUN_CMDSTAN_INTEGRATION,
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to execute the Stan state reconstruction.",
)
def test_stan_reconstructs_folded_speed_and_time_scaled_turn_innovations():
    window = replace(
        _dynamic_ctrv_window(),
        time_seconds=np.array([0.0, 1.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0]),
        observation_count=4,
    )
    model = batch_inference.compile_bayesian_ctrv_model()
    fit = model.sample(
        data=ctrv_model.build_stan_data(window),
        inits={
            "x_initial": 0.0,
            "y_initial": 0.0,
            "speed_initial": 0.1,
            "speed_innovation": [-2.0, 1.0],
            "heading_initial": 0.3,
            "turn_rate_initial": 0.02,
            "turn_rate_innovation": [-0.2, 0.3],
            "sigma_speed_process": 0.2,
            "sigma_turn_rate_process": 0.01,
            "sigma_position_observation_excess": 1.0,
        },
        fixed_param=True,
        adapt_engaged=False,
        chains=1,
        iter_warmup=0,
        iter_sampling=1,
        seed=42,
        show_progress=False,
    )
    np.testing.assert_allclose(fit.stan_variable("speed_state")[0], [0.1, 0.3, 0.7])
    np.testing.assert_allclose(
        fit.stan_variable("turn_rate_state")[0], [0.02, 0.018, 0.024]
    )
    np.testing.assert_allclose(
        fit.stan_variable("heading_state")[0], [0.3, 0.32, 0.392, 0.44]
    )


@pytest.mark.skipif(
    not RUN_CMDSTAN_INTEGRATION,
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to verify the physical posterior density.",
)
def test_noncentered_stan_preserves_physical_density_including_both_speed_branches():
    window = replace(
        _dynamic_ctrv_window(),
        time_seconds=np.array([0.0, 1.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0]),
        observation_count=3,
    )
    data = ctrv_model.build_stan_data(window)
    data["process_reference_interval_seconds"] = 1.0
    model = batch_inference.compile_bayesian_ctrv_model()
    actual, expected = [], []
    for speeds, turns, sigma_speed, sigma_turn, sigma_observation in (
        ([0.1, 0.3], [0.02, 0.018], 0.2, 0.01, 1.0),
        ([0.5, 0.7], [-0.04, -0.02], 0.4, 0.02, 0.8),
    ):
        parameters = dict(
            x_initial=0.0,
            y_initial=0.0,
            speed_initial=speeds[0],
            heading_initial=0.3,
            turn_rate_initial=turns[0],
            turn_rate_innovation=[(turns[1] - turns[0]) / sigma_turn],
            sigma_speed_process=sigma_speed,
            sigma_turn_rate_process=sigma_turn,
            sigma_position_observation_excess=sigma_observation,
        )
        # Both preimages contribute to the folded physical speed density.
        branches = []
        for signed_speed in (speeds[1], -speeds[1]):
            parameters["speed_innovation"] = [(signed_speed - speeds[0]) / sigma_speed]
            branches.append(
                model.log_prob(parameters, data=data, jacobian=False, sig_figs=18)[
                    "lp__"
                ].iloc[0]
            )
        actual.append(logsumexp(branches) - np.log(sigma_speed) - np.log(sigma_turn))

        # Independently evaluate the original physical-state model.
        state = np.array([[0.0, 0.0, speeds[0], 0.3, turns[0]]])
        positions = [state[0, :2].copy()]
        for index, dt in enumerate((1.0, 4.0)):
            state[0, 2], state[0, 4] = speeds[index], turns[index]
            state = ctrv_dynamics.transition_states(state, dt)
            positions.append(state[0, :2].copy())
        positions = np.asarray(positions)
        observation_scale = sigma_observation
        density = norm.logpdf(speeds[0], scale=data["speed_prior_scale"])
        density += norm.logpdf(turns[0], scale=data["turn_rate_prior_scale"])
        density += logsumexp(
            norm.logpdf([speeds[1], -speeds[1]], loc=speeds[0], scale=sigma_speed)
        )
        density += norm.logpdf(turns[1], loc=turns[0], scale=sigma_turn)
        for name, value in (
            ("sigma_speed_process", sigma_speed),
            ("sigma_turn_rate_process", sigma_turn),
            ("sigma_position_observation", sigma_observation),
        ):
            density += expon.logpdf(value, scale=1 / data[f"{name}_prior_rate"])
        density += norm.logpdf(
            data["x_observed"], loc=positions[:, 0], scale=observation_scale
        ).sum()
        density += norm.logpdf(
            data["y_observed"], loc=positions[:, 1], scale=observation_scale
        ).sum()
        expected.append(density)
    # Density differences cancel parameter-independent Stan normalization constants.
    assert actual[1] - actual[0] == pytest.approx(expected[1] - expected[0], abs=1e-8)


@pytest.mark.parametrize(
    ("proposal", "expected_speed"),
    ((4.0, 4.0), (-2.0, 2.0)),
)
def test_speed_reflection_preserves_heading(proposal, expected_speed):
    heading = 0.4
    states = np.asarray([[1.0, 2.0, proposal, heading, 0.01]])

    ctrv_dynamics.normalize_states(states)

    assert states[0, 2] == pytest.approx(expected_speed)
    assert states[0, 3] == pytest.approx(heading)


def test_speed_reflection_transforms_covariance_without_rotating_heading():
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

    transition_count = window.observation_count - 1
    assert initial_values["speed_innovation"].shape == (transition_count - 1,)
    assert initial_values["turn_rate_innovation"].shape == (transition_count - 1,)
    assert np.isfinite(initial_values["x_initial"])
    assert np.isfinite(initial_values["y_initial"])
    assert "x_true" not in initial_values
    assert "y_true" not in initial_values
    assert "sigma_motion_process" not in initial_values
    assert initial_values["speed_initial"] >= ctrv_dynamics.SPEED_STATE_LOWER_MPS
    assert initial_values["sigma_speed_process"] > 0
    assert initial_values["sigma_turn_rate_process"] > 0


def test_noncentered_initials_preserve_physical_states_with_irregular_time_gaps():
    window = replace(
        _dynamic_ctrv_window(),
        time_seconds=np.array([0.0, 1.0, 5.0, 14.0, 30.0, 31.0, 32.0, 33.0]),
    )
    data = ctrv_model.build_stan_data(window)
    data["process_reference_interval_seconds"] = 2.0
    initial = batch_inference._default_initial_values(data, seed=42)
    scales = np.sqrt(np.array([1.0, 4.0, 9.0]) / 2.0)
    speeds = [initial["speed_initial"]]
    turns = [initial["turn_rate_initial"]]
    for scale, speed_change, turn_change in zip(
        scales,
        initial["speed_innovation"],
        initial["turn_rate_innovation"],
        strict=True,
    ):
        speeds.append(
            abs(speeds[-1] + initial["sigma_speed_process"] * scale * speed_change)
        )
        turns.append(
            turns[-1] + initial["sigma_turn_rate_process"] * scale * turn_change
        )
    # Seeded physical initials without a fixed observation-noise floor.
    np.testing.assert_allclose(
        speeds,
        [7.317002917277321, 7.306634763552525, 7.2340047312170315, 7.342293507722877],
    )
    np.testing.assert_allclose(
        turns,
        [
            0.009401044223760728,
            0.011588165842882088,
            0.009223676086041936,
            0.012495930969340315,
        ],
    )


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
    return smc.SequentialMonteCarloCTRVFilter(
        config=smc.SequentialMonteCarloCTRVConfig(
            particle_count=particle_count,
            posterior_draw_count=draw_count,
        ),
        parameter_particles=parameter_particles,
        weights=np.full(particle_count, 1.0 / particle_count),
        state_particles=state_means,
        generator=np.random.default_rng(42),
        last_observation_time_seconds=0.0,
        processed_observation_count=1,
    )


def test_smc_parameter_particles_contain_only_remaining_noise_scales():
    online_filter = _controlled_online_filter(
        observation_noise=1.0,
        speed_process=0.5,
        turn_rate_process=0.01,
        draw_count=32,
    )

    assert online_filter.parameter_particles.shape == (128, 3)
    assert len(smc._parameter_values(online_filter.parameter_particles)) == 3


def test_smc_forecast_evolves_motion_after_the_completed_interval():
    future_times = np.asarray([10.0, 20.0])
    low_fit = _controlled_online_filter(
        observation_noise=0.1,
        speed_process=1e-9,
        turn_rate_process=1e-9,
    ).forecast(future_times, seed=43)
    high_fit = _controlled_online_filter(
        observation_noise=0.1,
        speed_process=2.0,
        turn_rate_process=1e-9,
    ).forecast(future_times, seed=43)

    np.testing.assert_allclose(
        low_fit.stan_variable("speed_at_origin"),
        high_fit.stan_variable("speed_at_origin"),
    )
    np.testing.assert_allclose(
        low_fit.stan_variable("x_prediction")[:, 0],
        high_fit.stan_variable("x_prediction")[:, 0],
    )
    assert not np.allclose(
        low_fit.stan_variable("x_prediction")[:, 1],
        high_fit.stan_variable("x_prediction")[:, 1],
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
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to run CmdStan VI.",
)
def test_ten_point_vi_does_not_collapse_into_observation_noise_mode():
    window = _ten_point_vi_regression_window()
    observations = position_observations.simulate_position_observations(
        window,
        position_noise_std_m=5.0,
        seed=2026,
    )
    fit = batch_inference.fit_bayesian_ctrv_model(
        window,
        position_observations=observations,
        inference_method="vi",
        seed=42,
        **inference.create_default_vi_config(),
    )

    observation_noise_median = np.median(
        reporting.posterior_variable_samples(
            fit,
            "sigma_position_observation",
        )
    )
    evaluation = metrics.evaluate_position_predictions(
        fit,
        window,
        credible_interval=0.9,
        position_variable_names=("x_prediction", "y_prediction"),
    )

    assert observation_noise_median < 20.0
    assert evaluation.ade_m < 100.0


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


@pytest.mark.skipif(
    not RUN_CMDSTAN_INTEGRATION,
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to run two-position batch fits.",
)
@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_two_position_batch_ctrv_integration(method):
    window = replace(_dynamic_ctrv_window(), observation_count=2)
    fit = batch_inference.fit_bayesian_ctrv_model(
        window,
        inference_method=method,
        iter=2_000,
        draws=64,
        grad_samples=20,
        elbo_samples=50,
        adapt_iter=50,
        tol_rel_obj=0.05,
        chains=1,
        parallel_chains=1,
        iter_warmup=100,
        iter_sampling=64,
        require_converged=False,
        seed=42,
    )

    _assert_dynamic_batch_fit_interface(fit, prediction_count=window.prediction_count)
    assert reporting.posterior_variable_samples(fit, "speed_state").shape[-1] == 1


@pytest.mark.skipif(
    not RUN_CMDSTAN_INTEGRATION,
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to run the failing BR24 MCMC fit.",
)
def test_low_speed_br24_mcmc_initialization_regression():
    window, observations, priors = _low_speed_br24_regression_case()
    fit = batch_inference.fit_bayesian_ctrv_model(
        window,
        position_observations=observations,
        priors=priors,
        inference_method="mcmc",
        seed=1073266673,
        mcmc_options={"show_progress": False},
        **inference.create_default_mcmc_config(),
    )

    _assert_dynamic_batch_fit_interface(fit, prediction_count=1)
    assert np.all(reporting.posterior_variable_samples(fit, "speed_state") > 0.0)
