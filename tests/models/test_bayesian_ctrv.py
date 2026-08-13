"""Tests for the Bayesian CTRV state-space model interface."""

from __future__ import annotations

import os
import re
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from ship_trajectory_prediction.evaluation.reporting import (
    posterior_variable_samples,
)
from ship_trajectory_prediction.models import bayesian_ctrv as model_module
from ship_trajectory_prediction.models.bayesian_ctrv import (
    DEFAULT_VI_ADAPT_ITER,
    MIN_INITIAL_SPEED_PRIOR_SCALE_MPS,
    NOISE_PARAMETER_NAMES,
    STAN_FILE,
    BayesianCTRVPriors,
    PositionObservations,
    VIRunResult,
    build_stan_data,
    compare_vi_runs,
    diagnose_observed_turn_rate,
    estimate_initial_speed_from_positions,
    estimate_initial_speed_prior_from_windows,
    fit_bayesian_ctrv_model,
    normalize_inference_method,
    simulate_position_observations,
    summarize_predictions,
    variational_converged,
)
from ship_trajectory_prediction.models.deterministic_ctrv import CTRVState
from ship_trajectory_prediction.simulation.synthetic_ctrv import (
    SyntheticCTRVNoise,
    simulate_synthetic_ctrv_data,
)
from ship_trajectory_prediction.trajectory import prepare_trajectory_window


def create_synthetic_window(*, speed=3.0, turn_rate=0.012, variable_dt=False):
    """Create one noise-free curved window for fast interface tests."""
    time_seconds = None
    if variable_dt:
        time_seconds = [0.0, 2.0, 5.0, 9.0, 12.0, 17.0, 21.0, 27.0, 32.0, 38.0]
    data = simulate_synthetic_ctrv_data(
        count=10,
        dt_seconds=5.0,
        time_seconds=time_seconds,
        initial_state=CTRVState(
            x=0.0,
            y=0.0,
            speed=speed,
            heading=0.3,
            turn_rate=turn_rate,
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
    return prepare_trajectory_window(
        data,
        observation_count=7,
        prediction_count=3,
    )


def create_linear_window(*, speed=3.0):
    """Create a window with exact position-only constant-speed motion."""
    window = create_synthetic_window(speed=speed, turn_rate=0.0)
    return replace(
        window,
        x_meters=speed * window.time_seconds,
        y_meters=np.zeros_like(window.y_meters),
    )


def test_build_stan_data_contains_only_position_history_and_future_times():
    """Stan data should expose position history but no GPS-speed input."""
    window = create_synthetic_window(variable_dt=True)

    stan_data = build_stan_data(window)

    assert stan_data["N_observed"] == 7
    assert stan_data["N_prediction"] == 3
    assert stan_data["time_observed"] == pytest.approx(
        [0.0, 2.0, 5.0, 9.0, 12.0, 17.0, 21.0]
    )
    assert stan_data["time_prediction"] == pytest.approx([27.0, 32.0, 38.0])
    assert "speed_observed" not in stan_data
    assert "sigma_speed_gps_prior_scale" not in stan_data
    assert stan_data["speed_initial_prior_mean"] == 0.0
    assert stan_data["turn_rate_initial_prior_mean"] == 0.0
    assert stan_data["turn_rate_state_prior_scale"] == pytest.approx(0.002)
    assert "heading_final" not in stan_data
    assert "turn_rate_final" not in stan_data
    assert "heading_final_prior_mean" not in stan_data
    assert "heading_final_prior_scale" not in stan_data
    assert "heading_initial_prior_mean" not in stan_data
    assert "turn_rate_limit" not in stan_data
    assert "x_state_prediction" not in stan_data
    assert "y_state_prediction" not in stan_data
    assert "speed_state_prediction" not in stan_data
    initial_values = model_module._default_initial_values(stan_data, seed=19)
    assert -np.pi <= initial_values["heading_final"] <= np.pi
    assert initial_values["turn_rate_state"].shape == (6,)


def test_simulate_position_observations_is_reproducible_and_keeps_window_clean():
    """Artificial noise should be deterministic and remain outside the window."""
    window = create_synthetic_window()
    clean_x = window.x_meters.copy()
    clean_y = window.y_meters.copy()

    first = simulate_position_observations(
        window,
        additional_noise_std_m=2.0,
        seed=2026,
    )
    second = simulate_position_observations(
        window,
        additional_noise_std_m=2.0,
        seed=2026,
    )
    different_seed = simulate_position_observations(
        window,
        additional_noise_std_m=2.0,
        seed=2027,
    )

    assert first.x_meters == pytest.approx(second.x_meters)
    assert first.y_meters == pytest.approx(second.y_meters)
    assert not np.array_equal(first.x_meters, different_seed.x_meters)
    assert not np.array_equal(first.y_meters, different_seed.y_meters)
    assert window.x_meters == pytest.approx(clean_x)
    assert window.y_meters == pytest.approx(clean_y)
    assert not first.x_meters.flags.writeable
    assert not first.y_meters.flags.writeable


def test_zero_additional_noise_preserves_observed_positions():
    """A zero standard deviation should provide a clean opt-out path."""
    window = create_synthetic_window()

    observations = simulate_position_observations(
        window,
        additional_noise_std_m=0.0,
        seed=2026,
    )

    assert observations.x_meters == pytest.approx(
        window.x_meters[window.observed_slice]
    )
    assert observations.y_meters == pytest.approx(
        window.y_meters[window.observed_slice]
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"additional_noise_std_m": -1.0}, "additional_noise_std_m"),
        ({"additional_noise_std_m": np.nan}, "additional_noise_std_m"),
        ({"seed": -1}, "seed"),
        ({"seed": 1.5}, "seed"),
    ],
)
def test_simulate_position_observations_rejects_invalid_noise_options(
    options,
    message,
):
    """Noise settings should fail clearly before any random data are generated."""
    with pytest.raises(ValueError, match=message):
        simulate_position_observations(create_synthetic_window(), **options)


def test_stan_data_and_derived_priors_use_supplied_position_observations_only():
    """Clean observed coordinates must not leak into noise-augmented fitting."""
    window = create_synthetic_window()
    observations = simulate_position_observations(
        window,
        additional_noise_std_m=2.0,
        seed=2026,
    )
    changed_x = window.x_meters.copy()
    changed_y = window.y_meters.copy()
    changed_x[window.observed_slice] += np.linspace(100.0, 700.0, 7)
    changed_y[window.observed_slice] -= np.linspace(50.0, 350.0, 7)
    changed_window = replace(window, x_meters=changed_x, y_meters=changed_y)

    reference_data = build_stan_data(
        window,
        position_observations=observations,
    )
    changed_data = build_stan_data(
        changed_window,
        position_observations=observations,
    )

    assert reference_data["x_observed"] == pytest.approx(observations.x_meters)
    assert reference_data["y_observed"] == pytest.approx(observations.y_meters)
    assert reference_data.keys() == changed_data.keys()
    for key in reference_data:
        assert changed_data[key] == pytest.approx(reference_data[key])


def test_build_stan_data_rejects_observations_from_a_different_window():
    """Externally supplied positions must match the window's observed timestamps."""
    window = create_synthetic_window()
    observations = PositionObservations(
        time_seconds=window.time_seconds[window.observed_slice] + 1.0,
        x_meters=window.x_meters[window.observed_slice],
        y_meters=window.y_meters[window.observed_slice],
        additional_noise_std_m=0.0,
        noise_seed=0,
    )

    with pytest.raises(ValueError, match="observed timestamps"):
        build_stan_data(window, position_observations=observations)


def test_build_stan_data_does_not_use_held_out_measurements():
    """Changing future GPS positions and speed must leave Stan data unchanged."""
    data = simulate_synthetic_ctrv_data(count=10, seed=9)
    reference_window = prepare_trajectory_window(
        data,
        observation_count=7,
        prediction_count=3,
    )
    changed = data.copy()
    changed.loc[7:, "gps_longitude"] += 2.0
    changed.loc[7:, "gps_latitude"] -= 2.0
    changed.loc[7:, "gps_speed"] = np.nan
    changed_window = prepare_trajectory_window(
        changed,
        observation_count=7,
        prediction_count=3,
    )

    reference_data = build_stan_data(reference_window)
    changed_data = build_stan_data(changed_window)

    assert reference_data.keys() == changed_data.keys()
    for key in reference_data:
        assert changed_data[key] == pytest.approx(reference_data[key])
    reference_initials = model_module._default_initial_values(reference_data, seed=19)
    changed_initials = model_module._default_initial_values(changed_data, seed=19)
    for name, values in reference_initials.items():
        assert changed_initials[name] == pytest.approx(values)


@pytest.mark.parametrize(
    "scale_name",
    [
        "position_initial_prior_scale",
        "speed_initial_prior_scale",
        "turn_rate_state_prior_scale",
        "sigma_position_gps_prior_scale",
        "sigma_position_process_prior_scale",
        "sigma_speed_process_prior_scale",
        "sigma_turn_rate_process_prior_scale",
    ],
)
def test_build_stan_data_rejects_non_positive_prior_scales(scale_name):
    """Every configurable prior scale should be positive and finite."""
    with pytest.raises(ValueError, match=scale_name):
        BayesianCTRVPriors(**{scale_name: 0.0})


def test_prior_configuration_rejects_negative_initial_speed_mean():
    """The configured center of a non-negative speed must not be negative."""
    with pytest.raises(ValueError, match="speed_initial_prior_mean"):
        BayesianCTRVPriors(speed_initial_prior_mean=-0.1)


def test_build_stan_data_uses_typed_prior_configuration():
    """One immutable configuration should supply every Stan prior scale."""
    priors = BayesianCTRVPriors(
        position_initial_prior_scale=4.0,
        speed_initial_prior_mean=2.5,
        speed_initial_prior_scale=0.6,
        turn_rate_initial_prior_mean=-0.003,
        turn_rate_state_prior_scale=0.004,
        sigma_position_gps_prior_scale=3.0,
        sigma_position_process_prior_scale=0.3,
        sigma_speed_process_prior_scale=0.04,
        sigma_turn_rate_process_prior_scale=0.0008,
    )

    stan_data = build_stan_data(create_synthetic_window(), priors=priors)

    for prior_field in fields(priors):
        field_name = prior_field.name
        assert stan_data[field_name] == pytest.approx(getattr(priors, field_name))


@pytest.mark.parametrize("prior_mean", [0.012, -0.012])
def test_turn_rate_prior_uses_configured_signed_center(prior_mean):
    """The full Bayesian prior center must not be learned from current data."""
    stan_data = build_stan_data(
        create_synthetic_window(turn_rate=-prior_mean),
        priors=BayesianCTRVPriors(turn_rate_initial_prior_mean=prior_mean),
    )

    assert stan_data["turn_rate_initial_prior_mean"] == pytest.approx(prior_mean)


def test_turn_rate_prior_is_zero_for_straight_motion():
    """A straight trajectory should center the initial turn rate at zero."""
    stan_data = build_stan_data(create_synthetic_window(turn_rate=0.0))

    assert stan_data["turn_rate_initial_prior_mean"] == pytest.approx(0.0, abs=1e-9)


def test_turn_rate_prior_center_must_be_finite():
    """A signed turn-rate center may be negative but never non-finite."""
    with pytest.raises(ValueError, match="turn_rate_initial_prior_mean"):
        BayesianCTRVPriors(turn_rate_initial_prior_mean=np.inf)


def test_turn_rate_diagnostics_are_robust_and_configurable():
    """Observed course changes should set a robust configurable state prior."""
    window = create_synthetic_window(turn_rate=0.012)

    derived = diagnose_observed_turn_rate(window)
    configured = diagnose_observed_turn_rate(
        window,
        turn_rate_state_prior_scale=0.004,
    )

    assert derived.sample_count == 5
    assert derived.median_rad_s == pytest.approx(0.012, abs=1e-8)
    assert derived.q90_absolute_rad_s == pytest.approx(0.012, abs=1e-8)
    assert derived.prior_scale_rad_s == pytest.approx(0.002)
    assert configured.prior_scale_rad_s == pytest.approx(0.004)


def test_turn_rate_diagnostics_do_not_clip_prior_center():
    """The unbounded state prior should retain a large observed turn rate."""
    diagnostics = diagnose_observed_turn_rate(
        create_synthetic_window(turn_rate=0.12),
    )

    assert diagnostics.median_rad_s == pytest.approx(0.12, abs=1e-8)


def test_turn_rate_diagnostics_reject_invalid_regularization():
    """Turn-rate prior scales must remain positive and finite."""
    with pytest.raises(ValueError, match="turn_rate_state_prior_scale"):
        diagnose_observed_turn_rate(
            create_synthetic_window(),
            turn_rate_state_prior_scale=0.0,
        )


def test_stationary_observations_use_neutral_heading_and_turn_rate_centers():
    """A stationary window should remain fit-ready without a false GPS course."""
    window = create_synthetic_window()
    stationary_window = replace(
        window,
        x_meters=np.zeros_like(window.x_meters),
        y_meters=np.zeros_like(window.y_meters),
    )

    stan_data = build_stan_data(stationary_window)

    assert "heading_final" not in stan_data
    assert "turn_rate_final" not in stan_data
    assert stan_data["turn_rate_initial_prior_mean"] == 0.0


def test_initial_speed_is_estimated_from_straight_position_motion():
    """Uniform straight displacement should recover its segment speed."""
    speed = estimate_initial_speed_from_positions(
        [0.0, 2.0, 4.0, 6.0, 8.0],
        [0.0, 6.0, 12.0, 18.0, 24.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    )

    assert speed == pytest.approx(3.0)


def test_initial_speed_supports_variable_time_intervals():
    """Local linear regression should handle irregular timestamps."""
    speed = estimate_initial_speed_from_positions(
        [0.0, 2.0, 5.0, 9.0],
        [0.0, 4.0, 10.0, 18.0],
        [0.0, 0.0, 0.0, 0.0],
        point_count=4,
    )

    assert speed == pytest.approx(2.0)


def test_initial_speed_retains_small_displacements_without_sensor_metadata():
    """Unknown sensor resolution must not impose an arbitrary zero threshold."""
    speed = estimate_initial_speed_from_positions(
        [0.0, 10.0, 20.0, 30.0],
        [0.0, 0.3, -0.2, 0.4],
        [0.0, -0.2, 0.1, 0.0],
        point_count=4,
    )

    assert speed == pytest.approx(np.hypot(0.007, 0.003))


@pytest.mark.parametrize(
    "time_seconds",
    ([0.0, 10.0, 10.0], [0.0, 10.0, 5.0], [0.0, np.nan, 20.0]),
)
def test_initial_speed_rejects_invalid_or_non_increasing_times(time_seconds):
    """Every position-derived segment requires a finite positive duration."""
    with pytest.raises(ValueError, match="time_seconds"):
        estimate_initial_speed_from_positions(
            time_seconds,
            [0.0, 2.0, 4.0],
            [0.0, 0.0, 0.0],
            point_count=3,
        )


def test_local_initial_speed_uses_only_first_configured_points():
    """Later positions must not enter the local initial-speed regression."""
    time_seconds = [0.0, 1.0, 2.0, 3.0, 4.0, -100.0]
    x_meters = [0.0, 2.0, 4.0, 6.0, 8.0, np.nan]

    local_speed = estimate_initial_speed_from_positions(
        time_seconds,
        x_meters,
        [0.0, 0.0, 0.0, 0.0, 0.0, np.nan],
        point_count=5,
    )

    assert local_speed == pytest.approx(2.0)


def test_historical_initial_speed_prior_uses_one_estimate_per_window():
    """Independent window starts should define the robust prior sample."""
    historical_windows = [
        create_linear_window(speed=2.0),
        create_linear_window(speed=4.0),
        create_linear_window(speed=6.0),
    ]

    prior = estimate_initial_speed_prior_from_windows(historical_windows)

    assert prior.window_estimates_mps == pytest.approx([2.0, 4.0, 6.0])
    assert len(prior.window_estimates_mps) == len(historical_windows)
    assert prior.mean_mps == pytest.approx(4.0)
    assert prior.scale_mps == pytest.approx(1.4826 * 2.0)


def test_historical_speed_prior_ignores_held_out_positions():
    """Changing prediction positions must leave historical statistics fixed."""
    window = create_linear_window(speed=3.0)
    changed_x = window.x_meters.copy()
    changed_y = window.y_meters.copy()
    changed_x[window.prediction_slice] += 100_000.0
    changed_y[window.prediction_slice] -= 100_000.0
    changed_window = replace(window, x_meters=changed_x, y_meters=changed_y)

    reference = estimate_initial_speed_prior_from_windows([window])
    changed = estimate_initial_speed_prior_from_windows([changed_window])

    assert changed == reference


def test_historical_speed_prior_is_robust_to_one_extreme_window():
    """Median and MAD should limit the influence of one implausible window."""
    moderate = [
        create_linear_window(speed=2.0),
        create_linear_window(speed=3.0),
        create_linear_window(speed=4.0),
    ]

    prior = estimate_initial_speed_prior_from_windows(
        [*moderate, create_linear_window(speed=1_000.0)]
    )

    assert prior.mean_mps == pytest.approx(3.5)
    assert prior.scale_mps == pytest.approx(1.4826)


def test_stationary_historical_windows_use_positive_scale_floor():
    """A stationary historical sample should yield a valid nondegenerate prior."""
    prior = estimate_initial_speed_prior_from_windows(
        [create_linear_window(speed=0.0) for _ in range(3)]
    )

    assert prior.window_estimates_mps == pytest.approx([0.0, 0.0, 0.0])
    assert prior.mean_mps == 0.0
    assert prior.scale_mps == pytest.approx(MIN_INITIAL_SPEED_PRIOR_SCALE_MPS)


def test_current_window_changes_initial_values_but_not_historical_prior():
    """Current positions may initialize numerics but never redefine the prior."""
    historical = estimate_initial_speed_prior_from_windows(
        [create_linear_window(speed=2.0), create_linear_window(speed=4.0)]
    )
    priors = BayesianCTRVPriors(
        speed_initial_prior_mean=historical.mean_mps,
        speed_initial_prior_scale=historical.scale_mps,
    )
    slow_data = build_stan_data(create_linear_window(speed=1.0), priors=priors)
    fast_data = build_stan_data(create_linear_window(speed=8.0), priors=priors)

    slow_initials = model_module._default_initial_values(slow_data, seed=19)
    fast_initials = model_module._default_initial_values(fast_data, seed=19)

    assert slow_data["speed_initial_prior_mean"] == pytest.approx(
        fast_data["speed_initial_prior_mean"]
    )
    assert slow_data["speed_initial_prior_scale"] == pytest.approx(
        fast_data["speed_initial_prior_scale"]
    )
    assert slow_initials["speed_state"][0] != pytest.approx(
        fast_initials["speed_state"][0]
    )


def test_historical_speed_prior_ignores_gps_speed():
    """GPS-speed values must not influence position-only prior calibration."""
    window = create_linear_window(speed=3.0)
    changed_window = replace(
        window,
        gps_speed_mps=np.full_like(window.gps_speed_mps, 10_000.0),
    )

    reference = estimate_initial_speed_prior_from_windows([window])
    changed = estimate_initial_speed_prior_from_windows([changed_window])

    assert changed == reference


def test_historical_speed_prior_requires_enough_observed_points():
    """The configured local regression length must fit every history window."""
    short_window = replace(create_linear_window(), observation_count=4)

    with pytest.raises(ValueError, match="at least 5"):
        estimate_initial_speed_prior_from_windows([short_window])


@pytest.mark.parametrize("external_speed", [np.nan, -0.1, 500.0])
def test_build_stan_data_ignores_external_gps_speed(external_speed):
    """GPS speed quality must not affect position-only Stan inputs."""
    window = create_synthetic_window()
    speed = window.gps_speed_mps.copy()
    speed[:] = external_speed
    changed_window = replace(window, gps_speed_mps=speed)

    reference_data = build_stan_data(window)
    changed_data = build_stan_data(changed_window)

    assert reference_data.keys() == changed_data.keys()
    for key in reference_data:
        assert changed_data[key] == pytest.approx(reference_data[key])
    reference_initials = model_module._default_initial_values(reference_data, seed=19)
    changed_initials = model_module._default_initial_values(changed_data, seed=19)
    for name, values in reference_initials.items():
        assert changed_initials[name] == pytest.approx(values)


def test_build_stan_data_rejects_invalid_observation_and_prediction_times():
    """Both parts of the state transition need positive time increments."""
    window = create_synthetic_window()
    invalid_observed_time = window.time_seconds.copy()
    invalid_observed_time[3] = invalid_observed_time[2]
    with pytest.raises(ValueError, match="Observed timestamps"):
        build_stan_data(replace(window, time_seconds=invalid_observed_time))

    invalid_prediction_time = window.time_seconds.copy()
    invalid_prediction_time[7] = invalid_prediction_time[6]
    with pytest.raises(ValueError, match="Prediction timestamps"):
        build_stan_data(replace(window, time_seconds=invalid_prediction_time))


def test_fit_forwards_explicit_meanfield_controls(monkeypatch):
    """The wrapper should expose the complete requested CmdStan VI interface."""
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module,
        "compile_bayesian_ctrv_model",
        lambda: fake_model,
    )

    fit = fit_bayesian_ctrv_model(
        create_synthetic_window(),
        algorithm="meanfield",
        iter=500,
        grad_samples=2,
        elbo_samples=20,
        eta=0.5,
        adapt_iter=25,
        tol_rel_obj=0.01,
        eval_elbo=50,
        draws=80,
        seed=17,
        require_converged=False,
        show_console=True,
    )

    assert fit is fake_model.result
    arguments = fake_model.arguments
    assert arguments["algorithm"] == "meanfield"
    assert arguments["iter"] == 500
    assert arguments["grad_samples"] == 2
    assert arguments["elbo_samples"] == 20
    assert arguments["eta"] == pytest.approx(0.5)
    assert arguments["adapt_iter"] == 25
    assert arguments["tol_rel_obj"] == pytest.approx(0.01)
    assert arguments["eval_elbo"] == 50
    assert arguments["draws"] == 80
    assert arguments["seed"] == 17
    assert arguments["require_converged"] is False
    assert arguments["show_console"] is True


def test_fit_uses_stable_default_vi_adaptation_length(monkeypatch):
    """The default VI fit should adapt long enough for positive speed states."""
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module,
        "compile_bayesian_ctrv_model",
        lambda: fake_model,
    )

    fit_bayesian_ctrv_model(create_synthetic_window())

    assert fake_model.arguments["adapt_iter"] == DEFAULT_VI_ADAPT_ITER


def test_fit_forwards_explicit_mcmc_controls_and_chain_initials(monkeypatch):
    """The wrapper should call NUTS with independent seeded chain initials."""
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module,
        "compile_bayesian_ctrv_model",
        lambda: fake_model,
    )

    fit = fit_bayesian_ctrv_model(
        create_synthetic_window(),
        inference_method="mcmc",
        chains=3,
        parallel_chains=2,
        iter_warmup=250,
        iter_sampling=300,
        adapt_delta=0.95,
        max_treedepth=12,
        seed=17,
        show_console=True,
    )

    assert fit is fake_model.mcmc_result
    arguments = fake_model.sample_arguments
    assert arguments["chains"] == 3
    assert arguments["parallel_chains"] == 2
    assert arguments["iter_warmup"] == 250
    assert arguments["iter_sampling"] == 300
    assert arguments["adapt_delta"] == pytest.approx(0.95)
    assert arguments["max_treedepth"] == 12
    assert arguments["seed"] == 17
    assert arguments["show_console"] is True
    assert len(arguments["inits"]) == 3
    assert not np.array_equal(
        arguments["inits"][0]["speed_state"],
        arguments["inits"][1]["speed_state"],
    )


def test_vi_and_mcmc_receive_the_same_supplied_position_observations(monkeypatch):
    """Inference selection must not regenerate the experimental input data."""
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module,
        "compile_bayesian_ctrv_model",
        lambda: fake_model,
    )
    window = create_synthetic_window()
    observations = simulate_position_observations(
        window,
        additional_noise_std_m=2.0,
        seed=2026,
    )

    fit_bayesian_ctrv_model(
        window,
        position_observations=observations,
        seed=17,
    )
    vi_data = fake_model.calls[-1]["data"]
    fit_bayesian_ctrv_model(
        window,
        position_observations=observations,
        inference_method="mcmc",
        chains=2,
        parallel_chains=2,
        seed=18,
    )
    mcmc_data = fake_model.sample_calls[-1]["data"]

    assert vi_data.keys() == mcmc_data.keys()
    for key in vi_data:
        assert mcmc_data[key] == pytest.approx(vi_data[key])
    assert vi_data["x_observed"] == pytest.approx(observations.x_meters)
    assert vi_data["y_observed"] == pytest.approx(observations.y_meters)


def test_fit_accepts_fullrank_and_reproducible_seeded_initials(monkeypatch):
    """A repeated seed should reproduce initial values for either VI family."""
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module,
        "compile_bayesian_ctrv_model",
        lambda: fake_model,
    )
    window = create_synthetic_window()

    fit_bayesian_ctrv_model(window, algorithm="fullrank", seed=31)
    first_arguments = fake_model.calls[-1]
    fit_bayesian_ctrv_model(window, algorithm="fullrank", seed=31)
    second_arguments = fake_model.calls[-1]

    assert first_arguments["algorithm"] == "fullrank"
    for name, first_value in first_arguments["inits"].items():
        assert second_arguments["inits"][name] == pytest.approx(first_value)


def test_fit_initializes_position_derived_speed_above_zero(monkeypatch):
    """Position-derived speeds must initialize above Stan's lower bound."""
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module,
        "compile_bayesian_ctrv_model",
        lambda: fake_model,
    )
    window = create_synthetic_window(speed=250.0)

    fit_bayesian_ctrv_model(window, seed=71)

    initial_speed = fake_model.arguments["inits"]["speed_state"]
    assert "speed_limit" not in fake_model.arguments["data"]
    assert np.all(initial_speed > 0)
    assert np.max(initial_speed) > 100.0


def test_gps_speed_does_not_change_seeded_vi_initial_values(monkeypatch):
    """Equal positions and times must give equal initials despite GPS speed."""
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module,
        "compile_bayesian_ctrv_model",
        lambda: fake_model,
    )
    window = create_synthetic_window()
    changed_window = replace(
        window,
        gps_speed_mps=np.linspace(50.0, 500.0, len(window.gps_speed_mps)),
    )

    fit_bayesian_ctrv_model(window, seed=71)
    reference_initials = fake_model.calls[-1]["inits"]
    fit_bayesian_ctrv_model(changed_window, seed=71)
    changed_initials = fake_model.calls[-1]["inits"]

    assert reference_initials.keys() == changed_initials.keys()
    for name, values in reference_initials.items():
        assert changed_initials[name] == pytest.approx(values)


@pytest.mark.parametrize(
    "options",
    [
        {"data": {}},
        {"seed": 1},
        {"algorithm": "fullrank"},
        {"tol_rel_obj": 0.2},
    ],
)
def test_fit_rejects_conflicting_variational_options(monkeypatch, options):
    """Generic options must not silently override the typed VI controls."""
    monkeypatch.setattr(
        model_module,
        "compile_bayesian_ctrv_model",
        lambda: FakeModel(),
    )

    with pytest.raises(ValueError, match="must not override"):
        fit_bayesian_ctrv_model(
            create_synthetic_window(),
            variational_options=options,
        )


@pytest.mark.parametrize(
    "options",
    [
        {"data": {}},
        {"seed": 1},
        {"chains": 2},
        {"adapt_delta": 0.95},
    ],
)
def test_fit_rejects_conflicting_mcmc_options(monkeypatch, options):
    """Generic MCMC options must not override typed NUTS controls."""
    monkeypatch.setattr(
        model_module,
        "compile_bayesian_ctrv_model",
        lambda: FakeModel(),
    )

    with pytest.raises(ValueError, match="must not override"):
        fit_bayesian_ctrv_model(
            create_synthetic_window(),
            inference_method="mcmc",
            mcmc_options=options,
        )


@pytest.mark.parametrize(
    ("inference_method", "option_name", "options"),
    [
        ("vi", "mcmc_options", {"show_progress": False}),
        ("mcmc", "variational_options", {"refresh": 0}),
    ],
)
def test_fit_rejects_options_for_inactive_inference_method(
    inference_method,
    option_name,
    options,
):
    """Configuration for the inactive inference method must not be ignored."""
    with pytest.raises(ValueError, match=option_name):
        fit_bayesian_ctrv_model(
            create_synthetic_window(),
            inference_method=inference_method,
            **{option_name: options},
        )


@pytest.mark.parametrize("inference_method", ["laplace", "nuts", ""])
def test_fit_rejects_unsupported_inference_methods(inference_method):
    """Only the explicit VI and MCMC interfaces should be accepted."""
    with pytest.raises(ValueError, match="inference_method"):
        fit_bayesian_ctrv_model(
            create_synthetic_window(),
            inference_method=inference_method,
        )


@pytest.mark.parametrize(
    ("inference_method", "expected"),
    [("vi", "vi"), (" VI ", "vi"), ("mcmc", "mcmc"), ("MCMC", "mcmc")],
)
def test_normalize_inference_method(inference_method, expected):
    """The shared normalizer should support CLI and programmatic spellings."""
    assert normalize_inference_method(inference_method) == expected


@pytest.mark.parametrize("inference_method", [None, "", "nuts"])
def test_normalize_inference_method_rejects_unsupported_values(inference_method):
    """The shared normalizer should reject unavailable inference methods."""
    with pytest.raises(ValueError, match="inference_method"):
        normalize_inference_method(inference_method)


@pytest.mark.parametrize(
    "arguments",
    [
        {"chains": 0},
        {"chains": 2, "parallel_chains": 3},
        {"iter_warmup": 0},
        {"iter_sampling": 0},
        {"adapt_delta": 1.0},
        {"max_treedepth": 0},
    ],
)
def test_fit_rejects_invalid_mcmc_controls(arguments):
    """NUTS controls should fail before compiling the Stan model."""
    with pytest.raises(ValueError):
        fit_bayesian_ctrv_model(
            create_synthetic_window(),
            inference_method="mcmc",
            **arguments,
        )


@pytest.mark.parametrize("algorithm", ["laplace", "mcmc", ""])
def test_fit_rejects_unsupported_inference_algorithms(algorithm):
    """This phase should expose only meanfield and fullrank VI."""
    with pytest.raises(ValueError, match="algorithm"):
        fit_bayesian_ctrv_model(create_synthetic_window(), algorithm=algorithm)


def test_summarize_predictions_includes_latent_states_and_position_observations():
    """Prediction summaries should distinguish latent and observed positions."""
    window = create_synthetic_window()
    base = np.arange(window.prediction_count, dtype=float)
    variables = {
        name: np.vstack([base + offset, base + offset + 1, base + offset + 2])
        for name, offset in (
            ("x_state_prediction", 0),
            ("y_state_prediction", 3),
            ("speed_state_prediction", 6),
            ("heading_state_prediction", 9),
            ("turn_rate_state_prediction", 12),
            ("x_observation_prediction", 15),
            ("y_observation_prediction", 18),
        )
    }

    summary = summarize_predictions(FakeFit(**variables), window, 0.8)

    assert len(summary) == window.prediction_count
    for prefix in (
        "x_state",
        "y_state",
        "speed_state",
        "heading_state",
        "turn_rate_state",
        "x_observation",
        "y_observation",
    ):
        assert {
            f"{prefix}_median",
            f"{prefix}_lower",
            f"{prefix}_upper",
        }.issubset(summary.columns)
    assert "speed_gps_reference" not in summary


def test_summarize_predictions_rejects_wrong_prediction_shape():
    """Malformed generated quantities should fail before evaluation."""
    window = create_synthetic_window()
    variables = {
        name: np.ones((5, window.prediction_count))
        for name in (
            "x_state_prediction",
            "y_state_prediction",
            "speed_state_prediction",
            "heading_state_prediction",
            "turn_rate_state_prediction",
            "x_observation_prediction",
            "y_observation_prediction",
        )
    }
    variables["heading_state_prediction"] = np.ones((5, 2))

    with pytest.raises(ValueError, match="unexpected shape"):
        summarize_predictions(FakeFit(**variables), window)


def test_compare_vi_runs_reports_noise_prediction_and_elbo_metrics(tmp_path):
    """Cross-seed comparison should expose all required stability quantities."""
    window = create_synthetic_window()
    prediction_count = window.prediction_count
    x_actual = window.x_meters[window.prediction_slice]
    y_actual = window.y_meters[window.prediction_slice]
    variables = {
        "x_observation_prediction": np.vstack([x_actual - 1, x_actual, x_actual + 1]),
        "y_observation_prediction": np.vstack([y_actual - 1, y_actual, y_actual + 1]),
        "x_state_prediction": np.vstack([x_actual - 1, x_actual, x_actual + 1]),
        "y_state_prediction": np.vstack([y_actual - 1, y_actual, y_actual + 1]),
    }
    for parameter_name in NOISE_PARAMETER_NAMES:
        variables[parameter_name] = np.asarray([0.1, 0.2, 0.3])
    stdout_path = tmp_path / "variational-stdout.txt"
    stdout_path.write_text(
        "  iter             ELBO   delta_ELBO_mean   delta_ELBO_med\n"
        "   100          -12.5             0.01             0.01\n",
        encoding="utf-8",
    )
    fit = FakeFit(stdout_path=stdout_path, **variables)

    comparison = compare_vi_runs(
        [
            VIRunResult(
                seed=4,
                algorithm="meanfield",
                fit=fit,
                runtime_seconds=1.25,
            )
        ],
        window,
    )

    assert comparison.loc[0, "final_elbo"] == pytest.approx(-12.5)
    assert comparison.loc[0, "ade_m"] == pytest.approx(0.0)
    assert comparison.loc[0, "fde_m"] == pytest.approx(0.0)
    assert comparison.loc[0, "radial_coverage"] == pytest.approx(1.0)
    assert comparison.loc[0, "sigma_position_gps_mean"] == pytest.approx(0.2)
    assert not any("sigma_speed_gps" in name for name in comparison.columns)
    assert prediction_count == 3


@pytest.mark.parametrize(("warning", "expected"), [(False, True), (True, False)])
def test_variational_converged_reads_cmdstan_status(tmp_path, warning, expected):
    """CmdStan's explicit VI warning should determine reported convergence."""
    transcript = "COMPLETED.\n"
    if warning:
        transcript = "The algorithm may not have converged.\n" + transcript
    stdout_path = tmp_path / "variational-status.txt"
    stdout_path.write_text(transcript, encoding="utf-8")

    assert variational_converged(FakeFit(stdout_path=stdout_path)) is expected


def test_synthetic_data_retains_truth_noise_and_is_seed_reproducible():
    """Synthetic validation data should retain every known latent quantity."""
    first = simulate_synthetic_ctrv_data(count=8, seed=101)
    second = simulate_synthetic_ctrv_data(count=8, seed=101)

    assert first.equals(second)
    required_columns = {
        "x_true",
        "y_true",
        "speed_true",
        "heading_true",
        "turn_rate_true",
        *(f"{name}_true" for name in NOISE_PARAMETER_NAMES),
    }
    assert required_columns.issubset(first.columns)


def test_stan_model_contains_ctrv_branches_and_variable_dt_diffusion():
    """The Stan source should retain exact, straight, and scaled transitions."""
    source = STAN_FILE.read_text(encoding="utf-8")
    data_block = re.search(r"data\s*\{(.*?)\n\}", source, flags=re.DOTALL).group(1)
    parameter_block = re.search(
        r"parameters\s*\{(.*?)\n\}",
        source,
        flags=re.DOTALL,
    ).group(1)

    assert "speed / turn_rate" in source
    assert "x + speed * dt * cos(heading)" in source
    assert "y + speed * dt * sin(heading)" in source
    assert "sigma_speed_process * sqrt(dt)" in source
    assert "sigma_turn_rate_process * sqrt(dt)" in source
    assert "vector[N_observed - 1] turn_rate_state" in source
    assert "turn_rate_limit" not in source
    assert "speed_limit" not in source
    assert "vector<lower=0>[N_observed] speed_state" in source
    assert "speed_state_raw" not in source
    assert "log1p_exp" not in source
    assert "real speed_current = fmax(" in source
    assert "real turn_rate_current = normal_rng(" in source
    assert "real heading_final" not in data_block
    assert "real turn_rate_final" not in data_block
    assert "real<lower=-pi(), upper=pi()> heading_final" in parameter_block
    assert "heading_state[N_observed] = heading_final" in source
    assert "heading_final ~ uniform(-pi(), pi())" in source
    assert "heading_final_prior" not in source
    assert "heading_initial" not in source
    assert "turn_rate_state ~ normal(turn_rate_initial_prior_mean" in source
    assert "real x_previous = x_state[N_observed]" in source
    assert "real y_previous = y_state[N_observed]" in source
    assert (
        "real turn_rate_forecast_origin = turn_rate_state[N_observed - 1]" in source
    )
    assert "real turn_rate_previous = turn_rate_forecast_origin" in source


def test_stan_ctrv_calls_never_use_observed_positions_as_transition_inputs():
    """Every fitted and future CTRV step should propagate latent positions."""
    source = STAN_FILE.read_text(encoding="utf-8")
    calls = re.findall(
        r"vector\[2\]\s+\w+\s*=\s*ctrv_position\((.*?)\);",
        source,
        flags=re.DOTALL,
    )

    assert len(calls) == 2
    assert all("x_observed" not in call for call in calls)
    assert all("y_observed" not in call for call in calls)
    assert "x_observed ~ normal(x_state, sigma_position_gps)" in source
    assert "y_observed ~ normal(y_state, sigma_position_gps)" in source


def test_stan_model_has_position_only_likelihood_and_named_predictions():
    """Stan must contain no GPS-speed likelihood or generated measurement."""
    source = STAN_FILE.read_text(encoding="utf-8")

    assert "speed_observed" not in source
    assert "sigma_speed_gps" not in source
    assert "vector[2 * N_observed] log_likelihood" in source
    for variable_name in (
        "x_state_prediction",
        "y_state_prediction",
        "speed_state_prediction",
        "heading_state_prediction",
        "turn_rate_state_prediction",
        "x_observation_prediction",
        "y_observation_prediction",
    ):
        assert f"vector[N_prediction] {variable_name}" in source


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_CMDSTAN_INTEGRATION") != "1",
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to run CmdStan inference.",
)
@pytest.mark.parametrize(
    ("algorithm", "seed", "grad_samples"),
    [("meanfield", 12, 1), ("fullrank", 13, 10)],
)
def test_small_synthetic_vi_fit_is_executable(algorithm, seed, grad_samples):
    """Both supported VI approximations should produce finite state draws."""
    fit = fit_bayesian_ctrv_model(
        create_synthetic_window(),
        algorithm=algorithm,
        iter=5_000,
        grad_samples=grad_samples,
        draws=100,
        seed=seed,
        require_converged=False,
    )

    assert posterior_samples(fit, "x_state").shape == (100, 7)
    assert posterior_samples(fit, "heading_state_prediction").shape == (100, 3)
    assert posterior_samples(fit, "speed_state_prediction").shape == (100, 3)
    assert posterior_samples(fit, "x_observation_prediction").shape == (100, 3)
    assert posterior_samples(fit, "log_likelihood").shape == (100, 14)
    assert np.all(np.isfinite(posterior_samples(fit, "sigma_position_gps")))


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_CMDSTAN_INTEGRATION") != "1",
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to run CmdStan inference.",
)
def test_small_synthetic_mcmc_fit_is_executable():
    """A small NUTS run should return finite position-only state draws."""
    fit = fit_bayesian_ctrv_model(
        create_synthetic_window(),
        inference_method="mcmc",
        chains=2,
        parallel_chains=2,
        iter_warmup=100,
        iter_sampling=100,
        adapt_delta=0.9,
        seed=14,
    )

    assert posterior_samples(fit, "x_state").shape == (200, 7)
    assert posterior_samples(fit, "speed_state_prediction").shape == (200, 3)
    assert posterior_samples(fit, "log_likelihood").shape == (200, 14)
    assert np.all(np.isfinite(posterior_samples(fit, "sigma_position_gps")))


def posterior_samples(fit, name):
    """Extract full VI or MCMC draws for the integration assertions."""
    return posterior_variable_samples(fit, name)


class FakeFit:
    """Minimal variational CmdStan result for reporting tests."""

    variational_sample = object()

    def __init__(self, stdout_path=None, **variables):
        self.variables = variables
        if stdout_path is not None:
            self.runset = SimpleNamespace(stdout_files=[str(stdout_path)])

    def stan_variable(self, name, mean=False):
        """Return one stored fake posterior variable."""
        del mean
        return self.variables[name]


class FakeModel:
    """Minimal CmdStan model that records VI and MCMC calls."""

    def __init__(self):
        self.result = object()
        self.mcmc_result = object()
        self.arguments = None
        self.calls = []
        self.sample_arguments = None
        self.sample_calls = []

    def variational(self, **kwargs):
        """Capture variational arguments and return a stable sentinel."""
        self.arguments = kwargs
        self.calls.append(kwargs)
        return self.result

    def sample(self, **kwargs):
        """Capture MCMC arguments and return a stable sentinel."""
        self.sample_arguments = kwargs
        self.sample_calls.append(kwargs)
        return self.mcmc_result
