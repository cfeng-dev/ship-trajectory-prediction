"""Tests for deterministic terminal motion in the hybrid Bayesian CTRV model."""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from ship_trajectory_prediction.models import hybrid_bayesian_ctrv as model_module
from ship_trajectory_prediction.models.deterministic_ctrv import CTRVState
from ship_trajectory_prediction.models.hybrid_bayesian_ctrv import (
    FINAL_MOTION_HISTORY_SECONDS,
    STAN_FILE,
    HybridBayesianCTRVConfig,
    HybridBayesianCTRVPriors,
    build_stan_data,
    estimate_final_motion_from_positions,
    fit_hybrid_bayesian_ctrv_model,
)
from ship_trajectory_prediction.observations import prepare_trajectory_window
from ship_trajectory_prediction.simulation.synthetic_ctrv import (
    SyntheticCTRVNoise,
    simulate_synthetic_ctrv_data,
)
from ship_trajectory_prediction.validation.reporting import (
    posterior_variable_samples,
)


def create_synthetic_window(*, turn_rate=0.012, variable_dt=False):
    """Create one noise-free curved window for hybrid interface tests."""
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
            speed=3.0,
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


def test_default_process_prior_scales_match_empirical_calibration():
    """Hybrid defaults should retain the calibrated process diffusion scales."""
    priors = HybridBayesianCTRVPriors()

    assert priors.sigma_position_process_prior_scale == pytest.approx(0.534)
    assert priors.sigma_speed_process_prior_scale == pytest.approx(0.0438)
    assert not hasattr(priors, "sigma_position_gps_prior_scale")


def test_hybrid_data_contains_deterministic_terminal_motion():
    """Hybrid Stan data should include the smoothed endpoint motion."""
    stan_data = build_stan_data(create_synthetic_window(variable_dt=True))

    assert stan_data["heading"] == pytest.approx(0.552, abs=1e-3)
    assert stan_data["turn_rate"] == pytest.approx(0.012, abs=1e-3)
    assert stan_data["sigma_position_observation"] == pytest.approx(2.0)


def test_hybrid_stan_data_omits_latent_turn_rate_priors():
    """Hybrid data should contain no priors for a latent turn-rate path."""
    stan_data = build_stan_data(
        create_synthetic_window(turn_rate=-0.012),
        priors=HybridBayesianCTRVPriors(),
    )

    assert "turn_rate_initial_prior_mean" not in stan_data
    assert "turn_rate_state_prior_scale" not in stan_data
    assert "sigma_turn_rate_process_prior_scale" not in stan_data


def test_hybrid_initial_values_omit_latent_turn_rate_parameters():
    """Hybrid inference should initialize only parameters present in its model."""
    stan_data = build_stan_data(create_synthetic_window())

    initial_values = model_module.bayesian_model._default_initial_values(
        stan_data,
        seed=15,
    )

    assert "turn_rate_state" not in initial_values
    assert "sigma_turn_rate_process" not in initial_values
    assert "sigma_position_gps" not in initial_values
    assert "heading_final" not in initial_values


def test_final_motion_recovers_quadratic_endpoint_velocity_and_turn_rate():
    """Quadratic position fits should provide deterministic terminal motion."""
    time_seconds = np.arange(5.0)
    heading, turn_rate = estimate_final_motion_from_positions(
        time_seconds,
        2.0 * time_seconds,
        0.5 * np.square(time_seconds),
    )

    assert heading == pytest.approx(np.arctan2(4.0, 2.0))
    assert turn_rate == pytest.approx(0.1)


def test_final_motion_ignores_positions_older_than_history_span():
    """Terminal motion should use time span rather than observation count."""
    recent_time = np.arange(
        -FINAL_MOTION_HISTORY_SECONDS,
        10.0,
        10.0,
    )
    time_seconds = np.concatenate(
        (([-FINAL_MOTION_HISTORY_SECONDS - 10.0]), recent_time)
    )
    x_meters = np.concatenate(
        (
            [10_000.0],
            500.0 + 4.0 * recent_time - 0.002 * np.square(recent_time),
        )
    )
    y_meters = np.concatenate(
        (
            [-10_000.0],
            1_000.0 + 3.0 * recent_time + 0.004 * np.square(recent_time),
        )
    )

    heading, turn_rate = estimate_final_motion_from_positions(
        time_seconds,
        x_meters,
        y_meters,
    )

    assert heading == pytest.approx(np.arctan2(3.0, 4.0))
    assert turn_rate == pytest.approx(0.00176)


def test_final_motion_uses_zero_turn_rate_at_insufficient_fitted_speed():
    """Slow terminal motion should not turn based on position jitter."""
    heading, turn_rate = estimate_final_motion_from_positions(
        np.arange(5.0),
        [0.0, 0.2, 0.4, 0.6, 0.8],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    )

    assert heading == pytest.approx(0.0)
    assert turn_rate == pytest.approx(0.0)


def test_final_motion_uses_configured_minimum_speed():
    """The hybrid speed guard should be configurable per experiment."""
    _, turn_rate = estimate_final_motion_from_positions(
        np.arange(5.0),
        2.0 * np.arange(5.0),
        0.5 * np.square(np.arange(5.0)),
        hybrid_config=HybridBayesianCTRVConfig(
            min_final_motion_speed_mps=10.0,
        ),
    )

    assert turn_rate == pytest.approx(0.0)


@pytest.mark.parametrize(
    "settings",
    [
        {"final_motion_history_seconds": 0.0},
        {"min_final_motion_speed_mps": 0.0},
    ],
)
def test_hybrid_configuration_requires_positive_values(settings):
    """Hybrid-specific motion settings should reject invalid values."""
    with pytest.raises(ValueError, match="positive finite"):
        HybridBayesianCTRVConfig(**settings)


def test_fit_forwards_hybrid_configuration_to_stan_data(monkeypatch):
    """The fitting wrapper should use the selected terminal-motion settings."""
    captured = {}
    expected_fit = object()

    def fake_fit(window, **kwargs):
        captured.update(kwargs)
        captured["window"] = window
        return expected_fit

    monkeypatch.setattr(
        model_module.bayesian_model,
        "fit_bayesian_ctrv_model",
        fake_fit,
    )
    window = create_synthetic_window(variable_dt=True)
    hybrid_config = HybridBayesianCTRVConfig(min_final_motion_speed_mps=10.0)

    fit = fit_hybrid_bayesian_ctrv_model(
        window,
        hybrid_config=hybrid_config,
    )
    stan_data = captured["_stan_data_builder"](window)

    assert fit is expected_fit
    assert captured["window"] is window
    assert stan_data["turn_rate"] == pytest.approx(0.0)


def test_final_motion_falls_back_to_two_point_heading():
    """Two observations should define heading with neutral turn rate."""
    heading, turn_rate = estimate_final_motion_from_positions(
        [0.0, 10.0],
        [0.0, 0.0],
        [0.0, 5.0],
    )

    assert heading == pytest.approx(np.pi / 2)
    assert turn_rate == 0.0


def test_hybrid_stan_model_uses_only_deterministic_turn_rate():
    """The hybrid should use fixed turn rate in its history and forecast."""
    source = STAN_FILE.read_text(encoding="utf-8")
    data_block = re.search(r"data\s*\{(.*?)\n\}", source, flags=re.DOTALL).group(1)
    parameter_block = re.search(
        r"parameters\s*\{(.*?)\n\}",
        source,
        flags=re.DOTALL,
    ).group(1)

    assert "real heading;" in data_block
    assert "real turn_rate;" in data_block
    assert "real<lower=1e-6> sigma_position_observation;" in data_block
    assert "sigma_position_gps_prior_scale" not in data_block
    assert "turn_rate_state_prior_scale" not in data_block
    assert "sigma_turn_rate_process_prior_scale" not in data_block
    assert "real heading;" not in parameter_block
    assert "real turn_rate;" not in parameter_block
    assert "turn_rate_state" not in parameter_block
    assert "sigma_turn_rate_process" not in parameter_block
    assert "sigma_position_gps" not in parameter_block
    assert "turn_rate_state" not in source
    assert "heading_final" not in source
    assert "turn_rate_final" not in source
    assert "heading_state[n + 1] - turn_rate * dt" in source
    assert "turn_rate_state ~ normal" not in source
    assert "turn_rate_forecast_origin" not in source
    assert "real turn_rate_previous = turn_rate;" in source
    assert "real turn_rate_current = turn_rate_previous" in source
    assert "normal_rng(turn_rate_previous" not in source


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_CMDSTAN_INTEGRATION") != "1",
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to run CmdStan inference.",
)
def test_small_hybrid_vi_fit_is_executable():
    """The separate hybrid model should compile and return finite draws."""
    fit = fit_hybrid_bayesian_ctrv_model(
        create_synthetic_window(),
        iter=5_000,
        draws=100,
        seed=15,
        require_converged=False,
    )

    assert posterior_variable_samples(fit, "x_state").shape == (100, 7)
    assert posterior_variable_samples(fit, "turn_rate_prediction").shape == (
        100,
        3,
    )
