"""Tests for the parametric Bayesian CTRV rolling evaluation."""

from types import SimpleNamespace

import numpy as np
import pytest

import experiments.model_evaluation.bayesian_ctrv as experiment
import ship_trajectory_prediction.models.bayesian_ctrv as model
import ship_trajectory_prediction.validation.bayesian_ctrv_workflow as workflow
from ship_trajectory_prediction.models.deterministic_ctrv import CTRVState
from ship_trajectory_prediction.observations import prepare_trajectory_window
from ship_trajectory_prediction.simulation.synthetic_ctrv import (
    SyntheticCTRVNoise,
    simulate_synthetic_ctrv_data,
)


def create_route():
    """Return one exact CTRV route long enough for K=20."""
    return simulate_synthetic_ctrv_data(
        count=23,
        dt_seconds=5.0,
        initial_state=CTRVState(
            x=0.0,
            y=0.0,
            speed=3.0,
            heading=0.3,
            turn_rate=0.01,
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


def test_main_applies_k_without_mutating_visible_configuration(monkeypatch):
    """The rolling entry point should pass an independent K=10 override."""
    captured = {}
    expected = object()
    monkeypatch.setattr(
        experiment.workflow,
        "run_bayesian_ctrv_evaluation",
        lambda **kwargs: captured.update(kwargs) or expected,
    )

    result = experiment.main(
        [
            "--history-positions",
            "10",
            "--turn-rate-prior-scale",
            "0.003",
        ]
    )

    assert result is expected
    assert experiment.EXPERIMENT.history_position_count == 20
    assert captured["options"].history_position_count == 10
    assert captured["options"].turn_rate_prior_scale == pytest.approx(0.003)


def test_k_comparison_reuses_route_noise_full_window_and_future_times():
    """Both K values should share one physical noisy observation realization."""
    route = create_route()
    window = prepare_trajectory_window(
        route,
        observation_count=20,
        prediction_count=3,
    )
    route_noise_x, route_noise_y = workflow._simulate_route_position_noise(
        len(route),
        additional_noise_std_m=5.0,
        seed=2026,
    )
    observations = workflow._build_window_position_observations(
        window,
        route_start_index=0,
        route_noise_x=route_noise_x,
        route_noise_y=route_noise_y,
        additional_noise_std_m=5.0,
        noise_seed=2026,
    )

    data_20 = model.build_stan_data(
        window,
        history_position_count=20,
        position_observations=observations,
    )
    data_10 = model.build_stan_data(
        window,
        history_position_count=10,
        position_observations=observations,
    )

    np.testing.assert_array_equal(data_10["x_observed"], data_20["x_observed"][-10:])
    np.testing.assert_array_equal(data_10["y_observed"], data_20["y_observed"][-10:])
    np.testing.assert_array_equal(
        data_10["time_prediction"], data_20["time_prediction"]
    )
    assert observations.noise_seed == 2026


def test_route_noise_is_seeded_once_and_reused_by_overlapping_windows():
    """The same route index must retain its noise in every rolling window."""
    route = create_route()
    route_noise_x, route_noise_y = workflow._simulate_route_position_noise(
        len(route),
        additional_noise_std_m=5.0,
        seed=2026,
    )
    first = prepare_trajectory_window(
        route,
        observation_count=10,
        prediction_count=3,
        start_index=0,
    )
    second = prepare_trajectory_window(
        route,
        observation_count=10,
        prediction_count=3,
        start_index=5,
    )
    first_observations = workflow._build_window_position_observations(
        first,
        route_start_index=0,
        route_noise_x=route_noise_x,
        route_noise_y=route_noise_y,
        additional_noise_std_m=5.0,
        noise_seed=2026,
    )
    second_observations = workflow._build_window_position_observations(
        second,
        route_start_index=5,
        route_noise_x=route_noise_x,
        route_noise_y=route_noise_y,
        additional_noise_std_m=5.0,
        noise_seed=2026,
    )

    first_x_noise = first_observations.x_meters - first.x_meters[:10]
    second_x_noise = second_observations.x_meters - second.x_meters[:10]
    np.testing.assert_allclose(first_x_noise[5:], second_x_noise[:5], atol=1e-12)


def test_diagnostics_report_only_global_motion_parameters():
    """The new rolling report should not expose removed scale diagnostics."""
    window = prepare_trajectory_window(
        create_route(),
        observation_count=20,
        prediction_count=3,
    )
    fit = FakeFit(
        speed=np.array([2.9, 3.0, 3.1]),
        heading_initial=np.array([0.2, 0.3, 0.4]),
        turn_rate=np.array([0.009, 0.01, 0.011]),
    )

    diagnostics = workflow._posterior_diagnostics(fit, window)

    assert set(diagnostics) == {
        "posterior_speed_median_mps",
        "posterior_heading_initial_median_rad",
        "posterior_turn_rate_median_rad_s",
        "forecast_heading_change_median_rad",
    }
    assert not any("process" in name for name in diagnostics)


class FakeFit:
    """Minimal posterior fit for diagnostics."""

    def __init__(self, **variables):
        self.variables = variables
        self.runset = SimpleNamespace(stdout_files=[])

    def stan_variable(self, name, mean=False):
        """Return one configured posterior variable."""
        del mean
        return self.variables[name]
