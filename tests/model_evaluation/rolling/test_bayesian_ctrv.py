"""Tests for the Bayesian CTRV rolling-evaluation experiment."""

import numpy as np
import pytest

import experiments.model_evaluation.rolling.bayesian_ctrv as experiment
from experiments.trajectory_prediction.config import (
    normalize_bayesian_ctrv_model_variant,
)
from ship_trajectory_prediction.models.bayesian_ctrv import BayesianCTRVPriors
from ship_trajectory_prediction.models.ctrv import CTRVState
from ship_trajectory_prediction.simulation.synthetic_ctrv import (
    SyntheticCTRVNoise,
    simulate_synthetic_ctrv_data,
)
from ship_trajectory_prediction.trajectory import prepare_trajectory_window


def _create_route():
    """Return one deterministic route for overlapping rolling windows."""
    return simulate_synthetic_ctrv_data(
        count=10,
        dt_seconds=5.0,
        initial_state=CTRVState(
            x=0.0,
            y=0.0,
            speed=3.0,
            heading=0.2,
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


def test_rolling_experiment_adds_reproducible_two_meter_position_noise():
    """The rolling default should match the single-fit 2 m noise scenario."""
    assert experiment.EXPERIMENT.additional_position_noise_std_m == 2.0
    assert experiment.EXPERIMENT.position_noise_seed == 2026

    first_x, first_y = experiment._simulate_route_position_noise(
        10,
        additional_noise_std_m=(experiment.EXPERIMENT.additional_position_noise_std_m),
        seed=experiment.EXPERIMENT.position_noise_seed,
    )
    second_x, second_y = experiment._simulate_route_position_noise(
        10,
        additional_noise_std_m=(experiment.EXPERIMENT.additional_position_noise_std_m),
        seed=experiment.EXPERIMENT.position_noise_seed,
    )

    np.testing.assert_array_equal(first_x, second_x)
    np.testing.assert_array_equal(first_y, second_y)
    assert np.any(first_x != 0.0)
    assert np.any(first_y != 0.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("bayesian", "bayesian"), (" HYBRID ", "hybrid")],
)
def test_model_variant_normalization(value, expected):
    """Shared experiments should select exactly one explicit model variant."""
    assert normalize_bayesian_ctrv_model_variant(value) == expected


def test_invalid_model_variant_is_rejected():
    """Ambiguous CTRV model labels should fail before fitting."""
    with pytest.raises(ValueError, match="bayesian.*hybrid"):
        normalize_bayesian_ctrv_model_variant("automatic")


def test_overlapping_windows_reuse_the_same_route_position_noise():
    """One physical observation must retain its perturbation across windows."""
    route = _create_route()
    route_noise_x, route_noise_y = experiment._simulate_route_position_noise(
        len(route),
        additional_noise_std_m=2.0,
        seed=2026,
    )
    first_window = prepare_trajectory_window(
        route,
        observation_count=5,
        prediction_count=2,
        start_index=0,
    )
    second_window = prepare_trajectory_window(
        route,
        observation_count=5,
        prediction_count=2,
        start_index=2,
    )

    first_observations = experiment._build_window_position_observations(
        first_window,
        route_start_index=0,
        route_noise_x=route_noise_x,
        route_noise_y=route_noise_y,
        additional_noise_std_m=2.0,
        noise_seed=2026,
    )
    second_observations = experiment._build_window_position_observations(
        second_window,
        route_start_index=2,
        route_noise_x=route_noise_x,
        route_noise_y=route_noise_y,
        additional_noise_std_m=2.0,
        noise_seed=2026,
    )

    first_x_noise = (
        first_observations.x_meters - first_window.x_meters[first_window.observed_slice]
    )
    first_y_noise = (
        first_observations.y_meters - first_window.y_meters[first_window.observed_slice]
    )
    second_x_noise = (
        second_observations.x_meters
        - second_window.x_meters[second_window.observed_slice]
    )
    second_y_noise = (
        second_observations.y_meters
        - second_window.y_meters[second_window.observed_slice]
    )

    np.testing.assert_allclose(first_x_noise[2:], second_x_noise[:3])
    np.testing.assert_allclose(first_y_noise[2:], second_y_noise[:3])
    np.testing.assert_allclose(first_x_noise, route_noise_x[:5])
    np.testing.assert_allclose(second_y_noise, route_noise_y[2:7])
    assert first_observations.additional_noise_std_m == 2.0
    assert first_observations.noise_seed == 2026


def test_zero_position_noise_keeps_the_recorded_route_unchanged():
    """Disabling the option should preserve the original observations exactly."""
    route = _create_route()
    window = prepare_trajectory_window(
        route,
        observation_count=5,
        prediction_count=2,
    )
    route_noise_x, route_noise_y = experiment._simulate_route_position_noise(
        len(route),
        additional_noise_std_m=0.0,
        seed=2026,
    )
    observations = experiment._build_window_position_observations(
        window,
        route_start_index=0,
        route_noise_x=route_noise_x,
        route_noise_y=route_noise_y,
        additional_noise_std_m=0.0,
        noise_seed=2026,
    )

    np.testing.assert_array_equal(
        observations.x_meters,
        window.x_meters[window.observed_slice],
    )
    np.testing.assert_array_equal(
        observations.y_meters,
        window.y_meters[window.observed_slice],
    )


class FakeFit:
    """Provide CmdStanVB-like scalar noise draws for retry tests."""

    variational_sample = object()

    def __init__(self, *, sigma_speed_process):
        self.variables = {
            "sigma_position_gps": np.array([5.0]),
            "sigma_position_process": np.array([0.5]),
            "sigma_speed_process": np.asarray(sigma_speed_process, dtype=float),
            "sigma_turn_rate_process": np.array([0.001]),
        }

    def stan_variable(self, name, mean=False):
        """Return one stored fake posterior variable."""
        del mean
        return self.variables[name]


def test_window_diagnostics_report_deterministic_forecast_turn_rate():
    """Rolling diagnostics should report the rate actually used by the forecast."""
    fit = FakeFit(sigma_speed_process=[0.05])
    fit.variables.update(
        {
            "turn_rate_forecast_origin": np.array([0.02, 0.02]),
            "heading_state": np.array([[0.3, 0.4], [0.3, 0.4]]),
            "heading_state_prediction": np.array(
                [[0.5, 0.7], [0.5, 0.7]],
            ),
        }
    )

    diagnostics = experiment._posterior_window_diagnostics(fit)

    assert diagnostics["forecast_origin_turn_rate_rad_s"] == pytest.approx(0.02)
    assert diagnostics["forecast_heading_change_rad"] == pytest.approx(0.3)
    assert "posterior_origin_turn_rate_median_rad_s" not in diagnostics


def test_numerically_exploded_vi_fit_is_retried_with_next_seed(
    monkeypatch,
    capsys,
):
    """A catastrophic VI tail should be discarded instead of breaking the plot."""
    unstable = FakeFit(sigma_speed_process=[0.05, 1e100])
    stable = FakeFit(sigma_speed_process=[0.04, 0.06])
    fits = iter((unstable, stable))
    used_seeds = []

    def fake_fit(*args, seed, **kwargs):
        del args, kwargs
        used_seeds.append(seed)
        return next(fits)

    monkeypatch.setattr(experiment, "fit_bayesian_ctrv_model", fake_fit)

    fit, seed = experiment._fit_rolling_window(
        object(),
        priors=BayesianCTRVPriors(),
        position_observations=object(),
        inference_method="vi",
        inference_config={},
        initial_seed=62,
    )

    assert fit is stable
    assert seed == 63
    assert used_seeds == [62, 63]
    assert "retrying with seed=63" in capsys.readouterr().out


def test_repeated_numerical_vi_instability_fails_clearly(monkeypatch):
    """Rolling evaluation must not plot an approximation that remains invalid."""
    unstable = FakeFit(sigma_speed_process=[1e100])
    monkeypatch.setattr(
        experiment,
        "fit_bayesian_ctrv_model",
        lambda *args, **kwargs: unstable,
    )

    with pytest.raises(RuntimeError, match="remained numerically unstable"):
        experiment._fit_rolling_window(
            object(),
            priors=BayesianCTRVPriors(),
            position_observations=object(),
            inference_method="vi",
            inference_config={},
            initial_seed=62,
        )


def test_mcmc_fit_is_never_subjected_to_vi_retry(monkeypatch):
    """Reference MCMC output should bypass the VI-specific stability policy."""
    fit = object()
    used_seeds = []

    def fake_fit(*args, seed, **kwargs):
        del args, kwargs
        used_seeds.append(seed)
        return fit

    monkeypatch.setattr(experiment, "fit_bayesian_ctrv_model", fake_fit)

    result, seed = experiment._fit_rolling_window(
        object(),
        priors=BayesianCTRVPriors(),
        position_observations=object(),
        inference_method="mcmc",
        inference_config={},
        initial_seed=62,
    )

    assert result is fit
    assert seed == 62
    assert used_seeds == [62]
