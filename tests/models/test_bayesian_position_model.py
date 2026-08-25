"""Tests for the local Bayesian Position Model wrapper and Stan source."""

import os

import numpy as np
import pandas as pd
import pytest

import ship_trajectory_prediction.models.bayesian_position_model as model
from ship_trajectory_prediction.observations.window import TrajectoryWindowData
from ship_trajectory_prediction.validation.reporting import posterior_variable_samples


def create_window(*, irregular_time=False):
    """Return 20 observed and three held-out position-only points."""
    count = 23
    time_seconds = np.arange(count, dtype=float) * 10.0
    if irregular_time:
        time_seconds[15:] += 1.0
    displacement_angle = np.arange(count - 1, dtype=float) * 0.02
    displacement_x = 10.0 * np.cos(displacement_angle)
    displacement_y = 10.0 * np.sin(displacement_angle)
    x_values = np.concatenate(([0.0], np.cumsum(displacement_x)))
    y_values = np.concatenate(([0.0], np.cumsum(displacement_y)))
    return TrajectoryWindowData(
        timestamps=pd.to_datetime(time_seconds, unit="s", utc=True),
        time_seconds=time_seconds,
        x_meters=x_values,
        y_meters=y_values,
        reference_longitude=0.0,
        reference_latitude=0.0,
        gps_speed_mps=np.full(count, 999.0),
        observation_count=20,
    )


@pytest.fixture
def priors():
    """Return the calibrated experiment priors."""
    return model.BayesianPositionModelPriors(
        log_displacement_scale_prior_scale=0.016354,
        rotation_angle_prior_scale=0.016980,
        sigma_displacement_residual_prior_scale=1.989083,
    )


@pytest.mark.parametrize("history_position_count", [20, 10])
def test_build_stan_data_uses_requested_trailing_position_history(
    priors,
    history_position_count,
):
    """K=20 and K=10 must select exactly the trailing observed positions."""
    window = create_window()
    stan_data = model.build_stan_data(
        window,
        priors=priors,
        history_position_count=history_position_count,
    )
    history_start = window.observation_count - history_position_count

    assert stan_data["N_history"] == history_position_count
    assert stan_data["N_prediction"] == 3
    np.testing.assert_allclose(
        stan_data["x_observed"],
        window.x_meters[history_start : window.observation_count],
    )
    np.testing.assert_allclose(
        stan_data["y_observed"],
        window.y_meters[history_start : window.observation_count],
    )
    assert "gps_speed_mps" not in stan_data
    assert "sigma_position_observation" not in stan_data


@pytest.mark.parametrize("history_position_count", [2, 21, 100])
def test_build_stan_data_rejects_invalid_history_lengths(
    priors,
    history_position_count,
):
    """The local history must contain 3..N_observed positions."""
    with pytest.raises(ValueError, match="history_position_count"):
        model.build_stan_data(
            create_window(),
            priors=priors,
            history_position_count=history_position_count,
        )


def test_build_stan_data_rejects_irregular_sampling(priors):
    """A per-observation rotation matrix requires one common time step."""
    with pytest.raises(ValueError, match="regular sampling interval"):
        model.build_stan_data(
            create_window(irregular_time=True),
            priors=priors,
            history_position_count=10,
        )


def test_position_observation_noise_is_used_only_to_perturb_input_positions(priors):
    """Known synthetic noise must not become a duplicate Stan likelihood scale."""
    window = create_window()
    observations = model.simulate_position_observations(
        window,
        additional_noise_std_m=5.0,
        seed=2026,
    )
    stan_data = model.build_stan_data(
        window,
        priors=priors,
        history_position_count=10,
        position_observations=observations,
    )

    np.testing.assert_allclose(stan_data["x_observed"], observations.x_meters[-10:])
    assert observations.observation_noise_std_m == 5.0
    assert "sigma_position_observation" not in stan_data


def test_default_initial_values_recover_rotation_scaling_motion(priors):
    """Numerical initials should follow the observed local complex coefficient."""
    stan_data = model.build_stan_data(
        create_window(),
        priors=priors,
        history_position_count=20,
    )
    initial_values = model._default_initial_values(stan_data, seed=42)

    assert initial_values["log_displacement_scale"] == pytest.approx(0.0, abs=0.01)
    assert initial_values["rotation_angle"] == pytest.approx(0.02, abs=0.01)
    assert initial_values["sigma_displacement_residual"] > 0


def test_fit_forwards_vi_and_mcmc_without_ctrv_states(monkeypatch, priors):
    """Both inference paths should receive only the new position-model data."""
    fake_model = FakeModel()
    monkeypatch.setattr(model, "compile_bayesian_position_model", lambda: fake_model)

    vi_fit = model.fit_bayesian_position_model(
        create_window(),
        priors=priors,
        history_position_count=10,
        draws=50,
        require_converged=False,
        seed=7,
    )
    mcmc_fit = model.fit_bayesian_position_model(
        create_window(),
        priors=priors,
        history_position_count=20,
        inference_method="mcmc",
        chains=2,
        parallel_chains=2,
        iter_warmup=10,
        iter_sampling=10,
        seed=8,
    )

    assert vi_fit is fake_model.vi_result
    assert mcmc_fit is fake_model.mcmc_result
    assert fake_model.vi_arguments["data"]["N_history"] == 10
    assert fake_model.mcmc_arguments["data"]["N_history"] == 20
    assert len(fake_model.mcmc_arguments["inits"]) == 2
    assert "x_state" not in fake_model.vi_arguments["data"]


def test_stan_model_contains_only_local_rotation_scaling_dynamics():
    """The new Stan source must not reintroduce latent CTRV state trajectories."""
    source = model.STAN_FILE.read_text(encoding="utf-8")

    assert "displacement_scale * cos(rotation_angle)" in source
    assert "displacement_scale * sin(rotation_angle)" in source
    assert "multi_normal_cholesky" in source
    assert "x_model_prediction" in source
    assert "x_observation_prediction" in source
    assert "displacement[N_history - 1]" in source
    for forbidden_name in (
        "x_state",
        "y_state",
        "speed_state",
        "heading_state",
        "turn_rate_state",
        "sigma_position_process",
        "sigma_speed_process",
        "sigma_turn_rate_process",
        "sigma_position_observation",
    ):
        assert forbidden_name not in source


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_CMDSTAN_INTEGRATION") != "1",
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to run CmdStan inference.",
)
@pytest.mark.parametrize("history_position_count", [20, 10])
def test_small_synthetic_vi_fit_is_executable(priors, history_position_count):
    """Both approved K options should produce finite recursive predictions."""
    fit = model.fit_bayesian_position_model(
        create_window(),
        priors=priors,
        history_position_count=history_position_count,
        iter=5_000,
        draws=100,
        seed=history_position_count,
        require_converged=False,
    )

    assert posterior_variable_samples(fit, "displacement_scale").shape == (100,)
    assert posterior_variable_samples(fit, "rotation_angle").shape == (100,)
    assert posterior_variable_samples(fit, "x_model_prediction").shape == (100, 3)
    assert posterior_variable_samples(fit, "x_observation_prediction").shape == (
        100,
        3,
    )
    assert posterior_variable_samples(fit, "log_likelihood").shape == (
        100,
        history_position_count - 2,
    )


class FakeModel:
    """Minimal CmdStan model recording both inference calls."""

    def __init__(self):
        self.vi_result = object()
        self.mcmc_result = object()
        self.vi_arguments = None
        self.mcmc_arguments = None

    def variational(self, **kwargs):
        """Record a VI call."""
        self.vi_arguments = kwargs
        return self.vi_result

    def sample(self, **kwargs):
        """Record an MCMC call."""
        self.mcmc_arguments = kwargs
        return self.mcmc_result
