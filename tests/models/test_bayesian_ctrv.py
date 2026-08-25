"""Tests for the three-parameter Bayesian CTRV model."""

from __future__ import annotations

import os
import re
from dataclasses import replace

import numpy as np
import pytest

import ship_trajectory_prediction.models.bayesian_ctrv as model
from ship_trajectory_prediction.models.deterministic_ctrv import CTRVState
from ship_trajectory_prediction.observations import prepare_trajectory_window
from ship_trajectory_prediction.simulation.synthetic_ctrv import (
    SyntheticCTRVNoise,
    simulate_synthetic_ctrv_data,
)
from ship_trajectory_prediction.validation.reporting import (
    posterior_variable_samples,
)


def create_synthetic_window():
    """Create a noise-free curved trajectory with 20 observed positions."""
    data = simulate_synthetic_ctrv_data(
        count=23,
        dt_seconds=5.0,
        initial_state=CTRVState(
            x=0.0,
            y=0.0,
            speed=3.0,
            heading=0.3,
            turn_rate=0.001,
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
        observation_count=20,
        prediction_count=3,
    )


def test_stan_model_has_exactly_three_parameters_and_no_process_terms():
    """The main Stan model should contain only global CTRV parameters."""
    source = model.STAN_FILE.read_text(encoding="utf-8")
    parameter_block = re.search(
        r"parameters\s*\{(.*?)\n\}",
        source,
        flags=re.DOTALL,
    ).group(1)

    declarations = [
        line.strip()
        for line in parameter_block.splitlines()
        if line.strip().endswith(";")
    ]
    assert len(declarations) == 3
    assert any(declaration.endswith(" speed;") for declaration in declarations)
    assert any(
        declaration.endswith(" heading_initial;") for declaration in declarations
    )
    assert declarations[-1].endswith(" turn_rate;")
    assert "multiplier=turn_rate_prior_scale" in declarations[-1]
    assert "process" not in source.lower()
    assert "_state" not in source


def test_stan_model_uses_fixed_anchor_and_excludes_it_from_likelihood():
    """History position one should anchor the curve but not be observed twice."""
    source = model.STAN_FILE.read_text(encoding="utf-8")

    assert "x_model[1] = x_observed[1]" in source
    assert "y_model[1] = y_observed[1]" in source
    assert "for (n in 2:N_history)" in source
    assert "x_observed[n] ~ normal(x_model[n]" in source
    assert "y_observed[n] ~ normal(y_model[n]" in source
    assert "x_observed[1] ~" not in source
    assert "y_observed[1] ~" not in source


def test_prediction_variables_separate_parameter_and_observation_uncertainty():
    """Only explicit sensor predictions should draw new measurement noise."""
    source = model.STAN_FILE.read_text(encoding="utf-8")

    assert "x_prediction[n] = position[1]" in source
    assert "y_prediction[n] = position[2]" in source
    assert "x_observation_prediction[n] = normal_rng(" in source
    assert "y_observation_prediction[n] = normal_rng(" in source
    assert "normal_rng(\n        x_prediction[n], sigma_position_observation)" in source
    assert "normal_rng(\n        y_prediction[n], sigma_position_observation)" in source


@pytest.mark.parametrize("history_position_count", [20, 10, 3])
def test_build_stan_data_uses_last_k_positions(history_position_count):
    """Every supported K should select a trailing slice of one full history."""
    window = create_synthetic_window()
    observations = model.simulate_position_observations(
        window,
        additional_noise_std_m=5.0,
        seed=2026,
    )

    stan_data = model.build_stan_data(
        window,
        history_position_count=history_position_count,
        position_observations=observations,
    )

    history_slice = slice(-history_position_count, None)
    assert stan_data["N_history"] == history_position_count
    assert stan_data["time_observed"] == pytest.approx(
        observations.time_seconds[history_slice]
    )
    assert stan_data["x_observed"] == pytest.approx(
        observations.x_meters[history_slice]
    )
    assert stan_data["y_observed"] == pytest.approx(
        observations.y_meters[history_slice]
    )
    assert stan_data["sigma_position_observation"] == pytest.approx(5.0)


@pytest.mark.parametrize("history_position_count", [2, 21, 100, True])
def test_build_stan_data_rejects_invalid_history_counts(history_position_count):
    """K must be an integer from three through the observation count."""
    with pytest.raises(ValueError, match="history_position_count"):
        model.build_stan_data(
            create_synthetic_window(),
            history_position_count=history_position_count,
        )


def test_k_20_and_k_10_reuse_full_observations_seed_and_held_out_points():
    """Changing K must only change the trailing fitted history slice."""
    window = create_synthetic_window()
    observations = model.simulate_position_observations(
        window,
        additional_noise_std_m=5.0,
        seed=2026,
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

    assert data_10["x_observed"] == pytest.approx(data_20["x_observed"][-10:])
    assert data_10["y_observed"] == pytest.approx(data_20["y_observed"][-10:])
    assert data_10["time_observed"] == pytest.approx(data_20["time_observed"][-10:])
    assert data_10["time_prediction"] == pytest.approx(data_20["time_prediction"])
    assert observations.noise_seed == 2026


def test_stan_data_ignores_held_out_positions_and_external_speed():
    """Neither future truth nor GPS speed may enter the fitted Stan data."""
    window = create_synthetic_window()
    changed_x = window.x_meters.copy()
    changed_y = window.y_meters.copy()
    changed_speed = window.gps_speed_mps.copy()
    changed_x[window.prediction_slice] += 100_000.0
    changed_y[window.prediction_slice] -= 100_000.0
    changed_speed[:] = 100_000.0
    changed_window = replace(
        window,
        x_meters=changed_x,
        y_meters=changed_y,
        gps_speed_mps=changed_speed,
    )

    reference = model.build_stan_data(window, history_position_count=10)
    changed = model.build_stan_data(changed_window, history_position_count=10)

    assert reference.keys() == changed.keys()
    for name in reference:
        assert changed[name] == pytest.approx(reference[name])
    assert not any("speed_observed" in name for name in reference)


def test_priors_are_typed_and_forwarded_without_calibration_claims():
    """The provisional prior object should map directly to Stan data."""
    priors = model.BayesianCTRVPriors(
        speed_prior_mean=2.5,
        speed_prior_scale=0.7,
        turn_rate_prior_mean=-0.002,
        turn_rate_prior_scale=0.004,
    )

    stan_data = model.build_stan_data(
        create_synthetic_window(),
        priors=priors,
        history_position_count=10,
    )

    for name in (
        "speed_prior_mean",
        "speed_prior_scale",
        "turn_rate_prior_mean",
        "turn_rate_prior_scale",
    ):
        assert stan_data[name] == pytest.approx(getattr(priors, name))
    assert "provisional" in model.BayesianCTRVPriors.__doc__.lower()


def test_fit_forwards_k_and_three_seeded_initial_values(monkeypatch):
    """The wrapper should pass selected history and only three initials."""
    fake_model = FakeModel()
    monkeypatch.setattr(model, "compile_bayesian_ctrv_model", lambda: fake_model)

    result = model.fit_bayesian_ctrv_model(
        create_synthetic_window(),
        history_position_count=10,
        iter=500,
        draws=80,
        seed=17,
        require_converged=False,
    )

    assert result is fake_model.result
    assert fake_model.arguments["data"]["N_history"] == 10
    assert set(fake_model.arguments["inits"]) == set(model.PARAMETER_NAMES)
    assert fake_model.arguments["seed"] == 17


def test_summarize_predictions_reports_model_and_sensor_positions():
    """Prediction reporting should preserve the two uncertainty contracts."""
    window = create_synthetic_window()
    base = np.arange(window.prediction_count, dtype=float)
    fit = FakeFit(
        x_prediction=np.vstack((base, base + 1.0, base + 2.0)),
        y_prediction=np.vstack((base + 3.0, base + 4.0, base + 5.0)),
        x_observation_prediction=np.vstack((base + 6.0, base + 7.0, base + 8.0)),
        y_observation_prediction=np.vstack((base + 9.0, base + 10.0, base + 11.0)),
    )

    summary = model.summarize_predictions(fit, window)

    assert len(summary) == window.prediction_count
    for prefix in ("x", "y", "x_observation", "y_observation"):
        assert f"{prefix}_median" in summary
        assert f"{prefix}_lower" in summary
        assert f"{prefix}_upper" in summary


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_CMDSTAN_INTEGRATION") != "1",
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to run CmdStan inference.",
)
@pytest.mark.parametrize("history_position_count", [20, 10])
def test_small_synthetic_vi_fit_is_executable(history_position_count):
    """Both comparison histories should compile and produce finite draws."""
    fit = model.fit_bayesian_ctrv_model(
        create_synthetic_window(),
        history_position_count=history_position_count,
        iter=5_000,
        grad_samples=2,
        draws=100,
        seed=42,
        require_converged=False,
    )

    assert posterior_variable_samples(fit, "speed").shape == (100,)
    assert posterior_variable_samples(fit, "x_prediction").shape == (100, 3)
    assert posterior_variable_samples(
        fit,
        "x_observation_prediction",
    ).shape == (100, 3)
    assert posterior_variable_samples(fit, "log_likelihood").shape == (
        100,
        2 * (history_position_count - 1),
    )


class FakeFit:
    """Minimal posterior object for prediction summaries."""

    def __init__(self, **variables):
        self.variables = variables

    def stan_variable(self, name, mean=False):
        """Return one configured variable."""
        del mean
        return self.variables[name]


class FakeModel:
    """Minimal CmdStan model recording a variational call."""

    def __init__(self):
        self.result = object()
        self.arguments = None

    def variational(self, **kwargs):
        """Capture the call and return a sentinel."""
        self.arguments = kwargs
        return self.result
