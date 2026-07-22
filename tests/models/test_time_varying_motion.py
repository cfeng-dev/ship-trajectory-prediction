"""Tests for the Bayesian time-varying motion model interface."""

import numpy as np
import pandas as pd
import pytest

from ship_trajectory_prediction.coordinates import local_to_gps_coordinates
from ship_trajectory_prediction.models import time_varying_motion as model_module
from ship_trajectory_prediction.models.time_varying_motion import (
    STAN_FILE,
    build_stan_data,
    fit_time_varying_motion_model,
    summarize_predictions,
)
from ship_trajectory_prediction.simulation.core import simulate_curved_trajectory
from ship_trajectory_prediction.trajectory import prepare_trajectory_window


def create_curved_trajectory_data():
    """Create GPS observations from a deterministic circular trajectory."""
    time_seconds = np.arange(12, dtype=float) * 10
    x_meters, y_meters = simulate_curved_trajectory(
        time_seconds,
        x0=0.0,
        y0=0.0,
        v=5.0,
        radius=200.0,
        theta=0.0,
    )
    longitude, latitude = local_to_gps_coordinates(
        x_meters,
        y_meters,
        reference_longitude=8.3122,
        reference_latitude=47.0515,
    )
    return pd.DataFrame(
        {
            "time": pd.Timestamp("2026-01-10 08:00:00", tz="UTC")
            + pd.to_timedelta(time_seconds, unit="s"),
            "run_id": 1,
            "gps_latitude": latitude,
            "gps_longitude": longitude,
            "gps_speed": 18.0,
        }
    )


def test_shared_trajectory_window_keeps_speed_values_in_meters_per_second():
    """The common window should retain converted GPS speeds for evaluation."""
    window = prepare_trajectory_window(
        create_curved_trajectory_data(),
        observation_count=8,
        prediction_count=4,
    )

    assert window.observation_count == 8
    assert window.prediction_count == 4
    assert window.gps_speed_mps == pytest.approx(np.full(12, 5.0))


def test_build_stan_data_holds_out_future_positions_and_speeds():
    """Stan should receive future times but no future measurements."""
    window = prepare_trajectory_window(
        create_curved_trajectory_data(),
        observation_count=8,
        prediction_count=4,
    )

    stan_data = build_stan_data(window)

    assert len(stan_data["x_observed"]) == 8
    assert len(stan_data["y_observed"]) == 8
    assert len(stan_data["speed_observed"]) == 8
    assert len(stan_data["time_prediction"]) == 4
    assert stan_data["speed_prior_median"] == pytest.approx(5.0)
    assert stan_data["heading_prior_mean"] == pytest.approx(0.5, abs=1e-6)
    assert stan_data["turn_rate_level"] == pytest.approx(0.025, abs=1e-6)
    assert stan_data["acceleration_state_scale"] == pytest.approx(0.02)
    assert stan_data["turn_rate_state_scale"] == pytest.approx(0.003)
    assert stan_data["turn_rate_decay_time"] == pytest.approx(600.0)
    assert "x_prediction" not in stan_data
    assert "y_prediction" not in stan_data
    assert "speed_prediction" not in stan_data


def test_turn_rate_level_uses_observed_positions_only():
    """Changing held-out positions must not alter the turn-rate level."""
    data = create_curved_trajectory_data()
    reference_window = prepare_trajectory_window(
        data, observation_count=8, prediction_count=4
    )
    changed_future = data.copy()
    changed_future.loc[8:, "gps_longitude"] += 1.0
    changed_future.loc[8:, "gps_latitude"] -= 1.0

    changed_window = prepare_trajectory_window(
        changed_future,
        observation_count=8,
        prediction_count=4,
    )

    reference_data = build_stan_data(reference_window)
    changed_data = build_stan_data(changed_window)
    assert changed_data["turn_rate_level"] == pytest.approx(
        reference_data["turn_rate_level"]
    )


def test_speed_prior_uses_first_three_positive_observed_values():
    """Later observed speeds should not shift the initial-speed prior."""
    data = create_curved_trajectory_data()
    data.loc[:7, "gps_speed"] = [0.0, 3.6, 7.2, 10.8, 180.0, 180.0, 180.0, 180.0]
    window = prepare_trajectory_window(
        data,
        observation_count=8,
        prediction_count=4,
    )

    stan_data = build_stan_data(window)

    assert stan_data["speed_prior_median"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "value_name",
    [
        "speed_prior_log_sd",
        "heading_prior_scale",
        "acceleration_initial_scale",
        "acceleration_state_scale",
        "acceleration_decay_time",
        "turn_rate_initial_scale",
        "turn_rate_state_scale",
        "turn_rate_decay_time",
        "sigma_position",
        "sigma_speed",
    ],
)
def test_build_stan_data_rejects_non_positive_scales(value_name):
    """Every configurable positive model value should be validated."""
    window = prepare_trajectory_window(
        create_curved_trajectory_data(),
        observation_count=8,
        prediction_count=4,
    )

    with pytest.raises(ValueError, match=value_name):
        build_stan_data(window, **{value_name: 0.0})


@pytest.mark.parametrize(("index", "value"), [(3, np.nan), (10, -1.0)])
def test_build_stan_data_rejects_invalid_speed_values(index, value):
    """The speed likelihood requires a valid complete evaluation window."""
    data = create_curved_trajectory_data()
    data.loc[index, "gps_speed"] = value
    window = prepare_trajectory_window(
        data,
        observation_count=8,
        prediction_count=4,
    )

    with pytest.raises(ValueError, match="gps_speed"):
        build_stan_data(window)


def test_fit_uses_stan_data_priors_for_default_initialization(monkeypatch):
    """Default sampler initials should reuse the observed-only Stan values."""
    window = prepare_trajectory_window(
        create_curved_trajectory_data(),
        observation_count=8,
        prediction_count=4,
    )
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module,
        "compile_time_varying_motion_model",
        lambda: fake_model,
    )

    fit = fit_time_varying_motion_model(
        window,
        chains=1,
        show_progress=False,
    )

    assert fit is fake_model.result
    inits = fake_model.sample_arguments["inits"]
    assert inits["speed_initial"] == pytest.approx(5.0)
    assert inits["heading_initial"] == pytest.approx(0.5, abs=1e-6)
    assert inits["turn_rate_initial"] == pytest.approx(0.025, abs=1e-6)
    assert inits["acceleration_innovation"] == pytest.approx(np.zeros(6))
    assert inits["turn_rate_innovation"] == pytest.approx(np.zeros(6))


def test_summarize_predictions_includes_speed_intervals():
    """Prediction summaries should cover positions and GPS speed."""
    window = prepare_trajectory_window(
        create_curved_trajectory_data(),
        observation_count=8,
        prediction_count=4,
    )
    base = np.arange(4, dtype=float)
    fit = FakeFit(
        x_prediction=np.vstack([base, base + 1, base + 2]),
        y_prediction=np.vstack([base + 3, base + 4, base + 5]),
        speed_prediction=np.vstack([base + 4, base + 5, base + 6]),
    )

    summary = summarize_predictions(fit, window, credible_interval=0.8)

    assert summary["x_median"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert summary["speed_median"].tolist() == [5.0, 6.0, 7.0, 8.0]
    assert summary["speed_actual"].tolist() == pytest.approx([5.0] * 4)
    assert len(summary) == window.prediction_count


def test_stan_model_file_exists():
    """The Python interface should resolve the repository Stan model."""
    assert STAN_FILE.is_file()


class FakeFit:
    """Minimal CmdStan-style fit object for prediction summary tests."""

    def __init__(self, **variables):
        self.variables = variables

    def stan_variable(self, name):
        """Return one stored fake posterior variable."""
        return self.variables[name]


class FakeModel:
    """Minimal CmdStan replacement for inspecting sampling arguments."""

    def __init__(self):
        self.result = object()
        self.sample_arguments = None

    def sample(self, **kwargs):
        """Capture sampler arguments and return a stable sentinel."""
        self.sample_arguments = kwargs
        return self.result
