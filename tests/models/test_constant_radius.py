"""Tests for the Bayesian constant-radius trajectory model interface."""

import numpy as np
import pandas as pd
import pytest

from ship_trajectory_prediction.coordinates import local_to_gps_coordinates
from ship_trajectory_prediction.models import constant_radius as model_module
from ship_trajectory_prediction.models.constant_radius import (
    STAN_FILE,
    build_stan_data,
    fit_constant_radius_model,
    summarize_predictions,
)
from ship_trajectory_prediction.simulation.core import simulate_curved_trajectory
from ship_trajectory_prediction.trajectory import prepare_trajectory_window


def create_curved_trajectory_data(run_id=1):
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
            "run_id": run_id,
            "gps_latitude": latitude,
            "gps_longitude": longitude,
            "gps_speed": 18.0,
        }
    )


def test_build_stan_data_derives_fixed_motion_values_without_turn_direction():
    """Stan data should not preclassify the observed turn direction."""
    window = prepare_trajectory_window(
        create_curved_trajectory_data(),
        observation_count=8,
        prediction_count=4,
    )
    stan_data = build_stan_data(window)

    assert window.observation_count == 8
    assert window.prediction_count == 4
    assert stan_data["speed"] == pytest.approx(5.0)
    assert stan_data["heading_initial"] == pytest.approx(0.5)
    assert stan_data["curvature_prior_scale"] == pytest.approx(0.002)
    assert "turn_direction" not in stan_data
    assert window.x_meters[0] == pytest.approx(0.0)
    assert window.y_meters[0] == pytest.approx(0.0)


def test_build_stan_data_keeps_future_positions_held_out():
    """Stan data should contain prediction times but no held-out positions."""
    window = prepare_trajectory_window(
        create_curved_trajectory_data(),
        observation_count=8,
        prediction_count=4,
    )

    stan_data = build_stan_data(window)

    assert stan_data["N_observed"] == 8
    assert stan_data["N_prediction"] == 4
    assert len(stan_data["x_observed"]) == 8
    assert len(stan_data["time_prediction"]) == 4
    assert "x_prediction" not in stan_data
    assert "y_prediction" not in stan_data


@pytest.mark.parametrize(
    "value_name",
    ["curvature_prior_scale", "sigma_prior_scale"],
)
@pytest.mark.parametrize("invalid_value", [0.0, np.nan])
def test_build_stan_data_rejects_invalid_prior_scales(
    value_name,
    invalid_value,
):
    """Every configurable prior scale should be positive and finite."""
    window = prepare_trajectory_window(
        create_curved_trajectory_data(),
        observation_count=8,
        prediction_count=4,
    )

    with pytest.raises(ValueError, match=value_name):
        build_stan_data(window, **{value_name: invalid_value})


def test_fit_model_uses_direction_neutral_curvature_initialization(monkeypatch):
    """Sampling should start at zero curvature without fixing its sign."""
    window = prepare_trajectory_window(
        create_curved_trajectory_data(),
        observation_count=8,
        prediction_count=4,
    )
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module,
        "compile_constant_radius_model",
        lambda: fake_model,
    )

    fit = fit_constant_radius_model(
        window,
        curvature_prior_scale=0.003,
        chains=1,
        show_progress=False,
    )

    assert fit is fake_model.result
    assert fake_model.sample_arguments["parallel_chains"] == 1
    assert fake_model.sample_arguments["data"][
        "curvature_prior_scale"
    ] == pytest.approx(0.003)
    assert fake_model.sample_arguments["inits"] == {
        "curvature": 0.0,
        "sigma": 10.0,
    }


def test_prepare_trajectory_window_rejects_multiple_runs():
    """One model window should never mix positions from different runs."""
    first_run = create_curved_trajectory_data(run_id=1)
    second_run = create_curved_trajectory_data(run_id=2)

    with pytest.raises(ValueError, match="exactly one run_id"):
        prepare_trajectory_window(
            pd.concat([first_run, second_run], ignore_index=True),
            observation_count=8,
            prediction_count=4,
        )


def test_summarize_predictions_uses_posterior_draws():
    """Prediction summaries should contain medians and credible intervals."""
    window = prepare_trajectory_window(
        create_curved_trajectory_data(),
        observation_count=8,
        prediction_count=4,
    )
    fit = FakeFit(
        x_prediction=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
            ]
        ),
        y_prediction=np.array(
            [
                [5.0, 6.0, 7.0, 8.0],
                [6.0, 7.0, 8.0, 9.0],
                [7.0, 8.0, 9.0, 10.0],
            ]
        ),
    )

    summary = summarize_predictions(fit, window, credible_interval=0.8)

    assert summary["x_median"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert summary["y_median"].tolist() == [6.0, 7.0, 8.0, 9.0]
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
    """Minimal CmdStan model replacement for checking sampling arguments."""

    def __init__(self):
        self.result = object()
        self.sample_arguments = None

    def sample(self, **kwargs):
        """Capture sampling arguments and return a stable sentinel."""
        self.sample_arguments = kwargs
        return self.result
