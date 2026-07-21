"""Tests for the Bayesian time-varying-radius model interface."""

import numpy as np
import pandas as pd
import pytest

from ship_trajectory_prediction.coordinates import local_to_gps_coordinates
from ship_trajectory_prediction.models import time_varying_radius as model_module
from ship_trajectory_prediction.models.time_varying_radius import (
    STAN_FILE,
    build_stan_data,
    fit_time_varying_radius_model,
    prepare_trajectory_window,
    summarize_predictions,
)
from ship_trajectory_prediction.simulation.core import simulate_curved_trajectory


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


def test_build_stan_data_keeps_future_positions_held_out():
    """Stan should receive observed positions and future timestamps only."""
    window = prepare_trajectory_window(
        create_curved_trajectory_data(),
        observation_count=8,
        prediction_count=4,
    )

    stan_data = build_stan_data(window)

    assert stan_data["N_observed"] == 8
    assert stan_data["N_prediction"] == 4
    assert len(stan_data["x_observed"]) == 8
    assert len(stan_data["y_observed"]) == 8
    assert len(stan_data["time_prediction"]) == 4
    assert stan_data["speed"] == pytest.approx(5.0)
    assert stan_data["integration_substeps"] == 4
    assert "x_prediction" not in stan_data
    assert "y_prediction" not in stan_data


def test_build_stan_data_uses_signed_curvature_prior():
    """The inferred turn direction should sign the initial curvature prior."""
    data = create_curved_trajectory_data()
    left_window = prepare_trajectory_window(
        data,
        observation_count=8,
        prediction_count=4,
        turn_direction=1,
    )
    right_window = prepare_trajectory_window(
        data,
        observation_count=8,
        prediction_count=4,
        turn_direction=-1,
    )

    assert build_stan_data(left_window)["curvature_prior_mean"] == pytest.approx(
        1 / 500
    )
    assert build_stan_data(right_window)["curvature_prior_mean"] == pytest.approx(
        -1 / 500
    )


@pytest.mark.parametrize(
    "value_name",
    [
        "radius_prior_median",
        "curvature_initial_prior_scale",
        "curvature_rate_prior_scale",
        "sigma_prior_scale",
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


@pytest.mark.parametrize("integration_substeps", [True, 1.5, 0, -1])
def test_build_stan_data_rejects_invalid_integration_substeps(
    integration_substeps,
):
    """Numerical integration requires a positive integer substep count."""
    window = prepare_trajectory_window(
        create_curved_trajectory_data(),
        observation_count=8,
        prediction_count=4,
    )

    with pytest.raises(ValueError, match="integration_substeps"):
        build_stan_data(window, integration_substeps=integration_substeps)


def test_fit_model_uses_prior_centered_default_initialization(monkeypatch):
    """Default initial values should represent no curvature change."""
    window = prepare_trajectory_window(
        create_curved_trajectory_data(),
        observation_count=8,
        prediction_count=4,
    )
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module,
        "compile_time_varying_radius_model",
        lambda: fake_model,
    )

    fit = fit_time_varying_radius_model(
        window,
        chains=1,
        show_progress=False,
    )

    assert fit is fake_model.result
    assert fake_model.sample_arguments["parallel_chains"] == 1
    assert fake_model.sample_arguments["inits"] == {
        "curvature_initial_raw": 0.0,
        "curvature_rate_raw": 0.0,
        "sigma": 10.0,
    }


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


class FakeModel:
    """Minimal CmdStan model replacement for checking sampling arguments."""

    def __init__(self):
        self.result = object()
        self.sample_arguments = None

    def sample(self, **kwargs):
        """Capture sampling arguments and return a stable sentinel."""
        self.sample_arguments = kwargs
        return self.result


class FakeFit:
    """Minimal CmdStan-style fit object for prediction summary tests."""

    def __init__(self, **variables):
        self.variables = variables

    def stan_variable(self, name):
        """Return one stored fake posterior variable."""
        return self.variables[name]
