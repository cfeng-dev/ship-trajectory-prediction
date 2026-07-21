"""Tests for the Bayesian time-varying motion model interface."""

import numpy as np
import pandas as pd
import pytest

from ship_trajectory_prediction.coordinates import local_to_gps_coordinates
from ship_trajectory_prediction.models.time_varying_motion import (
    STAN_FILE,
    build_stan_data,
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


def test_prepare_trajectory_window_keeps_speed_values_in_meters_per_second():
    """GPS speed should be converted and retained for evaluation."""
    window = prepare_trajectory_window(
        create_curved_trajectory_data(),
        observation_count=8,
        prediction_count=4,
    )

    assert window.observation_count == 8
    assert window.prediction_count == 4
    assert window.speed_mps == pytest.approx(np.full(12, 5.0))
    assert window.speed_prior_median == pytest.approx(5.0)
    assert window.turn_rate_level == pytest.approx(0.025, abs=1e-6)


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
    assert stan_data["turn_rate_level"] == pytest.approx(window.turn_rate_level)
    assert stan_data["acceleration_state_scale"] == pytest.approx(0.02)
    assert stan_data["turn_rate_state_scale"] == pytest.approx(0.003)
    assert stan_data["turn_rate_decay_time"] == pytest.approx(600.0)
    assert "x_prediction" not in stan_data
    assert "y_prediction" not in stan_data
    assert "speed_prediction" not in stan_data


def test_turn_rate_level_uses_observed_positions_only():
    """Changing held-out positions must not alter the turn-rate level."""
    data = create_curved_trajectory_data()
    reference = prepare_trajectory_window(
        data,
        observation_count=8,
        prediction_count=4,
    )
    changed_future = data.copy()
    changed_future.loc[8:, "gps_longitude"] += 1.0
    changed_future.loc[8:, "gps_latitude"] -= 1.0

    changed = prepare_trajectory_window(
        changed_future,
        observation_count=8,
        prediction_count=4,
    )

    assert changed.turn_rate_level == pytest.approx(reference.turn_rate_level)


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


def test_prepare_trajectory_window_rejects_invalid_speed_values():
    """The speed likelihood requires finite non-negative observations."""
    data = create_curved_trajectory_data()
    data.loc[3, "gps_speed"] = np.nan

    with pytest.raises(ValueError, match="gps_speed"):
        prepare_trajectory_window(
            data,
            observation_count=8,
            prediction_count=4,
        )


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
