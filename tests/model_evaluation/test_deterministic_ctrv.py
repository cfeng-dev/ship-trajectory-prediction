"""Tests for deterministic CTRV rolling evaluation."""

import numpy as np
import pandas as pd
import pytest

import experiments.model_evaluation.deterministic_ctrv as experiment
import ship_trajectory_prediction.validation.deterministic_ctrv_workflow as workflow
from ship_trajectory_prediction.models.deterministic_ctrv import CTRVState
from ship_trajectory_prediction.observations import prepare_trajectory_window
from ship_trajectory_prediction.simulation.synthetic_ctrv import (
    SyntheticCTRVNoise,
    simulate_synthetic_ctrv_data,
)


def _create_route():
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
        run_id=1,
    )


def test_main_evaluates_deterministic_rolling_windows(monkeypatch):
    route = _create_route()
    monkeypatch.setattr(
        workflow.observations_io,
        "read_ship_data",
        lambda *args, **kwargs: route,
    )

    predictions, summary = experiment.main(
        [
            "--observations",
            "5",
            "--predictions",
            "2",
            "--stride",
            "2",
            "--position-noise-std-m",
            "0",
            "--max-windows",
            "2",
            "--no-plot",
        ]
    )

    assert summary.window_count == 2
    assert summary.forecast_count == 4
    assert predictions["model_variant"].unique().tolist() == ["deterministic"]
    assert predictions["window_index"].tolist() == [0, 0, 1, 1]
    assert predictions["horizon_step"].tolist() == [1, 2, 1, 2]
    assert (predictions["window_runtime_seconds"] >= 0).all()
    assert summary.ade_m < 0.1
    assert summary.fde_m < 0.1


def test_route_noise_is_reproducible_and_can_be_disabled():
    first = workflow._simulate_route_position_noise(
        8,
        standard_deviation_m=2.0,
        seed=2026,
    )
    second = workflow._simulate_route_position_noise(
        8,
        standard_deviation_m=2.0,
        seed=2026,
    )
    disabled = workflow._simulate_route_position_noise(
        8,
        standard_deviation_m=0.0,
        seed=2026,
    )

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert np.any(first[0] != 0.0)
    np.testing.assert_array_equal(disabled[0], np.zeros(8))
    np.testing.assert_array_equal(disabled[1], np.zeros(8))


def test_observation_noise_does_not_modify_held_out_positions():
    window = prepare_trajectory_window(
        _create_route(),
        observation_count=5,
        prediction_count=2,
    )
    noise_x = np.arange(10, dtype=float)
    noise_y = -noise_x

    noisy = workflow._add_observation_noise(
        window,
        route_start_index=0,
        route_noise_x=noise_x,
        route_noise_y=noise_y,
    )

    np.testing.assert_allclose(
        noisy.x_meters[noisy.observed_slice],
        window.x_meters[window.observed_slice] + noise_x[:5],
    )
    np.testing.assert_allclose(
        noisy.y_meters[noisy.observed_slice],
        window.y_meters[window.observed_slice] + noise_y[:5],
    )
    np.testing.assert_array_equal(
        noisy.x_meters[noisy.prediction_slice],
        window.x_meters[window.prediction_slice],
    )
    np.testing.assert_array_equal(
        noisy.y_meters[noisy.prediction_slice],
        window.y_meters[window.prediction_slice],
    )


def test_summary_uses_mean_error_at_largest_horizon_as_fde():
    predictions = pd.DataFrame(
        {
            "window_index": [0, 0, 1, 1],
            "horizon_step": [1, 2, 1, 2],
            "horizon_seconds": [5.0, 10.0, 5.0, 10.0],
            "position_error_m": [1.0, 4.0, 3.0, 8.0],
            "window_runtime_seconds": [0.01, 0.01, 0.03, 0.03],
        }
    )

    summary = workflow.summarize_deterministic_predictions(predictions)

    assert summary.window_count == 2
    assert summary.forecast_count == 4
    assert summary.ade_m == pytest.approx(4.0)
    assert summary.fde_m == pytest.approx(6.0)
    assert summary.mean_window_runtime_seconds == pytest.approx(0.02)
    assert summary.median_window_runtime_seconds == pytest.approx(0.02)
    assert summary.total_computation_time_seconds == pytest.approx(0.04)


def test_summary_uses_aligned_metrics_and_compact_horizon_columns(capsys):
    """Deterministic reporting should match the compact Bayesian layout."""
    summary = workflow.DeterministicRollingSummary(
        window_count=114,
        forecast_count=341,
        ade_m=6.07,
        fde_m=6.31,
        mean_window_runtime_seconds=0.004,
        median_window_runtime_seconds=0.003,
        total_computation_time_seconds=0.456,
        per_horizon_table=pd.DataFrame(
            {
                "horizon_step": [1, 2, 3],
                "forecast_count": [114, 114, 113],
                "mean_horizon_seconds": [10.0, 20.0, 30.0],
                "ade_m": [3.1, 6.2, 8.9],
                "median_error_m": [2.8, 5.9, 8.4],
            }
        ),
    )

    workflow._print_summary(summary)

    output = capsys.readouterr().out
    metric_lines = [line for line in output.splitlines() if " : " in line]
    table_lines = output.partition("Per-horizon evaluation:\n")[2].splitlines()
    assert len(metric_lines) == 7
    assert len({line.index(":") for line in metric_lines}) == 1
    assert "Mean window runtime" in output
    assert "Total computation time" in output
    assert "horizon_step" not in output
    assert "Horizon[s]" in table_lines[0]
    assert max(map(len, table_lines)) <= 80
