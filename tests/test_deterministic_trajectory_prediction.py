"""Tests for the deterministic CTRV single-window entry point."""

import numpy as np
import pandas as pd

import experiments.trajectory_prediction.deterministic_ctrv as experiment
from ship_trajectory_prediction.evaluation.deterministic_ctrv import (
    DeterministicExperimentConfig,
)
from ship_trajectory_prediction.trajectory import TrajectoryWindowData


def _window():
    return TrajectoryWindowData(
        timestamps=pd.date_range("2026-01-01", periods=5, freq="s", tz="UTC"),
        time_seconds=np.arange(5, dtype=float),
        x_meters=np.arange(5, dtype=float),
        y_meters=np.arange(5, dtype=float) * 2,
        reference_longitude=10.0,
        reference_latitude=54.0,
        gps_speed_mps=np.ones(5),
        observation_count=3,
    )


def test_experiment_groups_window_and_position_noise_configuration():
    """The deterministic entry point should expose one visible experiment."""
    assert isinstance(experiment.EXPERIMENT, DeterministicExperimentConfig)
    assert experiment.EXPERIMENT.run_id == 1
    assert experiment.EXPERIMENT.start_index == 0
    assert experiment.EXPERIMENT.observation_count == 20
    assert experiment.EXPERIMENT.prediction_count == 5
    assert experiment.EXPERIMENT.additional_position_noise_std_m == 2.0
    assert experiment.EXPERIMENT.position_noise_seed == 2026


def test_position_observation_noise_is_reproducible_and_keeps_targets():
    """Noise should affect observations without changing held-out truth."""
    window = _window()

    first = experiment._add_position_observation_noise(
        window,
        additional_noise_std_m=2.0,
        seed=2026,
    )
    second = experiment._add_position_observation_noise(
        window,
        additional_noise_std_m=2.0,
        seed=2026,
    )

    np.testing.assert_array_equal(first.x_meters, second.x_meters)
    np.testing.assert_array_equal(first.y_meters, second.y_meters)
    assert not np.array_equal(
        first.x_meters[first.observed_slice],
        window.x_meters[window.observed_slice],
    )
    np.testing.assert_array_equal(
        first.x_meters[first.prediction_slice],
        window.x_meters[window.prediction_slice],
    )
    np.testing.assert_array_equal(
        first.y_meters[first.prediction_slice],
        window.y_meters[window.prediction_slice],
    )


def test_parse_arguments_uses_experiment_defaults_and_cli_overrides():
    """Position-noise CLI options should default to the experiment values."""
    defaults = experiment._parse_arguments([])
    overrides = experiment._parse_arguments(
        [
            "--position-noise-std-m",
            "0",
            "--position-noise-seed",
            "17",
            "--no-plot",
        ]
    )

    assert defaults.position_noise_std_m == 2.0
    assert defaults.position_noise_seed == 2026
    assert defaults.no_plot is False
    assert overrides.position_noise_std_m == 0.0
    assert overrides.position_noise_seed == 17
    assert overrides.no_plot is True
