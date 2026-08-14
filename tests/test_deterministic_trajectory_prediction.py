"""Tests for the deterministic CTRV single-window entry point."""

import experiments.trajectory_prediction.deterministic_ctrv as experiment
from ship_trajectory_prediction.forecasting.deterministic_ctrv import (
    DeterministicExperimentConfig,
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


def test_main_delegates_visible_experiment_configuration(monkeypatch):
    """The entry point should delegate its complete visible configuration."""
    captured = {}
    expected_result = object()

    def fake_run(**kwargs):
        captured.update(kwargs)
        return expected_result

    monkeypatch.setattr(experiment, "run_deterministic_ctrv_prediction", fake_run)

    result = experiment.main([])

    assert result is expected_result
    assert captured == {
        "data_file": experiment.DATA_FILE,
        "experiment": experiment.EXPERIMENT,
        "speed_estimation_points": experiment.SPEED_ESTIMATION_POINTS,
        "heading_estimation_segments": experiment.HEADING_ESTIMATION_SEGMENTS,
        "position_noise_std_m": (experiment.EXPERIMENT.additional_position_noise_std_m),
        "position_noise_seed": experiment.EXPERIMENT.position_noise_seed,
        "show_plot": True,
    }


def test_main_forwards_cli_overrides(monkeypatch):
    """The entry point should forward explicit deterministic CLI values."""
    captured = {}
    monkeypatch.setattr(
        experiment,
        "run_deterministic_ctrv_prediction",
        lambda **kwargs: captured.update(kwargs),
    )

    experiment.main(
        [
            "--position-noise-std-m",
            "0",
            "--position-noise-seed",
            "17",
            "--no-plot",
        ]
    )

    assert captured["position_noise_std_m"] == 0.0
    assert captured["position_noise_seed"] == 17
    assert captured["show_plot"] is False
