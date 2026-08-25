"""Tests for single-window CTRV command-line arguments."""

from ship_trajectory_prediction.forecasting.bayesian_ctrv import ExperimentConfig
from ship_trajectory_prediction.forecasting.cli import (
    parse_bayesian_ctrv_prediction_arguments,
)


def _experiment():
    return ExperimentConfig(
        run_id=1,
        start_index=0,
        observation_count=20,
        prediction_count=3,
        history_position_count=20,
        additional_position_noise_std_m=2.0,
        position_noise_seed=2026,
        inference_method="vi",
        inference_seed=42,
    )


def test_parse_bayesian_ctrv_prediction_arguments_uses_configured_defaults():
    """Empty CLI input should retain the visible experiment defaults."""
    experiment = _experiment()
    vi_config = {"algorithm": "meanfield", "require_converged": False}

    arguments = parse_bayesian_ctrv_prediction_arguments(
        description="Test prediction",
        experiment=experiment,
        vi_config=vi_config,
        plot_coordinate_mode="m",
        argv=[],
    )

    assert vars(arguments) == {
        "history_positions": 20,
        "inference": "vi",
        "vi_algorithm": "meanfield",
        "seed": 42,
        "position_noise_std_m": 2.0,
        "position_noise_seed": 2026,
        "require_converged": False,
        "plot_coordinates": "m",
    }


def test_parse_bayesian_ctrv_prediction_arguments_reads_cli_overrides():
    """Explicit CLI values should override every configurable default."""
    experiment = _experiment()
    vi_config = {"algorithm": "meanfield", "require_converged": False}

    arguments = parse_bayesian_ctrv_prediction_arguments(
        description="Test prediction",
        experiment=experiment,
        vi_config=vi_config,
        plot_coordinate_mode="m",
        argv=[
            "--history-positions",
            "10",
            "--inference",
            "mcmc",
            "--vi-algorithm",
            "fullrank",
            "--seed",
            "17",
            "--position-noise-std-m",
            "0",
            "--position-noise-seed",
            "18",
            "--require-converged",
            "--plot-coordinates",
            "km",
        ],
    )

    assert vars(arguments) == {
        "history_positions": 10,
        "inference": "mcmc",
        "vi_algorithm": "fullrank",
        "seed": 17,
        "position_noise_std_m": 0.0,
        "position_noise_seed": 18,
        "require_converged": True,
        "plot_coordinates": "km",
    }
