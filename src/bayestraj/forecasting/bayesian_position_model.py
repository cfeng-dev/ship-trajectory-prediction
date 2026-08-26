"""Configuration and CLI options for the Bayesian Position Model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import bayestraj.validation.prediction_plotting as prediction_plotting


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration of one Bayesian Position Model prediction."""

    run_id: int
    start_index: int
    observation_count: int
    prediction_count: int
    additional_position_noise_std_m: float
    position_noise_seed: int
    inference_method: str
    inference_seed: int


@dataclass(frozen=True, slots=True)
class RollingExperimentConfig:
    """Configuration of one rolling Bayesian Position Model evaluation."""

    run_id: int
    window_mode: str
    observation_count: int
    prediction_count: int
    additional_position_noise_std_m: float
    position_noise_seed: int
    stride: int | None
    inference_method: str
    inference_seed: int


@dataclass(frozen=True, slots=True)
class EvaluationOptions:
    """Runtime overrides for one rolling position-model evaluation."""

    window_mode: str
    observation_count: int
    prediction_count: int
    stride: int | None
    inference_method: str
    vi_algorithm: str
    inference_seed: int
    additional_position_noise_std_m: float
    position_noise_seed: int
    require_converged: bool
    max_windows: int | None
    plot_each_window: bool


def parse_prediction_arguments(
    *,
    description,
    experiment,
    vi_config,
    plot_coordinate_mode,
    argv=None,
):
    """Parse single-window position-model runtime options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--observations",
        type=int,
        default=experiment.observation_count,
        help="Use this many observed positions; at least 3 are required.",
    )
    parser.add_argument(
        "--inference",
        choices=("vi", "mcmc"),
        default=experiment.inference_method,
    )
    parser.add_argument(
        "--vi-algorithm",
        choices=("meanfield", "fullrank"),
        default=vi_config["algorithm"],
    )
    parser.add_argument("--seed", type=int, default=experiment.inference_seed)
    parser.add_argument(
        "--position-noise-std-m",
        type=float,
        default=experiment.additional_position_noise_std_m,
        help="Extra Gaussian standard deviation per local x/y axis; 0 disables it.",
    )
    parser.add_argument(
        "--position-noise-seed",
        type=int,
        default=experiment.position_noise_seed,
    )
    parser.add_argument(
        "--require-converged",
        action="store_true",
        default=vi_config["require_converged"],
    )
    parser.add_argument(
        "--plot-coordinates",
        metavar="{" + ",".join(prediction_plotting.PLOT_COORDINATE_MODES) + "}",
        default=plot_coordinate_mode,
    )
    return parser.parse_args(argv)


def parse_evaluation_arguments(
    *,
    description,
    experiment,
    vi_config,
    max_windows,
    plot_each_window,
    argv=None,
) -> EvaluationOptions:
    """Parse rolling position-model runtime options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--window-mode",
        choices=("sliding", "expanding"),
        default=experiment.window_mode,
    )
    parser.add_argument(
        "--observations", type=int, default=experiment.observation_count
    )
    parser.add_argument("--predictions", type=int, default=experiment.prediction_count)
    parser.add_argument("--stride", type=int, default=experiment.stride)
    parser.add_argument(
        "--inference",
        choices=("vi", "mcmc"),
        default=experiment.inference_method,
    )
    parser.add_argument(
        "--vi-algorithm",
        choices=("meanfield", "fullrank"),
        default=vi_config["algorithm"],
    )
    parser.add_argument("--seed", type=int, default=experiment.inference_seed)
    parser.add_argument(
        "--position-noise-std-m",
        type=float,
        default=experiment.additional_position_noise_std_m,
    )
    parser.add_argument(
        "--position-noise-seed",
        type=int,
        default=experiment.position_noise_seed,
    )
    parser.add_argument(
        "--require-converged",
        action="store_true",
        default=vi_config["require_converged"],
    )
    parser.add_argument("--max-windows", type=int, default=max_windows)
    parser.add_argument(
        "--plot-each-window",
        action="store_true",
        default=plot_each_window,
    )
    arguments = parser.parse_args(argv)
    return EvaluationOptions(
        window_mode=arguments.window_mode,
        observation_count=arguments.observations,
        prediction_count=arguments.predictions,
        stride=arguments.stride,
        inference_method=arguments.inference,
        vi_algorithm=arguments.vi_algorithm,
        inference_seed=arguments.seed,
        additional_position_noise_std_m=arguments.position_noise_std_m,
        position_noise_seed=arguments.position_noise_seed,
        require_converged=arguments.require_converged,
        max_windows=arguments.max_windows,
        plot_each_window=arguments.plot_each_window,
    )
