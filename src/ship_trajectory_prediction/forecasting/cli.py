"""Command-line arguments for single-window CTRV predictions."""

import argparse
from collections.abc import Mapping, Sequence
from typing import Any

from ship_trajectory_prediction.forecasting.bayesian_ctrv import ExperimentConfig
from ship_trajectory_prediction.forecasting.deterministic_ctrv import (
    DeterministicExperimentConfig,
)
from ship_trajectory_prediction.validation.prediction_plotting import (
    PLOT_COORDINATE_MODES,
)


def parse_bayesian_ctrv_prediction_arguments(
    *,
    description: str | None,
    experiment: ExperimentConfig,
    vi_config: Mapping[str, Any],
    plot_coordinate_mode: str,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the shared CLI options for one Bayesian CTRV prediction."""
    parser = argparse.ArgumentParser(description=description)
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
        help="Seed used only to generate the in-memory position perturbation.",
    )
    parser.add_argument(
        "--require-converged",
        action="store_true",
        default=vi_config["require_converged"],
        help="Abort instead of plotting if CmdStan reports non-converged VI.",
    )
    parser.add_argument(
        "--plot-coordinates",
        metavar="{" + ",".join(PLOT_COORDINATE_MODES) + "}",
        default=plot_coordinate_mode,
        help=(
            "Display local meters, local kilometers, or GPS coordinates; "
            "invalid values fall back to meters."
        ),
    )
    return parser.parse_args(argv)


def parse_deterministic_ctrv_prediction_arguments(
    *,
    description: str | None,
    experiment: DeterministicExperimentConfig,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse CLI options for one deterministic CTRV prediction."""
    parser = argparse.ArgumentParser(description=description)
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
        help="Seed used only to generate the in-memory position perturbation.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Print prediction metrics without opening a plot window.",
    )
    return parser.parse_args(argv)
