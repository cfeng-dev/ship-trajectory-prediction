"""Command-line arguments for deterministic CTRV prediction experiments."""

import argparse
from collections.abc import Sequence

from ship_trajectory_prediction.evaluation.deterministic_ctrv import (
    DeterministicExperimentConfig,
)


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
