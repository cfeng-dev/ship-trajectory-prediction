"""Command-line arguments for single-window CTRV predictions."""

import argparse
from collections.abc import Mapping, Sequence
from typing import Any

import bayestraj.forecasting.bayesian_ctrv as bayesian_forecasting
import bayestraj.forecasting.deterministic_ctrv as deterministic_forecasting
import bayestraj.inference.configuration as inference
import bayestraj.validation.prediction_plotting as prediction_plotting


def parse_bayesian_ctrv_prediction_arguments(
    *,
    description: str | None,
    experiment: bayesian_forecasting.ExperimentConfig,
    vi_config: Mapping[str, Any],
    plot_coordinate_mode: str,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse CLI options for one parametric Bayesian CTRV prediction."""
    return _parse_bayesian_prediction_arguments(
        description=description,
        experiment=experiment,
        vi_config=vi_config,
        plot_coordinate_mode=plot_coordinate_mode,
        argv=argv,
    )


def _parse_bayesian_prediction_arguments(
    *,
    description,
    experiment,
    vi_config,
    plot_coordinate_mode,
    argv,
):
    """Parse common Bayesian single-window options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--inference-method",
        "--inference",
        dest="inference_method",
        choices=(
            *inference.BATCH_INFERENCE_METHODS,
            *inference.CTRV_ONLINE_INFERENCE_METHODS,
        ),
        default=experiment.inference_method,
        help="VI/MCMC use batch inference; RBPF/SMC use online inference.",
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
        default=experiment.position_noise_std_m,
        help="Gaussian position-noise standard deviation per local x/y axis.",
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
        metavar="{" + ",".join(prediction_plotting.PLOT_COORDINATE_MODES) + "}",
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
    experiment: deterministic_forecasting.DeterministicExperimentConfig,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse CLI options for one deterministic CTRV prediction."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--position-noise-std-m",
        type=float,
        default=experiment.position_noise_std_m,
        help="Gaussian position-noise standard deviation per local x/y axis.",
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
