"""Command-line arguments for rolling trajectory evaluations."""

import argparse
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import ship_trajectory_prediction.forecasting.bayesian_ctrv as forecasting
import ship_trajectory_prediction.models.bayesian_ctrv as bayesian_model


@dataclasses.dataclass(frozen=True, slots=True)
class BayesianCTRVEvaluationOptions:
    """Runtime overrides for one Bayesian CTRV rolling evaluation."""

    window_mode: str
    observation_count: int
    prediction_count: int
    stride: int | None
    inference_method: str
    vi_algorithm: str
    turn_rate_state_prior_scale: float | None
    inference_seed: int
    additional_position_noise_std_m: float
    position_noise_seed: int
    require_converged: bool
    max_windows: int | None
    plot_each_window: bool


def parse_bayesian_ctrv_evaluation_arguments(
    *,
    description: str | None,
    experiment: forecasting.RollingExperimentConfig,
    priors: bayesian_model.BayesianCTRVPriors,
    vi_config: Mapping[str, Any],
    max_windows: int | None,
    plot_each_window: bool,
    argv: Sequence[str] | None = None,
) -> BayesianCTRVEvaluationOptions:
    """Parse shared options for one Bayesian CTRV rolling evaluation."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--window-mode",
        choices=("sliding", "expanding"),
        default=experiment.window_mode,
        help="Keep a fixed history or expand it from the beginning of the run.",
    )
    parser.add_argument(
        "--observations",
        type=int,
        default=experiment.observation_count,
    )
    parser.add_argument(
        "--predictions",
        type=int,
        default=experiment.prediction_count,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=experiment.stride,
        help="Forecast-origin step; defaults to the prediction horizon.",
    )
    parser.add_argument(
        "--inference",
        choices=("vi", "mcmc"),
        default=experiment.inference_method,
        help="Fast variational inference or reference MCMC for every window.",
    )
    parser.add_argument(
        "--vi-algorithm",
        choices=("meanfield", "fullrank"),
        default=vi_config["algorithm"],
    )
    parser.add_argument(
        "--turn-rate-prior-scale",
        type=float,
        default=priors.turn_rate_state_prior_scale,
        help="Optional fixed state-prior scale; defaults to observed-history MAD.",
    )
    parser.add_argument("--seed", type=int, default=experiment.inference_seed)
    parser.add_argument(
        "--position-noise-std-m",
        type=float,
        default=experiment.additional_position_noise_std_m,
        help="Additional Gaussian x/y observation noise in meters; 0 disables.",
    )
    parser.add_argument(
        "--position-noise-seed",
        type=int,
        default=experiment.position_noise_seed,
        help="Seed for one route-wide reproducible position perturbation.",
    )
    parser.add_argument(
        "--require-converged",
        action="store_true",
        default=vi_config["require_converged"],
        help="Abort when any rolling VI fit misses its convergence criterion.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=max_windows,
        help="Optional smoke-test limit; omit it to evaluate the complete run.",
    )
    parser.add_argument(
        "--plot-each-window",
        action="store_true",
        default=plot_each_window,
        help="Show each fitted window and continue after its plot is closed.",
    )
    arguments = parser.parse_args(argv)
    return BayesianCTRVEvaluationOptions(
        window_mode=arguments.window_mode,
        observation_count=arguments.observations,
        prediction_count=arguments.predictions,
        stride=arguments.stride,
        inference_method=arguments.inference,
        vi_algorithm=arguments.vi_algorithm,
        turn_rate_state_prior_scale=arguments.turn_rate_prior_scale,
        inference_seed=arguments.seed,
        additional_position_noise_std_m=arguments.position_noise_std_m,
        position_noise_seed=arguments.position_noise_seed,
        require_converged=arguments.require_converged,
        max_windows=arguments.max_windows,
        plot_each_window=arguments.plot_each_window,
    )
