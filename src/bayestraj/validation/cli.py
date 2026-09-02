"""Command-line arguments for rolling trajectory evaluations."""

import argparse
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import bayestraj.forecasting.bayesian_ctrv as forecasting
import bayestraj.forecasting.deterministic_ctrv as deterministic_forecasting
import bayestraj.forecasting.inference as inference
import bayestraj.models.bayesian_ctrv as bayesian_model


@dataclasses.dataclass(frozen=True, slots=True)
class BayesianCTRVEvaluationOptions:
    """Runtime overrides for one parametric Bayesian CTRV evaluation."""

    observation_count: int
    prediction_count: int
    stride: int | None
    inference_method: str
    vi_algorithm: str
    turn_rate_prior_abs_heading_change_deg: float
    inference_seed: int
    position_noise_std_m: float
    position_noise_seed: int
    require_converged: bool
    max_windows: int | None
    plot_each_window: bool
    window_mode: str | None = None

    def __post_init__(self) -> None:
        """Validate inference selection and its batch-only window mode."""
        _, inference_method, window_mode = inference.normalize_rolling_inference_method(
            self.inference_method,
            self.window_mode,
            online_inference_methods=inference.CTRV_ONLINE_INFERENCE_METHODS,
        )
        object.__setattr__(self, "inference_method", inference_method)
        object.__setattr__(self, "window_mode", window_mode)


@dataclasses.dataclass(frozen=True, slots=True)
class DeterministicCTRVEvaluationOptions:
    """Runtime overrides for one deterministic CTRV rolling evaluation."""

    window_mode: str
    observation_count: int
    prediction_count: int
    stride: int | None
    position_noise_std_m: float
    position_noise_seed: int
    max_windows: int | None
    show_plot: bool


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
    """Parse options for one parametric Bayesian CTRV rolling evaluation."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--window-mode",
        choices=inference.WINDOW_MODES,
        default=None,
        help="Batch-only evaluation history: fixed sliding or growing expanding.",
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
    parser.add_argument(
        "--turn-rate-prior-abs-heading-change-deg",
        type=float,
        default=priors.turn_rate_prior_abs_heading_change_deg,
        help=(
            "Absolute heading change over the configured reference interval "
            "that the prior exceeds with its configured tail probability."
        ),
    )
    parser.add_argument("--seed", type=int, default=experiment.inference_seed)
    parser.add_argument(
        "--position-noise-std-m",
        type=float,
        default=experiment.position_noise_std_m,
        help="Gaussian x/y position-noise standard deviation in meters; 0 disables.",
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
    window_mode = arguments.window_mode
    inference_mode, _ = inference.normalize_inference_method(
        arguments.inference_method,
        online_inference_methods=inference.CTRV_ONLINE_INFERENCE_METHODS,
    )
    if inference_mode == "batch" and window_mode is None:
        window_mode = experiment.window_mode
    try:
        return BayesianCTRVEvaluationOptions(
            window_mode=window_mode,
            observation_count=arguments.observations,
            prediction_count=arguments.predictions,
            stride=arguments.stride,
            inference_method=arguments.inference_method,
            vi_algorithm=arguments.vi_algorithm,
            turn_rate_prior_abs_heading_change_deg=(
                arguments.turn_rate_prior_abs_heading_change_deg
            ),
            inference_seed=arguments.seed,
            position_noise_std_m=arguments.position_noise_std_m,
            position_noise_seed=arguments.position_noise_seed,
            require_converged=arguments.require_converged,
            max_windows=arguments.max_windows,
            plot_each_window=arguments.plot_each_window,
        )
    except ValueError as error:
        parser.error(str(error))


def parse_deterministic_ctrv_evaluation_arguments(
    *,
    description: str | None,
    experiment: deterministic_forecasting.DeterministicRollingExperimentConfig,
    max_windows: int | None,
    show_plot: bool,
    argv: Sequence[str] | None = None,
) -> DeterministicCTRVEvaluationOptions:
    """Parse shared options for one deterministic CTRV rolling evaluation."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--window-mode",
        choices=("sliding", "expanding"),
        default=experiment.window_mode,
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
    parser.add_argument("--stride", type=int, default=experiment.stride)
    parser.add_argument(
        "--position-noise-std-m",
        type=float,
        default=experiment.position_noise_std_m,
    )
    parser.add_argument(
        "--position-noise-seed",
        type=int,
        default=experiment.position_noise_seed,
    )
    parser.add_argument("--max-windows", type=int, default=max_windows)
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="show_plot",
        default=show_plot,
        help="Print metrics without opening the route-wide plot.",
    )
    arguments = parser.parse_args(argv)
    return DeterministicCTRVEvaluationOptions(
        window_mode=arguments.window_mode,
        observation_count=arguments.observations,
        prediction_count=arguments.predictions,
        stride=arguments.stride,
        position_noise_std_m=arguments.position_noise_std_m,
        position_noise_seed=arguments.position_noise_seed,
        max_windows=arguments.max_windows,
        show_plot=arguments.show_plot,
    )
