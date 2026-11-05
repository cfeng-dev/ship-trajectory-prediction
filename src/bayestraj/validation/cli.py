"""Command-line arguments for rolling trajectory evaluations."""

import argparse
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import bayestraj.forecasting.bayesian_ctrv as forecasting
import bayestraj.inference.configuration as inference
import bayestraj.models.bayesian_ctrv as bayesian_model


@dataclasses.dataclass(frozen=True, slots=True)
class BayesianCTRVEvaluationOptions:
    """Runtime overrides for one parametric Bayesian CTRV evaluation."""

    observation_count: int
    prediction_count: int
    stride: int | None
    inference_method: str
    vi_algorithm: str
    turn_rate_prior_abs_rate_deg_s: float
    inference_seed: int
    position_noise_std_m: float
    position_noise_seed: int
    require_converged: bool
    max_windows: int | None

    def __post_init__(self) -> None:
        """Validate the combined rolling inference selection."""
        _, inference_method, _ = inference.normalize_ctrv_rolling_inference_method(
            self.inference_method
        )
        object.__setattr__(self, "inference_method", inference_method)


@dataclasses.dataclass(frozen=True, slots=True)
class IMMEvaluationOptions:
    """IMM evaluation settings; show_plot controls summary diagnostic plots."""

    observation_count: int
    prediction_count: int
    stride: int
    max_windows: int | None
    show_plot: bool


def parse_bayesian_ctrv_evaluation_arguments(
    *,
    description: str | None,
    experiment: forecasting.RollingExperimentConfig,
    priors: bayesian_model.BayesianCTRVPriors,
    vi_config: Mapping[str, Any] | None = None,
    max_windows: int | None,
    argv: Sequence[str] | None = None,
) -> BayesianCTRVEvaluationOptions:
    """Parse options for one parametric Bayesian CTRV rolling evaluation."""
    if vi_config is None:
        vi_config = inference.create_default_vi_config()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--observations",
        type=int,
        default=experiment.observation_count,
        help="Maximum past points per VI/MCMC fit; ignored by online methods.",
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
        help="Forecast-origin step; uses the configured stride (normally 1).",
    )
    parser.add_argument(
        "--inference-method",
        "--inference",
        dest="inference_method",
        choices=(
            *inference.CTRV_ROLLING_INFERENCE_METHODS,
            *inference.INFERENCE_METHOD_ALIASES,
        ),
        default=experiment.inference_method,
        help="Choose fixed-window VI/MCMC or online SMC/Sequential VI inference.",
    )
    parser.add_argument(
        "--vi-algorithm",
        choices=("meanfield", "fullrank"),
        default=vi_config["algorithm"],
    )
    parser.add_argument(
        "--turn-rate-prior-abs-rate-deg-s",
        type=float,
        default=priors.turn_rate_prior_abs_rate_deg_s,
        help=(
            "Absolute turn rate that the prior exceeds with its configured "
            "tail probability."
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
    arguments = parser.parse_args(argv)
    try:
        return BayesianCTRVEvaluationOptions(
            observation_count=arguments.observations,
            prediction_count=arguments.predictions,
            stride=arguments.stride,
            inference_method=arguments.inference_method,
            vi_algorithm=arguments.vi_algorithm,
            turn_rate_prior_abs_rate_deg_s=arguments.turn_rate_prior_abs_rate_deg_s,
            inference_seed=arguments.seed,
            position_noise_std_m=arguments.position_noise_std_m,
            position_noise_seed=arguments.position_noise_seed,
            require_converged=arguments.require_converged,
            max_windows=arguments.max_windows,
        )
    except ValueError as error:
        parser.error(str(error))


def parse_imm_evaluation_arguments(
    *,
    description: str | None,
    observation_count: int = 2,
    prediction_count: int,
    stride: int = 1,
    max_windows: int | None,
    show_plot: bool = True,
    argv: Sequence[str] | None = None,
) -> IMMEvaluationOptions:
    """Parse rolling controls; evaluate from N=2 like Bayesian CTRV by default."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--observations", type=int, default=observation_count)
    parser.add_argument("--predictions", type=int, default=prediction_count)
    parser.add_argument("--stride", type=int, default=stride)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=max_windows,
        help="Optional limit for a short local run.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="show_plot",
        default=show_plot,
        help="Print and export metrics without opening summary diagnostic plots.",
    )
    arguments = parser.parse_args(argv)
    for name, value in (
        ("observations", arguments.observations),
        ("predictions", arguments.predictions),
        ("stride", arguments.stride),
    ):
        _validate_imm_positive_integer(name, value)
    if arguments.max_windows is not None:
        _validate_imm_positive_integer("max_windows", arguments.max_windows)
    return IMMEvaluationOptions(
        observation_count=arguments.observations,
        prediction_count=arguments.predictions,
        stride=arguments.stride,
        max_windows=arguments.max_windows,
        show_plot=arguments.show_plot,
    )


def _validate_imm_positive_integer(name, value) -> None:
    """Raise a consistent CLI error for non-positive integer controls."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
