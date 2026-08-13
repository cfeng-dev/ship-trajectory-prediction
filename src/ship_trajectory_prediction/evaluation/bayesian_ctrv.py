"""Shared Bayesian CTRV evaluation configuration helpers."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ship_trajectory_prediction.models.bayesian_ctrv import (
    DEFAULT_VI_ADAPT_ITER,
    normalize_inference_method,
)

BAYESIAN_CTRV_MODEL_VARIANTS = ("bayesian", "hybrid")
DEFAULT_FULLRANK_GRAD_SAMPLES = 10


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration of one recorded-trajectory single-window experiment."""

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
    """Configuration of one recorded-trajectory rolling experiment."""

    run_id: int
    window_mode: str
    observation_count: int
    prediction_count: int
    additional_position_noise_std_m: float
    position_noise_seed: int
    stride: int | None
    inference_method: str
    inference_seed: int


def create_default_vi_config() -> dict[str, Any]:
    """Return independent default CmdStan variational-inference options."""
    return {
        "algorithm": "meanfield",
        "iter": 20_000,
        "grad_samples": 1,
        "elbo_samples": 100,
        "eta": 1.0,
        "adapt_iter": DEFAULT_VI_ADAPT_ITER,
        "tol_rel_obj": 0.01,
        "eval_elbo": 100,
        "draws": 1_000,
        "require_converged": False,
    }


def create_default_mcmc_config() -> dict[str, Any]:
    """Return independent default CmdStan MCMC options."""
    return {
        "chains": 4,
        "parallel_chains": 4,
        "iter_warmup": 1_000,
        "iter_sampling": 1_000,
        "adapt_delta": 0.9,
        "max_treedepth": 10,
    }


def normalize_bayesian_ctrv_model_variant(model_variant: str) -> str:
    """Return one supported fully Bayesian or hybrid model identifier."""
    if not isinstance(model_variant, str):
        raise ValueError("model_variant must be 'bayesian' or 'hybrid'.")
    normalized = model_variant.strip().lower()
    if normalized not in BAYESIAN_CTRV_MODEL_VARIANTS:
        raise ValueError("model_variant must be 'bayesian' or 'hybrid'.")
    return normalized


def select_bayesian_ctrv_inference_config(
    inference_method: str,
    *,
    vi_algorithm: str,
    require_converged: bool,
    vi_config: Mapping[str, Any],
    mcmc_config: Mapping[str, Any],
    fullrank_grad_samples: int,
) -> tuple[str, dict[str, Any]]:
    """Return the normalized method and its independent CmdStan options."""
    normalized_method = normalize_inference_method(inference_method)
    if normalized_method == "mcmc":
        return normalized_method, dict(mcmc_config)

    selected_config = dict(vi_config)
    selected_config.update(
        algorithm=vi_algorithm,
        require_converged=require_converged,
    )
    if vi_algorithm == "fullrank":
        selected_config["grad_samples"] = max(
            fullrank_grad_samples,
            selected_config["grad_samples"],
        )
    return normalized_method, selected_config
