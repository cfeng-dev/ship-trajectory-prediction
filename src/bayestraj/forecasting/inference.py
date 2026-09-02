"""Shared batch and online inference configuration for Bayesian forecasting."""

from collections.abc import Mapping
from typing import Any

import bayestraj.models.bayesian_ctrv as ctrv_model
import bayestraj.models.bayesian_inference as inference_support
import bayestraj.models.bayesian_position_model as position_model
import bayestraj.models.sequential_monte_carlo_ctrv as smc_model

DEFAULT_FULLRANK_GRAD_SAMPLES = 10
INFERENCE_MODES = ("batch", "online")
BATCH_INFERENCE_METHODS = ("vi", "mcmc")
ONLINE_INFERENCE_METHODS = ("rbpf",)
CTRV_ONLINE_INFERENCE_METHODS = ("rbpf", "smc")
WINDOW_MODES = ("sliding", "expanding")
_CTRV_ROLLING_BATCH_INFERENCE_METHODS = {
    "vi_sliding": ("vi", "sliding"),
    "vi_expanding": ("vi", "expanding"),
    "mcmc_sliding": ("mcmc", "sliding"),
    "mcmc_expanding": ("mcmc", "expanding"),
}
CTRV_ROLLING_INFERENCE_METHODS = (
    *_CTRV_ROLLING_BATCH_INFERENCE_METHODS,
    *CTRV_ONLINE_INFERENCE_METHODS,
)


def normalize_inference_method(
    inference_method: str,
    *,
    online_inference_methods: tuple[str, ...] = ONLINE_INFERENCE_METHODS,
) -> tuple[str, str]:
    """Validate an inference method and derive its batch or online mode."""
    if not isinstance(inference_method, str):
        raise ValueError("inference_method must be a supported inference method.")

    normalized_method = inference_method.strip().lower()
    if normalized_method in BATCH_INFERENCE_METHODS:
        return "batch", normalized_method
    if normalized_method in online_inference_methods:
        return "online", normalized_method

    allowed_methods = (*BATCH_INFERENCE_METHODS, *online_inference_methods)
    allowed = ", ".join(f"'{method}'" for method in allowed_methods)
    raise ValueError(f"inference_method must be one of {allowed}.")


def normalize_inference_configuration(
    inference_mode: str,
    inference_method: str,
    *,
    online_inference_methods: tuple[str, ...] = ONLINE_INFERENCE_METHODS,
) -> tuple[str, str]:
    """Validate and normalize one inference-mode/method combination."""
    if not isinstance(inference_mode, str):
        raise ValueError("inference_mode must be 'batch' or 'online'.")
    normalized_mode = inference_mode.strip().lower()
    if normalized_mode not in INFERENCE_MODES:
        raise ValueError("inference_mode must be 'batch' or 'online'.")

    if not isinstance(inference_method, str):
        raise ValueError("inference_method must be 'vi', 'mcmc', or 'rbpf'.")
    normalized_method = inference_method.strip().lower()
    allowed_methods = (
        BATCH_INFERENCE_METHODS
        if normalized_mode == "batch"
        else online_inference_methods
    )
    if normalized_method not in allowed_methods:
        allowed = " or ".join(f"'{method}'" for method in allowed_methods)
        raise ValueError(
            f"{normalized_mode.capitalize()} inference requires "
            f"inference_method to be {allowed}; got {inference_method!r}."
        )
    return normalized_mode, normalized_method


def normalize_rolling_inference_method(
    inference_method: str,
    window_mode: str | None,
    *,
    online_inference_methods: tuple[str, ...] = ONLINE_INFERENCE_METHODS,
) -> tuple[str, str, str | None]:
    """Derive the mode and validate its batch-only evaluation window mode."""
    normalized_mode, normalized_method = normalize_inference_method(
        inference_method,
        online_inference_methods=online_inference_methods,
    )
    if normalized_mode == "online":
        if window_mode is not None:
            raise ValueError(
                "Online inference does not use window_mode; set window_mode=None."
            )
        return normalized_mode, normalized_method, None

    if not isinstance(window_mode, str):
        raise ValueError(
            "Batch inference requires window_mode to be 'sliding' or 'expanding'."
        )
    normalized_window_mode = window_mode.strip().lower()
    if normalized_window_mode not in WINDOW_MODES:
        raise ValueError(
            "Batch inference requires window_mode to be 'sliding' or 'expanding'."
        )
    return normalized_mode, normalized_method, normalized_window_mode


def normalize_ctrv_rolling_inference_method(
    inference_method: str,
) -> tuple[str, str, str | None]:
    """Split one CTRV rolling selection into mode, method, and window mode."""
    if not isinstance(inference_method, str):
        raise ValueError("rolling inference_method must be a supported value.")

    normalized_selection = inference_method.strip().lower()
    if normalized_selection in CTRV_ONLINE_INFERENCE_METHODS:
        return "online", normalized_selection, None
    if normalized_selection in _CTRV_ROLLING_BATCH_INFERENCE_METHODS:
        method, window_mode = _CTRV_ROLLING_BATCH_INFERENCE_METHODS[
            normalized_selection
        ]
        return "batch", method, window_mode

    allowed = ", ".join(f"'{method}'" for method in CTRV_ROLLING_INFERENCE_METHODS)
    raise ValueError(f"rolling inference_method must be one of {allowed}.")


def normalize_rolling_inference_configuration(
    inference_mode: str,
    inference_method: str,
    window_mode: str | None,
    *,
    online_inference_methods: tuple[str, ...] = ONLINE_INFERENCE_METHODS,
) -> tuple[str, str, str | None]:
    """Validate inference selection and batch-only evaluation window mode."""
    normalized_mode, normalized_method = normalize_inference_configuration(
        inference_mode,
        inference_method,
        online_inference_methods=online_inference_methods,
    )
    if normalized_mode == "online":
        if window_mode is not None:
            raise ValueError(
                "Online inference does not use window_mode; set window_mode=None."
            )
        return normalized_mode, normalized_method, None

    if not isinstance(window_mode, str):
        raise ValueError(
            "Batch inference requires window_mode to be 'sliding' or 'expanding'."
        )
    normalized_window_mode = window_mode.strip().lower()
    if normalized_window_mode not in WINDOW_MODES:
        raise ValueError(
            "Batch inference requires window_mode to be 'sliding' or 'expanding'."
        )
    return normalized_mode, normalized_method, normalized_window_mode


def create_default_vi_config() -> dict[str, Any]:
    """Return independent default CmdStan variational-inference options."""
    return {
        "algorithm": "meanfield",
        "iter": 20_000,
        "grad_samples": inference_support.DEFAULT_MEANFIELD_GRAD_SAMPLES,
        "elbo_samples": 100,
        "eta": 1.0,
        "adapt_iter": inference_support.DEFAULT_VI_ADAPT_ITER,
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


def create_default_ctrv_rbpf_config() -> ctrv_model.SequentialCTRVFilterConfig:
    """Return independent default settings for the Bayesian CTRV RBPF."""
    return ctrv_model.SequentialCTRVFilterConfig()


def create_default_ctrv_smc_config() -> smc_model.SequentialMonteCarloCTRVConfig:
    """Return independent default settings for Bayesian CTRV bootstrap SMC."""
    return smc_model.SequentialMonteCarloCTRVConfig()


def create_default_position_rbpf_config() -> (
    position_model.SequentialPositionFilterConfig
):
    """Return independent default settings for the Bayesian position RBPF."""
    return position_model.SequentialPositionFilterConfig()


def select_inference_config(
    inference_mode: str,
    inference_method: str,
    *,
    vi_algorithm: str,
    require_converged: bool,
    vi_config: Mapping[str, Any],
    mcmc_config: Mapping[str, Any],
    fullrank_grad_samples: int,
) -> tuple[str, dict[str, Any]]:
    """Return one validated batch method and its independent CmdStan options."""
    normalized_mode, normalized_method = normalize_inference_configuration(
        inference_mode,
        inference_method,
    )
    if normalized_mode != "batch":
        raise ValueError("CmdStan inference configuration is only used in batch mode.")
    normalized_method = inference_support.normalize_inference_method(normalized_method)
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
