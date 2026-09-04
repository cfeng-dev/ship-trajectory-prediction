"""Shared CmdStan inference validation for Bayesian trajectory models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from cmdstanpy import CmdStanMCMC, CmdStanModel, CmdStanVB

from bayestraj.observations.position import (
    validate_positive_finite,
)

DEFAULT_MEANFIELD_GRAD_SAMPLES = 2
DEFAULT_VI_ADAPT_ITER = 100


def normalize_inference_method(inference_method: str) -> str:
    """Return a normalized supported inference method."""
    if not isinstance(inference_method, str):
        raise ValueError("inference_method must be 'vi' or 'mcmc'.")
    normalized = inference_method.strip().lower()
    if normalized not in {"vi", "mcmc"}:
        raise ValueError("inference_method must be 'vi' or 'mcmc'.")
    return normalized


def validate_variational_arguments(**arguments: Any) -> None:
    """Validate explicitly supported CmdStan VI controls."""
    if arguments["algorithm"] not in {"meanfield", "fullrank"}:
        raise ValueError("algorithm must be 'meanfield' or 'fullrank'.")
    for name in (
        "iter",
        "grad_samples",
        "elbo_samples",
        "adapt_iter",
        "eval_elbo",
        "draws",
        "seed",
    ):
        value = arguments[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    for name in ("eta", "tol_rel_obj"):
        validate_positive_finite(name, arguments[name])
    for name in ("require_converged", "show_console"):
        if not isinstance(arguments[name], bool):
            raise ValueError(f"{name} must be a boolean.")


def validate_mcmc_arguments(**arguments: Any) -> None:
    """Validate explicitly supported CmdStan NUTS controls."""
    for name in (
        "chains",
        "parallel_chains",
        "iter_warmup",
        "iter_sampling",
        "max_treedepth",
        "seed",
    ):
        value = arguments[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if arguments["parallel_chains"] > arguments["chains"]:
        raise ValueError("parallel_chains must not exceed chains.")
    try:
        adapt_delta = float(arguments["adapt_delta"])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("adapt_delta must be between 0 and 1.") from error
    if not np.isfinite(adapt_delta) or not 0 < adapt_delta < 1:
        raise ValueError("adapt_delta must be between 0 and 1.")
    if not isinstance(arguments["show_console"], bool):
        raise ValueError("show_console must be a boolean.")


def reject_conflicting_options(
    option_name: str,
    options: Mapping[str, Any],
    controlled_options: set[str],
) -> None:
    """Reject generic options that override explicit wrapper controls."""
    conflicting = controlled_options.intersection(options)
    if conflicting:
        names = ", ".join(sorted(conflicting))
        raise ValueError(f"{option_name} must not override: {names}.")


def run_variational_inference(
    model: CmdStanModel,
    data: Mapping[str, Any],
    *,
    algorithm: str,
    iter: int,
    grad_samples: int,
    elbo_samples: int,
    eta: float,
    adapt_iter: int,
    tol_rel_obj: float,
    eval_elbo: int,
    draws: int,
    seed: int,
    inits: Any,
    default_inits_factory: Callable[[], Any] | None,
    require_converged: bool,
    show_console: bool,
    options: Mapping[str, Any] | None = None,
) -> CmdStanVB:
    """Validate and run CmdStan variational inference."""
    validate_variational_arguments(
        algorithm=algorithm,
        iter=iter,
        grad_samples=grad_samples,
        elbo_samples=elbo_samples,
        eta=eta,
        adapt_iter=adapt_iter,
        tol_rel_obj=tol_rel_obj,
        eval_elbo=eval_elbo,
        draws=draws,
        seed=seed,
        require_converged=require_converged,
        show_console=show_console,
    )
    selected_options = dict(options or {})
    reject_conflicting_options(
        "variational_options",
        selected_options,
        {
            "data",
            "seed",
            "inits",
            "algorithm",
            "iter",
            "grad_samples",
            "elbo_samples",
            "eta",
            "adapt_iter",
            "tol_rel_obj",
            "eval_elbo",
            "draws",
            "require_converged",
            "show_console",
        },
    )
    if inits is None and default_inits_factory is not None:
        inits = default_inits_factory()
    return model.variational(
        data=data,
        seed=seed,
        inits=inits,
        algorithm=algorithm,
        iter=iter,
        grad_samples=grad_samples,
        elbo_samples=elbo_samples,
        eta=eta,
        adapt_iter=adapt_iter,
        tol_rel_obj=tol_rel_obj,
        eval_elbo=eval_elbo,
        draws=draws,
        require_converged=require_converged,
        show_console=show_console,
        **selected_options,
    )


def run_mcmc_inference(
    model: CmdStanModel,
    data: Mapping[str, Any],
    *,
    chains: int,
    parallel_chains: int | None,
    iter_warmup: int,
    iter_sampling: int,
    adapt_delta: float,
    max_treedepth: int,
    seed: int,
    inits: Any,
    default_inits_factory: Callable[[], Any] | None,
    show_console: bool,
    options: Mapping[str, Any] | None = None,
) -> CmdStanMCMC:
    """Validate and run CmdStan Markov chain Monte Carlo inference."""
    if parallel_chains is None:
        parallel_chains = chains
    validate_mcmc_arguments(
        chains=chains,
        parallel_chains=parallel_chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        seed=seed,
        show_console=show_console,
    )
    selected_options = dict(options or {})
    reject_conflicting_options(
        "mcmc_options",
        selected_options,
        {
            "data",
            "seed",
            "inits",
            "chains",
            "parallel_chains",
            "iter_warmup",
            "iter_sampling",
            "adapt_delta",
            "max_treedepth",
            "show_console",
        },
    )
    if inits is None and default_inits_factory is not None:
        inits = default_inits_factory()
    return model.sample(
        data=data,
        seed=seed,
        inits=inits,
        chains=chains,
        parallel_chains=parallel_chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        show_console=show_console,
        **selected_options,
    )


def variational_converged(fit: Any) -> bool:
    """Return whether CmdStan completed VI without its convergence warning."""
    if not hasattr(fit, "variational_sample") or not hasattr(fit, "runset"):
        raise TypeError("fit must be a CmdStan variational result.")
    stdout_files = fit.runset.stdout_files
    if not stdout_files:
        raise ValueError("Variational fit does not provide a stdout file.")
    transcript = Path(stdout_files[0]).read_text(encoding="utf-8")
    return (
        "COMPLETED." in transcript
        and "The algorithm may not have converged." not in transcript
    )
