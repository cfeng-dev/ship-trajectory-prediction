"""Single-window hybrid Bayesian CTRV prediction workflow."""

from collections.abc import Mapping
from functools import partial
from typing import Any

import numpy as np

import ship_trajectory_prediction.forecasting.bayesian_ctrv as forecasting
import ship_trajectory_prediction.forecasting.bayesian_ctrv_workflow as shared_workflow
import ship_trajectory_prediction.models.hybrid_bayesian_ctrv as hybrid_model


def run_hybrid_bayesian_ctrv_prediction(
    *,
    data_file,
    experiment: forecasting.ExperimentConfig,
    priors: hybrid_model.HybridBayesianCTRVPriors,
    hybrid_config: hybrid_model.HybridBayesianCTRVConfig,
    vi_config: Mapping[str, Any],
    mcmc_config: Mapping[str, Any],
    fullrank_grad_samples: int,
    credible_interval: float,
    inference_method: str,
    vi_algorithm: str,
    seed: int,
    position_noise_std_m: float,
    position_noise_seed: int,
    require_converged: bool,
    plot_coordinate_mode: str,
) -> None:
    """Fit and evaluate one hybrid Bayesian CTRV prediction."""
    return shared_workflow.run_bayesian_ctrv_prediction(
        data_file=data_file,
        experiment=experiment,
        priors=priors,
        vi_config=vi_config,
        mcmc_config=mcmc_config,
        fullrank_grad_samples=fullrank_grad_samples,
        credible_interval=credible_interval,
        inference_method=inference_method,
        vi_algorithm=vi_algorithm,
        seed=seed,
        position_noise_std_m=position_noise_std_m,
        position_noise_seed=position_noise_seed,
        require_converged=require_converged,
        plot_coordinate_mode=plot_coordinate_mode,
        model_label="Hybrid Bayesian CTRV",
        stan_data_builder=partial(
            hybrid_model.build_stan_data,
            hybrid_config=hybrid_config,
        ),
        fit_model=partial(
            hybrid_model.fit_hybrid_bayesian_ctrv_model,
            hybrid_config=hybrid_config,
        ),
        build_motion_setup_rows=partial(
            _hybrid_motion_setup_rows,
            history_seconds=hybrid_config.final_motion_history_seconds,
        ),
        build_motion_state_rows=_hybrid_motion_state_rows,
        noise_parameter_names=hybrid_model.NOISE_PARAMETER_NAMES,
    )


def _hybrid_motion_setup_rows(stan_data, *, history_seconds):
    """Describe the deterministic motion inputs of the hybrid model."""
    return [
        (
            "Deterministic heading",
            f"{stan_data['heading']:.4f} rad "
            f"({np.degrees(stan_data['heading']):.1f} deg)",
        ),
        (
            "Deterministic turn rate",
            f"{stan_data['turn_rate']:.5f} rad/s",
        ),
        ("Motion history", f"{history_seconds:g} s"),
    ]


def _hybrid_motion_state_rows(fit, stan_data):
    """Return the fixed turn-rate row of the hybrid model."""
    del fit
    return [("Turn rate [rad/s]", f"fixed at {stan_data['turn_rate']:.5f}")]
