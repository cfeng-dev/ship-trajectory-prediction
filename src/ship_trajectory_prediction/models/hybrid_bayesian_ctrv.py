"""Hybrid Bayesian CTRV model with deterministic terminal motion."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from cmdstanpy import CmdStanMCMC, CmdStanModel, CmdStanVB

from ship_trajectory_prediction.models import bayesian_ctrv as _bayesian
from ship_trajectory_prediction.models.bayesian_ctrv import (
    DEFAULT_VI_ADAPT_ITER,
    NOISE_PARAMETER_NAMES,
    BayesianCTRVPriors,
    HistoricalInitialSpeedPrior,
    PositionObservations,
    TurnRateDiagnostics,
    VIRunResult,
    compare_vi_runs,
    diagnose_observed_turn_rate,
    estimate_initial_speed_from_positions,
    estimate_initial_speed_prior_from_windows,
    normalize_inference_method,
    simulate_position_observations,
    summarize_predictions,
    variational_converged,
)
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import TrajectoryWindowData

STAN_FILE = project_path("stan/models/hybrid_bayesian_ctrv.stan")

# Terminal-motion settings for the deterministic part of the hybrid model.
FINAL_MOTION_HISTORY_SECONDS = 60
MIN_FINAL_MOTION_SPEED_MPS = 1.0


@dataclass(frozen=True, slots=True)
class HybridBayesianCTRVConfig:
    """Settings for deterministic terminal-motion estimation."""

    final_motion_history_seconds: float = FINAL_MOTION_HISTORY_SECONDS
    min_final_motion_speed_mps: float = MIN_FINAL_MOTION_SPEED_MPS

    def __post_init__(self) -> None:
        """Validate and normalize the hybrid-specific settings."""
        for name in (
            "final_motion_history_seconds",
            "min_final_motion_speed_mps",
        ):
            value = getattr(self, name)
            _bayesian._validate_positive_finite(name, value)
            object.__setattr__(self, name, float(value))


DEFAULT_HYBRID_CONFIG = HybridBayesianCTRVConfig()


def build_stan_data(
    window: TrajectoryWindowData,
    *,
    priors: BayesianCTRVPriors | None = None,
    position_observations: PositionObservations | None = None,
    hybrid_config: HybridBayesianCTRVConfig = DEFAULT_HYBRID_CONFIG,
) -> dict[str, Any]:
    """Build hybrid data with deterministic terminal heading and turn rate."""
    stan_data = _bayesian.build_stan_data(
        window,
        priors=priors,
        position_observations=position_observations,
    )
    observations = _bayesian._resolve_position_observations(
        window,
        position_observations,
    )
    turn_rate_diagnostics = _bayesian.diagnose_observed_turn_rate(
        window,
        turn_rate_state_prior_scale=(
            None if priors is None else priors.turn_rate_state_prior_scale
        ),
        position_observations=position_observations,
    )
    stan_data["turn_rate_initial_prior_mean"] = turn_rate_diagnostics.median_rad_s
    try:
        heading_final, turn_rate_final = estimate_final_motion_from_positions(
            observations.time_seconds,
            observations.x_meters,
            observations.y_meters,
            hybrid_config=hybrid_config,
        )
    except ValueError:
        heading_final = 0.0
        turn_rate_final = 0.0
        stan_data["turn_rate_initial_prior_mean"] = 0.0
    stan_data["heading_final"] = heading_final
    stan_data["turn_rate_final"] = turn_rate_final
    return stan_data


def estimate_final_motion_from_positions(
    time_seconds,
    x_meters,
    y_meters,
    *,
    hybrid_config=DEFAULT_HYBRID_CONFIG,
):
    """Estimate deterministic terminal motion over the configured history."""
    time_seconds = np.asarray(time_seconds, dtype=float)
    x_meters = np.asarray(x_meters, dtype=float)
    y_meters = np.asarray(y_meters, dtype=float)
    _bayesian._validate_matching_position_time_arrays(
        time_seconds,
        x_meters,
        y_meters,
    )
    if time_seconds.size < 2:
        raise ValueError("At least two observed positions are required.")
    if np.any(np.diff(time_seconds) <= 0):
        raise ValueError("time_seconds must be strictly increasing.")

    delta_x_final = float(x_meters[-1] - x_meters[-2])
    delta_y_final = float(y_meters[-1] - y_meters[-2])
    if np.hypot(delta_x_final, delta_y_final) <= 1e-8:
        raise ValueError("Final observed positions do not contain movement.")
    final_course = float(np.arctan2(delta_y_final, delta_x_final))

    history_start_time = time_seconds[-1] - hybrid_config.final_motion_history_seconds
    history_mask = time_seconds >= history_start_time
    history_time = time_seconds[history_mask]
    history_x = x_meters[history_mask]
    history_y = y_meters[history_mask]
    point_count = history_time.size
    if point_count < 3:
        return final_course, 0.0

    centered_time = history_time - time_seconds[-1]
    design_matrix = np.column_stack(
        (np.ones(point_count), centered_time, np.square(centered_time))
    )
    x_coefficients = np.linalg.lstsq(
        design_matrix,
        history_x,
        rcond=None,
    )[0]
    y_coefficients = np.linalg.lstsq(
        design_matrix,
        history_y,
        rcond=None,
    )[0]
    velocity_x = float(x_coefficients[1])
    velocity_y = float(y_coefficients[1])
    acceleration_x = float(2.0 * x_coefficients[2])
    acceleration_y = float(2.0 * y_coefficients[2])
    speed_squared = velocity_x**2 + velocity_y**2
    if speed_squared < hybrid_config.min_final_motion_speed_mps**2:
        return final_course, 0.0

    heading_final = float(np.arctan2(velocity_y, velocity_x))
    turn_rate_final = float(
        (velocity_x * acceleration_y - velocity_y * acceleration_x) / speed_squared
    )
    return heading_final, turn_rate_final


def compile_hybrid_bayesian_ctrv_model(
    stan_file: str | Path = STAN_FILE,
) -> CmdStanModel:
    """Compile and return the hybrid Bayesian CTRV CmdStan model."""
    return _bayesian.compile_bayesian_ctrv_model(stan_file)


def fit_hybrid_bayesian_ctrv_model(
    window: TrajectoryWindowData,
    *,
    hybrid_config: HybridBayesianCTRVConfig = DEFAULT_HYBRID_CONFIG,
    **kwargs: Any,
) -> CmdStanVB | CmdStanMCMC:
    """Fit the hybrid model through the shared VI/MCMC implementation."""
    conflicting = {"_stan_data_builder", "_model_compiler"}.intersection(kwargs)
    if conflicting:
        names = ", ".join(sorted(conflicting))
        raise ValueError(f"Hybrid backend options are internal: {names}.")
    return _bayesian.fit_bayesian_ctrv_model(
        window,
        _stan_data_builder=partial(
            build_stan_data,
            hybrid_config=hybrid_config,
        ),
        _model_compiler=compile_hybrid_bayesian_ctrv_model,
        **kwargs,
    )


__all__ = [
    "DEFAULT_VI_ADAPT_ITER",
    "DEFAULT_HYBRID_CONFIG",
    "FINAL_MOTION_HISTORY_SECONDS",
    "MIN_FINAL_MOTION_SPEED_MPS",
    "NOISE_PARAMETER_NAMES",
    "STAN_FILE",
    "BayesianCTRVPriors",
    "HistoricalInitialSpeedPrior",
    "HybridBayesianCTRVConfig",
    "PositionObservations",
    "TurnRateDiagnostics",
    "VIRunResult",
    "build_stan_data",
    "compare_vi_runs",
    "compile_hybrid_bayesian_ctrv_model",
    "diagnose_observed_turn_rate",
    "estimate_final_motion_from_positions",
    "estimate_initial_speed_from_positions",
    "estimate_initial_speed_prior_from_windows",
    "fit_hybrid_bayesian_ctrv_model",
    "normalize_inference_method",
    "simulate_position_observations",
    "summarize_predictions",
    "variational_converged",
]
