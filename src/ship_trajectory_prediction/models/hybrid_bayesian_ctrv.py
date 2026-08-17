"""Hybrid Bayesian CTRV model with deterministic terminal motion."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
from cmdstanpy import CmdStanMCMC, CmdStanModel, CmdStanVB

import ship_trajectory_prediction.models.bayesian_ctrv as bayesian_model
import ship_trajectory_prediction.models.paths as paths
import ship_trajectory_prediction.observations.window as observation_window

STAN_FILE = paths.stan_path("models/hybrid_bayesian_ctrv.stan")

# Preserve the model's established convenience exports.
DEFAULT_VI_ADAPT_ITER = bayesian_model.DEFAULT_VI_ADAPT_ITER
NOISE_PARAMETER_NAMES = (
    "sigma_position_process",
    "sigma_speed_process",
)
HistoricalInitialSpeedPrior = bayesian_model.HistoricalInitialSpeedPrior
PositionObservations = bayesian_model.PositionObservations
estimate_initial_speed_from_positions = (
    bayesian_model.estimate_initial_speed_from_positions
)
estimate_initial_speed_prior_from_windows = (
    bayesian_model.estimate_initial_speed_prior_from_windows
)
normalize_inference_method = bayesian_model.normalize_inference_method
simulate_position_observations = bayesian_model.simulate_position_observations
variational_converged = bayesian_model.variational_converged

# Terminal-motion settings for the deterministic part of the hybrid model.
FINAL_MOTION_HISTORY_SECONDS = 60
MIN_FINAL_MOTION_SPEED_MPS = 1.0
MOTION_ESTIMATORS = ("polynomial", "ctrv_fit")
MotionEstimator = Literal["polynomial", "ctrv_fit"]

_CTRV_FIT_INITIAL_GRID_SIZE = 257
_CTRV_FIT_REFINEMENT_GRID_SIZE = 65
_CTRV_FIT_REFINEMENT_STEPS = 2


@dataclass(frozen=True, slots=True)
class HybridBayesianCTRVPriors:
    """Priors for latent position, speed, and noise in the hybrid model."""

    position_initial_prior_scale: float = 5.0
    speed_initial_prior_mean: float = 0.0
    speed_initial_prior_scale: float = 0.75
    sigma_position_process_prior_scale: float = 0.534
    sigma_speed_process_prior_scale: float = 0.0438

    def __post_init__(self) -> None:
        """Normalize and validate every hybrid prior value."""
        speed_mean = bayesian_model._validate_non_negative_finite(
            "speed_initial_prior_mean",
            self.speed_initial_prior_mean,
        )
        object.__setattr__(self, "speed_initial_prior_mean", speed_mean)
        for name in (
            "position_initial_prior_scale",
            "speed_initial_prior_scale",
            "sigma_position_process_prior_scale",
            "sigma_speed_process_prior_scale",
        ):
            value = getattr(self, name)
            bayesian_model._validate_positive_finite(name, value)
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True, slots=True)
class HybridBayesianCTRVConfig:
    """Settings for deterministic terminal-motion estimation."""

    motion_estimator: MotionEstimator = "polynomial"
    final_motion_history_seconds: float = FINAL_MOTION_HISTORY_SECONDS
    min_final_motion_speed_mps: float = MIN_FINAL_MOTION_SPEED_MPS

    def __post_init__(self) -> None:
        """Validate and normalize the hybrid-specific settings."""
        if not isinstance(self.motion_estimator, str):
            raise ValueError("motion_estimator must be 'polynomial' or 'ctrv_fit'.")
        motion_estimator = self.motion_estimator.strip().lower()
        if motion_estimator not in MOTION_ESTIMATORS:
            raise ValueError("motion_estimator must be 'polynomial' or 'ctrv_fit'.")
        object.__setattr__(self, "motion_estimator", motion_estimator)
        for name in (
            "final_motion_history_seconds",
            "min_final_motion_speed_mps",
        ):
            value = getattr(self, name)
            bayesian_model._validate_positive_finite(name, value)
            object.__setattr__(self, name, float(value))


DEFAULT_HYBRID_CONFIG = HybridBayesianCTRVConfig()


def build_stan_data(
    window: observation_window.TrajectoryWindowData,
    *,
    priors: HybridBayesianCTRVPriors | None = None,
    position_observations: PositionObservations | None = None,
    hybrid_config: HybridBayesianCTRVConfig = DEFAULT_HYBRID_CONFIG,
) -> dict[str, Any]:
    """Build hybrid data with deterministic terminal heading and turn rate."""
    if priors is None:
        priors = HybridBayesianCTRVPriors()
    if not isinstance(priors, HybridBayesianCTRVPriors):
        raise TypeError("priors must be a HybridBayesianCTRVPriors instance or None.")
    stan_data, observations = bayesian_model._build_position_speed_stan_data(
        window,
        priors=priors,
        position_observations=position_observations,
    )
    try:
        heading, turn_rate = estimate_final_motion_from_positions(
            observations.time_seconds,
            observations.x_meters,
            observations.y_meters,
            hybrid_config=hybrid_config,
        )
    except ValueError:
        heading = 0.0
        turn_rate = 0.0
    stan_data["heading"] = heading
    stan_data["turn_rate"] = turn_rate
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
    bayesian_model._validate_matching_position_time_arrays(
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

    if hybrid_config.motion_estimator == "ctrv_fit":
        return _estimate_final_motion_with_ctrv_fit(
            history_time,
            history_x,
            history_y,
            final_course=final_course,
            minimum_speed_mps=hybrid_config.min_final_motion_speed_mps,
        )
    return _estimate_final_motion_with_polynomial(
        history_time,
        history_x,
        history_y,
        final_course=final_course,
        minimum_speed_mps=hybrid_config.min_final_motion_speed_mps,
    )


def _estimate_final_motion_with_polynomial(
    history_time,
    history_x,
    history_y,
    *,
    final_course,
    minimum_speed_mps,
):
    """Estimate endpoint motion from separate quadratic position fits."""
    centered_time = history_time - history_time[-1]
    point_count = history_time.size
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
    if speed_squared < minimum_speed_mps**2:
        return final_course, 0.0

    heading = float(np.arctan2(velocity_y, velocity_x))
    turn_rate = float(
        (velocity_x * acceleration_y - velocity_y * acceleration_x) / speed_squared
    )
    return heading, turn_rate


def _estimate_final_motion_with_ctrv_fit(
    history_time,
    history_x,
    history_y,
    *,
    final_course,
    minimum_speed_mps,
):
    """Estimate endpoint motion by fitting one constant-speed CTRV path."""
    centered_time = history_time - history_time[-1]
    centered_x = history_x - history_x[-1]
    centered_y = history_y - history_y[-1]
    turn_rate, coefficients = _fit_constant_turn_motion(
        centered_time,
        centered_x,
        centered_y,
    )
    velocity_x = float(coefficients[2])
    velocity_y = float(coefficients[3])
    speed = float(np.hypot(velocity_x, velocity_y))
    if speed < minimum_speed_mps:
        return final_course, 0.0
    heading = float(np.arctan2(velocity_y, velocity_x))
    return heading, float(turn_rate)


def _fit_constant_turn_motion(centered_time, centered_x, centered_y):
    """Return the least-squares terminal motion of one direct CTRV fit."""
    maximum_time_step = float(np.max(np.diff(centered_time)))
    # Avoid turn-rate aliases exceeding half a revolution between observations.
    turn_rate_limit = np.pi / maximum_time_step
    lower_bound = -turn_rate_limit
    upper_bound = turn_rate_limit
    best_turn_rate = 0.0
    best_coefficients = None

    grid_sizes = (_CTRV_FIT_INITIAL_GRID_SIZE,) + (
        (_CTRV_FIT_REFINEMENT_GRID_SIZE,) * _CTRV_FIT_REFINEMENT_STEPS
    )
    for grid_size in grid_sizes:
        candidates = np.linspace(lower_bound, upper_bound, grid_size)
        scores = np.empty(grid_size)
        coefficients_by_candidate = []
        for index, candidate in enumerate(candidates):
            score, coefficients = _solve_ctrv_at_turn_rate(
                centered_time,
                centered_x,
                centered_y,
                float(candidate),
            )
            scores[index] = score
            coefficients_by_candidate.append(coefficients)

        best_index = int(np.argmin(scores))
        best_turn_rate = float(candidates[best_index])
        best_coefficients = coefficients_by_candidate[best_index]
        lower_index = max(0, best_index - 1)
        upper_index = min(grid_size - 1, best_index + 1)
        lower_bound = float(candidates[lower_index])
        upper_bound = float(candidates[upper_index])

    return best_turn_rate, best_coefficients


def _solve_ctrv_at_turn_rate(
    centered_time,
    centered_x,
    centered_y,
    turn_rate,
):
    """Solve terminal position and velocity for one fixed turn rate."""
    if abs(turn_rate) <= 1e-12:
        along_heading = centered_time
        across_heading = np.zeros_like(centered_time)
    else:
        turn_angle = turn_rate * centered_time
        along_heading = centered_time * np.sinc(turn_angle / np.pi)
        across_heading = -2.0 * np.square(np.sin(0.5 * turn_angle)) / turn_rate

    point_count = centered_time.size
    design_matrix = np.zeros((2 * point_count, 4))
    design_matrix[0::2, 0] = 1.0
    design_matrix[1::2, 1] = 1.0
    design_matrix[0::2, 2] = along_heading
    design_matrix[0::2, 3] = across_heading
    design_matrix[1::2, 2] = -across_heading
    design_matrix[1::2, 3] = along_heading

    observed_positions = np.empty(2 * point_count)
    observed_positions[0::2] = centered_x
    observed_positions[1::2] = centered_y
    coefficients = np.linalg.lstsq(
        design_matrix,
        observed_positions,
        rcond=None,
    )[0]
    residuals = observed_positions - design_matrix @ coefficients
    return float(residuals @ residuals), coefficients


def summarize_predictions(
    fit: Any,
    window: observation_window.TrajectoryWindowData,
    credible_interval: float = 0.9,
):
    """Summarize hybrid predictions with deterministic turn-rate output."""
    return bayesian_model.summarize_predictions(
        fit,
        window,
        credible_interval,
        prediction_variables={
            "x_state": "x_state_prediction",
            "y_state": "y_state_prediction",
            "speed_state": "speed_state_prediction",
            "heading_state": "heading_state_prediction",
            "turn_rate": "turn_rate_prediction",
            "x_observation": "x_observation_prediction",
            "y_observation": "y_observation_prediction",
        },
    )


def compile_hybrid_bayesian_ctrv_model(
    stan_file: str | Path = STAN_FILE,
) -> CmdStanModel:
    """Compile and return the hybrid Bayesian CTRV CmdStan model."""
    return bayesian_model.compile_bayesian_ctrv_model(stan_file)


def fit_hybrid_bayesian_ctrv_model(
    window: observation_window.TrajectoryWindowData,
    *,
    hybrid_config: HybridBayesianCTRVConfig = DEFAULT_HYBRID_CONFIG,
    **kwargs: Any,
) -> CmdStanVB | CmdStanMCMC:
    """Fit the hybrid model through the shared VI/MCMC implementation."""
    conflicting = {"_stan_data_builder", "_model_compiler"}.intersection(kwargs)
    if conflicting:
        names = ", ".join(sorted(conflicting))
        raise ValueError(f"Hybrid backend options are internal: {names}.")
    return bayesian_model.fit_bayesian_ctrv_model(
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
    "HybridBayesianCTRVPriors",
    "MIN_FINAL_MOTION_SPEED_MPS",
    "MOTION_ESTIMATORS",
    "MotionEstimator",
    "NOISE_PARAMETER_NAMES",
    "STAN_FILE",
    "HistoricalInitialSpeedPrior",
    "HybridBayesianCTRVConfig",
    "PositionObservations",
    "build_stan_data",
    "compile_hybrid_bayesian_ctrv_model",
    "estimate_final_motion_from_positions",
    "estimate_initial_speed_from_positions",
    "estimate_initial_speed_prior_from_windows",
    "fit_hybrid_bayesian_ctrv_model",
    "normalize_inference_method",
    "simulate_position_observations",
    "summarize_predictions",
    "variational_converged",
]
