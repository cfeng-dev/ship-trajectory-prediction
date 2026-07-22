"""Bayesian trajectory prediction with time-varying motion states."""

from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanMCMC, CmdStanModel

from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import TrajectoryWindowData
from ship_trajectory_prediction.trajectory.window import (
    estimate_initial_heading,
)

STAN_FILE = project_path("stan/models/time_varying_motion.stan")


def build_stan_data(
    window: TrajectoryWindowData,
    *,
    speed_prior_log_sd=0.5,
    heading_prior_scale=0.5,
    acceleration_initial_scale=0.1,
    acceleration_state_scale=0.02,
    acceleration_decay_time=60.0,
    turn_rate_initial_scale=0.01,
    turn_rate_state_scale=0.003,
    turn_rate_decay_time=600.0,
    sigma_position=5.0,
    sigma_speed=0.2,
):
    """Build CmdStan data for the time-varying state-space model."""
    positive_values = {
        "speed_prior_log_sd": speed_prior_log_sd,
        "heading_prior_scale": heading_prior_scale,
        "acceleration_initial_scale": acceleration_initial_scale,
        "acceleration_state_scale": acceleration_state_scale,
        "acceleration_decay_time": acceleration_decay_time,
        "turn_rate_initial_scale": turn_rate_initial_scale,
        "turn_rate_state_scale": turn_rate_state_scale,
        "turn_rate_decay_time": turn_rate_decay_time,
        "sigma_position": sigma_position,
        "sigma_speed": sigma_speed,
    }
    for name, value in positive_values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite value.")

    speed_mps = window.gps_speed_mps
    if not np.all(np.isfinite(speed_mps)) or np.any(speed_mps < 0):
        raise ValueError("gps_speed must contain finite non-negative values.")

    observed = window.observed_slice
    prediction = window.prediction_slice
    speed_prior_median = _estimate_initial_speed_prior(speed_mps[observed])
    heading_prior_mean = estimate_initial_heading(
        window.x_meters[observed],
        window.y_meters[observed],
    )
    turn_rate_level = _estimate_turn_rate_level(
        window.time_seconds[observed],
        window.x_meters[observed],
        window.y_meters[observed],
    )
    return {
        "N_observed": window.observation_count,
        "time_observed": window.time_seconds[observed],
        "x_observed": window.x_meters[observed],
        "y_observed": window.y_meters[observed],
        "speed_observed": speed_mps[observed],
        "N_prediction": window.prediction_count,
        "time_prediction": window.time_seconds[prediction],
        "x_initial": float(window.x_meters[0]),
        "y_initial": float(window.y_meters[0]),
        "speed_prior_median": speed_prior_median,
        "heading_prior_mean": heading_prior_mean,
        "turn_rate_level": turn_rate_level,
        **positive_values,
    }


def compile_time_varying_motion_model(stan_file=STAN_FILE):
    """Compile and return the time-varying CmdStan model."""
    stan_file = Path(stan_file)
    if not stan_file.is_file():
        raise FileNotFoundError(f"Stan model not found: {stan_file}")
    return CmdStanModel(stan_file=str(stan_file))


def fit_time_varying_motion_model(
    window: TrajectoryWindowData,
    *,
    speed_prior_log_sd=0.5,
    heading_prior_scale=0.5,
    acceleration_initial_scale=0.1,
    acceleration_state_scale=0.02,
    acceleration_decay_time=60.0,
    turn_rate_initial_scale=0.01,
    turn_rate_state_scale=0.003,
    turn_rate_decay_time=600.0,
    sigma_position=5.0,
    sigma_speed=0.2,
    chains=4,
    parallel_chains=None,
    iter_warmup=750,
    iter_sampling=1000,
    seed=42,
    show_progress=True,
    metric="dense_e",
    adapt_delta=0.95,
    max_treedepth=12,
    inits=None,
):
    """Fit smoothly varying acceleration and turn-rate states.

    Fixed state and measurement scales keep the short trajectory identifiable.
    Decay times regularize neighboring states toward zero instead of estimating
    an independent acceleration and turn rate for every interval.
    """
    stan_data = build_stan_data(
        window,
        speed_prior_log_sd=speed_prior_log_sd,
        heading_prior_scale=heading_prior_scale,
        acceleration_initial_scale=acceleration_initial_scale,
        acceleration_state_scale=acceleration_state_scale,
        acceleration_decay_time=acceleration_decay_time,
        turn_rate_initial_scale=turn_rate_initial_scale,
        turn_rate_state_scale=turn_rate_state_scale,
        turn_rate_decay_time=turn_rate_decay_time,
        sigma_position=sigma_position,
        sigma_speed=sigma_speed,
    )
    model = compile_time_varying_motion_model()
    if parallel_chains is None:
        parallel_chains = chains
    if inits is None:
        innovation_count = stan_data["N_observed"] - 2
        inits = {
            "speed_initial": stan_data["speed_prior_median"],
            "heading_initial": stan_data["heading_prior_mean"],
            "acceleration_initial": 0.0,
            "acceleration_innovation": np.zeros(innovation_count),
            "turn_rate_initial": stan_data["turn_rate_level"],
            "turn_rate_innovation": np.zeros(innovation_count),
        }

    return model.sample(
        data=stan_data,
        chains=chains,
        parallel_chains=parallel_chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        seed=seed,
        show_progress=show_progress,
        metric=metric,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        inits=inits,
    )


def summarize_predictions(fit, window: TrajectoryWindowData, credible_interval=0.9):
    """Summarize position and speed predictions against held-out data."""
    if not 0 < credible_interval < 1:
        raise ValueError("credible_interval must be between 0 and 1.")

    x_samples = _prediction_samples(fit, "x_prediction", window.prediction_count)
    y_samples = _prediction_samples(fit, "y_prediction", window.prediction_count)
    speed_samples = _prediction_samples(
        fit,
        "speed_prediction",
        window.prediction_count,
    )
    lower_probability = (1 - credible_interval) / 2
    upper_probability = 1 - lower_probability
    prediction = window.prediction_slice

    return pd.DataFrame(
        {
            "time": window.timestamps[prediction],
            "t": window.time_seconds[prediction],
            "x_actual": window.x_meters[prediction],
            "y_actual": window.y_meters[prediction],
            "speed_actual": window.gps_speed_mps[prediction],
            "x_median": np.median(x_samples, axis=0),
            "y_median": np.median(y_samples, axis=0),
            "speed_median": np.median(speed_samples, axis=0),
            "x_lower": np.quantile(x_samples, lower_probability, axis=0),
            "x_upper": np.quantile(x_samples, upper_probability, axis=0),
            "y_lower": np.quantile(y_samples, lower_probability, axis=0),
            "y_upper": np.quantile(y_samples, upper_probability, axis=0),
            "speed_lower": np.quantile(
                speed_samples,
                lower_probability,
                axis=0,
            ),
            "speed_upper": np.quantile(
                speed_samples,
                upper_probability,
                axis=0,
            ),
        }
    )


def _prediction_samples(fit, variable_name, prediction_count):
    """Extract and validate one posterior prediction matrix."""
    if not isinstance(fit, CmdStanMCMC) and not hasattr(fit, "stan_variable"):
        raise TypeError("fit must provide CmdStan-style posterior variables.")

    samples = np.asarray(fit.stan_variable(variable_name), dtype=float)
    if samples.ndim != 2 or samples.shape[1] != prediction_count:
        raise ValueError(
            f"Posterior variable {variable_name!r} has an unexpected shape."
        )
    return samples


def _estimate_initial_speed_prior(speed_values):
    """Estimate a robust initial-speed prior from the first moving samples."""
    positive = speed_values[speed_values > 0]
    if len(positive) == 0:
        raise ValueError("Observed gps_speed must contain positive values.")
    return float(np.median(positive[: min(3, len(positive))]))


def _estimate_turn_rate_level(time_seconds, x_meters, y_meters):
    """Estimate the persistent turn-rate trend from observed course changes."""
    delta_x = np.diff(x_meters)
    delta_y = np.diff(y_meters)
    moving = np.hypot(delta_x, delta_y) > 1e-8
    if np.count_nonzero(moving) < 2:
        raise ValueError("Observed positions do not contain enough movement for turn.")

    segment_time = 0.5 * (time_seconds[:-1] + time_seconds[1:])
    heading = np.unwrap(np.arctan2(delta_y[moving], delta_x[moving]))
    turn_rate = float(np.polyfit(segment_time[moving], heading, 1)[0])
    return float(np.clip(turn_rate, -0.1, 0.1))
