"""Bayesian constant-turn-rate trajectory prediction with derived radius."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanMCMC, CmdStanModel

from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory.window import (
    estimate_initial_heading,
    estimate_positive_speed_median,
)
from ship_trajectory_prediction.trajectory.window import (
    prepare_trajectory_window as _prepare_base_trajectory_window,
)

STAN_FILE = project_path("stan/models/constant_turn_rate.stan")


@dataclass(frozen=True)
class TrajectoryWindow:
    """Prepared observation and evaluation window for one ship trajectory."""

    timestamps: pd.DatetimeIndex
    time_seconds: np.ndarray
    x_meters: np.ndarray
    y_meters: np.ndarray
    observation_count: int
    speed_prior_median: float
    heading_prior_mean: float

    @property
    def prediction_count(self):
        """Return the number of held-out future observations."""
        return len(self.time_seconds) - self.observation_count

    @property
    def observed_slice(self):
        """Return the slice selecting observations used for inference."""
        return slice(0, self.observation_count)

    @property
    def prediction_slice(self):
        """Return the slice selecting held-out observations."""
        return slice(self.observation_count, None)


def prepare_trajectory_window(
    data,
    observation_count=20,
    prediction_count=10,
    *,
    start_index=0,
    speed_unit="km/h",
):
    """Prepare one real-data window for constant-turn-rate inference.

    The input must contain exactly one trajectory run. GPS coordinates are
    converted to local meters using the first selected position as origin.
    Observed speed and initial heading provide prior centers; Stan estimates
    the actual speed, heading, and signed turn rate. The remaining positions
    are retained exclusively for evaluation.
    """
    base_window = _prepare_base_trajectory_window(
        data,
        observation_count=observation_count,
        prediction_count=prediction_count,
        start_index=start_index,
        speed_unit=speed_unit,
    )

    observed = base_window.observed_slice
    try:
        speed_prior_median = estimate_positive_speed_median(
            base_window.gps_speed_mps[observed]
        )
    except ValueError as error:
        raise ValueError(
            "Observed gps_speed must contain positive finite values."
        ) from error
    heading_prior_mean = estimate_initial_heading(
        base_window.x_meters[observed],
        base_window.y_meters[observed],
    )
    return TrajectoryWindow(
        timestamps=base_window.timestamps,
        time_seconds=base_window.time_seconds,
        x_meters=base_window.x_meters,
        y_meters=base_window.y_meters,
        observation_count=base_window.observation_count,
        speed_prior_median=speed_prior_median,
        heading_prior_mean=heading_prior_mean,
    )


def build_stan_data(
    window,
    *,
    speed_prior_log_sd=0.5,
    heading_prior_scale=0.5,
    turn_rate_prior_scale=0.01,
    sigma_prior_scale=20.0,
):
    """Build the CmdStan data dictionary for a prepared trajectory window."""
    prior_values = {
        "speed_prior_median": window.speed_prior_median,
        "speed_prior_log_sd": speed_prior_log_sd,
        "heading_prior_scale": heading_prior_scale,
        "turn_rate_prior_scale": turn_rate_prior_scale,
        "sigma_prior_scale": sigma_prior_scale,
    }
    for name, value in prior_values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite value.")

    observed = window.observed_slice
    prediction = window.prediction_slice
    return {
        "N_observed": window.observation_count,
        "time_observed": window.time_seconds[observed],
        "x_observed": window.x_meters[observed],
        "y_observed": window.y_meters[observed],
        "N_prediction": window.prediction_count,
        "time_prediction": window.time_seconds[prediction],
        "x_initial": float(window.x_meters[0]),
        "y_initial": float(window.y_meters[0]),
        "heading_prior_mean": window.heading_prior_mean,
        **prior_values,
    }


def compile_constant_turn_rate_model(stan_file=STAN_FILE):
    """Compile and return the constant-turn-rate CmdStan model."""
    stan_file = Path(stan_file)
    if not stan_file.is_file():
        raise FileNotFoundError(f"Stan model not found: {stan_file}")
    return CmdStanModel(stan_file=str(stan_file))


def fit_constant_turn_rate_model(
    window,
    *,
    speed_prior_log_sd=0.5,
    heading_prior_scale=0.5,
    turn_rate_prior_scale=0.01,
    sigma_prior_scale=20.0,
    chains=4,
    parallel_chains=None,
    iter_warmup=500,
    iter_sampling=1000,
    seed=42,
    show_progress=True,
    inits=None,
):
    """Estimate speed, heading, and signed turn rate for a prepared window.

    Radius is derived for every posterior draw as ``speed / abs(turn_rate)``;
    it is not sampled as an independent parameter. Positive turn rates denote
    counterclockwise (left) turns and negative values clockwise (right) turns.
    """
    stan_data = build_stan_data(
        window,
        speed_prior_log_sd=speed_prior_log_sd,
        heading_prior_scale=heading_prior_scale,
        turn_rate_prior_scale=turn_rate_prior_scale,
        sigma_prior_scale=sigma_prior_scale,
    )
    model = compile_constant_turn_rate_model()
    if parallel_chains is None:
        parallel_chains = chains
    if inits is None:
        inits = {
            "speed": window.speed_prior_median,
            "heading_initial": window.heading_prior_mean,
            "turn_rate": 0.0,
            "sigma": sigma_prior_scale / 2,
        }

    return model.sample(
        data=stan_data,
        chains=chains,
        parallel_chains=parallel_chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        seed=seed,
        show_progress=show_progress,
        inits=inits,
    )


def summarize_predictions(fit, window, credible_interval=0.9):
    """Summarize posterior predictive positions against held-out observations."""
    if not 0 < credible_interval < 1:
        raise ValueError("credible_interval must be between 0 and 1.")

    x_samples = _prediction_samples(fit, "x_prediction", window.prediction_count)
    y_samples = _prediction_samples(fit, "y_prediction", window.prediction_count)
    lower_probability = (1 - credible_interval) / 2
    upper_probability = 1 - lower_probability
    prediction = window.prediction_slice

    return pd.DataFrame(
        {
            "time": window.timestamps[prediction],
            "t": window.time_seconds[prediction],
            "x_actual": window.x_meters[prediction],
            "y_actual": window.y_meters[prediction],
            "x_median": np.median(x_samples, axis=0),
            "y_median": np.median(y_samples, axis=0),
            "x_lower": np.quantile(x_samples, lower_probability, axis=0),
            "x_upper": np.quantile(x_samples, upper_probability, axis=0),
            "y_lower": np.quantile(y_samples, lower_probability, axis=0),
            "y_upper": np.quantile(y_samples, upper_probability, axis=0),
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
