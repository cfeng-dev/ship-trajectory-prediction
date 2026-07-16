"""Bayesian constant-radius trajectory prediction with CmdStanPy."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanMCMC, CmdStanModel

from ship_trajectory_prediction.coordinates import gps_to_local_coordinates
from ship_trajectory_prediction.paths import project_path

KILOMETERS_PER_HOUR_TO_METERS_PER_SECOND = 1 / 3.6
STAN_FILE = project_path("stan/models/constant_radius.stan")


@dataclass(frozen=True)
class TrajectoryWindow:
    """Prepared observation and evaluation window for one ship trajectory."""

    timestamps: pd.DatetimeIndex
    time_seconds: np.ndarray
    x_meters: np.ndarray
    y_meters: np.ndarray
    observation_count: int
    speed_mps: float
    initial_heading: float
    turn_direction: int

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
    turn_direction=None,
):
    """Prepare one real-data window for constant-radius inference.

    The input must contain exactly one trajectory run. GPS coordinates are
    converted to local meters using the first selected position as origin.
    Speed and initial heading are estimated from the observed part only; the
    remaining positions are retained exclusively for evaluation.
    """
    _validate_window_arguments(
        observation_count,
        prediction_count,
        start_index,
        speed_unit,
        turn_direction,
    )

    required_columns = {
        "time",
        "run_id",
        "gps_latitude",
        "gps_longitude",
        "gps_speed",
    }
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data["run_id"].nunique(dropna=False) != 1:
        raise ValueError("Trajectory data must contain exactly one run_id.")

    prepared_data = data.copy()
    prepared_data["time"] = pd.to_datetime(
        prepared_data["time"],
        utc=True,
        format="mixed",
    )
    prepared_data = prepared_data.sort_values("time").reset_index(drop=True)

    window_size = observation_count + prediction_count
    stop_index = start_index + window_size
    if stop_index > len(prepared_data):
        raise ValueError(
            "Not enough trajectory rows for the requested observation and "
            "prediction window."
        )

    window_data = prepared_data.iloc[start_index:stop_index].copy()
    timestamps = pd.DatetimeIndex(window_data["time"])
    time_seconds = (timestamps - timestamps[0]).total_seconds().to_numpy(dtype=float)
    if np.any(np.diff(time_seconds) <= 0):
        raise ValueError("Trajectory timestamps must be strictly increasing.")

    longitude = pd.to_numeric(
        window_data["gps_longitude"],
        errors="coerce",
    ).to_numpy(dtype=float)
    latitude = pd.to_numeric(
        window_data["gps_latitude"],
        errors="coerce",
    ).to_numpy(dtype=float)
    x_meters, y_meters = gps_to_local_coordinates(
        longitude,
        latitude,
        unit="m",
    )

    observed_slice = slice(0, observation_count)
    speed_mps = _estimate_speed(
        window_data["gps_speed"].iloc[observed_slice],
        speed_unit,
    )
    initial_heading = _estimate_initial_heading(
        x_meters[observed_slice],
        y_meters[observed_slice],
    )
    if turn_direction is None:
        turn_direction = _infer_turn_direction(
            x_meters[observed_slice],
            y_meters[observed_slice],
        )

    return TrajectoryWindow(
        timestamps=timestamps,
        time_seconds=time_seconds,
        x_meters=x_meters,
        y_meters=y_meters,
        observation_count=observation_count,
        speed_mps=speed_mps,
        initial_heading=initial_heading,
        turn_direction=turn_direction,
    )


def build_stan_data(
    window,
    *,
    radius_prior_median=500.0,
    radius_prior_log_sd=1.0,
    sigma_prior_scale=20.0,
):
    """Build the CmdStan data dictionary for a prepared trajectory window."""
    prior_values = {
        "radius_prior_median": radius_prior_median,
        "radius_prior_log_sd": radius_prior_log_sd,
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
        "speed": window.speed_mps,
        "heading_initial": window.initial_heading,
        "turn_direction": window.turn_direction,
        **prior_values,
    }


def compile_constant_radius_model(stan_file=STAN_FILE):
    """Compile and return the constant-radius CmdStan model."""
    stan_file = Path(stan_file)
    if not stan_file.is_file():
        raise FileNotFoundError(f"Stan model not found: {stan_file}")
    return CmdStanModel(stan_file=str(stan_file))


def fit_constant_radius_model(
    window,
    *,
    radius_prior_median=500.0,
    radius_prior_log_sd=1.0,
    sigma_prior_scale=20.0,
    chains=4,
    parallel_chains=None,
    iter_warmup=500,
    iter_sampling=1000,
    seed=42,
    show_progress=True,
    inits=None,
):
    """Fit the Bayesian constant-radius model to a prepared window."""
    stan_data = build_stan_data(
        window,
        radius_prior_median=radius_prior_median,
        radius_prior_log_sd=radius_prior_log_sd,
        sigma_prior_scale=sigma_prior_scale,
    )
    model = compile_constant_radius_model()
    if parallel_chains is None:
        parallel_chains = chains
    if inits is None:
        inits = {
            "radius": radius_prior_median,
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


def _estimate_speed(speed_values, speed_unit):
    """Estimate constant positive speed from observed GPS speed values."""
    speed_values = pd.to_numeric(speed_values, errors="coerce").to_numpy(dtype=float)
    valid_speeds = speed_values[np.isfinite(speed_values) & (speed_values > 0)]
    if len(valid_speeds) == 0:
        raise ValueError("Observed gps_speed must contain positive finite values.")

    speed = float(np.median(valid_speeds))
    if speed_unit == "km/h":
        speed *= KILOMETERS_PER_HOUR_TO_METERS_PER_SECOND
    return speed


def _estimate_initial_heading(x_meters, y_meters):
    """Estimate initial heading from the first few observed positions."""
    endpoint = min(4, len(x_meters) - 1)
    while endpoint > 0:
        delta_x = x_meters[endpoint] - x_meters[0]
        delta_y = y_meters[endpoint] - y_meters[0]
        if np.hypot(delta_x, delta_y) > 1e-8:
            return float(np.arctan2(delta_y, delta_x))
        endpoint -= 1
    raise ValueError("Observed positions do not contain enough movement for heading.")


def _infer_turn_direction(x_meters, y_meters):
    """Infer clockwise or counterclockwise motion from observed headings."""
    delta_x = np.diff(x_meters)
    delta_y = np.diff(y_meters)
    moving = np.hypot(delta_x, delta_y) > 1e-8
    headings = np.unwrap(np.arctan2(delta_y[moving], delta_x[moving]))
    if len(headings) < 2:
        raise ValueError("Observed positions do not contain enough movement for turn.")

    total_heading_change = float(headings[-1] - headings[0])
    return -1 if total_heading_change < 0 else 1


def _validate_window_arguments(
    observation_count,
    prediction_count,
    start_index,
    speed_unit,
    turn_direction,
):
    """Validate trajectory window configuration."""
    integer_arguments = {
        "observation_count": (observation_count, 3),
        "prediction_count": (prediction_count, 1),
        "start_index": (start_index, 0),
    }
    for name, (value, minimum) in integer_arguments.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"{name} must be an integer greater than or equal to {minimum}."
            )

    if speed_unit not in {"km/h", "m/s"}:
        raise ValueError("speed_unit must be 'km/h' or 'm/s'.")
    if turn_direction not in {None, -1, 1}:
        raise ValueError("turn_direction must be -1, 1, or None.")
