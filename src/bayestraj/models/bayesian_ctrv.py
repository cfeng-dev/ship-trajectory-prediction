"""Bayesian CTRV with stochastic kinematics and conditional positions.

Position is deterministic conditional on latent speed, heading, and turn rate;
there is no independent additive Cartesian position-process noise.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from statistics import NormalDist
from typing import Any

import numpy as np

import bayestraj.numeric_validation as numeric_validation
import bayestraj.observations.position as observation_support
import bayestraj.observations.window as observation_window
import bayestraj.stan as stan_resources
from bayestraj.models.ctrv import (
    PROCESS_REFERENCE_INTERVAL_SECONDS,
    SPEED_STATE_LOWER_MPS,
)

STAN_FILE = stan_resources.stan_path("models/bayesian_ctrv.stan")

MIN_OBSERVATION_COUNT = 3

PositionObservations = observation_support.PositionObservations
simulate_position_observations = observation_support.simulate_position_observations

NOISE_PARAMETER_NAMES = (
    "sigma_position_observation",
    "sigma_speed_process",
    "sigma_turn_rate_process",
)
PARAMETER_NAMES = (
    "speed_at_origin",
    "heading_at_origin",
    "turn_rate_at_origin",
    *NOISE_PARAMETER_NAMES,
)


@dataclass(frozen=True, slots=True)
class BayesianCTRVPriors:
    """Ship-independent priors for Bayesian CTRV state dynamics."""

    speed_prior_upper_mps: float = 20.0
    speed_prior_tail_probability: float = 0.05
    turn_rate_prior_abs_heading_change_deg: float = 45.0
    turn_rate_prior_reference_interval_seconds: float = 10.0
    turn_rate_prior_tail_probability: float = 0.05
    sigma_position_observation_prior_upper_m: float = 20.0
    sigma_position_observation_prior_tail_probability: float = 0.05
    sigma_speed_process_prior_upper_mps: float = 5.0
    sigma_speed_process_prior_tail_probability: float = 0.05
    sigma_turn_rate_process_prior_upper_deg_s: float = 4.5
    sigma_turn_rate_process_prior_tail_probability: float = 0.05

    def __post_init__(self) -> None:
        """Validate and normalize all configured prior values."""
        for prior_field in fields(self):
            name = prior_field.name
            value = getattr(self, name)
            if name.endswith("_tail_probability"):
                value = numeric_validation.validate_finite_scalar(name, value)
                if not 0.0 < value < 1.0:
                    raise ValueError(f"{name} must be strictly between zero and one.")
            elif name == "turn_rate_prior_abs_heading_change_deg":
                value = numeric_validation.validate_positive_finite(name, value)
                if value > 180.0:
                    raise ValueError(f"{name} must not exceed 180 degrees.")
            else:
                value = numeric_validation.validate_positive_finite(name, value)
            object.__setattr__(self, name, float(value))

    @property
    def speed_prior_scale(self) -> float:
        """Return the half-normal scale implied by the speed tail statement."""
        return _two_sided_normal_scale(
            self.speed_prior_upper_mps,
            self.speed_prior_tail_probability,
        )

    @property
    def turn_rate_prior_scale(self) -> float:
        """Return the normal turn-rate scale implied by a heading-change tail."""
        absolute_upper_rad_s = (
            np.deg2rad(self.turn_rate_prior_abs_heading_change_deg)
            / self.turn_rate_prior_reference_interval_seconds
        )
        return _two_sided_normal_scale(
            absolute_upper_rad_s,
            self.turn_rate_prior_tail_probability,
        )

    @property
    def sigma_position_observation_prior_rate(self) -> float:
        """Return the exponential rate implied by one prior tail statement."""
        return _exponential_rate_from_tail(
            self.sigma_position_observation_prior_upper_m,
            self.sigma_position_observation_prior_tail_probability,
        )

    @property
    def sigma_speed_process_prior_rate(self) -> float:
        """Return the speed-process exponential rate from its tail statement."""
        return _exponential_rate_from_tail(
            self.sigma_speed_process_prior_upper_mps,
            self.sigma_speed_process_prior_tail_probability,
        )

    @property
    def sigma_turn_rate_process_prior_rate(self) -> float:
        """Return the turn-rate-process exponential rate in seconds per radian."""
        upper_rad_s = np.deg2rad(self.sigma_turn_rate_process_prior_upper_deg_s)
        return _exponential_rate_from_tail(
            upper_rad_s,
            self.sigma_turn_rate_process_prior_tail_probability,
        )


def build_stan_data(
    window: observation_window.TrajectoryWindowData,
    *,
    priors: BayesianCTRVPriors | None = None,
    position_observations: PositionObservations | None = None,
) -> dict[str, Any]:
    """Build position-only Stan data from the complete observed window."""
    if priors is None:
        priors = BayesianCTRVPriors()
    if not isinstance(priors, BayesianCTRVPriors):
        raise TypeError("priors must be a BayesianCTRVPriors instance or None.")
    if window.observation_count < MIN_OBSERVATION_COUNT:
        raise ValueError(
            f"window must contain at least {MIN_OBSERVATION_COUNT} observed positions."
        )
    if window.prediction_count < 1:
        raise ValueError("window must contain at least one prediction position.")

    position_observations = observation_support.resolve_position_observations(
        window,
        position_observations,
    )
    time_observed = np.asarray(
        position_observations.time_seconds,
        dtype=float,
    )
    x_observed = np.asarray(
        position_observations.x_meters,
        dtype=float,
    )
    y_observed = np.asarray(
        position_observations.y_meters,
        dtype=float,
    )
    time_prediction = np.asarray(
        window.time_seconds[window.prediction_slice],
        dtype=float,
    )
    _validate_time_arrays(time_observed, time_prediction)
    numeric_validation.validate_finite_vector("x_observed", x_observed)
    numeric_validation.validate_finite_vector("y_observed", y_observed)

    return {
        "N_history": window.observation_count,
        "time_observed": time_observed,
        "x_observed": x_observed,
        "y_observed": y_observed,
        "sigma_position_observation_prior_rate": (
            priors.sigma_position_observation_prior_rate
        ),
        "sigma_speed_process_prior_rate": priors.sigma_speed_process_prior_rate,
        "sigma_turn_rate_process_prior_rate": (
            priors.sigma_turn_rate_process_prior_rate
        ),
        "process_reference_interval_seconds": PROCESS_REFERENCE_INTERVAL_SECONDS,
        "speed_state_lower_mps": SPEED_STATE_LOWER_MPS,
        "N_prediction": window.prediction_count,
        "time_prediction": time_prediction,
        "speed_prior_scale": priors.speed_prior_scale,
        "turn_rate_prior_scale": priors.turn_rate_prior_scale,
    }


def estimate_constant_motion_from_positions(
    time_seconds,
    x_meters,
    y_meters,
) -> tuple[float, float, float]:
    """Estimate numerical initials for speed, initial heading, and turn rate."""
    time_seconds = np.asarray(time_seconds, dtype=float)
    x_meters = np.asarray(x_meters, dtype=float)
    y_meters = np.asarray(y_meters, dtype=float)
    observation_support.validate_matching_position_time_arrays(
        time_seconds,
        x_meters,
        y_meters,
    )
    time_steps = np.diff(time_seconds)
    if np.any(time_steps <= 0):
        raise ValueError("time_seconds must be strictly increasing.")
    delta_x = np.diff(x_meters)
    delta_y = np.diff(y_meters)
    displacement = np.hypot(delta_x, delta_y)
    speed = float(np.median(displacement / time_steps))
    moving = displacement > 1e-9
    if not np.any(moving):
        return 0.0, 0.0, 0.0

    segment_heading = np.unwrap(np.arctan2(delta_y[moving], delta_x[moving]))
    segment_mid_time = (
        0.5 * (time_seconds[:-1] + time_seconds[1:])[moving] - time_seconds[0]
    )
    if segment_heading.size < 2 or np.ptp(segment_mid_time) <= 0:
        turn_rate = 0.0
        heading_initial = float(segment_heading[0])
    else:
        turn_rate, heading_initial = np.polyfit(
            segment_mid_time,
            segment_heading,
            deg=1,
        )
    heading_initial = float(
        np.arctan2(np.sin(heading_initial), np.cos(heading_initial))
    )
    return speed, heading_initial, float(turn_rate)


def _validate_time_arrays(time_observed, time_prediction) -> None:
    """Validate selected history and future timestamps."""
    numeric_validation.validate_finite_vector("time_observed", time_observed)
    numeric_validation.validate_finite_vector("time_prediction", time_prediction)
    if np.any(np.diff(time_observed) <= 0):
        raise ValueError("Observed timestamps must be strictly increasing.")
    if time_prediction[0] <= time_observed[-1] or np.any(np.diff(time_prediction) <= 0):
        raise ValueError(
            "Prediction timestamps must be strictly increasing and follow "
            "the observed timestamps."
        )


def _two_sided_normal_scale(absolute_upper: float, tail_probability: float) -> float:
    """Return a zero-centered normal scale from a two-sided tail statement."""
    quantile = NormalDist().inv_cdf(1.0 - tail_probability / 2.0)
    return float(absolute_upper / quantile)


def _exponential_rate_from_tail(upper: float, tail_probability: float) -> float:
    """Return an exponential rate from one upper-tail probability statement."""
    return float(-np.log(tail_probability) / upper)
