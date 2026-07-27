"""Synthetic noisy CTRV trajectories for Bayesian model validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ship_trajectory_prediction.coordinates import local_to_gps_coordinates
from ship_trajectory_prediction.models.ctrv import CTRVState, ctrv_step


@dataclass(frozen=True, slots=True)
class SyntheticCTRVNoise:
    """Known measurement and diffusion scales used by the simulator.

    GPS position is in meters and GPS speed in meters per second. Process
    scales are state units per square-root second because each transition uses
    ``sigma * sqrt(dt)``.
    """

    sigma_position_gps: float = 1.5
    sigma_speed_gps: float = 0.15
    sigma_position_process: float = 0.15
    sigma_speed_process: float = 0.02
    sigma_turn_rate_process: float = 0.0003

    def __post_init__(self) -> None:
        """Reject negative or non-finite simulation scales."""
        for field_name in self.__dataclass_fields__:
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be a finite non-negative value.")
            object.__setattr__(self, field_name, value)


def simulate_synthetic_ctrv_data(
    *,
    count: int = 18,
    dt_seconds: float = 5.0,
    time_seconds: Sequence[float] | None = None,
    initial_state: CTRVState | None = None,
    noise: SyntheticCTRVNoise | None = None,
    seed: int = 42,
    run_id: int = 0,
    reference_longitude: float = 8.55,
    reference_latitude: float = 47.37,
) -> pd.DataFrame:
    """Simulate latent CTRV states and noisy GPS observations.

    The returned frame is directly consumable by ``prepare_trajectory_window``.
    It also retains the aligned latent truth and all known noise scales for
    recovery and coverage checks. ``gps_speed`` follows the repository CSV
    convention and is stored in kilometers per hour.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 3:
        raise ValueError("count must be an integer greater than or equal to 3.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")

    if time_seconds is None:
        if not np.isfinite(dt_seconds) or dt_seconds <= 0:
            raise ValueError("dt_seconds must be a positive finite value.")
        times = np.arange(count, dtype=float) * float(dt_seconds)
    else:
        times = np.asarray(time_seconds, dtype=float)
        if times.shape != (count,) or not np.all(np.isfinite(times)):
            raise ValueError("time_seconds must contain one finite value per row.")
        if times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError(
                "time_seconds must start at zero and be strictly increasing."
            )

    if initial_state is None:
        initial_state = CTRVState(
            x=0.0,
            y=0.0,
            speed=3.0,
            heading=0.35,
            turn_rate=0.012,
        )
    if not isinstance(initial_state, CTRVState):
        raise TypeError("initial_state must be a CTRVState instance.")
    if noise is None:
        noise = SyntheticCTRVNoise()
    if not isinstance(noise, SyntheticCTRVNoise):
        raise TypeError("noise must be a SyntheticCTRVNoise instance.")

    generator = np.random.default_rng(seed)
    truth = [initial_state]
    for index in range(1, count):
        dt = float(times[index] - times[index - 1])
        deterministic = ctrv_step(truth[-1], dt)
        truth.append(
            CTRVState(
                x=deterministic.x
                + generator.normal(
                    0,
                    noise.sigma_position_process * np.sqrt(dt),
                ),
                y=deterministic.y
                + generator.normal(
                    0,
                    noise.sigma_position_process * np.sqrt(dt),
                ),
                speed=max(
                    0.001,
                    deterministic.speed
                    + generator.normal(
                        0,
                        noise.sigma_speed_process * np.sqrt(dt),
                    ),
                ),
                heading=deterministic.heading,
                turn_rate=float(
                    np.clip(
                        deterministic.turn_rate
                        + generator.normal(
                            0,
                            noise.sigma_turn_rate_process * np.sqrt(dt),
                        ),
                        -0.1,
                        0.1,
                    )
                ),
            )
        )

    x_true_raw = np.asarray([state.x for state in truth])
    y_true_raw = np.asarray([state.y for state in truth])
    speed_true = np.asarray([state.speed for state in truth])
    heading_true = np.asarray([state.heading for state in truth])
    turn_rate_true = np.asarray([state.turn_rate for state in truth])
    x_observed_raw = x_true_raw + generator.normal(
        0,
        noise.sigma_position_gps,
        count,
    )
    y_observed_raw = y_true_raw + generator.normal(
        0,
        noise.sigma_position_gps,
        count,
    )
    speed_observed = np.maximum(
        0,
        speed_true + generator.normal(0, noise.sigma_speed_gps, count),
    )

    # Window preprocessing defines the first GPS observation as (0, 0). Align
    # both observation and truth columns to that exact reference frame.
    x_reference = x_observed_raw[0]
    y_reference = y_observed_raw[0]
    x_observed = x_observed_raw - x_reference
    y_observed = y_observed_raw - y_reference
    x_true = x_true_raw - x_reference
    y_true = y_true_raw - y_reference
    longitude, latitude = local_to_gps_coordinates(
        x_observed,
        y_observed,
        reference_longitude,
        reference_latitude,
        unit="m",
    )

    start_time = pd.Timestamp("2025-01-01T12:00:00Z")
    result = pd.DataFrame(
        {
            "time": start_time + pd.to_timedelta(times, unit="s"),
            "run_id": run_id,
            "gps_latitude": latitude,
            "gps_longitude": longitude,
            "gps_speed": speed_observed * 3.6,
            "x_observed": x_observed,
            "y_observed": y_observed,
            "speed_observed": speed_observed,
            "x_true": x_true,
            "y_true": y_true,
            "speed_true": speed_true,
            "heading_true": heading_true,
            "turn_rate_true": turn_rate_true,
        }
    )
    for field_name in noise.__dataclass_fields__:
        result[f"{field_name}_true"] = getattr(noise, field_name)
    return result
