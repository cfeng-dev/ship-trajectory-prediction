"""Synthetic position-only data for validating a switching CTRV model.

The scenario contains four consecutive transition phases: stop, cruise,
maneuver, and cruise.  True continuous states and modes are retained solely
for evaluation.  Model-ready inputs remain timestamps and noisy GPS position
proxies; ``gps_speed`` is an independently noisy external reference and is not
needed to generate positions or modes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from ship_trajectory_prediction.coordinates import local_to_gps_coordinates
from ship_trajectory_prediction.models.ctrv import CTRVState, ctrv_step
from ship_trajectory_prediction.models.switching_bayesian_ctrv import (
    MODE_CRUISE,
    MODE_MANEUVER,
    MODE_NAMES,
    MODE_STOP,
)

MODE_INITIAL = 0
MAX_TIME_STEP_SECONDS = 15.0
DEFAULT_PHASE_TRANSITION_COUNTS = (6, 8, 8, 8)

# Fixed multipliers mirror the identifiable stop/cruise/maneuver ordering used
# by the switching model.  Values are indexed by the public one-based modes.
POSITION_PROCESS_MULTIPLIERS = (0.5, 1.0, 3.0)
SPEED_PROCESS_MULTIPLIERS = (0.25, 1.0, 4.0)
TURN_RATE_PROCESS_MULTIPLIERS = (0.25, 1.0, 4.0)


@dataclass(frozen=True, slots=True)
class SyntheticSwitchingCTRVNoise:
    """Known standard deviations used by the switching simulator.

    Position measurement noise is in meters and GPS-speed reference noise is
    in meters per second.  Process scales use state units per square-root
    second and are multiplied by ``sqrt(dt)`` and the fixed mode multiplier.
    """

    sigma_position_gps: float = 1.0
    sigma_speed_gps: float = 0.15
    sigma_position_process: float = 0.08
    sigma_speed_process: float = 0.02
    sigma_turn_rate_process: float = 0.0003

    def __post_init__(self) -> None:
        """Normalize and reject negative or non-finite noise scales."""
        for noise_field in fields(self):
            value = _non_negative_finite(
                noise_field.name,
                getattr(self, noise_field.name),
            )
            object.__setattr__(self, noise_field.name, value)


def simulate_synthetic_switching_ctrv_data(
    *,
    phase_transition_counts: Sequence[int] = DEFAULT_PHASE_TRANSITION_COUNTS,
    dt_seconds: float = 5.0,
    time_seconds: Sequence[float] | None = None,
    initial_state: CTRVState | None = None,
    cruise_speed_mps: float = 3.0,
    maneuver_turn_rate_rad_s: float = 0.012,
    stop_decay_time_seconds: float = 6.0,
    stop_turn_decay_time_seconds: float = 6.0,
    turn_rate_limit_rad_s: float = 0.02,
    noise: SyntheticSwitchingCTRVNoise | None = None,
    seed: int = 42,
    run_id: int = 0,
    reference_longitude: float = 8.55,
    reference_latitude: float = 47.37,
) -> pd.DataFrame:
    """Simulate a reproducible stop-cruise-maneuver-cruise trajectory.

    ``phase_transition_counts`` gives the number of destination rows in each
    phase.  Consequently, the returned frame contains one additional initial
    row.  ``mode_true[0]`` is the sentinel ``0`` because no transition enters
    the initial state; subsequent modes describe the transition from the
    preceding row to their own row.

    Cruise and maneuver positions follow the public deterministic CTRV step.
    Stop transitions hold expected position fixed while speed and turn rate
    decay exponentially.  At phase boundaries, speed or turn rate is moved to
    the new regime before mode-scaled process noise is added.  These known
    states and modes are evaluation truth only and must not be supplied to the
    position-only inference model.

    The usual CSV-compatible ``gps_speed`` column is stored in kilometers per
    hour as a noisy external reference.  It has no influence on simulated
    positions, phase labels, or continuous-state evolution.
    """
    phase_transition_counts = _phase_transition_counts(phase_transition_counts)
    mode_sequence = _mode_sequence(phase_transition_counts)
    sample_count = len(mode_sequence) + 1
    times = _time_values(
        sample_count,
        dt_seconds=dt_seconds,
        time_seconds=time_seconds,
    )
    seed = _non_negative_integer("seed", seed)
    run_id = _integer("run_id", run_id)
    cruise_speed_mps = _positive_finite(
        "cruise_speed_mps",
        cruise_speed_mps,
    )
    maneuver_turn_rate_rad_s = _finite_float(
        "maneuver_turn_rate_rad_s",
        maneuver_turn_rate_rad_s,
    )
    if maneuver_turn_rate_rad_s == 0:
        raise ValueError("maneuver_turn_rate_rad_s must be non-zero.")
    stop_decay_time_seconds = _positive_finite(
        "stop_decay_time_seconds",
        stop_decay_time_seconds,
    )
    stop_turn_decay_time_seconds = _positive_finite(
        "stop_turn_decay_time_seconds",
        stop_turn_decay_time_seconds,
    )
    turn_rate_limit_rad_s = _positive_finite(
        "turn_rate_limit_rad_s",
        turn_rate_limit_rad_s,
    )
    if abs(maneuver_turn_rate_rad_s) >= turn_rate_limit_rad_s:
        raise ValueError(
            "abs(maneuver_turn_rate_rad_s) must be smaller than turn_rate_limit_rad_s."
        )

    if initial_state is None:
        initial_state = CTRVState(
            x=0.0,
            y=0.0,
            speed=0.0,
            heading=0.25,
            turn_rate=0.0,
        )
    if not isinstance(initial_state, CTRVState):
        raise TypeError("initial_state must be a CTRVState instance.")
    if noise is None:
        noise = SyntheticSwitchingCTRVNoise()
    if not isinstance(noise, SyntheticSwitchingCTRVNoise):
        raise TypeError("noise must be a SyntheticSwitchingCTRVNoise instance.")

    generator = np.random.default_rng(seed)
    truth = [initial_state]
    previous_mode = MODE_INITIAL
    for index, mode in enumerate(mode_sequence, start=1):
        dt = float(times[index] - times[index - 1])
        expected = _expected_state(
            truth[-1],
            dt,
            mode=mode,
            previous_mode=previous_mode,
            cruise_speed_mps=cruise_speed_mps,
            maneuver_turn_rate_rad_s=maneuver_turn_rate_rad_s,
            stop_decay_time_seconds=stop_decay_time_seconds,
            stop_turn_decay_time_seconds=stop_turn_decay_time_seconds,
        )
        truth.append(
            _add_process_noise(
                expected,
                dt,
                mode=mode,
                noise=noise,
                turn_rate_limit_rad_s=turn_rate_limit_rad_s,
                generator=generator,
            )
        )
        previous_mode = mode

    x_true_raw = np.asarray([state.x for state in truth], dtype=float)
    y_true_raw = np.asarray([state.y for state in truth], dtype=float)
    speed_true = np.asarray([state.speed for state in truth], dtype=float)
    heading_true = np.asarray([state.heading for state in truth], dtype=float)
    turn_rate_true = np.asarray(
        [state.turn_rate for state in truth],
        dtype=float,
    )
    x_observed_raw = x_true_raw + generator.normal(
        0.0,
        noise.sigma_position_gps,
        sample_count,
    )
    y_observed_raw = y_true_raw + generator.normal(
        0.0,
        noise.sigma_position_gps,
        sample_count,
    )
    speed_observed = np.maximum(
        0.0,
        speed_true + generator.normal(0.0, noise.sigma_speed_gps, sample_count),
    )

    # Shared window preprocessing makes the first GPS proxy the local origin.
    # Align truth and explicit proxy columns to that same reference frame.
    x_reference = float(x_observed_raw[0])
    y_reference = float(y_observed_raw[0])
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

    modes = np.concatenate(([MODE_INITIAL], mode_sequence)).astype(int)
    mode_names = np.asarray(
        ["initial", *(_mode_name(mode) for mode in mode_sequence)],
        dtype=object,
    )
    position_multipliers = _aligned_mode_values(
        mode_sequence,
        POSITION_PROCESS_MULTIPLIERS,
    )
    speed_multipliers = _aligned_mode_values(
        mode_sequence,
        SPEED_PROCESS_MULTIPLIERS,
    )
    turn_rate_multipliers = _aligned_mode_values(
        mode_sequence,
        TURN_RATE_PROCESS_MULTIPLIERS,
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
            "mode_true": modes,
            "mode_name_true": mode_names,
            "position_process_multiplier_true": position_multipliers,
            "speed_process_multiplier_true": speed_multipliers,
            "turn_rate_process_multiplier_true": turn_rate_multipliers,
        }
    )
    for noise_field in fields(noise):
        result[f"{noise_field.name}_true"] = getattr(noise, noise_field.name)
    return result


def _expected_state(
    previous: CTRVState,
    dt: float,
    *,
    mode: int,
    previous_mode: int,
    cruise_speed_mps: float,
    maneuver_turn_rate_rad_s: float,
    stop_decay_time_seconds: float,
    stop_turn_decay_time_seconds: float,
) -> CTRVState:
    """Return one deterministic regime transition before process noise."""
    if mode == MODE_STOP:
        return CTRVState(
            x=previous.x,
            y=previous.y,
            speed=previous.speed * np.exp(-dt / stop_decay_time_seconds),
            heading=previous.heading + previous.turn_rate * dt,
            turn_rate=previous.turn_rate * np.exp(-dt / stop_turn_decay_time_seconds),
        )

    propagated = ctrv_step(previous, dt)
    speed = propagated.speed
    turn_rate = propagated.turn_rate
    if mode == MODE_CRUISE and previous_mode != MODE_CRUISE:
        speed = cruise_speed_mps
        turn_rate = 0.0
    elif mode == MODE_MANEUVER and previous_mode != MODE_MANEUVER:
        speed = max(speed, cruise_speed_mps)
        turn_rate = maneuver_turn_rate_rad_s
    return CTRVState(
        x=propagated.x,
        y=propagated.y,
        speed=speed,
        heading=propagated.heading,
        turn_rate=turn_rate,
    )


def _add_process_noise(
    expected: CTRVState,
    dt: float,
    *,
    mode: int,
    noise: SyntheticSwitchingCTRVNoise,
    turn_rate_limit_rad_s: float,
    generator: np.random.Generator,
) -> CTRVState:
    """Draw one mode-scaled continuous state around its expectation."""
    mode_index = mode - 1
    sqrt_dt = np.sqrt(dt)
    position_scale = (
        noise.sigma_position_process
        * POSITION_PROCESS_MULTIPLIERS[mode_index]
        * sqrt_dt
    )
    speed_scale = (
        noise.sigma_speed_process * SPEED_PROCESS_MULTIPLIERS[mode_index] * sqrt_dt
    )
    turn_rate_scale = (
        noise.sigma_turn_rate_process
        * TURN_RATE_PROCESS_MULTIPLIERS[mode_index]
        * sqrt_dt
    )
    return CTRVState(
        x=expected.x + generator.normal(0.0, position_scale),
        y=expected.y + generator.normal(0.0, position_scale),
        speed=max(0.0, expected.speed + generator.normal(0.0, speed_scale)),
        heading=expected.heading,
        turn_rate=float(
            np.clip(
                expected.turn_rate + generator.normal(0.0, turn_rate_scale),
                -turn_rate_limit_rad_s,
                turn_rate_limit_rad_s,
            )
        ),
    )


def _phase_transition_counts(values: Sequence[int]) -> tuple[int, int, int, int]:
    """Return four positive phase-transition counts."""
    if isinstance(values, (str, bytes)):
        raise ValueError("phase_transition_counts must contain four integers.")
    try:
        counts = tuple(values)
    except TypeError as error:
        raise ValueError(
            "phase_transition_counts must contain four integers."
        ) from error
    if len(counts) != 4 or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
        for value in counts
    ):
        raise ValueError("phase_transition_counts must contain four positive integers.")
    return tuple(int(value) for value in counts)


def _mode_sequence(phase_transition_counts: Sequence[int]) -> np.ndarray:
    """Return destination-aligned modes for all four configured phases."""
    first_cruise_count, maneuver_count, final_cruise_count = phase_transition_counts[1:]
    return np.concatenate(
        (
            np.full(phase_transition_counts[0], MODE_STOP, dtype=int),
            np.full(first_cruise_count, MODE_CRUISE, dtype=int),
            np.full(maneuver_count, MODE_MANEUVER, dtype=int),
            np.full(final_cruise_count, MODE_CRUISE, dtype=int),
        )
    )


def _time_values(
    sample_count: int,
    *,
    dt_seconds: float,
    time_seconds: Sequence[float] | None,
) -> np.ndarray:
    """Return finite timestamps compatible with shared window validation."""
    if time_seconds is None:
        dt_seconds = _positive_finite("dt_seconds", dt_seconds)
        if dt_seconds > MAX_TIME_STEP_SECONDS:
            raise ValueError(
                f"dt_seconds must not exceed {MAX_TIME_STEP_SECONDS:g} seconds."
            )
        return np.arange(sample_count, dtype=float) * dt_seconds

    times = np.asarray(time_seconds, dtype=float)
    if times.shape != (sample_count,) or not np.all(np.isfinite(times)):
        raise ValueError(
            "time_seconds must contain one finite value per synthetic row."
        )
    differences = np.diff(times)
    if times[0] != 0 or np.any(differences <= 0):
        raise ValueError("time_seconds must start at zero and be strictly increasing.")
    if np.any(differences > MAX_TIME_STEP_SECONDS):
        raise ValueError(
            "time_seconds must not contain gaps greater than "
            f"{MAX_TIME_STEP_SECONDS:g} seconds."
        )
    return times.copy()


def _aligned_mode_values(
    mode_sequence: np.ndarray,
    values: Sequence[float],
) -> np.ndarray:
    """Align one mode-indexed value sequence and prepend the row-zero sentinel."""
    aligned = np.asarray([values[mode - 1] for mode in mode_sequence], dtype=float)
    return np.concatenate(([0.0], aligned))


def _mode_name(mode: int) -> str:
    """Resolve one public mode name from a mapping or ordered sequence."""
    if isinstance(MODE_NAMES, Mapping):
        return str(MODE_NAMES[mode])
    names = tuple(MODE_NAMES)
    if len(names) == 3:
        return str(names[mode - 1])
    return str(names[mode])


def _integer(name: str, value: int) -> int:
    """Return one integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _non_negative_integer(name: str, value: int) -> int:
    """Return one non-negative integer."""
    result = _integer(name, value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


def _finite_float(name: str, value: float) -> float:
    """Return one finite scalar while rejecting booleans and strings."""
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _non_negative_finite(name: str, value: float) -> float:
    """Return one finite scalar greater than or equal to zero."""
    result = _finite_float(name, value)
    if result < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return result


def _positive_finite(name: str, value: float) -> float:
    """Return one finite scalar strictly greater than zero."""
    result = _finite_float(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return result


__all__ = [
    "DEFAULT_PHASE_TRANSITION_COUNTS",
    "MAX_TIME_STEP_SECONDS",
    "MODE_INITIAL",
    "POSITION_PROCESS_MULTIPLIERS",
    "SPEED_PROCESS_MULTIPLIERS",
    "SyntheticSwitchingCTRVNoise",
    "TURN_RATE_PROCESS_MULTIPLIERS",
    "simulate_synthetic_switching_ctrv_data",
]
