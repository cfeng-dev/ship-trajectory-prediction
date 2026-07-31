"""Bayesian CTRV state-space model fitted with variational inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel, CmdStanVB

from ship_trajectory_prediction.evaluation.metrics import (
    evaluate_position_predictions,
)
from ship_trajectory_prediction.evaluation.reporting import (
    posterior_variable_samples,
    variational_elbo_history,
)
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import TrajectoryWindowData
from ship_trajectory_prediction.trajectory.window import (
    estimate_initial_heading,
)

STAN_FILE = project_path("stan/models/bayesian_ctrv.stan")

SPEED_STATE_INITIAL_LOWER = 0.001
SPEED_STATE_INITIAL_UPPER = 99.99
INITIAL_SPEED_INTERVAL_COUNT = 3
POSITION_JITTER_THRESHOLD_METERS = 1.0
# At 10-second sampling, 0.02 rad/s permits at most 11.46 degrees per step.
DEFAULT_TURN_RATE_LIMIT = 0.02
# Robust window-specific scales are kept informative but not degenerate.
MIN_TURN_RATE_PRIOR_SCALE = 0.002
MAX_TURN_RATE_PRIOR_SCALE = 0.01
TURN_RATE_PRIOR_SCALE_MULTIPLIER = 2.0
MIN_COURSE_DISPLACEMENT_METERS = 1.0

NOISE_PARAMETER_NAMES = (
    "sigma_position_gps",
    "sigma_position_process",
    "sigma_speed_process",
    "sigma_turn_rate_process",
)


@dataclass(frozen=True, slots=True)
class BayesianCTRVPriors:
    """Configurable prior scales for the Bayesian CTRV state-space model."""

    position_initial_prior_scale: float = 5.0
    speed_initial_prior_scale: float = 0.75
    heading_initial_prior_scale: float = 0.35
    turn_rate_state_prior_scale: float | None = None
    sigma_position_gps_prior_scale: float = 5.0
    sigma_position_process_prior_scale: float = 0.5
    sigma_speed_process_prior_scale: float = 0.05
    sigma_turn_rate_process_prior_scale: float = 0.001

    def __post_init__(self) -> None:
        """Normalize and validate every explicitly configured prior scale."""
        for prior_field in fields(self):
            field_name = prior_field.name
            value = getattr(self, field_name)
            if field_name == "turn_rate_state_prior_scale" and value is None:
                continue
            _validate_positive_finite(field_name, value)
            object.__setattr__(self, field_name, float(value))


@dataclass(frozen=True)
class VIRunResult:
    """One fitted VI run together with reproducibility metadata."""

    seed: int
    algorithm: str
    fit: CmdStanVB
    runtime_seconds: float
    converged: bool = True


@dataclass(frozen=True, slots=True)
class TurnRateDiagnostics:
    """Robust observed-history diagnostics used to regularize turn rate."""

    sample_count: int
    median_rad_s: float
    robust_scale_rad_s: float
    q90_absolute_rad_s: float
    prior_scale_rad_s: float
    limit_rad_s: float


def build_stan_data(
    window: TrajectoryWindowData,
    *,
    priors: BayesianCTRVPriors | None = None,
    turn_rate_limit: float = DEFAULT_TURN_RATE_LIMIT,
) -> dict[str, Any]:
    """Build position-only data and priors for one observed trajectory window.

    All data-derived prior centers use only the observed portion of ``window``.
    The available GPS positions act as noisy proxy observations for a later
    externally observed target-vessel trajectory. GPS speed is deliberately
    excluded from Stan data, prior construction, and initialization. Position
    is measured in meters, latent speed in meters per second, heading in radians,
    turn rate in radians per second, and time in seconds. Process-noise standard
    deviations are multiplied by ``sqrt(dt)`` in Stan.
    """
    if priors is None:
        priors = BayesianCTRVPriors()
    if not isinstance(priors, BayesianCTRVPriors):
        raise TypeError("priors must be a BayesianCTRVPriors instance or None.")

    if window.observation_count < 2:
        raise ValueError("window must contain at least two observed positions.")
    if window.prediction_count < 1:
        raise ValueError("window must contain at least one prediction position.")

    observed = window.observed_slice
    prediction = window.prediction_slice
    time_observed = np.asarray(window.time_seconds[observed], dtype=float)
    time_prediction = np.asarray(window.time_seconds[prediction], dtype=float)
    x_observed = np.asarray(window.x_meters[observed], dtype=float)
    y_observed = np.asarray(window.y_meters[observed], dtype=float)

    _validate_time_arrays(time_observed, time_prediction)
    _validate_finite_vector("x_observed", x_observed)
    _validate_finite_vector("y_observed", y_observed)
    speed_initial_prior_mean = estimate_initial_speed_from_positions(
        time_observed,
        x_observed,
        y_observed,
    )
    turn_rate_diagnostics = diagnose_observed_turn_rate(
        window,
        turn_rate_state_prior_scale=priors.turn_rate_state_prior_scale,
        turn_rate_limit=turn_rate_limit,
    )
    try:
        heading_initial_prior_mean = estimate_initial_heading(
            x_observed,
            y_observed,
        )
    except ValueError:
        # Heading and turn rate are not identifiable while the ship is
        # stationary. Neutral centers keep such windows valid without deriving
        # arbitrary motion from GPS jitter.
        heading_initial_prior_mean = 0.0
        turn_rate_initial_prior_mean = 0.0
    else:
        turn_rate_initial_prior_mean = turn_rate_diagnostics.median_rad_s

    prior_scales = {
        "position_initial_prior_scale": priors.position_initial_prior_scale,
        "speed_initial_prior_scale": priors.speed_initial_prior_scale,
        "heading_initial_prior_scale": priors.heading_initial_prior_scale,
        "turn_rate_state_prior_scale": turn_rate_diagnostics.prior_scale_rad_s,
        "sigma_position_gps_prior_scale": priors.sigma_position_gps_prior_scale,
        "sigma_position_process_prior_scale": (
            priors.sigma_position_process_prior_scale
        ),
        "sigma_speed_process_prior_scale": priors.sigma_speed_process_prior_scale,
        "sigma_turn_rate_process_prior_scale": (
            priors.sigma_turn_rate_process_prior_scale
        ),
    }
    return {
        "N_observed": window.observation_count,
        "time_observed": time_observed,
        "x_observed": x_observed,
        "y_observed": y_observed,
        "N_prediction": window.prediction_count,
        "time_prediction": time_prediction,
        "x_initial_prior_mean": float(x_observed[0]),
        "y_initial_prior_mean": float(y_observed[0]),
        "speed_initial_prior_mean": speed_initial_prior_mean,
        "heading_initial_prior_mean": heading_initial_prior_mean,
        "turn_rate_initial_prior_mean": turn_rate_initial_prior_mean,
        "turn_rate_limit": turn_rate_diagnostics.limit_rad_s,
        **prior_scales,
    }


def estimate_initial_speed_from_positions(
    time_seconds,
    x_meters,
    y_meters,
    *,
    interval_count: int = INITIAL_SPEED_INTERVAL_COUNT,
    jitter_threshold_meters: float = POSITION_JITTER_THRESHOLD_METERS,
) -> float:
    """Estimate initial latent speed from early position-only intervals.

    The estimate is the median of at most the first ``interval_count`` segment
    speeds. A segment whose displacement is no greater than
    ``jitter_threshold_meters`` contributes exactly zero, which prevents small
    position jitter from forcing a moving-state prior while retaining a prior
    centered near zero for stationary targets. Times must be finite and strictly
    increasing; all positions must be finite and shape-aligned.
    """
    if (
        isinstance(interval_count, bool)
        or not isinstance(interval_count, (int, np.integer))
        or interval_count < 1
    ):
        raise ValueError("interval_count must be a positive integer.")
    jitter_threshold_meters = _validate_non_negative_finite(
        "jitter_threshold_meters",
        jitter_threshold_meters,
    )
    time_seconds = np.asarray(time_seconds, dtype=float)
    x_meters = np.asarray(x_meters, dtype=float)
    y_meters = np.asarray(y_meters, dtype=float)
    _validate_matching_position_time_arrays(time_seconds, x_meters, y_meters)

    time_differences = np.diff(time_seconds)
    if np.any(time_differences <= 0):
        raise ValueError("time_seconds must be strictly increasing.")
    displacements = np.hypot(np.diff(x_meters), np.diff(y_meters))
    segment_speeds = np.divide(displacements, time_differences)
    segment_speeds[displacements <= jitter_threshold_meters] = 0.0
    selected = segment_speeds[: min(interval_count, segment_speeds.size)]
    return float(np.median(selected))


def compile_bayesian_ctrv_model(
    stan_file: str | Path = STAN_FILE,
) -> CmdStanModel:
    """Compile and return the Bayesian CTRV CmdStan model."""
    stan_path = Path(stan_file)
    if not stan_path.is_file():
        raise FileNotFoundError(f"Stan model not found: {stan_path}")
    return CmdStanModel(stan_file=str(stan_path))


def fit_bayesian_ctrv_model(
    window: TrajectoryWindowData,
    *,
    priors: BayesianCTRVPriors | None = None,
    turn_rate_limit: float = DEFAULT_TURN_RATE_LIMIT,
    algorithm: str = "meanfield",
    iter: int = 20_000,
    grad_samples: int = 1,
    elbo_samples: int = 100,
    eta: float = 1.0,
    adapt_iter: int = 50,
    tol_rel_obj: float = 0.01,
    eval_elbo: int = 100,
    draws: int = 1_000,
    seed: int = 42,
    inits: Mapping[str, Any] | str | float | None = None,
    require_converged: bool = True,
    show_console: bool = False,
    variational_options: Mapping[str, Any] | None = None,
) -> CmdStanVB:
    """Fit the Bayesian CTRV model with CmdStan variational inference.

    ``meanfield`` is the default approximation. ``fullrank`` is available for
    sensitivity analysis, but this function intentionally exposes no MCMC path.
    With ``require_converged=True``, CmdStanPy raises if its VI convergence
    criterion is not reached.
    """
    _validate_variational_arguments(
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
    options = dict(variational_options or {})
    controlled_options = {
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
    }
    conflicting = controlled_options.intersection(options)
    if conflicting:
        names = ", ".join(sorted(conflicting))
        raise ValueError(f"variational_options must not override: {names}.")

    stan_data = build_stan_data(
        window,
        priors=priors,
        turn_rate_limit=turn_rate_limit,
    )
    if inits is None:
        inits = _default_initial_values(stan_data, seed=seed)

    model = compile_bayesian_ctrv_model()
    return model.variational(
        data=stan_data,
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
        **options,
    )


def summarize_predictions(
    fit: Any,
    window: TrajectoryWindowData,
    credible_interval: float = 0.9,
    *,
    include_speed_gps_reference: bool = False,
) -> pd.DataFrame:
    """Summarize future latent states and noisy position observations.

    ``speed_gps_reference`` can be included for a post-fit plausibility check.
    It is not an observed model variable and is never used for model fitting.
    """
    if not np.isfinite(credible_interval) or not 0 < credible_interval < 1:
        raise ValueError("credible_interval must be between 0 and 1.")
    if not isinstance(include_speed_gps_reference, bool):
        raise TypeError("include_speed_gps_reference must be a boolean.")

    prediction_variables = {
        "x_state": "x_state_prediction",
        "y_state": "y_state_prediction",
        "speed_state": "speed_state_prediction",
        "heading_state": "heading_state_prediction",
        "turn_rate_state": "turn_rate_state_prediction",
        "x_observation": "x_observation_prediction",
        "y_observation": "y_observation_prediction",
    }
    prediction_samples = {
        prefix: _prediction_samples(fit, variable_name, window.prediction_count)
        for prefix, variable_name in prediction_variables.items()
    }
    lower_probability = (1 - credible_interval) / 2
    upper_probability = 1 - lower_probability
    prediction = window.prediction_slice
    table_data: dict[str, Any] = {
        "time": window.timestamps[prediction],
        "t": window.time_seconds[prediction],
        "x_actual": window.x_meters[prediction],
        "y_actual": window.y_meters[prediction],
    }
    if include_speed_gps_reference:
        table_data["speed_gps_reference"] = window.gps_speed_mps[prediction]
    for prefix, samples in prediction_samples.items():
        table_data[f"{prefix}_median"] = np.median(samples, axis=0)
        table_data[f"{prefix}_lower"] = np.quantile(
            samples,
            lower_probability,
            axis=0,
        )
        table_data[f"{prefix}_upper"] = np.quantile(
            samples,
            upper_probability,
            axis=0,
        )
    return pd.DataFrame(table_data)


def compare_vi_runs(
    runs: Sequence[VIRunResult],
    window: TrajectoryWindowData,
    *,
    credible_interval: float = 0.9,
) -> pd.DataFrame:
    """Compare VI seeds using posterior, ELBO, accuracy, and coverage metrics."""
    if not runs:
        raise ValueError("runs must contain at least one variational fit.")

    rows = []
    for run in runs:
        if run.algorithm not in {"meanfield", "fullrank"}:
            raise ValueError("Each VI run must use 'meanfield' or 'fullrank'.")
        if not np.isfinite(run.runtime_seconds) or run.runtime_seconds < 0:
            raise ValueError("runtime_seconds must be a finite non-negative value.")

        history = variational_elbo_history(run.fit)
        final_elbo = history.iloc[-1]
        evaluation = evaluate_position_predictions(
            run.fit,
            window,
            credible_interval=credible_interval,
            position_variable_names=(
                "x_observation_prediction",
                "y_observation_prediction",
            ),
        )
        x_prediction = _prediction_samples(
            run.fit,
            "x_state_prediction",
            window.prediction_count,
        )
        y_prediction = _prediction_samples(
            run.fit,
            "y_state_prediction",
            window.prediction_count,
        )
        row = {
            "seed": run.seed,
            "algorithm": run.algorithm,
            "converged": run.converged,
            "runtime_seconds": run.runtime_seconds,
            "final_iteration": int(final_elbo["iteration"]),
            "final_elbo": float(final_elbo["elbo"]),
            "final_delta_elbo_mean": float(final_elbo["delta_elbo_mean"]),
            "final_delta_elbo_median": float(final_elbo["delta_elbo_median"]),
            "endpoint_x_m": float(np.median(x_prediction[:, -1])),
            "endpoint_y_m": float(np.median(y_prediction[:, -1])),
            "ade_m": evaluation.ade_m,
            "fde_m": evaluation.fde_m,
            "mean_interval_width_m": (evaluation.mean_marginal_interval_width_m),
            "radial_coverage": evaluation.radial_coverage,
        }
        for parameter_name in NOISE_PARAMETER_NAMES:
            samples = posterior_variable_samples(run.fit, parameter_name)
            if samples.ndim != 1 or samples.size == 0:
                raise ValueError(
                    f"Posterior variable {parameter_name!r} must contain scalar draws."
                )
            row[f"{parameter_name}_mean"] = float(np.mean(samples))
            row[f"{parameter_name}_std"] = float(
                np.std(samples, ddof=1) if samples.size > 1 else 0.0
            )
        rows.append(row)

    return pd.DataFrame(rows)


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


def diagnose_observed_turn_rate(
    window: TrajectoryWindowData,
    *,
    turn_rate_state_prior_scale: float | None = None,
    turn_rate_limit: float = DEFAULT_TURN_RATE_LIMIT,
) -> TurnRateDiagnostics:
    """Summarize course-derived turn rates from observed positions only.

    The median supplies the signed prior center. A MAD-based robust scale keeps
    isolated GPS course changes from making the state prior arbitrarily wide.
    The held-out part of ``window`` is never inspected.
    """
    _validate_positive_finite("turn_rate_limit", turn_rate_limit)
    if turn_rate_state_prior_scale is not None:
        _validate_positive_finite(
            "turn_rate_state_prior_scale",
            turn_rate_state_prior_scale,
        )

    observed = window.observed_slice
    rates = _observed_turn_rates(
        np.asarray(window.time_seconds[observed], dtype=float),
        np.asarray(window.x_meters[observed], dtype=float),
        np.asarray(window.y_meters[observed], dtype=float),
    )
    if rates.size == 0:
        median = 0.0
        robust_scale = 0.0
        q90_absolute = 0.0
    else:
        median = float(np.median(rates))
        median_absolute_deviation = float(np.median(np.abs(rates - median)))
        robust_scale = 1.4826 * median_absolute_deviation
        q90_absolute = float(np.quantile(np.abs(rates), 0.9))

    prior_scale = turn_rate_state_prior_scale
    if prior_scale is None:
        prior_scale = float(
            np.clip(
                TURN_RATE_PRIOR_SCALE_MULTIPLIER * robust_scale,
                MIN_TURN_RATE_PRIOR_SCALE,
                MAX_TURN_RATE_PRIOR_SCALE,
            )
        )
    center_limit = 0.95 * turn_rate_limit
    return TurnRateDiagnostics(
        sample_count=int(rates.size),
        median_rad_s=float(np.clip(median, -center_limit, center_limit)),
        robust_scale_rad_s=robust_scale,
        q90_absolute_rad_s=q90_absolute,
        prior_scale_rad_s=prior_scale,
        limit_rad_s=float(turn_rate_limit),
    )


def _prediction_samples(fit: Any, variable_name: str, prediction_count: int):
    """Extract and validate one finite posterior prediction matrix."""
    samples = posterior_variable_samples(fit, variable_name)
    if samples.ndim != 2 or samples.shape[1] != prediction_count:
        raise ValueError(
            f"Posterior variable {variable_name!r} has an unexpected shape."
        )
    if samples.shape[0] == 0 or not np.all(np.isfinite(samples)):
        raise ValueError(
            f"Posterior variable {variable_name!r} must contain finite draws."
        )
    return samples


def _observed_turn_rates(time_seconds, x_meters, y_meters) -> np.ndarray:
    """Return signed course changes from sufficiently separated GPS points."""
    delta_x = np.diff(x_meters)
    delta_y = np.diff(y_meters)
    moving = np.hypot(delta_x, delta_y) >= MIN_COURSE_DISPLACEMENT_METERS
    if np.count_nonzero(moving) < 2:
        return np.asarray([], dtype=float)

    segment_times = 0.5 * (time_seconds[:-1] + time_seconds[1:])
    moving_times = segment_times[moving]
    headings = np.unwrap(np.arctan2(delta_y[moving], delta_x[moving]))
    time_differences = np.diff(moving_times)
    valid = time_differences > 0
    if not np.any(valid):
        return np.asarray([], dtype=float)
    rates = np.diff(headings)[valid] / time_differences[valid]
    return rates[np.isfinite(rates)]


def _default_initial_values(stan_data: Mapping[str, Any], *, seed: int):
    """Create seeded VI initials from observed positions and times only.

    Stan maps an exact constrained zero to negative infinity on its internal
    scale. Speed initials are therefore kept slightly inside the zero-inclusive
    model bounds even when the position-derived path is stationary.
    """
    generator = np.random.default_rng(seed)
    time_observed = np.asarray(stan_data["time_observed"], dtype=float)
    x_observed = np.asarray(stan_data["x_observed"], dtype=float)
    y_observed = np.asarray(stan_data["y_observed"], dtype=float)
    speed_initial = _position_derived_speed_initials(
        time_observed,
        x_observed,
        y_observed,
        initial_prior_mean=float(stan_data["speed_initial_prior_mean"]),
    )
    turn_rate_center = float(stan_data["turn_rate_initial_prior_mean"])
    state_count = int(stan_data["N_observed"])

    x_jitter = min(0.1, 0.02 * stan_data["position_initial_prior_scale"])
    speed_jitter = min(0.02, 0.02 * stan_data["speed_initial_prior_scale"])
    turn_jitter = min(
        1e-4,
        0.02 * stan_data["turn_rate_state_prior_scale"],
    )
    return {
        "x_state": x_observed + generator.normal(0, x_jitter, state_count),
        "y_state": y_observed + generator.normal(0, x_jitter, state_count),
        "speed_state": np.clip(
            speed_initial + generator.normal(0, speed_jitter, state_count),
            SPEED_STATE_INITIAL_LOWER,
            SPEED_STATE_INITIAL_UPPER,
        ),
        "heading_initial": float(stan_data["heading_initial_prior_mean"])
        + generator.normal(0, 0.01),
        "turn_rate_state": np.clip(
            turn_rate_center + generator.normal(0, turn_jitter, state_count),
            -0.99 * stan_data["turn_rate_limit"],
            0.99 * stan_data["turn_rate_limit"],
        ),
        "sigma_position_gps": max(
            1e-3,
            0.5 * stan_data["sigma_position_gps_prior_scale"],
        ),
        "sigma_position_process": max(
            1e-3,
            0.5 * stan_data["sigma_position_process_prior_scale"],
        ),
        "sigma_speed_process": max(
            1e-4,
            0.5 * stan_data["sigma_speed_process_prior_scale"],
        ),
        "sigma_turn_rate_process": max(
            1e-5,
            0.5 * stan_data["sigma_turn_rate_process_prior_scale"],
        ),
    }


def _position_derived_speed_initials(
    time_seconds,
    x_meters,
    y_meters,
    *,
    initial_prior_mean: float,
) -> np.ndarray:
    """Map position-derived segment speeds onto latent state timestamps."""
    time_seconds = np.asarray(time_seconds, dtype=float)
    x_meters = np.asarray(x_meters, dtype=float)
    y_meters = np.asarray(y_meters, dtype=float)
    _validate_matching_position_time_arrays(time_seconds, x_meters, y_meters)
    time_differences = np.diff(time_seconds)
    if np.any(time_differences <= 0):
        raise ValueError("time_seconds must be strictly increasing.")

    displacements = np.hypot(np.diff(x_meters), np.diff(y_meters))
    segment_speeds = displacements / time_differences
    segment_speeds[displacements <= POSITION_JITTER_THRESHOLD_METERS] = 0.0
    state_speeds = np.empty(time_seconds.size, dtype=float)
    state_speeds[0] = _validate_non_negative_finite(
        "initial_prior_mean",
        initial_prior_mean,
    )
    state_speeds[1:] = segment_speeds
    return state_speeds


def _validate_variational_arguments(**arguments: Any) -> None:
    """Validate the explicitly supported CmdStan VI controls."""
    algorithm = arguments["algorithm"]
    if algorithm not in {"meanfield", "fullrank"}:
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
        _validate_positive_finite(name, arguments[name])
    for name in ("require_converged", "show_console"):
        if not isinstance(arguments[name], bool):
            raise ValueError(f"{name} must be a boolean.")


def _validate_time_arrays(time_observed, time_prediction) -> None:
    """Validate observed and future timestamps before handing them to Stan."""
    _validate_finite_vector("time_observed", time_observed)
    _validate_finite_vector("time_prediction", time_prediction)
    if np.any(np.diff(time_observed) <= 0):
        raise ValueError("Observed timestamps must be strictly increasing.")
    if time_prediction[0] <= time_observed[-1] or np.any(np.diff(time_prediction) <= 0):
        raise ValueError(
            "Prediction timestamps must be strictly increasing and follow "
            "the observed timestamps."
        )


def _validate_finite_vector(name: str, values) -> None:
    """Validate one non-empty, one-dimensional, finite array."""
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a non-empty finite vector.")


def _validate_matching_position_time_arrays(time_seconds, x_meters, y_meters) -> None:
    """Validate matching finite position and time vectors with two points."""
    for name, values in (
        ("time_seconds", time_seconds),
        ("x_meters", x_meters),
        ("y_meters", y_meters),
    ):
        _validate_finite_vector(name, values)
    if (
        time_seconds.size < 2
        or x_meters.shape != time_seconds.shape
        or y_meters.shape != time_seconds.shape
    ):
        raise ValueError(
            "time_seconds, x_meters, and y_meters must be matching vectors "
            "with at least two values."
        )


def _validate_non_negative_finite(name: str, value: float) -> float:
    """Validate and return a non-negative finite scalar."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a non-negative finite value.") from error
    if not np.isfinite(numeric_value) or numeric_value < 0:
        raise ValueError(f"{name} must be a non-negative finite value.")
    return numeric_value


def _validate_positive_finite(name: str, value: float) -> None:
    """Validate a positive finite scalar."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite value.") from error
    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be a positive finite value.")
