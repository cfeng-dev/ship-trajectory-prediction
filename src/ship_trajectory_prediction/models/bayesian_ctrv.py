"""Bayesian CTRV state-space model fitted with variational inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

SPEED_STATE_INITIAL_LOWER = 0.01
SPEED_STATE_INITIAL_UPPER = 99.99

NOISE_PARAMETER_NAMES = (
    "sigma_position_gps",
    "sigma_speed_gps",
    "sigma_position_process",
    "sigma_speed_process",
    "sigma_turn_rate_process",
)


@dataclass(frozen=True)
class VIRunResult:
    """One fitted VI run together with reproducibility metadata."""

    seed: int
    algorithm: str
    fit: CmdStanVB
    runtime_seconds: float
    converged: bool = True


def build_stan_data(
    window: TrajectoryWindowData,
    *,
    position_initial_prior_scale: float = 5.0,
    speed_initial_prior_scale: float = 0.75,
    heading_initial_prior_scale: float = 0.35,
    turn_rate_initial_prior_scale: float = 0.01,
    sigma_position_gps_prior_scale: float = 5.0,
    sigma_speed_gps_prior_scale: float = 0.5,
    sigma_position_process_prior_scale: float = 0.5,
    sigma_speed_process_prior_scale: float = 0.05,
    sigma_turn_rate_process_prior_scale: float = 0.001,
) -> dict[str, Any]:
    """Build data and weakly informative priors for one observed window.

    All data-derived prior centers use only the observed portion of ``window``.
    Position is measured in meters, speed in meters per second, heading in
    radians, turn rate in radians per second, and time in seconds. Process-noise
    standard deviations are multiplied by ``sqrt(dt)`` in Stan. Their units are
    therefore state units per square-root second.
    """
    prior_scales = {
        "position_initial_prior_scale": position_initial_prior_scale,
        "speed_initial_prior_scale": speed_initial_prior_scale,
        "heading_initial_prior_scale": heading_initial_prior_scale,
        "turn_rate_initial_prior_scale": turn_rate_initial_prior_scale,
        "sigma_position_gps_prior_scale": sigma_position_gps_prior_scale,
        "sigma_speed_gps_prior_scale": sigma_speed_gps_prior_scale,
        "sigma_position_process_prior_scale": (sigma_position_process_prior_scale),
        "sigma_speed_process_prior_scale": sigma_speed_process_prior_scale,
        "sigma_turn_rate_process_prior_scale": (sigma_turn_rate_process_prior_scale),
    }
    for name, value in prior_scales.items():
        _validate_positive_finite(name, value)

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
    speed_observed = np.asarray(window.gps_speed_mps[observed], dtype=float)

    _validate_time_arrays(time_observed, time_prediction)
    _validate_finite_vector("x_observed", x_observed)
    _validate_finite_vector("y_observed", y_observed)
    _validate_finite_vector("speed_observed", speed_observed)
    if np.any(speed_observed < 0):
        raise ValueError("Observed gps_speed must contain non-negative values.")

    positive_speeds = speed_observed[speed_observed > 0]
    if positive_speeds.size == 0:
        raise ValueError("Observed gps_speed must contain positive values.")
    initial_speed_count = min(3, positive_speeds.size)
    speed_initial_prior_mean = float(np.median(positive_speeds[:initial_speed_count]))
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
        turn_rate_initial_prior_mean = _estimate_turn_rate_prior_mean(
            time_observed,
            x_observed,
            y_observed,
        )

    return {
        "N_observed": window.observation_count,
        "time_observed": time_observed,
        "x_observed": x_observed,
        "y_observed": y_observed,
        "speed_observed": speed_observed,
        "N_prediction": window.prediction_count,
        "time_prediction": time_prediction,
        "x_initial_prior_mean": float(x_observed[0]),
        "y_initial_prior_mean": float(y_observed[0]),
        "speed_initial_prior_mean": speed_initial_prior_mean,
        "heading_initial_prior_mean": heading_initial_prior_mean,
        "turn_rate_initial_prior_mean": turn_rate_initial_prior_mean,
        **prior_scales,
    }


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
    **stan_data_options: float,
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

    stan_data = build_stan_data(window, **stan_data_options)
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
) -> pd.DataFrame:
    """Summarize posterior-predictive states against held-out observations."""
    if not np.isfinite(credible_interval) or not 0 < credible_interval < 1:
        raise ValueError("credible_interval must be between 0 and 1.")

    prediction_samples = {
        name: _prediction_samples(fit, name, window.prediction_count)
        for name in (
            "x_prediction",
            "y_prediction",
            "speed_prediction",
            "heading_prediction",
            "turn_rate_prediction",
        )
    }
    lower_probability = (1 - credible_interval) / 2
    upper_probability = 1 - lower_probability
    prediction = window.prediction_slice
    table_data: dict[str, Any] = {
        "time": window.timestamps[prediction],
        "t": window.time_seconds[prediction],
        "x_actual": window.x_meters[prediction],
        "y_actual": window.y_meters[prediction],
        "speed_actual": window.gps_speed_mps[prediction],
    }
    for variable_name, samples in prediction_samples.items():
        prefix = variable_name.removesuffix("_prediction")
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
        )
        x_prediction = _prediction_samples(
            run.fit,
            "x_prediction",
            window.prediction_count,
        )
        y_prediction = _prediction_samples(
            run.fit,
            "y_prediction",
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


def _estimate_turn_rate_prior_mean(time_seconds, x_meters, y_meters) -> float:
    """Estimate signed turn rate from observed course changes only."""
    delta_x = np.diff(x_meters)
    delta_y = np.diff(y_meters)
    moving = np.hypot(delta_x, delta_y) > 1e-8
    if np.count_nonzero(moving) < 2:
        return 0.0

    segment_times = 0.5 * (time_seconds[:-1] + time_seconds[1:])
    moving_times = segment_times[moving]
    headings = np.unwrap(np.arctan2(delta_y[moving], delta_x[moving]))
    time_differences = np.diff(moving_times)
    valid = time_differences > 0
    if not np.any(valid):
        return 0.0
    rates = np.diff(headings)[valid] / time_differences[valid]
    return float(np.clip(np.median(rates), -0.1, 0.1))


def _default_initial_values(stan_data: Mapping[str, Any], *, seed: int):
    """Create deterministic-with-seed, slightly jittered VI initial values."""
    generator = np.random.default_rng(seed)
    x_observed = np.asarray(stan_data["x_observed"], dtype=float)
    y_observed = np.asarray(stan_data["y_observed"], dtype=float)
    speed_observed = np.asarray(stan_data["speed_observed"], dtype=float)
    turn_rate_center = float(stan_data["turn_rate_initial_prior_mean"])
    state_count = int(stan_data["N_observed"])

    x_jitter = min(0.1, 0.02 * stan_data["position_initial_prior_scale"])
    speed_jitter = min(0.02, 0.02 * stan_data["speed_initial_prior_scale"])
    turn_jitter = min(
        1e-4,
        0.02 * stan_data["turn_rate_initial_prior_scale"],
    )
    return {
        "x_state": x_observed + generator.normal(0, x_jitter, state_count),
        "y_state": y_observed + generator.normal(0, x_jitter, state_count),
        "speed_state": np.clip(
            speed_observed + generator.normal(0, speed_jitter, state_count),
            SPEED_STATE_INITIAL_LOWER,
            SPEED_STATE_INITIAL_UPPER,
        ),
        "heading_initial": float(stan_data["heading_initial_prior_mean"])
        + generator.normal(0, 0.01),
        "turn_rate_state": np.clip(
            turn_rate_center + generator.normal(0, turn_jitter, state_count),
            -0.099,
            0.099,
        ),
        "sigma_position_gps": max(
            1e-3,
            0.5 * stan_data["sigma_position_gps_prior_scale"],
        ),
        "sigma_speed_gps": max(
            1e-3,
            0.5 * stan_data["sigma_speed_gps_prior_scale"],
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


def _validate_positive_finite(name: str, value: float) -> None:
    """Validate a positive finite scalar."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite value.") from error
    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be a positive finite value.")
