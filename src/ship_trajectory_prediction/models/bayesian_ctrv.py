"""Bayesian CTRV state-space model fitted with VI or MCMC."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanMCMC, CmdStanModel, CmdStanVB

import ship_trajectory_prediction.models.paths as model_paths
import ship_trajectory_prediction.observations.window as observation_window
import ship_trajectory_prediction.validation.reporting as reporting

STAN_FILE = model_paths.stan_path("models/bayesian_ctrv.stan")

SPEED_STATE_INITIAL_LOWER = 0.001
DEFAULT_INITIAL_SPEED_POINT_COUNT = 5
ROBUST_MAD_SCALE_FACTOR = 1.4826
MIN_INITIAL_SPEED_PRIOR_SCALE_MPS = 0.001
DEFAULT_VI_ADAPT_ITER = 100
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
    """Configurable prior parameters for the Bayesian CTRV state-space model.

    Fixed prior parameters should be calibrated from independent historical
    windows. Forecast-origin heading and turn rate remain latent in the fully
    Bayesian model.
    """

    position_initial_prior_scale: float = 5.0
    speed_initial_prior_mean: float = 0.0
    speed_initial_prior_scale: float = 0.75
    turn_rate_initial_prior_mean: float = 0.0
    turn_rate_state_prior_scale: float | None = None
    sigma_position_gps_prior_scale: float = 5.0
    sigma_position_process_prior_scale: float = 0.5
    sigma_speed_process_prior_scale: float = 0.05
    sigma_turn_rate_process_prior_scale: float = 0.001

    def __post_init__(self) -> None:
        """Normalize and validate every explicitly configured prior value."""
        for prior_field in fields(self):
            field_name = prior_field.name
            value = getattr(self, field_name)
            if field_name == "turn_rate_state_prior_scale" and value is None:
                continue
            if field_name == "speed_initial_prior_mean":
                value = _validate_non_negative_finite(field_name, value)
            elif field_name == "turn_rate_initial_prior_mean":
                value = _validate_finite_scalar(field_name, value)
            else:
                _validate_positive_finite(field_name, value)
            object.__setattr__(self, field_name, float(value))


@dataclass(frozen=True, slots=True)
class HistoricalInitialSpeedPrior:
    """Robust initial-speed prior estimated from independent windows."""

    mean_mps: float
    scale_mps: float
    window_estimates_mps: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the robust summary and its per-window speed sample."""
        mean_mps = _validate_non_negative_finite("mean_mps", self.mean_mps)
        _validate_positive_finite("scale_mps", self.scale_mps)
        estimates = tuple(
            _validate_non_negative_finite("window_estimates_mps", value)
            for value in self.window_estimates_mps
        )
        if not estimates:
            raise ValueError("window_estimates_mps must not be empty.")
        object.__setattr__(self, "mean_mps", mean_mps)
        object.__setattr__(self, "scale_mps", float(self.scale_mps))
        object.__setattr__(self, "window_estimates_mps", estimates)


@dataclass(frozen=True, slots=True)
class PositionObservations:
    """Immutable observed positions supplied to the Bayesian CTRV fit.

    The arrays contain only the observed part of one trajectory window. The
    additional noise metadata records the experimental perturbation applied in
    memory; it is not another parameter of the Stan observation model.
    """

    time_seconds: np.ndarray
    x_meters: np.ndarray
    y_meters: np.ndarray
    additional_noise_std_m: float
    noise_seed: int

    def __post_init__(self) -> None:
        """Copy, validate, and make all observation arrays read-only."""
        time_seconds = np.asarray(self.time_seconds, dtype=float).copy()
        x_meters = np.asarray(self.x_meters, dtype=float).copy()
        y_meters = np.asarray(self.y_meters, dtype=float).copy()
        _validate_matching_position_time_arrays(
            time_seconds,
            x_meters,
            y_meters,
        )
        if np.any(np.diff(time_seconds) <= 0):
            raise ValueError("time_seconds must be strictly increasing.")
        additional_noise_std_m = _validate_non_negative_finite(
            "additional_noise_std_m",
            self.additional_noise_std_m,
        )
        noise_seed = _validate_non_negative_integer("noise_seed", self.noise_seed)
        for values in (time_seconds, x_meters, y_meters):
            values.setflags(write=False)
        object.__setattr__(self, "time_seconds", time_seconds)
        object.__setattr__(self, "x_meters", x_meters)
        object.__setattr__(self, "y_meters", y_meters)
        object.__setattr__(
            self,
            "additional_noise_std_m",
            additional_noise_std_m,
        )
        object.__setattr__(self, "noise_seed", noise_seed)


@dataclass(frozen=True, slots=True)
class TurnRateDiagnostics:
    """Robust observed-history diagnostics used to regularize turn rate."""

    sample_count: int
    median_rad_s: float
    robust_scale_rad_s: float
    q90_absolute_rad_s: float
    prior_scale_rad_s: float


def simulate_position_observations(
    window: observation_window.TrajectoryWindowData,
    *,
    additional_noise_std_m: float = 0.0,
    seed: int = 2026,
) -> PositionObservations:
    """Create reproducible position observations without changing ``window``.

    Independent zero-mean Gaussian noise with the configured standard
    deviation is added once to each local x and y coordinate. A value of zero
    preserves the converted GPS positions exactly. Only the observed slice is
    copied; held-out positions remain untouched for later evaluation.
    """
    additional_noise_std_m = _validate_non_negative_finite(
        "additional_noise_std_m",
        additional_noise_std_m,
    )
    seed = _validate_non_negative_integer("seed", seed)
    if window.observation_count < 2:
        raise ValueError("window must contain at least two observed positions.")

    observed = window.observed_slice
    time_seconds = np.asarray(window.time_seconds[observed], dtype=float).copy()
    x_meters = np.asarray(window.x_meters[observed], dtype=float).copy()
    y_meters = np.asarray(window.y_meters[observed], dtype=float).copy()
    if additional_noise_std_m > 0:
        generator = np.random.default_rng(seed)
        x_meters += generator.normal(0.0, additional_noise_std_m, x_meters.size)
        y_meters += generator.normal(0.0, additional_noise_std_m, y_meters.size)

    return PositionObservations(
        time_seconds=time_seconds,
        x_meters=x_meters,
        y_meters=y_meters,
        additional_noise_std_m=additional_noise_std_m,
        noise_seed=seed,
    )


def build_stan_data(
    window: observation_window.TrajectoryWindowData,
    *,
    priors: BayesianCTRVPriors | None = None,
    position_observations: PositionObservations | None = None,
) -> dict[str, Any]:
    """Build position-only data and priors for one observed trajectory window.

    The initial-speed mean and scale are fixed values from ``priors`` and must
    be calibrated independently of ``window``. Data-derived turn-rate prior
    diagnostics use only ``position_observations``; terminal heading and turn
    rate remain latent parameters of the Stan model.
    If observations are omitted, the observed portion of ``window`` is copied
    without adding noise. Position is measured in meters, latent speed in
    meters per second, heading in radians, turn rate in radians per second, and
    time in seconds. Process-noise standard deviations are multiplied by
    ``sqrt(dt)`` in Stan.
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
    window_time_observed = np.asarray(window.time_seconds[observed], dtype=float)
    time_prediction = np.asarray(window.time_seconds[prediction], dtype=float)
    _validate_time_arrays(window_time_observed, time_prediction)
    position_observations = _resolve_position_observations(
        window,
        position_observations,
    )
    time_observed = position_observations.time_seconds
    x_observed = position_observations.x_meters
    y_observed = position_observations.y_meters

    _validate_finite_vector("x_observed", x_observed)
    _validate_finite_vector("y_observed", y_observed)
    turn_rate_diagnostics = diagnose_observed_turn_rate(
        window,
        turn_rate_state_prior_scale=priors.turn_rate_state_prior_scale,
        position_observations=position_observations,
    )
    prior_parameters = {
        "position_initial_prior_scale": priors.position_initial_prior_scale,
        "speed_initial_prior_mean": priors.speed_initial_prior_mean,
        "speed_initial_prior_scale": priors.speed_initial_prior_scale,
        "turn_rate_initial_prior_mean": priors.turn_rate_initial_prior_mean,
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
        **prior_parameters,
    }


def estimate_initial_speed_from_positions(
    time_seconds,
    x_meters,
    y_meters,
    *,
    point_count: int = DEFAULT_INITIAL_SPEED_POINT_COUNT,
) -> float:
    """Estimate local initial speed by regressing the first positions on time.

    Separate linear fits ``x(t)`` and ``y(t)`` provide the two initial velocity
    components. Only the first ``point_count`` values are used. This estimator
    supports both independent historical prior calibration and current-window
    numerical initialization; its caller determines which role the data serve.
    """
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, (int, np.integer))
        or point_count < 2
    ):
        raise ValueError("point_count must be an integer of at least 2.")
    time_seconds = np.asarray(time_seconds, dtype=float)
    x_meters = np.asarray(x_meters, dtype=float)
    y_meters = np.asarray(y_meters, dtype=float)
    if (
        time_seconds.ndim != 1
        or x_meters.shape != time_seconds.shape
        or y_meters.shape != time_seconds.shape
        or time_seconds.size < point_count
    ):
        raise ValueError(
            "time_seconds, x_meters, and y_meters must be matching vectors "
            f"with at least {point_count} values."
        )

    selected_time = time_seconds[:point_count]
    selected_x = x_meters[:point_count]
    selected_y = y_meters[:point_count]
    _validate_matching_position_time_arrays(selected_time, selected_x, selected_y)
    time_differences = np.diff(selected_time)
    if np.any(time_differences <= 0):
        raise ValueError("time_seconds must be strictly increasing.")

    centered_time = selected_time - np.mean(selected_time)
    time_sum_of_squares = float(np.dot(centered_time, centered_time))
    velocity_x_mps = float(
        np.dot(centered_time, selected_x - np.mean(selected_x)) / time_sum_of_squares
    )
    velocity_y_mps = float(
        np.dot(centered_time, selected_y - np.mean(selected_y)) / time_sum_of_squares
    )
    initial_speed_mps = float(np.hypot(velocity_x_mps, velocity_y_mps))
    return _validate_non_negative_finite("initial_speed_mps", initial_speed_mps)


def estimate_initial_speed_prior_from_windows(
    historical_windows: Sequence[observation_window.TrajectoryWindowData],
    *,
    point_count: int = DEFAULT_INITIAL_SPEED_POINT_COUNT,
    minimum_scale_mps: float = MIN_INITIAL_SPEED_PRIOR_SCALE_MPS,
) -> HistoricalInitialSpeedPrior:
    """Estimate a robust position-only speed prior from historical windows.

    Exactly one local initial-speed estimate is obtained from the first
    ``point_count`` observed positions of each independent historical window.
    Prediction positions and GPS-speed values are never read. The returned
    center is the median and the scale is ``1.4826 * MAD`` with only a small
    positive numerical floor.
    """
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, (int, np.integer))
        or point_count < 2
    ):
        raise ValueError("point_count must be an integer of at least 2.")
    _validate_positive_finite("minimum_scale_mps", minimum_scale_mps)
    minimum_scale_mps = float(minimum_scale_mps)
    try:
        windows = tuple(historical_windows)
    except TypeError as error:
        raise TypeError(
            "historical_windows must be a collection of TrajectoryWindowData."
        ) from error
    if not windows:
        raise ValueError("historical_windows must contain at least one window.")

    estimates = []
    for window_index, window in enumerate(windows):
        if not isinstance(window, observation_window.TrajectoryWindowData):
            raise TypeError(
                "historical_windows must contain only TrajectoryWindowData "
                f"instances; item {window_index} has type "
                f"{type(window).__name__}."
            )
        if window.observation_count < point_count:
            raise ValueError(
                f"Historical window {window_index} has "
                f"{window.observation_count} observed positions; at least "
                f"{point_count} are required."
            )
        observed = window.observed_slice
        estimate_mps = estimate_initial_speed_from_positions(
            window.time_seconds[observed],
            window.x_meters[observed],
            window.y_meters[observed],
            point_count=point_count,
        )
        estimates.append(estimate_mps)

    estimates_array = np.asarray(estimates, dtype=float)
    prior_mean_mps = _validate_non_negative_finite(
        "speed_initial_prior_mean",
        np.median(estimates_array),
    )
    median_absolute_deviation = float(
        np.median(np.abs(estimates_array - prior_mean_mps))
    )
    prior_scale_mps = max(
        ROBUST_MAD_SCALE_FACTOR * median_absolute_deviation,
        minimum_scale_mps,
    )
    _validate_positive_finite("speed_initial_prior_scale", prior_scale_mps)
    return HistoricalInitialSpeedPrior(
        mean_mps=prior_mean_mps,
        scale_mps=float(prior_scale_mps),
        window_estimates_mps=tuple(float(value) for value in estimates_array),
    )


def compile_bayesian_ctrv_model(
    stan_file: str | Path = STAN_FILE,
) -> CmdStanModel:
    """Compile and return the Bayesian CTRV CmdStan model."""
    stan_path = Path(stan_file)
    if not stan_path.is_file():
        raise FileNotFoundError(f"Stan model not found: {stan_path}")
    return CmdStanModel(stan_file=str(stan_path))


def fit_bayesian_ctrv_model(
    window: observation_window.TrajectoryWindowData,
    *,
    priors: BayesianCTRVPriors | None = None,
    position_observations: PositionObservations | None = None,
    inference_method: str = "vi",
    algorithm: str = "meanfield",
    iter: int = 20_000,
    grad_samples: int = 1,
    elbo_samples: int = 100,
    eta: float = 1.0,
    adapt_iter: int = DEFAULT_VI_ADAPT_ITER,
    tol_rel_obj: float = 0.01,
    eval_elbo: int = 100,
    draws: int = 1_000,
    chains: int = 4,
    parallel_chains: int | None = None,
    iter_warmup: int = 1_000,
    iter_sampling: int = 1_000,
    adapt_delta: float = 0.9,
    max_treedepth: int = 10,
    seed: int = 42,
    inits: (
        Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | float | None
    ) = None,
    require_converged: bool = True,
    show_console: bool = False,
    variational_options: Mapping[str, Any] | None = None,
    mcmc_options: Mapping[str, Any] | None = None,
    _stan_data_builder: Callable[..., dict[str, Any]] | None = None,
    _model_compiler: Callable[[], CmdStanModel] | None = None,
) -> CmdStanVB | CmdStanMCMC:
    """Fit the Bayesian CTRV model with selectable VI or MCMC inference.

    ``inference_method="vi"`` uses CmdStan variational inference with either a
    ``meanfield`` or ``fullrank`` approximation. ``inference_method="mcmc"``
    uses NUTS through CmdStan's sampler. VI and MCMC have separate option maps,
    and options for the inactive method are rejected instead of silently ignored.
    ``require_converged`` applies only to the VI path; MCMC quality is assessed
    through sampler diagnostics, effective sample sizes, and R-hat values.
    """
    inference_method = normalize_inference_method(inference_method)
    if inference_method == "vi":
        if mcmc_options:
            raise ValueError("mcmc_options can only be used with MCMC inference.")
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
        _reject_conflicting_options(
            "variational_options",
            options,
            controlled_options,
        )
    else:
        if variational_options:
            raise ValueError("variational_options can only be used with VI inference.")
        if parallel_chains is None:
            parallel_chains = chains
        _validate_mcmc_arguments(
            chains=chains,
            parallel_chains=parallel_chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            adapt_delta=adapt_delta,
            max_treedepth=max_treedepth,
            seed=seed,
            show_console=show_console,
        )
        options = dict(mcmc_options or {})
        controlled_options = {
            "data",
            "seed",
            "inits",
            "chains",
            "parallel_chains",
            "iter_warmup",
            "iter_sampling",
            "adapt_delta",
            "max_treedepth",
            "show_console",
        }
        _reject_conflicting_options("mcmc_options", options, controlled_options)

    if _stan_data_builder is None:
        _stan_data_builder = build_stan_data
    if _model_compiler is None:
        _model_compiler = compile_bayesian_ctrv_model
    stan_data = _stan_data_builder(
        window,
        priors=priors,
        position_observations=position_observations,
    )
    model = _model_compiler()
    if inference_method == "vi":
        if inits is None:
            inits = _default_initial_values(stan_data, seed=seed)
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

    if inits is None:
        inits = [
            _default_initial_values(stan_data, seed=seed + chain_index)
            for chain_index in range(chains)
        ]
    return model.sample(
        data=stan_data,
        seed=seed,
        inits=inits,
        chains=chains,
        parallel_chains=parallel_chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        show_console=show_console,
        **options,
    )


def summarize_predictions(
    fit: Any,
    window: observation_window.TrajectoryWindowData,
    credible_interval: float = 0.9,
) -> pd.DataFrame:
    """Summarize future latent states and noisy position observations."""
    if not np.isfinite(credible_interval) or not 0 < credible_interval < 1:
        raise ValueError("credible_interval must be between 0 and 1.")

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
    window: observation_window.TrajectoryWindowData,
    *,
    turn_rate_state_prior_scale: float | None = None,
    position_observations: PositionObservations | None = None,
) -> TurnRateDiagnostics:
    """Summarize course-derived turn rates from observed positions only.

    The median reports a signed empirical center that the hybrid model may use.
    A MAD-based robust scale provides a configurable diagnostic candidate.
    Supplied position observations override the clean observed coordinates;
    the held-out part of ``window`` is never inspected.
    """
    if turn_rate_state_prior_scale is not None:
        _validate_positive_finite(
            "turn_rate_state_prior_scale",
            turn_rate_state_prior_scale,
        )

    position_observations = _resolve_position_observations(
        window,
        position_observations,
    )
    rates = _observed_turn_rates(
        position_observations.time_seconds,
        position_observations.x_meters,
        position_observations.y_meters,
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
    return TurnRateDiagnostics(
        sample_count=int(rates.size),
        median_rad_s=median,
        robust_scale_rad_s=robust_scale,
        q90_absolute_rad_s=q90_absolute,
        prior_scale_rad_s=prior_scale,
    )


def _prediction_samples(fit: Any, variable_name: str, prediction_count: int):
    """Extract and validate one finite posterior prediction matrix."""
    samples = reporting.posterior_variable_samples(fit, variable_name)
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

    The current-window speed estimate is only a numerical starting guess. It
    does not alter the fixed historical Bayesian prior stored in ``stan_data``.
    Stan maps an exact constrained zero to negative infinity on its internal
    scale. Speed initials are therefore kept slightly above the zero lower
    bound even when the position-derived path is stationary.
    """
    generator = np.random.default_rng(seed)
    time_observed = np.asarray(stan_data["time_observed"], dtype=float)
    x_observed = np.asarray(stan_data["x_observed"], dtype=float)
    y_observed = np.asarray(stan_data["y_observed"], dtype=float)
    initial_speed_mps = estimate_initial_speed_from_positions(
        time_observed,
        x_observed,
        y_observed,
        point_count=min(DEFAULT_INITIAL_SPEED_POINT_COUNT, time_observed.size),
    )
    speed_initial = _position_derived_speed_initials(
        time_observed,
        x_observed,
        y_observed,
        initial_speed_mps=initial_speed_mps,
    )
    turn_rate_center = float(stan_data["turn_rate_initial_prior_mean"])
    state_count = int(stan_data["N_observed"])
    turn_rate_count = state_count if "heading_final" in stan_data else state_count - 1

    x_jitter = min(0.1, 0.02 * stan_data["position_initial_prior_scale"])
    speed_jitter = min(0.02, 0.02 * stan_data["speed_initial_prior_scale"])
    turn_jitter = min(
        1e-4,
        0.02 * stan_data["turn_rate_state_prior_scale"],
    )
    initial_values = {
        "x_state": x_observed + generator.normal(0, x_jitter, state_count),
        "y_state": y_observed + generator.normal(0, x_jitter, state_count),
        "speed_state": np.maximum(
            speed_initial + generator.normal(0, speed_jitter, state_count),
            SPEED_STATE_INITIAL_LOWER,
        ),
        "turn_rate_state": turn_rate_center
        + generator.normal(0, turn_jitter, turn_rate_count),
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
    if "heading_final" not in stan_data:
        final_course = float(
            np.arctan2(
                y_observed[-1] - y_observed[-2],
                x_observed[-1] - x_observed[-2],
            )
        )
        heading_jitter = generator.normal(0.0, 0.01)
        initial_values["heading_final"] = float(
            np.arctan2(
                np.sin(final_course + heading_jitter),
                np.cos(final_course + heading_jitter),
            )
        )
    return initial_values


def _position_derived_speed_initials(
    time_seconds,
    x_meters,
    y_meters,
    *,
    initial_speed_mps: float,
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
    state_speeds = np.empty(time_seconds.size, dtype=float)
    state_speeds[0] = _validate_non_negative_finite(
        "initial_speed_mps",
        initial_speed_mps,
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


def normalize_inference_method(inference_method: str) -> str:
    """Return a normalized supported inference method."""
    if not isinstance(inference_method, str):
        raise ValueError("inference_method must be 'vi' or 'mcmc'.")
    normalized = inference_method.strip().lower()
    if normalized not in {"vi", "mcmc"}:
        raise ValueError("inference_method must be 'vi' or 'mcmc'.")
    return normalized


def _validate_mcmc_arguments(**arguments: Any) -> None:
    """Validate the explicitly supported CmdStan NUTS controls."""
    for name in (
        "chains",
        "parallel_chains",
        "iter_warmup",
        "iter_sampling",
        "max_treedepth",
        "seed",
    ):
        value = arguments[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if arguments["parallel_chains"] > arguments["chains"]:
        raise ValueError("parallel_chains must not exceed chains.")
    adapt_delta = arguments["adapt_delta"]
    if isinstance(adapt_delta, (bool, str, bytes)):
        raise ValueError("adapt_delta must be between 0 and 1.")
    try:
        adapt_delta = float(adapt_delta)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("adapt_delta must be between 0 and 1.") from error
    if not np.isfinite(adapt_delta) or not 0 < adapt_delta < 1:
        raise ValueError("adapt_delta must be between 0 and 1.")
    if not isinstance(arguments["show_console"], bool):
        raise ValueError("show_console must be a boolean.")


def _reject_conflicting_options(
    option_name: str,
    options: Mapping[str, Any],
    controlled_options: set[str],
) -> None:
    """Reject generic options that override the wrapper's explicit controls."""
    conflicting = controlled_options.intersection(options)
    if conflicting:
        names = ", ".join(sorted(conflicting))
        raise ValueError(f"{option_name} must not override: {names}.")


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


def _resolve_position_observations(
    window: observation_window.TrajectoryWindowData,
    position_observations: PositionObservations | None,
) -> PositionObservations:
    """Return observations aligned with the observed part of ``window``."""
    if position_observations is None:
        position_observations = simulate_position_observations(
            window,
            additional_noise_std_m=0.0,
            seed=0,
        )
    if not isinstance(position_observations, PositionObservations):
        raise TypeError(
            "position_observations must be a PositionObservations instance or None."
        )

    expected_time = np.asarray(
        window.time_seconds[window.observed_slice],
        dtype=float,
    )
    if (
        position_observations.time_seconds.shape != expected_time.shape
        or not np.array_equal(position_observations.time_seconds, expected_time)
    ):
        raise ValueError(
            "position_observations must match the observed timestamps in window."
        )
    return position_observations


def _validate_non_negative_integer(name: str, value: int) -> int:
    """Validate and return a non-negative integer seed-like value."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _validate_non_negative_finite(name: str, value: float) -> float:
    """Validate and return a non-negative finite scalar."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a non-negative finite value.") from error
    if not np.isfinite(numeric_value) or numeric_value < 0:
        raise ValueError(f"{name} must be a non-negative finite value.")
    return numeric_value


def _validate_finite_scalar(name: str, value: float) -> float:
    """Validate and return a signed finite scalar."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite value.") from error
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be a finite value.")
    return numeric_value


def _validate_positive_finite(name: str, value: float) -> None:
    """Validate a positive finite scalar."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite value.") from error
    if not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be a positive finite value.")
