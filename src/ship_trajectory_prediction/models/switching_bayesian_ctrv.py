"""Switching Bayesian CTRV state-space model fitted with variational inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
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
from ship_trajectory_prediction.models.bayesian_ctrv import (
    DEFAULT_TURN_RATE_LIMIT,
    NOISE_PARAMETER_NAMES,
    POSITION_JITTER_THRESHOLD_METERS,
    SPEED_STATE_INITIAL_LOWER,
    SPEED_STATE_INITIAL_UPPER,
    BayesianCTRVPriors,
    PositionObservations,
    VIRunResult,
)
from ship_trajectory_prediction.models.bayesian_ctrv import (
    build_stan_data as build_bayesian_ctrv_stan_data,
)
from ship_trajectory_prediction.models.ctrv import CTRVState, ctrv_step
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import TrajectoryWindowData

STAN_FILE = project_path("stan/models/switching_bayesian_ctrv.stan")

MODE_COUNT = 3
MODE_STOP = 1
MODE_CRUISE = 2
MODE_MANEUVER = 3
MODE_NAMES: Mapping[int, str] = MappingProxyType(
    {
        MODE_STOP: "stop",
        MODE_CRUISE: "cruise",
        MODE_MANEUVER: "maneuver",
    }
)

DEFAULT_INITIAL_MODE_PROBABILITY = (1 / 3, 1 / 3, 1 / 3)
DEFAULT_ALPHA_TRANSITION = (
    (20.0, 1.0, 1.0),
    (1.0, 20.0, 1.0),
    (1.0, 1.0, 20.0),
)
DEFAULT_POSITION_PROCESS_MULTIPLIER = (0.5, 1.0, 3.0)
DEFAULT_SPEED_PROCESS_MULTIPLIER = (0.25, 1.0, 4.0)
DEFAULT_TURN_RATE_PROCESS_MULTIPLIER = (0.25, 1.0, 4.0)

_PROBABILITY_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class SwitchingCTRVConfig:
    """Fixed mode semantics and transition-prior hyperparameters.

    Multipliers scale global process-noise standard deviations for stop,
    cruise, and maneuver in this order. Decay times are expressed in seconds.
    The transition concentrations are dimensionless Dirichlet parameters.
    """

    initial_mode_probability: tuple[float, float, float] = (
        DEFAULT_INITIAL_MODE_PROBABILITY
    )
    alpha_transition: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = DEFAULT_ALPHA_TRANSITION
    position_process_multiplier: tuple[float, float, float] = (
        DEFAULT_POSITION_PROCESS_MULTIPLIER
    )
    speed_process_multiplier: tuple[float, float, float] = (
        DEFAULT_SPEED_PROCESS_MULTIPLIER
    )
    turn_rate_process_multiplier: tuple[float, float, float] = (
        DEFAULT_TURN_RATE_PROCESS_MULTIPLIER
    )
    stop_speed_decay_time: float = 20.0
    stop_turn_decay_time: float = 10.0

    def __post_init__(self) -> None:
        """Normalize and validate every switching hyperparameter."""
        initial_probability = _positive_simplex(
            "initial_mode_probability",
            self.initial_mode_probability,
            size=MODE_COUNT,
        )
        alpha_transition = _positive_matrix(
            "alpha_transition",
            self.alpha_transition,
            shape=(MODE_COUNT, MODE_COUNT),
        )
        position_multiplier = _ordered_positive_vector(
            "position_process_multiplier",
            self.position_process_multiplier,
        )
        speed_multiplier = _ordered_positive_vector(
            "speed_process_multiplier",
            self.speed_process_multiplier,
        )
        turn_rate_multiplier = _ordered_positive_vector(
            "turn_rate_process_multiplier",
            self.turn_rate_process_multiplier,
        )
        speed_decay_time = _positive_finite(
            "stop_speed_decay_time",
            self.stop_speed_decay_time,
        )
        turn_decay_time = _positive_finite(
            "stop_turn_decay_time",
            self.stop_turn_decay_time,
        )

        object.__setattr__(self, "initial_mode_probability", initial_probability)
        object.__setattr__(self, "alpha_transition", alpha_transition)
        object.__setattr__(
            self,
            "position_process_multiplier",
            position_multiplier,
        )
        object.__setattr__(self, "speed_process_multiplier", speed_multiplier)
        object.__setattr__(
            self,
            "turn_rate_process_multiplier",
            turn_rate_multiplier,
        )
        object.__setattr__(self, "stop_speed_decay_time", speed_decay_time)
        object.__setattr__(self, "stop_turn_decay_time", turn_decay_time)


def build_stan_data(
    window: TrajectoryWindowData,
    *,
    priors: BayesianCTRVPriors | None = None,
    config: SwitchingCTRVConfig | None = None,
    turn_rate_limit: float = DEFAULT_TURN_RATE_LIMIT,
    position_observations: PositionObservations | None = None,
) -> dict[str, Any]:
    """Build position-only data for the three-mode switching CTRV model.

    The common continuous-state priors and position inputs are constructed by
    the public Bayesian CTRV data builder. Switching hyperparameters are fixed
    experiment inputs and therefore never depend on held-out positions or GPS
    speed. Mode order is stop, cruise, and maneuver.
    """
    if config is None:
        config = SwitchingCTRVConfig()
    if not isinstance(config, SwitchingCTRVConfig):
        raise TypeError("config must be a SwitchingCTRVConfig instance or None.")

    stan_data = build_bayesian_ctrv_stan_data(
        window,
        priors=priors,
        turn_rate_limit=turn_rate_limit,
        position_observations=position_observations,
    )
    x_observed = np.asarray(stan_data["x_observed"], dtype=float)
    y_observed = np.asarray(stan_data["y_observed"], dtype=float)
    substantial_index = _substantial_displacement_index(
        x_observed,
        y_observed,
    )
    stan_data["heading_initial_prior_mean"] = _position_derived_heading_initial(
        x_observed,
        y_observed,
        fallback=float(stan_data["heading_initial_prior_mean"]),
    )
    if substantial_index is not None and substantial_index > 0:
        # While initially stationary, heading and turn rate are not observed.
        # The first clear later displacement supplies heading; a neutral turn
        # center avoids rotating through early position jitter.
        stan_data["turn_rate_initial_prior_mean"] = 0.0
    return {
        **stan_data,
        "K": MODE_COUNT,
        "initial_mode_probability": np.asarray(
            config.initial_mode_probability,
            dtype=float,
        ),
        "alpha_transition": np.asarray(config.alpha_transition, dtype=float),
        "position_process_multiplier": np.asarray(
            config.position_process_multiplier,
            dtype=float,
        ),
        "speed_process_multiplier": np.asarray(
            config.speed_process_multiplier,
            dtype=float,
        ),
        "turn_rate_process_multiplier": np.asarray(
            config.turn_rate_process_multiplier,
            dtype=float,
        ),
        "stop_speed_decay_time": config.stop_speed_decay_time,
        "stop_turn_decay_time": config.stop_turn_decay_time,
    }


def compile_switching_bayesian_ctrv_model(
    stan_file: str | Path = STAN_FILE,
) -> CmdStanModel:
    """Compile and return the switching Bayesian CTRV CmdStan model."""
    stan_path = Path(stan_file)
    if not stan_path.is_file():
        raise FileNotFoundError(f"Stan model not found: {stan_path}")
    return CmdStanModel(stan_file=str(stan_path))


def fit_switching_bayesian_ctrv_model(
    window: TrajectoryWindowData,
    *,
    priors: BayesianCTRVPriors | None = None,
    config: SwitchingCTRVConfig | None = None,
    turn_rate_limit: float = DEFAULT_TURN_RATE_LIMIT,
    position_observations: PositionObservations | None = None,
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
    """Fit the switching model using mean-field or full-rank CmdStan VI."""
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
    _reject_conflicting_options(
        options,
        {
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
        },
    )
    stan_data = build_stan_data(
        window,
        priors=priors,
        config=config,
        turn_rate_limit=turn_rate_limit,
        position_observations=position_observations,
    )
    if inits is None:
        inits = _default_initial_values(stan_data, seed=seed)

    model = compile_switching_bayesian_ctrv_model()
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


def forward_log_probabilities(
    transition_log_density,
    transition_probability,
    initial_mode_probability,
) -> np.ndarray:
    """Return unnormalized HMM forward log probabilities for each transition.

    ``transition_log_density[t, k]`` is the continuous-state transition log
    density under mode ``k``. Computation stays in the log domain and mirrors
    the analytically marginalized Stan recursion.
    """
    log_density = np.asarray(transition_log_density, dtype=float)
    if (
        log_density.ndim != 2
        or log_density.shape[0] < 1
        or log_density.shape[1] != MODE_COUNT
        or not np.all(np.isfinite(log_density))
    ):
        raise ValueError(
            "transition_log_density must be a finite matrix with three columns."
        )
    transition_probability = _strict_probability_matrix(
        "transition_probability",
        transition_probability,
        shape=(MODE_COUNT, MODE_COUNT),
    )
    initial_mode_probability = np.asarray(
        _positive_simplex(
            "initial_mode_probability",
            initial_mode_probability,
            size=MODE_COUNT,
        ),
        dtype=float,
    )

    log_transition = np.log(transition_probability)
    forward = np.empty_like(log_density)
    forward[0] = np.log(initial_mode_probability) + log_density[0]
    for transition_index in range(1, log_density.shape[0]):
        for current_mode in range(MODE_COUNT):
            forward[transition_index, current_mode] = log_density[
                transition_index, current_mode
            ] + np.logaddexp.reduce(
                forward[transition_index - 1] + log_transition[:, current_mode]
            )
    if not np.all(np.isfinite(forward)):
        raise ValueError("Forward recursion produced non-finite values.")
    return forward


def expected_mode_transition(
    state: CTRVState,
    dt_seconds: float,
    mode: int,
    *,
    config: SwitchingCTRVConfig | None = None,
) -> CTRVState:
    """Return the deterministic transition mean for one movement mode.

    Cruise and maneuver share the CTRV mean and differ through their fixed
    process-noise multipliers. Stop retains position while speed and turn rate
    decay exponentially. Heading remains unwrapped in all modes.
    """
    if not isinstance(state, CTRVState):
        raise TypeError("state must be a CTRVState instance.")
    dt_seconds = _positive_finite("dt_seconds", dt_seconds)
    mode = _validate_mode(mode)
    if config is None:
        config = SwitchingCTRVConfig()
    if not isinstance(config, SwitchingCTRVConfig):
        raise TypeError("config must be a SwitchingCTRVConfig instance or None.")

    if mode in {MODE_CRUISE, MODE_MANEUVER}:
        return ctrv_step(state, dt_seconds)

    return CTRVState(
        x=state.x,
        y=state.y,
        speed=state.speed * np.exp(-dt_seconds / config.stop_speed_decay_time),
        heading=state.heading + state.turn_rate * dt_seconds,
        turn_rate=(state.turn_rate * np.exp(-dt_seconds / config.stop_turn_decay_time)),
    )


def summarize_switching_predictions(
    fit: Any,
    window: TrajectoryWindowData,
    credible_interval: float = 0.9,
    *,
    include_speed_gps_reference: bool = False,
) -> pd.DataFrame:
    """Summarize future states, observations, and sampled movement modes."""
    credible_interval = _credible_interval(credible_interval)
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
    samples = {
        prefix: _posterior_samples(
            fit,
            variable_name,
            expected_tail=(window.prediction_count,),
        )
        for prefix, variable_name in prediction_variables.items()
    }
    mode_prediction = _mode_prediction_samples(fit, window.prediction_count)
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
    for prefix, values in samples.items():
        table_data[f"{prefix}_median"] = np.median(values, axis=0)
        table_data[f"{prefix}_lower"] = np.quantile(
            values,
            lower_probability,
            axis=0,
        )
        table_data[f"{prefix}_upper"] = np.quantile(
            values,
            upper_probability,
            axis=0,
        )

    mode_probabilities = np.column_stack(
        [
            np.mean(mode_prediction == mode, axis=0)
            for mode in (MODE_STOP, MODE_CRUISE, MODE_MANEUVER)
        ]
    )
    for mode_index, mode_name in MODE_NAMES.items():
        table_data[f"mode_{mode_name}_probability"] = mode_probabilities[
            :, mode_index - 1
        ]
    most_likely_mode = np.argmax(mode_probabilities, axis=1) + 1
    table_data["most_likely_mode"] = most_likely_mode
    table_data["most_likely_mode_name"] = [
        MODE_NAMES[int(mode)] for mode in most_likely_mode
    ]
    return pd.DataFrame(table_data)


def summarize_mode_probabilities(
    fit: Any,
    window: TrajectoryWindowData,
    credible_interval: float = 0.9,
) -> pd.DataFrame:
    """Summarize smoothed mode probabilities at observed transitions."""
    credible_interval = _credible_interval(credible_interval)
    transition_count = window.observation_count - 1
    samples = _mode_probability_samples(fit, transition_count)
    lower_probability = (1 - credible_interval) / 2
    upper_probability = 1 - lower_probability
    observed_endpoints = slice(1, window.observation_count)
    table_data: dict[str, Any] = {
        "transition_index": np.arange(1, transition_count + 1),
        "time": window.timestamps[observed_endpoints],
        "t": window.time_seconds[observed_endpoints],
    }
    mean_probability = np.mean(samples, axis=0)
    for mode_index, mode_name in MODE_NAMES.items():
        mode_samples = samples[:, :, mode_index - 1]
        table_data[f"{mode_name}_probability"] = mean_probability[:, mode_index - 1]
        table_data[f"{mode_name}_median"] = np.median(mode_samples, axis=0)
        table_data[f"{mode_name}_lower"] = np.quantile(
            mode_samples,
            lower_probability,
            axis=0,
        )
        table_data[f"{mode_name}_upper"] = np.quantile(
            mode_samples,
            upper_probability,
            axis=0,
        )
    most_likely_mode = np.argmax(mean_probability, axis=1) + 1
    table_data["most_likely_mode"] = most_likely_mode
    table_data["most_likely_mode_name"] = [
        MODE_NAMES[int(mode)] for mode in most_likely_mode
    ]
    return pd.DataFrame(table_data)


def summarize_transition_probabilities(
    fit: Any,
    credible_interval: float = 0.9,
) -> pd.DataFrame:
    """Return a long-form posterior summary of the transition matrix."""
    credible_interval = _credible_interval(credible_interval)
    samples = _transition_probability_samples(fit)
    lower_probability = (1 - credible_interval) / 2
    upper_probability = 1 - lower_probability
    rows = []
    for from_mode, from_name in MODE_NAMES.items():
        for to_mode, to_name in MODE_NAMES.items():
            values = samples[:, from_mode - 1, to_mode - 1]
            rows.append(
                {
                    "from_mode": from_mode,
                    "from_mode_name": from_name,
                    "to_mode": to_mode,
                    "to_mode_name": to_name,
                    "probability_mean": float(np.mean(values)),
                    "probability_median": float(np.median(values)),
                    "probability_lower": float(np.quantile(values, lower_probability)),
                    "probability_upper": float(np.quantile(values, upper_probability)),
                }
            )
    return pd.DataFrame(rows)


def compare_switching_vi_runs(
    runs: Sequence[VIRunResult],
    window: TrajectoryWindowData,
    *,
    credible_interval: float = 0.9,
) -> pd.DataFrame:
    """Compare switching VI runs using fit, prediction, and mode stability."""
    if not runs:
        raise ValueError("runs must contain at least one variational fit.")
    credible_interval = _credible_interval(credible_interval)

    rows = []
    for run in runs:
        if not isinstance(run, VIRunResult):
            raise TypeError("Each run must be a VIRunResult instance.")
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
        x_prediction = _posterior_samples(
            run.fit,
            "x_state_prediction",
            expected_tail=(window.prediction_count,),
        )
        y_prediction = _posterior_samples(
            run.fit,
            "y_state_prediction",
            expected_tail=(window.prediction_count,),
        )
        mode_probability = _mode_probability_samples(
            run.fit,
            window.observation_count - 1,
        )
        mode_prediction = _mode_prediction_samples(
            run.fit,
            window.prediction_count,
        )
        transition_probability = _transition_probability_samples(run.fit)

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
            "endpoint_x_std_m": _sample_standard_deviation(x_prediction[:, -1]),
            "endpoint_y_std_m": _sample_standard_deviation(y_prediction[:, -1]),
            "ade_m": evaluation.ade_m,
            "fde_m": evaluation.fde_m,
            "mean_interval_width_m": (evaluation.mean_marginal_interval_width_m),
            "radial_coverage": evaluation.radial_coverage,
        }
        for mode_index, mode_name in MODE_NAMES.items():
            row[f"observed_{mode_name}_probability_mean"] = float(
                np.mean(mode_probability[:, :, mode_index - 1])
            )
            row[f"predicted_{mode_name}_probability_mean"] = float(
                np.mean(mode_prediction == mode_index)
            )
        for from_mode, from_name in MODE_NAMES.items():
            for to_mode, to_name in MODE_NAMES.items():
                values = transition_probability[
                    :,
                    from_mode - 1,
                    to_mode - 1,
                ]
                prefix = f"transition_{from_name}_to_{to_name}"
                row[f"{prefix}_mean"] = float(np.mean(values))
                row[f"{prefix}_std"] = _sample_standard_deviation(values)
        for parameter_name in NOISE_PARAMETER_NAMES:
            values = _posterior_samples(
                run.fit,
                parameter_name,
                expected_tail=(),
            )
            row[f"{parameter_name}_mean"] = float(np.mean(values))
            row[f"{parameter_name}_std"] = _sample_standard_deviation(values)
        rows.append(row)
    return pd.DataFrame(rows)


def _default_initial_values(stan_data: Mapping[str, Any], *, seed: int):
    """Create seeded position-only initials strictly inside Stan bounds."""
    generator = np.random.default_rng(seed)
    time_observed = np.asarray(stan_data["time_observed"], dtype=float)
    x_observed = np.asarray(stan_data["x_observed"], dtype=float)
    y_observed = np.asarray(stan_data["y_observed"], dtype=float)
    x_initial = _median_smoothed_positions(x_observed)
    y_initial = _median_smoothed_positions(y_observed)
    state_count = int(stan_data["N_observed"])
    speed_initial = _position_derived_speed_initials(
        time_observed,
        x_initial,
        y_initial,
        initial_prior_mean=float(stan_data["speed_initial_prior_mean"]),
    )
    heading_initial = _position_derived_heading_initial(
        x_initial,
        y_initial,
        fallback=float(stan_data["heading_initial_prior_mean"]),
    )
    turn_rate_center = (
        0.0
        if speed_initial[0] <= SPEED_STATE_INITIAL_LOWER
        else float(stan_data["turn_rate_initial_prior_mean"])
    )
    x_jitter = min(0.1, 0.02 * stan_data["position_initial_prior_scale"])
    speed_jitter = min(0.02, 0.02 * stan_data["speed_initial_prior_scale"])
    turn_jitter = min(
        1e-4,
        0.02 * stan_data["turn_rate_state_prior_scale"],
    )
    alpha_transition = np.asarray(stan_data["alpha_transition"], dtype=float)
    transition_initial = alpha_transition / np.sum(
        alpha_transition,
        axis=1,
        keepdims=True,
    )
    return {
        "x_state": x_initial + generator.normal(0, x_jitter, state_count),
        "y_state": y_initial + generator.normal(0, x_jitter, state_count),
        "speed_state": np.clip(
            speed_initial + generator.normal(0, speed_jitter, state_count),
            SPEED_STATE_INITIAL_LOWER,
            SPEED_STATE_INITIAL_UPPER,
        ),
        "heading_initial": heading_initial + generator.normal(0, 0.01),
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
        "transition_probability": transition_initial,
    }


def _median_smoothed_positions(values: np.ndarray, *, width: int = 5) -> np.ndarray:
    """Return position-only median-smoothed initials without changing data."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Position initials require a finite one-dimensional vector.")
    radius = min(width // 2, (values.size - 1) // 2)
    if radius == 0:
        return values.copy()
    result = values.copy()
    for index in range(radius, values.size - radius):
        result[index] = np.median(values[index - radius : index + radius + 1])
    return result


def _position_derived_heading_initial(
    x_meters: np.ndarray,
    y_meters: np.ndarray,
    *,
    fallback: float,
) -> float:
    """Use the earliest substantial observed displacement as heading initial."""
    delta_x = np.diff(np.asarray(x_meters, dtype=float))
    delta_y = np.diff(np.asarray(y_meters, dtype=float))
    index = _substantial_displacement_index(x_meters, y_meters)
    if index is None:
        return float(fallback)
    return float(np.arctan2(delta_y[index], delta_x[index]))


def _substantial_displacement_index(
    x_meters: np.ndarray,
    y_meters: np.ndarray,
) -> int | None:
    """Return the first segment clearly larger than position jitter."""
    displacement = np.hypot(
        np.diff(np.asarray(x_meters, dtype=float)),
        np.diff(np.asarray(y_meters, dtype=float)),
    )
    if displacement.size == 0 or not np.all(np.isfinite(displacement)):
        return None
    maximum = float(np.max(displacement))
    if maximum <= 3 * POSITION_JITTER_THRESHOLD_METERS:
        return None
    threshold = max(
        POSITION_JITTER_THRESHOLD_METERS,
        0.5 * maximum,
    )
    candidates = np.flatnonzero(displacement >= threshold)
    return None if candidates.size == 0 else int(candidates[0])


def _position_derived_speed_initials(
    time_seconds,
    x_meters,
    y_meters,
    *,
    initial_prior_mean: float,
) -> np.ndarray:
    """Map position-derived segment speeds onto state timestamps."""
    time_seconds = np.asarray(time_seconds, dtype=float)
    x_meters = np.asarray(x_meters, dtype=float)
    y_meters = np.asarray(y_meters, dtype=float)
    if (
        time_seconds.ndim != 1
        or time_seconds.size < 2
        or x_meters.shape != time_seconds.shape
        or y_meters.shape != time_seconds.shape
        or not np.all(np.isfinite(time_seconds))
        or not np.all(np.isfinite(x_meters))
        or not np.all(np.isfinite(y_meters))
    ):
        raise ValueError("Position-derived initials require matching finite vectors.")
    time_differences = np.diff(time_seconds)
    if np.any(time_differences <= 0):
        raise ValueError("time_seconds must be strictly increasing.")
    displacement = np.hypot(np.diff(x_meters), np.diff(y_meters))
    segment_speed = displacement / time_differences
    segment_speed[displacement <= POSITION_JITTER_THRESHOLD_METERS] = 0.0
    state_speed = np.empty(time_seconds.size, dtype=float)
    state_speed[0] = _non_negative_finite(
        "initial_prior_mean",
        initial_prior_mean,
    )
    state_speed[1:] = segment_speed
    return state_speed


def _posterior_samples(
    fit: Any,
    variable_name: str,
    *,
    expected_tail: tuple[int, ...],
) -> np.ndarray:
    """Extract one finite posterior array with an explicit draw dimension."""
    samples = posterior_variable_samples(fit, variable_name)
    expected_ndim = len(expected_tail) + 1
    if (
        samples.ndim != expected_ndim
        or samples.shape[0] < 1
        or samples.shape[1:] != expected_tail
        or not np.all(np.isfinite(samples))
    ):
        raise ValueError(
            f"Posterior variable {variable_name!r} has an unexpected shape "
            "or non-finite draws."
        )
    return samples


def _mode_probability_samples(fit: Any, transition_count: int) -> np.ndarray:
    """Extract and validate smoothed observed-mode probabilities."""
    samples = _posterior_samples(
        fit,
        "mode_probability",
        expected_tail=(transition_count, MODE_COUNT),
    )
    _validate_probability_draws("mode_probability", samples)
    return samples


def _transition_probability_samples(fit: Any) -> np.ndarray:
    """Extract and validate posterior transition-matrix draws."""
    samples = _posterior_samples(
        fit,
        "transition_probability",
        expected_tail=(MODE_COUNT, MODE_COUNT),
    )
    _validate_probability_draws("transition_probability", samples)
    return samples


def _mode_prediction_samples(fit: Any, prediction_count: int) -> np.ndarray:
    """Extract future sampled modes as a validated integer-valued matrix."""
    samples = _posterior_samples(
        fit,
        "mode_prediction",
        expected_tail=(prediction_count,),
    )
    rounded = np.rint(samples)
    if not np.array_equal(samples, rounded) or np.any(
        (rounded < MODE_STOP) | (rounded > MODE_MANEUVER)
    ):
        raise ValueError("mode_prediction must contain integer modes 1, 2, or 3.")
    return rounded.astype(int)


def _validate_probability_draws(name: str, samples: np.ndarray) -> None:
    """Validate simplex values along the final array dimension."""
    if np.any(samples < -_PROBABILITY_TOLERANCE) or np.any(
        samples > 1 + _PROBABILITY_TOLERANCE
    ):
        raise ValueError(f"{name} must contain probabilities between zero and one.")
    if not np.allclose(
        np.sum(samples, axis=-1),
        1.0,
        rtol=0.0,
        atol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError(f"Every {name} row must sum to one.")


def _validate_variational_arguments(**arguments: Any) -> None:
    """Validate the explicitly supported CmdStan VI controls."""
    if arguments["algorithm"] not in {"meanfield", "fullrank"}:
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
        _positive_finite(name, arguments[name])
    for name in ("require_converged", "show_console"):
        if not isinstance(arguments[name], bool):
            raise ValueError(f"{name} must be a boolean.")


def _reject_conflicting_options(
    options: Mapping[str, Any],
    controlled_options: set[str],
) -> None:
    """Reject generic options that override explicit wrapper controls."""
    conflicting = controlled_options.intersection(options)
    if conflicting:
        names = ", ".join(sorted(conflicting))
        raise ValueError(f"variational_options must not override: {names}.")


def _positive_simplex(name: str, values, *, size: int) -> tuple[float, ...]:
    """Return a finite, strictly positive simplex as an immutable tuple."""
    array = np.asarray(values, dtype=float)
    if (
        array.shape != (size,)
        or not np.all(np.isfinite(array))
        or np.any(array <= 0)
        or not np.isclose(
            np.sum(array),
            1.0,
            rtol=0.0,
            atol=_PROBABILITY_TOLERANCE,
        )
    ):
        raise ValueError(f"{name} must be a positive simplex of length {size}.")
    return tuple(float(value) for value in array)


def _strict_probability_matrix(
    name: str,
    values,
    *,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return a finite matrix of strictly positive simplex rows."""
    array = np.asarray(values, dtype=float)
    if (
        array.shape != shape
        or not np.all(np.isfinite(array))
        or np.any(array <= 0)
        or not np.allclose(
            np.sum(array, axis=1),
            1.0,
            rtol=0.0,
            atol=_PROBABILITY_TOLERANCE,
        )
    ):
        raise ValueError(f"{name} must contain positive simplex rows.")
    return array


def _positive_matrix(
    name: str,
    values,
    *,
    shape: tuple[int, int],
) -> tuple[tuple[float, ...], ...]:
    """Return one immutable matrix of positive finite values."""
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(
            f"{name} must be a positive finite {shape[0]}x{shape[1]} matrix."
        )
    return tuple(tuple(float(value) for value in row) for row in array)


def _ordered_positive_vector(name: str, values) -> tuple[float, float, float]:
    """Validate fixed stop, cruise, and maneuver process-noise ordering."""
    array = np.asarray(values, dtype=float)
    if (
        array.shape != (MODE_COUNT,)
        or not np.all(np.isfinite(array))
        or np.any(array <= 0)
        or not array[MODE_STOP - 1] < array[MODE_CRUISE - 1] < array[MODE_MANEUVER - 1]
    ):
        raise ValueError(
            f"{name} must contain positive stop < cruise < maneuver values."
        )
    return tuple(float(value) for value in array)


def _validate_mode(mode: int) -> int:
    """Return one supported one-based mode index."""
    if isinstance(mode, bool) or not isinstance(mode, (int, np.integer)):
        raise ValueError("mode must be MODE_STOP, MODE_CRUISE, or MODE_MANEUVER.")
    mode = int(mode)
    if mode not in MODE_NAMES:
        raise ValueError("mode must be MODE_STOP, MODE_CRUISE, or MODE_MANEUVER.")
    return mode


def _credible_interval(value: float) -> float:
    """Return one probability strictly between zero and one."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("credible_interval must be between 0 and 1.") from error
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("credible_interval must be between 0 and 1.")
    return value


def _sample_standard_deviation(values: np.ndarray) -> float:
    """Return the sample deviation, or zero for a single draw."""
    return float(np.std(values, ddof=1) if values.size > 1 else 0.0)


def _positive_finite(name: str, value: float) -> float:
    """Return one positive finite scalar."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive finite value.") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite value.")
    return value


def _non_negative_finite(name: str, value: float) -> float:
    """Return one non-negative finite scalar."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a non-negative finite value.") from error
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite value.")
    return value


__all__ = [
    "MODE_COUNT",
    "MODE_CRUISE",
    "MODE_MANEUVER",
    "MODE_NAMES",
    "MODE_STOP",
    "STAN_FILE",
    "SwitchingCTRVConfig",
    "build_stan_data",
    "compare_switching_vi_runs",
    "compile_switching_bayesian_ctrv_model",
    "expected_mode_transition",
    "fit_switching_bayesian_ctrv_model",
    "forward_log_probabilities",
    "summarize_mode_probabilities",
    "summarize_switching_predictions",
    "summarize_transition_probabilities",
]
