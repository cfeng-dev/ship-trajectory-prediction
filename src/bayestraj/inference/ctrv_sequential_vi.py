"""Sequential variational inference for the Bayesian CTRV model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

import bayestraj.inference.ctrv_cmdstan as batch_inference
import bayestraj.inference.particle_utils as particle_utils
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.numeric_validation as numeric_validation
import bayestraj.observations.position as observation_support
import bayestraj.observations.window as observation_window
import bayestraj.stan as stan_resources
from bayestraj.inference import cmdstan
from bayestraj.models import ctrv as ctrv_dynamics

CARRY_SIZE = 8
MINIMUM_POSITIVE_SCALE = 1e-9
SEQUENTIAL_STAN_FILE = stan_resources.stan_path("models/sequential_bayesian_ctrv.stan")


@dataclass(frozen=True, slots=True)
class SequentialVIConfig:
    """Validated controls for bootstrap and one-step full-rank ADVI."""

    n_bootstrap: int = 10
    algorithm: str = "fullrank"
    iter: int = cmdstan.DEFAULT_VI_ITER
    grad_samples: int = cmdstan.DEFAULT_VI_GRAD_SAMPLES
    elbo_samples: int = cmdstan.DEFAULT_VI_ELBO_SAMPLES
    eta: float = cmdstan.DEFAULT_VI_ETA
    adapt_iter: int = cmdstan.DEFAULT_VI_ADAPT_ITER
    tol_rel_obj: float = cmdstan.DEFAULT_VI_TOL_REL_OBJ
    eval_elbo: int = cmdstan.DEFAULT_VI_EVAL_ELBO
    draws: int = cmdstan.DEFAULT_VI_DRAWS
    require_converged: bool = False
    show_console: bool = False
    covariance_jitter: float = 1e-9
    regularization_attempts: int = 8
    predictive_log_density_threshold: float | None = -10.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_bootstrap, bool)
            or not isinstance(self.n_bootstrap, (int, np.integer))
            or self.n_bootstrap < 3
        ):
            raise ValueError(
                "n_bootstrap must be an integer greater than or equal to 3."
            )
        if self.algorithm != "fullrank":
            raise ValueError("algorithm must be 'fullrank' for Sequential VI.")
        if (
            isinstance(self.draws, bool)
            or not isinstance(self.draws, (int, np.integer))
            or self.draws < 2
        ):
            raise ValueError("draws must be an integer greater than or equal to 2.")
        jitter = float(self.covariance_jitter)
        if not np.isfinite(jitter) or jitter <= 0.0:
            raise ValueError("covariance_jitter must be strictly positive and finite.")
        if (
            isinstance(self.regularization_attempts, bool)
            or not isinstance(self.regularization_attempts, (int, np.integer))
            or self.regularization_attempts < 1
        ):
            raise ValueError("regularization_attempts must be a positive integer.")
        cmdstan.validate_variational_arguments(
            algorithm=self.algorithm,
            iter=self.iter,
            grad_samples=self.grad_samples,
            elbo_samples=self.elbo_samples,
            eta=self.eta,
            adapt_iter=self.adapt_iter,
            tol_rel_obj=self.tol_rel_obj,
            eval_elbo=self.eval_elbo,
            draws=self.draws,
            seed=1,
            require_converged=self.require_converged,
            show_console=self.show_console,
        )
        threshold = self.predictive_log_density_threshold
        if threshold is not None:
            threshold = _finite_scalar(
                threshold,
                name="predictive_log_density_threshold",
            )
        object.__setattr__(self, "covariance_jitter", jitter)
        object.__setattr__(
            self,
            "regularization_attempts",
            int(self.regularization_attempts),
        )
        object.__setattr__(self, "predictive_log_density_threshold", threshold)


def _frozen_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = array.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class GaussianCarry:
    """Full-covariance Gaussian approximation in constraint-safe coordinates."""

    mean: np.ndarray
    cholesky: np.ndarray
    heading_reference: float
    draw_count: int

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        cholesky = np.asarray(self.cholesky, dtype=float)
        if mean.shape != (CARRY_SIZE,):
            raise ValueError(f"mean must have shape ({CARRY_SIZE},).")
        if cholesky.shape != (CARRY_SIZE, CARRY_SIZE):
            raise ValueError(f"cholesky must have shape ({CARRY_SIZE}, {CARRY_SIZE}).")
        if not np.all(np.isfinite(mean)):
            raise ValueError("mean must contain only finite values.")
        if not np.all(np.isfinite(cholesky)):
            raise ValueError("cholesky must contain only finite values.")
        if not np.allclose(cholesky, np.tril(cholesky)):
            raise ValueError("cholesky must be lower triangular.")
        if np.any(np.diag(cholesky) <= 0.0):
            raise ValueError("cholesky must represent a positive definite covariance.")
        heading_reference = float(self.heading_reference)
        if not np.isfinite(heading_reference):
            raise ValueError("heading_reference must be finite.")
        if (
            isinstance(self.draw_count, bool)
            or not isinstance(self.draw_count, (int, np.integer))
            or self.draw_count < 2
        ):
            raise ValueError(
                "draw_count must be an integer greater than or equal to 2."
            )

        frozen_mean = mean.copy()
        frozen_cholesky = cholesky.copy()
        frozen_mean.flags.writeable = False
        frozen_cholesky.flags.writeable = False
        object.__setattr__(self, "mean", frozen_mean)
        object.__setattr__(self, "cholesky", frozen_cholesky)
        object.__setattr__(
            self,
            "heading_reference",
            float((heading_reference + np.pi) % (2.0 * np.pi) - np.pi),
        )
        object.__setattr__(self, "draw_count", int(self.draw_count))


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Scalar posterior summary."""

    mean: float
    median: float
    standard_deviation: float
    lower_95: float
    upper_95: float

    def __post_init__(self) -> None:
        for summary_field in fields(self):
            value = float(getattr(self, summary_field.name))
            if not np.isfinite(value):
                raise ValueError(f"{summary_field.name} must be finite.")
            object.__setattr__(self, summary_field.name, value)
        if self.standard_deviation < 0.0:
            raise ValueError("standard_deviation must be non-negative.")
        if not self.lower_95 <= self.median <= self.upper_95:
            raise ValueError(
                "summary interval must satisfy lower_95 <= median <= upper_95."
            )


def _validate_summary_fields(instance: object) -> None:
    for summary_field in fields(instance):
        if not isinstance(getattr(instance, summary_field.name), DistributionSummary):
            raise TypeError(f"{summary_field.name} must be a DistributionSummary.")


@dataclass(frozen=True, slots=True)
class SequentialVIStateSummary:
    """Posterior summaries for one CTRV state."""

    x: DistributionSummary
    y: DistributionSummary
    speed: DistributionSummary
    heading: DistributionSummary
    turn_rate: DistributionSummary

    def __post_init__(self) -> None:
        _validate_summary_fields(self)


@dataclass(frozen=True, slots=True)
class SequentialVIParameterSummary:
    """Posterior summaries for the three CTRV noise scales."""

    sigma_position_observation: DistributionSummary
    sigma_speed_process: DistributionSummary
    sigma_turn_rate_process: DistributionSummary

    def __post_init__(self) -> None:
        _validate_summary_fields(self)


@dataclass(frozen=True, slots=True)
class SequentialVIPredictiveDraws:
    """One-step observation-predictive draws retained for scoring."""

    x: np.ndarray
    y: np.ndarray
    sigma_position_observation: np.ndarray

    def __post_init__(self) -> None:
        x = _frozen_vector(self.x, name="x")
        y = _frozen_vector(self.y, name="y")
        sigma = _frozen_vector(
            self.sigma_position_observation,
            name="sigma_position_observation",
        )
        if not x.shape == y.shape == sigma.shape:
            raise ValueError("predictive draw arrays must have matching shapes.")
        if np.any(sigma <= 0.0):
            raise ValueError("sigma_position_observation must be strictly positive.")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "sigma_position_observation", sigma)


def _predictive_log_density(
    predictive_draws: SequentialVIPredictiveDraws,
    *,
    x_observed: float,
    y_observed: float,
) -> float:
    """Evaluate the predictive position mixture before the variational update."""
    squared_error = (x_observed - predictive_draws.x) ** 2 + (
        y_observed - predictive_draws.y
    ) ** 2
    observation_noise = predictive_draws.sigma_position_observation
    log_components = (
        -np.log(2.0 * np.pi)
        - 2.0 * np.log(observation_noise)
        - 0.5 * squared_error / observation_noise**2
    )
    return float(_logsumexp(log_components) - np.log(log_components.size))


def _logsumexp(log_values: np.ndarray) -> float:
    """Return a stable logarithm of a finite positive sum of exponentials."""
    maximum_log_value = float(np.max(log_values))
    if not np.isfinite(maximum_log_value):
        raise RuntimeError("Sequential VI predictive density collapsed.")
    shifted_sum = float(np.sum(np.exp(log_values - maximum_log_value)))
    if not np.isfinite(shifted_sum) or shifted_sum <= 0.0:
        raise RuntimeError("Sequential VI predictive density collapsed.")
    return maximum_log_value + float(np.log(shifted_sum))


@dataclass(frozen=True, slots=True)
class SequentialVIUpdate:
    """Diagnostics and posterior summaries for one attempted online update."""

    observation_index: int
    time_seconds: float
    x_observed: float
    y_observed: float
    predictive_state: SequentialVIStateSummary
    filtered_state: SequentialVIStateSummary
    predictive_parameters: SequentialVIParameterSummary
    filtered_parameters: SequentialVIParameterSummary
    predictive_draws: SequentialVIPredictiveDraws
    runtime_seconds: float
    cumulative_runtime_seconds: float
    converged: bool
    error_message: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.observation_index, bool)
            or not isinstance(self.observation_index, (int, np.integer))
            or self.observation_index < 0
        ):
            raise ValueError("observation_index must be a non-negative integer.")
        for name in (
            "time_seconds",
            "x_observed",
            "y_observed",
            "runtime_seconds",
            "cumulative_runtime_seconds",
        ):
            value = _finite_scalar(getattr(self, name), name=name)
            if name.endswith("runtime_seconds") and value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)
        if not isinstance(self.converged, bool):
            raise ValueError("converged must be a boolean.")
        if self.error_message is not None and not isinstance(self.error_message, str):
            raise ValueError("error_message must be a string or None.")
        object.__setattr__(self, "observation_index", int(self.observation_index))


@dataclass(frozen=True, slots=True)
class SequentialVIResult:
    """Result of a bootstrap followed by attempted sequential updates."""

    bootstrap_runtime_seconds: float
    bootstrap_converged: bool
    bootstrap_draw_count: int
    updates: tuple[SequentialVIUpdate, ...]
    processed_observation_count: int
    completed: bool
    failure_index: int | None
    failure_message: str | None

    def __post_init__(self) -> None:
        runtime = _finite_scalar(
            self.bootstrap_runtime_seconds,
            name="bootstrap_runtime_seconds",
        )
        if runtime < 0.0:
            raise ValueError("bootstrap_runtime_seconds must be non-negative.")
        for name in ("bootstrap_draw_count", "processed_observation_count"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer.")
            object.__setattr__(self, name, int(value))
        if not isinstance(self.bootstrap_converged, bool):
            raise ValueError("bootstrap_converged must be a boolean.")
        if not isinstance(self.completed, bool):
            raise ValueError("completed must be a boolean.")
        if not isinstance(self.updates, tuple) or not all(
            isinstance(update, SequentialVIUpdate) for update in self.updates
        ):
            raise TypeError("updates must be a tuple of SequentialVIUpdate values.")
        object.__setattr__(self, "bootstrap_runtime_seconds", runtime)


def _posterior_draws(fit: object, name: str) -> np.ndarray:
    if hasattr(fit, "variational_sample"):
        return np.asarray(fit.stan_variable(name, mean=False), dtype=float)
    return np.asarray(fit.stan_variable(name), dtype=float)


def _posterior_vector(fit: object, name: str) -> np.ndarray:
    try:
        values = _posterior_draws(fit, name)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Posterior variable {name!r} is unavailable.") from error
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"Posterior variable {name!r} must be one-dimensional.")
    if vector.size < 2:
        raise ValueError(
            f"Posterior variable {name!r} must contain at least two draws."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"Posterior variable {name!r} must contain only finite draws.")
    return vector


def _positive_posterior_vector(fit: object, name: str) -> np.ndarray:
    vector = _posterior_vector(fit, name)
    if np.any(vector <= 0.0):
        raise ValueError(f"Posterior variable {name!r} must be strictly positive.")
    return vector


def _posterior_origin_position(fit: object, axis: str) -> np.ndarray:
    explicit_name = f"{axis}_at_origin"
    try:
        return _posterior_vector(fit, explicit_name)
    except ValueError as explicit_error:
        state_name = f"{axis}_state"
        try:
            states = _posterior_draws(fit, state_name)
        except (KeyError, ValueError) as error:
            raise explicit_error from error
        if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 1:
            raise ValueError(
                f"Posterior variable {state_name!r} must be a non-empty draw matrix."
            ) from None
        if not np.all(np.isfinite(states)):
            raise ValueError(
                f"Posterior variable {state_name!r} must contain only finite draws."
            ) from None
        return states[:, -1]


def transformed_draws(
    fit: object,
    *,
    minimum_positive: float = MINIMUM_POSITIVE_SCALE,
    position_observation_noise_floor_m: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Return posterior draws in the Gaussian carry coordinate system."""
    minimum_positive = float(minimum_positive)
    if not np.isfinite(minimum_positive) or minimum_positive <= 0.0:
        raise ValueError("minimum_positive must be strictly positive and finite.")
    position_observation_noise_floor_m = (
        numeric_validation.validate_non_negative_finite(
            "position_observation_noise_floor_m",
            position_observation_noise_floor_m,
        )
    )

    headings = _posterior_vector(fit, "heading_at_origin")
    heading_reference = float(
        np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings)))
    )
    local_headings = ctrv_dynamics.wrap_angles(headings - heading_reference)
    speed = np.maximum(
        _posterior_vector(fit, "speed_at_origin"),
        minimum_positive,
    )
    observation_noise = _positive_posterior_vector(
        fit,
        "sigma_position_observation",
    )
    observation_noise_excess = np.sqrt(
        np.maximum(
            observation_noise**2 - position_observation_noise_floor_m**2,
            minimum_positive**2,
        )
    )
    vectors = (
        _posterior_origin_position(fit, "x"),
        _posterior_origin_position(fit, "y"),
        np.log(speed),
        local_headings,
        _posterior_vector(fit, "turn_rate_at_origin"),
        np.log(observation_noise_excess),
        np.log(_positive_posterior_vector(fit, "sigma_speed_process")),
        np.log(_positive_posterior_vector(fit, "sigma_turn_rate_process")),
    )
    draw_counts = {vector.shape[0] for vector in vectors}
    if len(draw_counts) != 1:
        raise ValueError("Posterior variables must have matching draw counts.")
    transformed = np.column_stack(vectors)
    if not np.all(np.isfinite(transformed)):
        raise ValueError("Transformed posterior draws must be finite.")
    return transformed, heading_reference


def _regularized_cholesky(
    covariance: np.ndarray,
    *,
    covariance_jitter: float,
    regularization_attempts: int,
) -> np.ndarray:
    covariance_jitter = float(covariance_jitter)
    if not np.isfinite(covariance_jitter) or covariance_jitter <= 0.0:
        raise ValueError("covariance_jitter must be strictly positive and finite.")
    if (
        isinstance(regularization_attempts, bool)
        or not isinstance(regularization_attempts, (int, np.integer))
        or regularization_attempts < 1
    ):
        raise ValueError("regularization_attempts must be a positive integer.")
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (CARRY_SIZE, CARRY_SIZE):
        raise ValueError(f"covariance must have shape ({CARRY_SIZE}, {CARRY_SIZE}).")
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must contain only finite values.")

    identity = np.eye(CARRY_SIZE)
    jitter = covariance_jitter
    for _ in range(int(regularization_attempts)):
        try:
            return np.linalg.cholesky(covariance + jitter * identity)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise ValueError("covariance could not be regularized to positive definite form.")


def build_gaussian_carry(
    fit: object,
    *,
    covariance_jitter: float = 1e-9,
    regularization_attempts: int = 8,
    position_observation_noise_floor_m: float = 0.0,
) -> GaussianCarry:
    """Project a posterior fit to a full-covariance Gaussian carry."""
    transformed, heading_reference = transformed_draws(
        fit,
        position_observation_noise_floor_m=position_observation_noise_floor_m,
    )
    covariance = np.cov(transformed, rowvar=False, ddof=1)
    cholesky = _regularized_cholesky(
        covariance,
        covariance_jitter=covariance_jitter,
        regularization_attempts=regularization_attempts,
    )
    return GaussianCarry(
        mean=np.mean(transformed, axis=0),
        cholesky=cholesky,
        heading_reference=heading_reference,
        draw_count=transformed.shape[0],
    )


def _physical_draw_mapping(
    transformed: np.ndarray,
    heading_reference: float,
    *,
    position_observation_noise_floor_m: float = 0.0,
) -> dict[str, np.ndarray]:
    maximum_log = float(np.log(np.finfo(float).max))
    position_observation_noise_floor_m = (
        numeric_validation.validate_non_negative_finite(
            "position_observation_noise_floor_m",
            position_observation_noise_floor_m,
        )
    )
    return {
        "x_at_origin": transformed[:, 0],
        "y_at_origin": transformed[:, 1],
        "speed_at_origin": np.exp(np.minimum(transformed[:, 2], maximum_log)),
        "heading_at_origin": ctrv_dynamics.wrap_angles(
            heading_reference + transformed[:, 3]
        ),
        "turn_rate_at_origin": transformed[:, 4],
        "sigma_position_observation": np.hypot(
            np.exp(np.minimum(transformed[:, 5], maximum_log)),
            position_observation_noise_floor_m,
        ),
        "sigma_speed_process": np.exp(np.minimum(transformed[:, 6], maximum_log)),
        "sigma_turn_rate_process": np.exp(np.minimum(transformed[:, 7], maximum_log)),
    }


def sample_gaussian_carry(
    carry: GaussianCarry,
    *,
    draw_count: int,
    generator: np.random.Generator,
    position_observation_noise_floor_m: float = 0.0,
) -> dict[str, np.ndarray]:
    """Sample physical state and parameter draws from a Gaussian carry."""
    if not isinstance(carry, GaussianCarry):
        raise TypeError("carry must be a GaussianCarry.")
    if (
        isinstance(draw_count, bool)
        or not isinstance(draw_count, (int, np.integer))
        or draw_count < 1
    ):
        raise ValueError("draw_count must be a positive integer.")
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be a numpy.random.Generator.")
    transformed = (
        carry.mean
        + generator.normal(size=(int(draw_count), CARRY_SIZE)) @ carry.cholesky.T
    )
    return _physical_draw_mapping(
        transformed,
        carry.heading_reference,
        position_observation_noise_floor_m=position_observation_noise_floor_m,
    )


def _finite_scalar(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}scalar.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}scalar.") from error
    if not np.isfinite(numeric) or (positive and numeric <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}scalar.")
    return numeric


def build_sequential_update_data(
    carry: GaussianCarry,
    *,
    process_interval_seconds: float,
    observation_interval_seconds: float,
    x_observed: float,
    y_observed: float,
    position_observation_noise_floor_m: float = 0.0,
) -> dict[str, object]:
    """Build scalar Stan data for one position-observation update."""
    if not isinstance(carry, GaussianCarry):
        raise TypeError("carry must be a GaussianCarry.")
    return {
        "carry_mean": carry.mean.copy(),
        "carry_cholesky": carry.cholesky.copy(),
        "heading_reference": carry.heading_reference,
        "process_interval_seconds": _finite_scalar(
            process_interval_seconds,
            name="process_interval_seconds",
            positive=True,
        ),
        "observation_interval_seconds": _finite_scalar(
            observation_interval_seconds,
            name="observation_interval_seconds",
            positive=True,
        ),
        "x_observed": _finite_scalar(x_observed, name="x_observed"),
        "y_observed": _finite_scalar(y_observed, name="y_observed"),
        "process_reference_interval_seconds": (
            ctrv_dynamics.PROCESS_REFERENCE_INTERVAL_SECONDS
        ),
        "minimum_positive_scale": MINIMUM_POSITIVE_SCALE,
        "sigma_position_observation_floor_m": (
            numeric_validation.validate_non_negative_finite(
                "position_observation_noise_floor_m",
                position_observation_noise_floor_m,
            )
        ),
    }


def compile_sequential_bayesian_ctrv_model(
    stan_file: str | Path = SEQUENTIAL_STAN_FILE,
) -> CmdStanModel:
    """Compile and return the one-observation Sequential VI model."""
    stan_path = Path(stan_file)
    if not stan_path.is_file():
        raise FileNotFoundError(f"Stan model not found: {stan_path}")
    return CmdStanModel(stan_file=str(stan_path))


def _sequential_initial_values(carry: GaussianCarry) -> dict[str, Any]:
    return {
        "previous_standardized": np.zeros(CARRY_SIZE),
        "speed_for_interval": float(np.exp(carry.mean[2])),
        "turn_rate_for_interval": float(carry.mean[4]),
    }


def fit_sequential_update(
    carry: GaussianCarry,
    *,
    process_interval_seconds: float,
    observation_interval_seconds: float,
    x_observed: float,
    y_observed: float,
    config: SequentialVIConfig,
    seed: int,
    model: CmdStanModel | None = None,
    position_observation_noise_floor_m: float = 0.0,
):
    """Fit the one-observation posterior update with full-rank ADVI."""
    if not isinstance(config, SequentialVIConfig):
        raise TypeError("config must be a SequentialVIConfig.")
    if model is None:
        model = compile_sequential_bayesian_ctrv_model()
    data = build_sequential_update_data(
        carry,
        process_interval_seconds=process_interval_seconds,
        observation_interval_seconds=observation_interval_seconds,
        x_observed=x_observed,
        y_observed=y_observed,
        position_observation_noise_floor_m=position_observation_noise_floor_m,
    )
    return cmdstan.run_variational_inference(
        model,
        data,
        algorithm="fullrank",
        iter=config.iter,
        grad_samples=config.grad_samples,
        elbo_samples=config.elbo_samples,
        eta=config.eta,
        adapt_iter=config.adapt_iter,
        tol_rel_obj=config.tol_rel_obj,
        eval_elbo=config.eval_elbo,
        draws=config.draws,
        seed=seed,
        inits=_sequential_initial_values(carry),
        default_inits_factory=None,
        require_converged=config.require_converged,
        show_console=config.show_console,
    )


def summarize_draws(
    draws: np.ndarray,
    *,
    circular: bool = False,
) -> DistributionSummary:
    """Summarize scalar draws, optionally in a local circular coordinate."""
    vector = _frozen_vector(draws, name="draws")
    if not isinstance(circular, bool):
        raise ValueError("circular must be a boolean.")
    if circular:
        reference = float(np.arctan2(np.mean(np.sin(vector)), np.mean(np.cos(vector))))
        summarized = reference + ctrv_dynamics.wrap_angles(vector - reference)
        mean = reference
    else:
        summarized = vector
        mean = float(np.mean(vector))
    lower, median, upper = np.quantile(summarized, (0.025, 0.5, 0.975))
    return DistributionSummary(
        mean=mean,
        median=float(median),
        standard_deviation=float(np.std(summarized, ddof=1)),
        lower_95=float(lower),
        upper_95=float(upper),
    )


def _state_summary(draws: dict[str, np.ndarray]) -> SequentialVIStateSummary:
    return SequentialVIStateSummary(
        x=summarize_draws(draws["x_at_origin"]),
        y=summarize_draws(draws["y_at_origin"]),
        speed=summarize_draws(draws["speed_at_origin"]),
        heading=summarize_draws(draws["heading_at_origin"], circular=True),
        turn_rate=summarize_draws(draws["turn_rate_at_origin"]),
    )


def _parameter_summary(
    draws: dict[str, np.ndarray],
) -> SequentialVIParameterSummary:
    return SequentialVIParameterSummary(
        sigma_position_observation=summarize_draws(draws["sigma_position_observation"]),
        sigma_speed_process=summarize_draws(draws["sigma_speed_process"]),
        sigma_turn_rate_process=summarize_draws(draws["sigma_turn_rate_process"]),
    )


def _physical_fit_draws(fit: object) -> dict[str, np.ndarray]:
    return {
        "x_at_origin": _posterior_origin_position(fit, "x"),
        "y_at_origin": _posterior_origin_position(fit, "y"),
        "speed_at_origin": _posterior_vector(fit, "speed_at_origin"),
        "heading_at_origin": _posterior_vector(fit, "heading_at_origin"),
        "turn_rate_at_origin": _posterior_vector(fit, "turn_rate_at_origin"),
        "sigma_position_observation": _positive_posterior_vector(
            fit, "sigma_position_observation"
        ),
        "sigma_speed_process": _positive_posterior_vector(fit, "sigma_speed_process"),
        "sigma_turn_rate_process": _positive_posterior_vector(
            fit, "sigma_turn_rate_process"
        ),
    }


def _variational_fit_converged(fit: object) -> bool:
    explicit = getattr(fit, "converged", None)
    if isinstance(explicit, bool):
        return explicit
    return cmdstan.variational_converged(fit)


def _forecast_from_origin_draws(
    draws: dict[str, np.ndarray],
    *,
    origin_time_seconds: float,
    future_time_seconds,
    generator: np.random.Generator,
) -> particle_utils.SequentialCTRVFit:
    future_times = particle_utils.validate_future_times(
        future_time_seconds,
        after=origin_time_seconds,
    )
    draw_count = draws["speed_at_origin"].size
    states = np.column_stack(
        (
            draws["x_at_origin"],
            draws["y_at_origin"],
            draws["speed_at_origin"],
            draws["heading_at_origin"],
            draws["turn_rate_at_origin"],
        )
    )
    speed_at_origin = states[:, ctrv_dynamics.STATE_SPEED_INDEX].copy()
    heading_at_origin = states[:, ctrv_dynamics.STATE_HEADING_INDEX].copy()
    turn_rate_at_origin = states[:, ctrv_dynamics.STATE_TURN_RATE_INDEX].copy()
    prediction_count = future_times.size
    x_prediction = np.empty((draw_count, prediction_count))
    y_prediction = np.empty_like(x_prediction)
    x_observation_prediction = np.empty_like(x_prediction)
    y_observation_prediction = np.empty_like(x_prediction)
    current_time = origin_time_seconds
    for prediction_index, prediction_time in enumerate(future_times):
        interval = float(prediction_time - current_time)
        states = ctrv_dynamics.transition_states(states, interval)
        x_prediction[:, prediction_index] = states[:, ctrv_dynamics.STATE_X_INDEX]
        y_prediction[:, prediction_index] = states[:, ctrv_dynamics.STATE_Y_INDEX]
        observation_innovation = generator.normal(
            0.0,
            draws["sigma_position_observation"][:, None],
            size=(draw_count, 2),
        )
        x_observation_prediction[:, prediction_index] = (
            states[:, ctrv_dynamics.STATE_X_INDEX] + observation_innovation[:, 0]
        )
        y_observation_prediction[:, prediction_index] = (
            states[:, ctrv_dynamics.STATE_Y_INDEX] + observation_innovation[:, 1]
        )
        process_scale = ctrv_dynamics.process_time_scale(interval)
        states[:, ctrv_dynamics.STATE_SPEED_INDEX] += generator.normal(
            0.0,
            draws["sigma_speed_process"] * process_scale,
        )
        states[:, ctrv_dynamics.STATE_TURN_RATE_INDEX] += generator.normal(
            0.0,
            draws["sigma_turn_rate_process"] * process_scale,
        )
        ctrv_dynamics.normalize_states(states)
        current_time = float(prediction_time)

    return particle_utils.SequentialCTRVFit(
        {
            "speed_at_origin": speed_at_origin,
            "heading_at_origin": heading_at_origin,
            "turn_rate_at_origin": turn_rate_at_origin,
            "sigma_position_observation": draws["sigma_position_observation"],
            "sigma_speed_process": draws["sigma_speed_process"],
            "sigma_turn_rate_process": draws["sigma_turn_rate_process"],
            "x_prediction": x_prediction,
            "y_prediction": y_prediction,
            "x_observation_prediction": x_observation_prediction,
            "y_observation_prediction": y_observation_prediction,
        }
    )


def forecast_ctrv_prior_predictive(
    *,
    x_at_origin: float,
    y_at_origin: float,
    origin_time_seconds: float,
    future_time_seconds,
    priors: bayesian_model.BayesianCTRVPriors,
    draw_count: int,
    seed: int,
) -> particle_utils.SequentialCTRVFit:
    """Draw a CTRV forecast from the configured priors at one observed origin."""
    if not isinstance(priors, bayesian_model.BayesianCTRVPriors):
        raise TypeError("priors must be a BayesianCTRVPriors instance.")
    if (
        isinstance(draw_count, bool)
        or not isinstance(draw_count, (int, np.integer))
        or draw_count < 2
    ):
        raise ValueError("draw_count must be an integer greater than or equal to 2.")
    seed = numeric_validation.validate_non_negative_integer("seed", seed)
    x_origin = _finite_scalar(x_at_origin, name="x_at_origin")
    y_origin = _finite_scalar(y_at_origin, name="y_at_origin")
    origin_time = _finite_scalar(origin_time_seconds, name="origin_time_seconds")
    generator = np.random.default_rng(seed)
    draw_count = int(draw_count)
    draws = {
        "x_at_origin": np.full(draw_count, x_origin),
        "y_at_origin": np.full(draw_count, y_origin),
        "speed_at_origin": np.abs(
            generator.normal(0.0, priors.speed_prior_scale, draw_count)
        ),
        "heading_at_origin": generator.uniform(-np.pi, np.pi, draw_count),
        "turn_rate_at_origin": generator.normal(
            0.0,
            priors.turn_rate_prior_scale,
            draw_count,
        ),
        "sigma_position_observation": np.hypot(
            generator.exponential(
                1.0 / priors.sigma_position_observation_prior_rate,
                draw_count,
            ),
            priors.sigma_position_observation_floor_m,
        ),
        "sigma_speed_process": generator.exponential(
            1.0 / priors.sigma_speed_process_prior_rate,
            draw_count,
        ),
        "sigma_turn_rate_process": generator.exponential(
            1.0 / priors.sigma_turn_rate_process_prior_rate,
            draw_count,
        ),
    }
    return _forecast_from_origin_draws(
        draws,
        origin_time_seconds=origin_time,
        future_time_seconds=future_time_seconds,
        generator=generator,
    )


@dataclass(slots=True)
class SequentialCTRVVI:
    """Persistent one-observation-at-a-time variational CTRV approximation."""

    config: SequentialVIConfig
    priors: bayesian_model.BayesianCTRVPriors
    carry: GaussianCarry
    last_observation_time_seconds: float
    last_process_interval_seconds: float
    processed_observation_count: int
    bootstrap_runtime_seconds: float
    bootstrap_converged: bool
    bootstrap_draw_count: int
    _next_seed: int
    _update_fitter: Callable[..., object] = field(repr=False)
    _model: CmdStanModel | None = field(default=None, repr=False)
    _updates: list[SequentialVIUpdate] = field(default_factory=list, repr=False)
    last_log_predictive_density: float | None = None
    predictive_log_density_history: list[float] = field(default_factory=list)
    track_lost: bool | None = None
    last_observed_position: np.ndarray | None = None
    reinitialization_count: int = 0

    @classmethod
    def initialize(
        cls,
        time_seconds,
        x_observed,
        y_observed,
        *,
        priors: bayesian_model.BayesianCTRVPriors,
        config: SequentialVIConfig | None = None,
        seed: int = 42,
        batch_fitter: Callable[..., object] | None = None,
        update_fitter: Callable[..., object] | None = None,
        model: CmdStanModel | None = None,
    ) -> SequentialCTRVVI:
        """Bootstrap once, then assimilate only observations after bootstrap."""
        if not isinstance(priors, bayesian_model.BayesianCTRVPriors):
            raise TypeError("priors must be a BayesianCTRVPriors instance.")
        if config is None:
            config = SequentialVIConfig()
        if not isinstance(config, SequentialVIConfig):
            raise TypeError("config must be a SequentialVIConfig instance or None.")
        seed = numeric_validation.validate_non_negative_integer("seed", seed)
        time_values, x_values, y_values = (
            particle_utils.validate_sequential_observations(
                time_seconds,
                x_observed,
                y_observed,
                minimum_count=config.n_bootstrap,
            )
        )
        if batch_fitter is None:
            batch_fitter = batch_inference.fit_bayesian_ctrv_model
        if update_fitter is None:
            update_fitter = fit_sequential_update

        bootstrap_count = config.n_bootstrap
        bootstrap_times = time_values[:bootstrap_count]
        last_interval = float(bootstrap_times[-1] - bootstrap_times[-2])
        synthetic_time = float(bootstrap_times[-1] + last_interval)
        window_times = np.concatenate((bootstrap_times, [synthetic_time]))
        window_x = np.concatenate(
            (x_values[:bootstrap_count], [x_values[bootstrap_count - 1]])
        )
        window_y = np.concatenate(
            (y_values[:bootstrap_count], [y_values[bootstrap_count - 1]])
        )
        window = observation_window.TrajectoryWindowData(
            timestamps=pd.DatetimeIndex(
                pd.to_datetime(window_times, unit="s", origin="unix", utc=True)
            ),
            time_seconds=window_times,
            x_meters=window_x,
            y_meters=window_y,
            reference_longitude=0.0,
            reference_latitude=0.0,
            gps_speed_mps=np.full(window_times.shape, np.nan),
            observation_count=bootstrap_count,
        )
        position_observations = observation_support.PositionObservations(
            time_seconds=bootstrap_times,
            x_meters=x_values[:bootstrap_count],
            y_meters=y_values[:bootstrap_count],
            position_noise_std_m=0.0,
            noise_seed=seed,
        )
        started = perf_counter()
        bootstrap_fit = batch_fitter(
            window,
            priors=priors,
            position_observations=position_observations,
            inference_method="vi",
            algorithm="fullrank",
            iter=config.iter,
            grad_samples=config.grad_samples,
            elbo_samples=config.elbo_samples,
            eta=config.eta,
            adapt_iter=config.adapt_iter,
            tol_rel_obj=config.tol_rel_obj,
            eval_elbo=config.eval_elbo,
            draws=config.draws,
            seed=seed,
            require_converged=config.require_converged,
            show_console=config.show_console,
        )
        bootstrap_runtime = perf_counter() - started
        carry = build_gaussian_carry(
            bootstrap_fit,
            covariance_jitter=config.covariance_jitter,
            regularization_attempts=config.regularization_attempts,
            position_observation_noise_floor_m=(
                priors.sigma_position_observation_floor_m
            ),
        )
        instance = cls(
            config=config,
            priors=priors,
            carry=carry,
            last_observation_time_seconds=float(bootstrap_times[-1]),
            last_process_interval_seconds=last_interval,
            processed_observation_count=bootstrap_count,
            bootstrap_runtime_seconds=bootstrap_runtime,
            bootstrap_converged=_variational_fit_converged(bootstrap_fit),
            bootstrap_draw_count=carry.draw_count,
            _next_seed=seed + 1,
            _update_fitter=update_fitter,
            _model=model,
            last_observed_position=np.array(
                [x_values[bootstrap_count - 1], y_values[bootstrap_count - 1]],
                dtype=float,
            ),
        )
        instance.update_many(
            time_values[bootstrap_count:],
            x_values[bootstrap_count:],
            y_values[bootstrap_count:],
        )
        return instance

    @property
    def updates(self) -> tuple[SequentialVIUpdate, ...]:
        """Return immutable records for successful updates."""
        return tuple(self._updates)

    def _predictive_summary(
        self,
        observation_interval_seconds: float,
        *,
        seed: int,
    ) -> tuple[
        SequentialVIStateSummary,
        SequentialVIParameterSummary,
        SequentialVIPredictiveDraws,
    ]:
        generator = np.random.default_rng(seed)
        draws = sample_gaussian_carry(
            self.carry,
            draw_count=self.config.draws,
            generator=generator,
            position_observation_noise_floor_m=(
                self.priors.sigma_position_observation_floor_m
            ),
        )
        process_scale = ctrv_dynamics.process_time_scale(
            self.last_process_interval_seconds
        )
        speed = np.abs(
            generator.normal(
                draws["speed_at_origin"],
                draws["sigma_speed_process"] * process_scale,
            )
        )
        turn_rate = generator.normal(
            draws["turn_rate_at_origin"],
            draws["sigma_turn_rate_process"] * process_scale,
        )
        states = np.column_stack(
            (
                draws["x_at_origin"],
                draws["y_at_origin"],
                speed,
                draws["heading_at_origin"],
                turn_rate,
            )
        )
        states = ctrv_dynamics.transition_states(
            states,
            observation_interval_seconds,
        )
        predictive_mapping = {
            **draws,
            "x_at_origin": states[:, ctrv_dynamics.STATE_X_INDEX],
            "y_at_origin": states[:, ctrv_dynamics.STATE_Y_INDEX],
            "speed_at_origin": states[:, ctrv_dynamics.STATE_SPEED_INDEX],
            "heading_at_origin": states[:, ctrv_dynamics.STATE_HEADING_INDEX],
            "turn_rate_at_origin": states[:, ctrv_dynamics.STATE_TURN_RATE_INDEX],
        }
        predictive_draws = SequentialVIPredictiveDraws(
            x=predictive_mapping["x_at_origin"],
            y=predictive_mapping["y_at_origin"],
            sigma_position_observation=predictive_mapping["sigma_position_observation"],
        )
        return (
            _state_summary(predictive_mapping),
            _parameter_summary(predictive_mapping),
            predictive_draws,
        )

    def update(
        self,
        time_seconds: float,
        x_observed: float,
        y_observed: float,
    ) -> SequentialVIUpdate:
        """Assimilate one new position atomically."""
        time_value = _finite_scalar(time_seconds, name="time_seconds")
        if time_value <= self.last_observation_time_seconds:
            raise ValueError("time_seconds must follow the previous observation.")
        x_value = _finite_scalar(x_observed, name="x_observed")
        y_value = _finite_scalar(y_observed, name="y_observed")
        observation_interval = time_value - self.last_observation_time_seconds
        started = perf_counter()
        predictive_state, predictive_parameters, predictive_draws = (
            self._predictive_summary(observation_interval, seed=self._next_seed)
        )
        log_predictive_density = _predictive_log_density(
            predictive_draws,
            x_observed=x_value,
            y_observed=y_value,
        )
        self.last_log_predictive_density = log_predictive_density
        self.predictive_log_density_history.append(log_predictive_density)
        threshold = self.config.predictive_log_density_threshold
        self.track_lost = (
            None
            if threshold is None
            else bool(log_predictive_density < threshold)
        )
        if self.track_lost:
            next_carry = self._reinitialize_carry_at_observation(
                x_observed=x_value,
                y_observed=y_value,
                observation_interval_seconds=observation_interval,
            )
            filtered_draws = sample_gaussian_carry(
                next_carry,
                draw_count=self.config.draws,
                generator=np.random.default_rng(self._next_seed),
                position_observation_noise_floor_m=(
                    self.priors.sigma_position_observation_floor_m
                ),
            )
            filtered_state = _state_summary(filtered_draws)
            filtered_parameters = _parameter_summary(filtered_draws)
            converged = False
            self.reinitialization_count += 1
        else:
            fitted = self._update_fitter(
                self.carry,
                process_interval_seconds=self.last_process_interval_seconds,
                observation_interval_seconds=observation_interval,
                x_observed=x_value,
                y_observed=y_value,
                config=self.config,
                seed=self._next_seed,
                model=self._model,
                position_observation_noise_floor_m=(
                    self.priors.sigma_position_observation_floor_m
                ),
            )
            next_carry = build_gaussian_carry(
                fitted,
                covariance_jitter=self.config.covariance_jitter,
                regularization_attempts=self.config.regularization_attempts,
                position_observation_noise_floor_m=(
                    self.priors.sigma_position_observation_floor_m
                ),
            )
            filtered_draws = _physical_fit_draws(fitted)
            filtered_state = _state_summary(filtered_draws)
            filtered_parameters = _parameter_summary(filtered_draws)
            converged = _variational_fit_converged(fitted)
        runtime = perf_counter() - started
        cumulative_runtime = (
            self.bootstrap_runtime_seconds
            + sum(update.runtime_seconds for update in self._updates)
            + runtime
        )
        record = SequentialVIUpdate(
            observation_index=self.processed_observation_count,
            time_seconds=time_value,
            x_observed=x_value,
            y_observed=y_value,
            predictive_state=predictive_state,
            filtered_state=filtered_state,
            predictive_parameters=predictive_parameters,
            filtered_parameters=filtered_parameters,
            predictive_draws=predictive_draws,
            runtime_seconds=runtime,
            cumulative_runtime_seconds=cumulative_runtime,
            converged=converged,
        )

        self.carry = next_carry
        self.last_observation_time_seconds = time_value
        self.last_process_interval_seconds = observation_interval
        self.processed_observation_count += 1
        self._next_seed += 1
        self._updates.append(record)
        self.last_observed_position = np.array([x_value, y_value], dtype=float)
        return record

    def _reinitialize_carry_at_observation(
        self,
        *,
        x_observed: float,
        y_observed: float,
        observation_interval_seconds: float,
    ) -> GaussianCarry:
        """Construct a new variational carry anchored at a lost-track fix."""
        generator = np.random.default_rng(self._next_seed)
        source_draws = sample_gaussian_carry(
            self.carry,
            draw_count=self.config.draws,
            generator=generator,
            position_observation_noise_floor_m=(
                self.priors.sigma_position_observation_floor_m
            ),
        )
        observation_noise = source_draws["sigma_position_observation"]
        previous_position = self.last_observed_position
        if previous_position is None:
            displacement = np.zeros(2, dtype=float)
        else:
            previous_position = np.asarray(previous_position, dtype=float)
            if previous_position.shape != (2,) or not np.all(
                np.isfinite(previous_position)
            ):
                raise RuntimeError("Sequential VI last observation became invalid.")
            displacement = np.array([x_observed, y_observed]) - previous_position
        distance = float(np.hypot(*displacement))
        speed_scale = np.sqrt(2.0) * observation_noise / observation_interval_seconds
        heading_observable = distance > 2.0 * float(np.median(observation_noise))
        if heading_observable:
            speed = np.maximum(
                np.abs(generator.normal(distance / observation_interval_seconds, speed_scale)),
                MINIMUM_POSITIVE_SCALE,
            )
            heading_reference = float(np.arctan2(displacement[1], displacement[0]))
            heading_local = generator.normal(
                0.0,
                np.minimum(
                    np.pi,
                    speed_scale / (distance / observation_interval_seconds),
                ),
            )
        else:
            speed = np.maximum(
                np.abs(generator.normal(0.0, speed_scale)),
                MINIMUM_POSITIVE_SCALE,
            )
            heading_reference = 0.0
            heading_local = generator.normal(0.0, np.pi, self.config.draws)
        observation_noise_excess = np.sqrt(
            np.maximum(
                observation_noise**2
                - self.priors.sigma_position_observation_floor_m**2,
                MINIMUM_POSITIVE_SCALE**2,
            )
        )
        transformed = np.column_stack(
            (
                generator.normal(x_observed, observation_noise),
                generator.normal(y_observed, observation_noise),
                np.log(speed),
                heading_local,
                generator.normal(
                    0.0,
                    source_draws["sigma_turn_rate_process"]
                    * ctrv_dynamics.process_time_scale(observation_interval_seconds),
                ),
                np.log(observation_noise_excess),
                np.log(source_draws["sigma_speed_process"]),
                np.log(source_draws["sigma_turn_rate_process"]),
            )
        )
        covariance = np.cov(transformed, rowvar=False, ddof=1)
        mean = np.mean(transformed, axis=0)
        mean[0] = x_observed
        mean[1] = y_observed
        return GaussianCarry(
            mean=mean,
            cholesky=_regularized_cholesky(
                covariance,
                covariance_jitter=self.config.covariance_jitter,
                regularization_attempts=self.config.regularization_attempts,
            ),
            heading_reference=heading_reference,
            draw_count=self.config.draws,
        )

    def update_many(self, time_seconds, x_observed, y_observed) -> None:
        """Assimilate matching new observations exactly once."""
        time_values, x_values, y_values = (
            particle_utils.validate_sequential_observations(
                time_seconds,
                x_observed,
                y_observed,
                minimum_count=0,
            )
        )
        if time_values.size and time_values[0] <= self.last_observation_time_seconds:
            raise ValueError(
                "Sequential timestamps must follow processed observations."
            )
        for time_value, x_value, y_value in zip(
            time_values,
            x_values,
            y_values,
            strict=True,
        ):
            self.update(float(time_value), float(x_value), float(y_value))

    def sample_current_posterior(
        self,
        *,
        seed: int,
    ) -> particle_utils.SequentialCTRVFit:
        """Sample the latest carried state through the shared fit interface."""
        seed = numeric_validation.validate_non_negative_integer("seed", seed)
        draws = sample_gaussian_carry(
            self.carry,
            draw_count=self.config.draws,
            generator=np.random.default_rng(seed),
            position_observation_noise_floor_m=(
                self.priors.sigma_position_observation_floor_m
            ),
        )
        return particle_utils.SequentialCTRVFit(
            {name: draws[name] for name in bayesian_model.PARAMETER_NAMES}
        )

    def forecast(
        self,
        future_time_seconds,
        *,
        seed: int,
    ) -> particle_utils.SequentialCTRVFit:
        """Draw future latent trajectories and noisy position observations."""
        seed = numeric_validation.validate_non_negative_integer("seed", seed)
        generator = np.random.default_rng(seed)
        draws = sample_gaussian_carry(
            self.carry,
            draw_count=self.config.draws,
            generator=generator,
            position_observation_noise_floor_m=(
                self.priors.sigma_position_observation_floor_m
            ),
        )
        return _forecast_from_origin_draws(
            draws,
            origin_time_seconds=self.last_observation_time_seconds,
            future_time_seconds=future_time_seconds,
            generator=generator,
        )

    def fit(self, time_seconds, x_observed, y_observed) -> SequentialVIResult:
        """Assimilate a sequence and return a partial result on terminal failure."""
        time_values, x_values, y_values = (
            particle_utils.validate_sequential_observations(
                time_seconds,
                x_observed,
                y_observed,
                minimum_count=0,
            )
        )
        if time_values.size and time_values[0] <= self.last_observation_time_seconds:
            raise ValueError(
                "Sequential timestamps must follow processed observations."
            )
        failed_record = None
        failure_index = None
        failure_message = None
        for time_value, x_value, y_value in zip(
            time_values,
            x_values,
            y_values,
            strict=True,
        ):
            attempted_index = self.processed_observation_count
            attempted_interval = float(time_value - self.last_observation_time_seconds)
            started = perf_counter()
            try:
                self.update(float(time_value), float(x_value), float(y_value))
            except Exception as error:
                runtime = perf_counter() - started
                predictive_state, predictive_parameters, predictive_draws = (
                    self._predictive_summary(
                        attempted_interval,
                        seed=self._next_seed,
                    )
                )
                current_draws = sample_gaussian_carry(
                    self.carry,
                    draw_count=self.config.draws,
                    generator=np.random.default_rng(self._next_seed),
                    position_observation_noise_floor_m=(
                        self.priors.sigma_position_observation_floor_m
                    ),
                )
                failure_message = str(error)
                failure_index = attempted_index
                failed_record = SequentialVIUpdate(
                    observation_index=attempted_index,
                    time_seconds=float(time_value),
                    x_observed=float(x_value),
                    y_observed=float(y_value),
                    predictive_state=predictive_state,
                    filtered_state=_state_summary(current_draws),
                    predictive_parameters=predictive_parameters,
                    filtered_parameters=_parameter_summary(current_draws),
                    predictive_draws=predictive_draws,
                    runtime_seconds=runtime,
                    cumulative_runtime_seconds=(
                        self.bootstrap_runtime_seconds
                        + sum(update.runtime_seconds for update in self._updates)
                        + runtime
                    ),
                    converged=False,
                    error_message=failure_message,
                )
                break
        records = self.updates
        if failed_record is not None:
            records = (*records, failed_record)
        return SequentialVIResult(
            bootstrap_runtime_seconds=self.bootstrap_runtime_seconds,
            bootstrap_converged=self.bootstrap_converged,
            bootstrap_draw_count=self.bootstrap_draw_count,
            updates=records,
            processed_observation_count=self.processed_observation_count,
            completed=failed_record is None,
            failure_index=failure_index,
            failure_message=failure_message,
        )
