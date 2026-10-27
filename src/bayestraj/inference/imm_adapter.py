"""Read-only IMM posterior adapters shared by evaluation and desktop tools.

The optional local tracker keeps its filter equations and state ownership. This
module only exposes model moments and coordinatewise mixture medians; it has no
application imports, GUI dependencies, or application-specific file defaults.
"""

from __future__ import annotations

import runpy
from dataclasses import dataclass
from math import fsum
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr


@dataclass(frozen=True, slots=True)
class IMMStatePosterior:
    """Read-only [x, y, heading, turn rate, speed] Gaussian state moments."""

    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float).copy()
        covariance = np.asarray(self.covariance, dtype=float).copy()
        if mean.shape != (5,) or not np.all(np.isfinite(mean)):
            raise ValueError("IMM posterior mean must be a finite five-element vector.")
        if covariance.shape != (5, 5) or not np.all(np.isfinite(covariance)):
            raise ValueError("IMM posterior covariance must be a finite 5-by-5 matrix.")
        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(covariance))))
        if not np.allclose(covariance, covariance.T, rtol=0, atol=tolerance):
            raise ValueError("IMM posterior covariance must be symmetric.")
        if np.min(np.linalg.eigvalsh(covariance)) < -tolerance:
            raise ValueError("IMM posterior covariance must be positive semidefinite.")
        mean.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True, slots=True)
class IMMStateMixture:
    """Cached CV/CTRV/CA moments in the MATLAB common CTRV coordinates."""

    weights: np.ndarray
    components: tuple[IMMStatePosterior, ...]

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float).copy()
        components = tuple(self.components)
        if (
            weights.shape != (3,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-10)
        ):
            raise ValueError(
                "IMM mixture needs three non-negative probabilities summing to one."
            )
        if len(components) != 3 or not all(
            isinstance(component, IMMStatePosterior) for component in components
        ):
            raise ValueError("IMM mixture needs CV, CTRV and CA state moments.")
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "components", components)

    def marginal_parameters(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return active weights, means and standard deviations in native SI units."""
        active = self.weights > 0
        weights = self.weights[active]
        means = np.array([component.mean[index] for component in self.components])[
            active
        ]
        sigmas = np.sqrt(
            np.maximum(
                [component.covariance[index, index] for component in self.components], 0
            )
        )[active]
        if index == 2:
            # Keep the same local heading branch for densities and summaries.
            center = np.angle(np.sum(weights * np.exp(1j * means)))
            means = center + (means - center + np.pi) % (2 * np.pi) - np.pi
        return weights, means, sigmas

    def marginal_median(self, index: int) -> float:
        """Return the lower 0.5 quantile, retaining discrete probability jumps."""
        weights, means, sigmas = self.marginal_parameters(index)
        continuous = sigmas > 0

        def balance(value: float, *, before_atom=False) -> float:
            # Sign of F(value)-0.5, scaled in log space so tiny Gaussian tails
            # neither cancel against 0.5 nor underflow between distant modes.
            left = means <= value
            if before_atom:
                left &= continuous | (means != value)
            base = float(np.sum(weights[left])) - 0.5
            log_terms = []
            signs = []
            if base:
                log_terms.append(float(np.log(abs(base))))
                signs.append(float(np.sign(base)))
            for weight, mean, sigma, on_left in zip(
                weights[continuous],
                means[continuous],
                sigmas[continuous],
                left[continuous],
                strict=True,
            ):
                log_terms.append(
                    float(np.log(weight) + log_ndtr(-abs(value - mean) / sigma))
                )
                signs.append(-1.0 if on_left else 1.0)
            if not log_terms:
                return 0.0
            scale = max(log_terms)
            if not np.isfinite(scale):
                return 0.0
            return fsum(
                sign * np.exp(term - scale)
                for sign, term in zip(signs, log_terms, strict=True)
            )

        for atom in np.unique(means[~continuous]):
            if balance(float(atom), before_atom=True) <= 0 <= balance(float(atom)):
                return float(atom)
        lower, upper = float(np.min(means)), float(np.max(means))
        if lower == upper:
            return lower
        padding = 8 * float(np.max(sigmas))
        return float(
            brentq(balance, lower - padding, upper + padding, xtol=1e-12, maxiter=200)
        )


def forecast_position_medians(tracker, forecast) -> np.ndarray | None:
    """Read coordinatewise mixture medians independently of display path samples."""
    if forecast is None:
        return None
    reader = getattr(tracker, "forecast_mixtures", None)
    mixtures = reader(forecast.time_seconds) if callable(reader) else None
    if mixtures is None:
        # Older backends expose only a fused Gaussian: its marginal mean = median.
        positions = np.asarray(forecast.position_means, dtype=float).copy()
    else:
        positions = np.array(
            [
                [mixture.marginal_median(index) for index in (0, 1)]
                for mixture in mixtures
            ]
        )
    if positions.shape != np.shape(forecast.position_means) or not np.all(
        np.isfinite(positions)
    ):
        raise ValueError(
            "IMM forecast medians must be finite matching (time, xy) positions."
        )
    positions.setflags(write=False)
    return positions


def load_imm_tracker(imm_python: str | Path):
    """Load an explicit optional backend and expose read-only posterior moments."""
    imm_python = Path(imm_python)
    if not imm_python.is_file():
        raise FileNotFoundError(
            f"Optional IMM implementation not found: {imm_python}. "
            "Restore the local external IMM files to use this tracker."
        )
    module = runpy.run_path(str(imm_python))
    try:
        tracker_type = module["BR24IMMTracker"]
    except KeyError as error:
        raise RuntimeError(
            "The local IMM implementation must expose BR24IMMTracker."
        ) from error
    fuse = module.get("_mix_to_ctrv")
    if not callable(fuse):
        return tracker_type
    to_ctrv = module.get("_to_ctrv")

    def state_mixture(state) -> IMMStateMixture | None:
        if not callable(to_ctrv):
            return None
        return IMMStateMixture(
            state.mode_probabilities,
            (
                IMMStatePosterior(*to_ctrv(state.cv_state, state.cv_covariance)),
                IMMStatePosterior(state.ctrv_state, state.ctrv_covariance),
                IMMStatePosterior(*to_ctrv(state.ca_state, state.ca_covariance)),
            ),
        )

    class DisplayTracker(tracker_type):
        """Expose output moments without modifying the optional filter implementation."""

        def posterior_state(self) -> IMMStatePosterior:
            mean, covariance = fuse(self._state, self._state.mode_probabilities)
            return IMMStatePosterior(mean, covariance)

        def posterior_mixture(self) -> IMMStateMixture | None:
            """Read the same model conversions as output fusion, without collapsing."""
            return state_mixture(self._state)

        def forecast_mixtures(
            self, future_time_seconds
        ) -> tuple[IMMStateMixture, ...] | None:
            """Read model components through the unchanged backend forecast equations."""
            to_world = module.get("_state_in_world_coordinates")
            predict = module.get("_predict_imm")
            state_type = module.get("_IMMState")
            validate_times = module.get("_validate_future_times")
            if not all(
                callable(function)
                for function in (
                    to_ctrv,
                    to_world,
                    predict,
                    state_type,
                    validate_times,
                )
            ):
                return None
            times = validate_times(
                future_time_seconds, after=self._last_observation_time_seconds
            )
            state = to_world(self._state, self._sensor_position)
            current_time = self._last_observation_time_seconds
            mixtures = []
            for future_time in times:
                predictions, _, probabilities = predict(
                    state, float(future_time - current_time), np.zeros(2)
                )
                cv, ctrv, ca = predictions
                state = state_type(*cv, *ctrv, *ca, probabilities)
                mixtures.append(state_mixture(state))
                current_time = float(future_time)
            return tuple(mixtures)

    return DisplayTracker
