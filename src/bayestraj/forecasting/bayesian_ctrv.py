"""Configuration for the parametric Bayesian CTRV model."""

from dataclasses import dataclass

import bayestraj.forecasting.inference as inference


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration of one single-window parametric CTRV experiment."""

    run_id: int
    start_index: int
    observation_count: int
    prediction_count: int
    position_noise_std_m: float
    position_noise_seed: int
    inference_method: str
    inference_seed: int

    def __post_init__(self) -> None:
        """Validate batch inference or a full-state online particle filter."""
        _, inference_method = inference.normalize_inference_method(
            self.inference_method,
            online_inference_methods=inference.CTRV_ONLINE_INFERENCE_METHODS,
        )
        object.__setattr__(self, "inference_method", inference_method)


@dataclass(frozen=True, slots=True)
class RollingExperimentConfig:
    """Configuration of one rolling parametric CTRV experiment."""

    run_id: int
    observation_count: int
    prediction_count: int
    position_noise_std_m: float
    position_noise_seed: int
    stride: int | None
    inference_method: str
    inference_seed: int

    def __post_init__(self) -> None:
        """Validate the combined rolling inference selection."""
        _, inference_method, window_mode = (
            inference.normalize_ctrv_rolling_inference_method(self.inference_method)
        )
        normalized_selection = (
            inference_method
            if window_mode is None
            else f"{inference_method}_{window_mode}"
        )
        object.__setattr__(self, "inference_method", normalized_selection)
