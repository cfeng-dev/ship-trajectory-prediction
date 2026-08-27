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
    inference_mode: str
    inference_method: str
    inference_seed: int

    def __post_init__(self) -> None:
        """Require the batch inference supported by single-window fitting."""
        inference_mode, inference_method = inference.normalize_inference_configuration(
            self.inference_mode,
            self.inference_method,
        )
        if inference_mode != "batch":
            raise ValueError("Single-window prediction supports batch inference only.")
        object.__setattr__(self, "inference_mode", inference_mode)
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
    inference_mode: str
    inference_method: str
    inference_seed: int
    window_mode: str | None = None

    def __post_init__(self) -> None:
        """Validate inference selection and its batch-only window mode."""
        inference_mode, inference_method, window_mode = (
            inference.normalize_rolling_inference_configuration(
                self.inference_mode,
                self.inference_method,
                self.window_mode,
            )
        )
        object.__setattr__(self, "inference_mode", inference_mode)
        object.__setattr__(self, "inference_method", inference_method)
        object.__setattr__(self, "window_mode", window_mode)
