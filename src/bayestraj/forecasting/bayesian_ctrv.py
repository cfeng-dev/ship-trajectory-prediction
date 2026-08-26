"""Configuration for the parametric Bayesian CTRV model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration of one single-window parametric CTRV experiment."""

    run_id: int
    start_index: int
    observation_count: int
    prediction_count: int
    additional_position_noise_std_m: float
    position_noise_seed: int
    inference_method: str
    inference_seed: int


@dataclass(frozen=True, slots=True)
class RollingExperimentConfig:
    """Configuration of one rolling parametric CTRV experiment."""

    run_id: int
    window_mode: str
    observation_count: int
    prediction_count: int
    additional_position_noise_std_m: float
    position_noise_seed: int
    stride: int | None
    inference_method: str
    inference_seed: int
