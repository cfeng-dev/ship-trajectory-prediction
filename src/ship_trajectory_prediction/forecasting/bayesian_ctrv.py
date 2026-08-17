"""Experiment configuration for Bayesian CTRV forecasting."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration of one recorded-trajectory single-window experiment."""

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
    """Configuration of one recorded-trajectory rolling experiment."""

    run_id: int
    window_mode: str
    observation_count: int
    prediction_count: int
    additional_position_noise_std_m: float
    position_noise_seed: int
    stride: int | None
    inference_method: str
    inference_seed: int
