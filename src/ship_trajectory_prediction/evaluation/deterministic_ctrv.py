"""Configuration types for deterministic CTRV evaluation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeterministicExperimentConfig:
    """Configuration of one deterministic recorded-trajectory experiment."""

    run_id: int
    start_index: int
    observation_count: int
    prediction_count: int
    additional_position_noise_std_m: float
    position_noise_seed: int
