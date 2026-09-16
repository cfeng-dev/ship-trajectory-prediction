"""Configuration for the parametric Bayesian CTRV model."""

from dataclasses import dataclass

import bayestraj.inference.configuration as inference


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration of one single-window parametric CTRV experiment.

    Attributes:
        run_id: Identifier of the ship trajectory to evaluate.

        start_index: Index of the first observed position in the selected run.

        observation_count: Number of observed positions used for inference.

        prediction_count: Number of future positions forecast after inference.

        position_noise_std_m: Gaussian position-noise standard deviation per local x/y coordinate in metres; 0 disables artificial noise.

        position_noise_seed: Random seed for reproducible position noise.

        inference_method: ``vi``, ``mcmc``, ``rbpf``, or ``smc``.

        inference_seed: Random seed for reproducible inference.
    """

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
    """Configuration of one rolling parametric CTRV experiment.

    Attributes:
        run_id: Identifier of the ship trajectory to evaluate.

        observation_count: Initial number of observed positions. For a sliding window, this is also the fixed window size.

        prediction_count: Number of future positions forecast per window.

        position_noise_std_m: Gaussian position-noise standard deviation per local x/y coordinate in metres; 0 disables artificial noise.

        position_noise_seed: Random seed for reproducible position noise.

        stride: Number of newly observed positions between forecast origins.

        inference_method: ``vi_sliding``, ``vi_expanding``, ``mcmc_sliding``, ``mcmc_expanding``, ``rbpf``, or ``smc``.

        inference_seed: Random seed for reproducible inference.
    """

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
