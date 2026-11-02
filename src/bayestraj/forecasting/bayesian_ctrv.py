"""Configuration for the parametric Bayesian CTRV model."""

from dataclasses import dataclass

import bayestraj.inference.configuration as inference


@dataclass(frozen=True, slots=True)
class RollingExperimentConfig:
    """Configuration of one rolling parametric CTRV experiment.

    Attributes:
        observation_count: Maximum past positions (raw radar cycles for BR24) per VI/MCMC fit. The window grows until this size is reached. Ignored by online SMC and Sequential VI; evaluation starts at N=2.

        prediction_count: Number of future positions forecast per window.

        position_noise_std_m: Gaussian position-noise standard deviation per local x/y coordinate in metres; 0 disables artificial noise.

        position_noise_seed: Random seed for reproducible position noise.

        stride: Number of newly observed positions between forecast origins.

        inference_method: ``vi``, ``mcmc``, ``smc``, or ``seq_vi`` (Sequential VI). VI and MCMC use a fixed sliding window.

        inference_seed: Random seed for reproducible inference.
    """

    prediction_count: int
    position_noise_std_m: float
    position_noise_seed: int
    stride: int | None
    inference_method: str
    inference_seed: int
    observation_count: int = inference.DEFAULT_CTRV_ROLLING_OBSERVATION_COUNT

    def __post_init__(self) -> None:
        """Validate the combined rolling inference selection."""
        _, inference_method, _ = inference.normalize_ctrv_rolling_inference_method(
            self.inference_method
        )
        object.__setattr__(self, "inference_method", inference_method)
