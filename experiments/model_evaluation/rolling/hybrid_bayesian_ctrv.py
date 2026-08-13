"""Evaluate hybrid Bayesian CTRV forecasts across rolling windows."""

from ship_trajectory_prediction.evaluation.bayesian_ctrv import (
    RollingExperimentConfig,
)
from ship_trajectory_prediction.models.bayesian_ctrv import (
    DEFAULT_VI_ADAPT_ITER,
    BayesianCTRVPriors,
)
from ship_trajectory_prediction.models.hybrid_bayesian_ctrv import (
    HybridBayesianCTRVConfig,
)
from ship_trajectory_prediction.paths import project_path

if __package__:
    from .bayesian_ctrv import run_cli
else:
    from bayesian_ctrv import run_cli

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

EXPERIMENT = RollingExperimentConfig(
    run_id=1,  # Trajectory run to evaluate.
    window_mode="sliding",  # Fixed "sliding" or growing "expanding" history.
    observation_count=20,  # Position points used by the first fit.
    prediction_count=3,  # Held-out future points per rolling forecast.
    additional_position_noise_std_m=2.0,  # Per x/y axis [m]; 0 disables.
    position_noise_seed=2026,  # Reproduces route-wide added position noise.
    stride=None,  # Forecast-origin step; None uses prediction_count.
    inference_method="vi",  # Fast "vi" or reference "mcmc".
    inference_seed=42,  # Reproduces every rolling VI or MCMC fit.
)
PRIORS = BayesianCTRVPriors(
    position_initial_prior_scale=5.0,  # Initial x/y uncertainty [m].
    # Historical calibration from Run IDs 0-99; keep evaluation runs disjoint.
    speed_initial_prior_mean=3.524,  # Robust initial speed center [m/s].
    speed_initial_prior_scale=0.365,  # Robust initial speed scale [m/s].
    turn_rate_initial_prior_mean=0.0,  # Neutral independent center [rad/s].
    turn_rate_state_prior_scale=0.001698,  # Robust turn-rate scale [rad/s].
    sigma_position_gps_prior_scale=5.0,  # Measurement-noise scale [m].
    sigma_position_process_prior_scale=0.5,  # Position drift [m/sqrt(s)].
    sigma_speed_process_prior_scale=0.05,  # Speed drift [(m/s)/sqrt(s)].
    sigma_turn_rate_process_prior_scale=0.001,  # Turn drift [(rad/s)/sqrt(s)].
)
HYBRID_CONFIG = HybridBayesianCTRVConfig(
    final_motion_history_seconds=60.0,  # Recent positions used for endpoint motion.
    min_final_motion_speed_mps=1.0,  # Below this, use neutral turn rate [m/s].
)
VI_CONFIG = {
    "algorithm": "meanfield",  # "meanfield" or "fullrank".
    "iter": 20_000,  # Maximum optimization iterations.
    "grad_samples": 1,  # Samples per gradient estimate.
    "elbo_samples": 100,  # Samples per ELBO estimate.
    "eta": 1.0,  # Initial step size.
    "adapt_iter": DEFAULT_VI_ADAPT_ITER,  # Step-size adaptation iterations.
    "tol_rel_obj": 0.01,  # Relative ELBO stopping tolerance.
    "eval_elbo": 100,  # ELBO evaluation interval.
    "draws": 1_000,  # Posterior draws to save.
    "require_converged": False,  # Allow preliminary non-converged VI.
}
FULLRANK_GRAD_SAMPLES = 10
MCMC_CONFIG = {
    "chains": 4,  # Independent NUTS chains.
    "parallel_chains": 4,  # Chains run concurrently.
    "iter_warmup": 1_000,  # Warmup iterations per chain.
    "iter_sampling": 1_000,  # Saved draws per chain.
    "adapt_delta": 0.9,  # Target acceptance probability.
    "max_treedepth": 10,  # Maximum NUTS tree depth.
}
CREDIBLE_INTERVAL = 0.9  # Central 90% posterior-predictive region.
MAX_WINDOWS = None  # Optional smoke-test limit; None evaluates every window.
PLOT_EACH_WINDOW = False  # Show the individual fit of every rolling window.
SAMPLE_TRAJECTORIES_PER_FORECAST = 15  # Posterior paths shown per forecast.
MODEL_VARIANT = "hybrid"


def main():
    """Run the shared rolling evaluation with the hybrid Bayesian model."""
    run_cli(
        model_variant=MODEL_VARIANT,
        description=__doc__,
        data_file=DATA_FILE,
        experiment=EXPERIMENT,
        priors=PRIORS,
        vi_config=VI_CONFIG,
        mcmc_config=MCMC_CONFIG,
        fullrank_grad_samples=FULLRANK_GRAD_SAMPLES,
        credible_interval=CREDIBLE_INTERVAL,
        max_windows=MAX_WINDOWS,
        plot_each_window=PLOT_EACH_WINDOW,
        sample_trajectories_per_forecast=SAMPLE_TRAJECTORIES_PER_FORECAST,
        hybrid_config=HYBRID_CONFIG,
    )


if __name__ == "__main__":
    main()
