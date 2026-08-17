"""Evaluate hybrid Bayesian CTRV forecasts across rolling windows."""

import ship_trajectory_prediction.forecasting.bayesian_ctrv as config
import ship_trajectory_prediction.forecasting.inference as inference
import ship_trajectory_prediction.models.hybrid_bayesian_ctrv as hybrid_model
import ship_trajectory_prediction.observations.paths as paths
import ship_trajectory_prediction.validation.bayesian_ctrv_workflow as workflow
import ship_trajectory_prediction.validation.cli as cli

DATA_FILE = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

EXPERIMENT = config.RollingExperimentConfig(
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
PRIORS = hybrid_model.HybridBayesianCTRVPriors(
    position_initial_prior_scale=5.0,  # Initial x/y uncertainty [m].
    # Historical calibration from Run IDs 0-99; keep evaluation runs disjoint.
    speed_initial_prior_mean=3.524,  # Robust initial speed center [m/s].
    speed_initial_prior_scale=0.365,  # Robust initial speed scale [m/s].
    sigma_position_gps_prior_scale=5.0,  # Measurement-noise scale [m].
    sigma_position_process_prior_scale=0.5,  # Position drift [m/sqrt(s)].
    sigma_speed_process_prior_scale=0.05,  # Speed drift [(m/s)/sqrt(s)].
)
HYBRID_CONFIG = hybrid_model.HybridBayesianCTRVConfig(
    final_motion_history_seconds=60.0,  # Recent positions used for endpoint motion.
    min_final_motion_speed_mps=1.0,  # Below this, use neutral turn rate [m/s].
)
VI_CONFIG = inference.create_default_vi_config()
MCMC_CONFIG = inference.create_default_mcmc_config()
CREDIBLE_INTERVAL = 0.9  # Central 90% posterior-predictive region.
MAX_WINDOWS = None  # Optional smoke-test limit; None evaluates every window.
PLOT_EACH_WINDOW = False  # Show the individual fit of every rolling window.
SAMPLE_TRAJECTORIES_PER_FORECAST = 15  # Posterior paths shown per forecast.


def main(argv=None):
    """Run the configured hybrid Bayesian CTRV rolling evaluation."""
    options = cli.parse_bayesian_ctrv_evaluation_arguments(
        description=__doc__,
        experiment=EXPERIMENT,
        priors=PRIORS,
        vi_config=VI_CONFIG,
        max_windows=MAX_WINDOWS,
        plot_each_window=PLOT_EACH_WINDOW,
        include_turn_rate_prior=False,
        argv=argv,
    )
    return workflow.run_hybrid_bayesian_ctrv_evaluation(
        data_file=DATA_FILE,
        experiment=EXPERIMENT,
        priors=PRIORS,
        hybrid_config=HYBRID_CONFIG,
        vi_config=VI_CONFIG,
        mcmc_config=MCMC_CONFIG,
        fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
        credible_interval=CREDIBLE_INTERVAL,
        sample_trajectories_per_forecast=SAMPLE_TRAJECTORIES_PER_FORECAST,
        options=options,
    )


if __name__ == "__main__":
    main()
