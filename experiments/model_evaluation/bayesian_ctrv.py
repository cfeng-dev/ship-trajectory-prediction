"""Evaluate parametric Bayesian CTRV forecasts across rolling windows."""

import bayestraj.forecasting.bayesian_ctrv as config
import bayestraj.forecasting.inference as inference
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.observations.paths as paths
import bayestraj.validation.bayesian_ctrv_workflow as workflow
import bayestraj.validation.cli as cli

DATA_FILE = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

EXPERIMENT = config.RollingExperimentConfig(
    run_id=102,
    window_mode="sliding",
    observation_count=5,
    prediction_count=3,
    position_noise_std_m=5.0,
    position_noise_seed=2026,
    stride=None,
    inference_method="vi",
    inference_seed=42,
)
PRIORS = bayesian_model.BayesianCTRVPriors(
    speed_prior_upper_mps=20.0,
    speed_prior_tail_probability=0.05,
    turn_rate_prior_abs_heading_change_deg=45.0,
    turn_rate_prior_reference_interval_seconds=10.0,
    turn_rate_prior_tail_probability=0.05,
    sigma_motion_process_prior_upper_m=20.0,
    sigma_motion_process_prior_tail_probability=0.05,
    sigma_position_observation_prior_upper_m=20.0,
    sigma_position_observation_prior_tail_probability=0.05,
)
VI_CONFIG = inference.create_default_vi_config()
MCMC_CONFIG = inference.create_default_mcmc_config()
CREDIBLE_INTERVAL = 0.9
MAX_WINDOWS = None
PLOT_EACH_WINDOW = False
SAMPLE_TRAJECTORIES_PER_FORECAST = 15
SHOW_TIME_LABELS = False  # Avoid repeated labels across all rolling windows.


def main(argv=None):
    """Run the configured parametric Bayesian CTRV rolling evaluation."""
    options = cli.parse_bayesian_ctrv_evaluation_arguments(
        description=__doc__,
        experiment=EXPERIMENT,
        priors=PRIORS,
        vi_config=VI_CONFIG,
        max_windows=MAX_WINDOWS,
        plot_each_window=PLOT_EACH_WINDOW,
        argv=argv,
    )
    return workflow.run_bayesian_ctrv_evaluation(
        data_file=DATA_FILE,
        experiment=EXPERIMENT,
        priors=PRIORS,
        vi_config=VI_CONFIG,
        mcmc_config=MCMC_CONFIG,
        fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
        credible_interval=CREDIBLE_INTERVAL,
        sample_trajectories_per_forecast=SAMPLE_TRAJECTORIES_PER_FORECAST,
        options=options,
        show_time_labels=SHOW_TIME_LABELS,
    )


if __name__ == "__main__":
    main()
