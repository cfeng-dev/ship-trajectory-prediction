"""Evaluate Bayesian latent-position forecasts across rolling windows."""

import bayestraj.forecasting.bayesian_position_model as config
import bayestraj.forecasting.inference as inference
import bayestraj.models.bayesian_position_model as position_model
import bayestraj.observations.paths as paths
import bayestraj.validation.bayesian_position_model_workflow as workflow

DATA_FILE = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

EXPERIMENT = config.RollingExperimentConfig(
    run_id=102,
    observation_count=5,
    prediction_count=3,
    position_noise_std_m=5.0,
    position_noise_seed=2026,
    stride=None,
    inference_mode="online",  # Choose "batch" or "online".
    inference_method="rbpf",  # Batch: "vi"/"mcmc"; online: "rbpf".
    inference_seed=42,
)
PRIORS = position_model.BayesianPositionModelPriors(
    displacement_scale_prior_factor=2.0,
    displacement_scale_prior_tail_probability=0.05,
    rotation_angle_prior_abs_upper_deg=45.0,
    rotation_angle_prior_tail_probability=0.05,
    sigma_position_observation_prior_upper_m=20.0,
    sigma_position_observation_prior_tail_probability=0.05,
    sigma_motion_residual_prior_upper_m=20.0,
    sigma_motion_residual_prior_tail_probability=0.05,
)
VI_CONFIG = inference.create_default_vi_config()
MCMC_CONFIG = inference.create_default_mcmc_config()
RBPF_CONFIG = position_model.SequentialPositionFilterConfig(
    particle_count=4_000,
    posterior_draw_count=1_000,
    resample_ess_fraction=0.5,
    rejuvenation_scale=0.05,
)
CREDIBLE_INTERVAL = 0.9
MAX_WINDOWS = None
PLOT_EACH_WINDOW = False
SAMPLE_TRAJECTORIES_PER_FORECAST = 15
SHOW_TIME_LABELS = False  # Avoid repeated labels across all rolling windows.


def main(argv=None):
    """Run the configured rolling latent-position model evaluation."""
    options = config.parse_evaluation_arguments(
        description=__doc__,
        experiment=EXPERIMENT,
        vi_config=VI_CONFIG,
        max_windows=MAX_WINDOWS,
        plot_each_window=PLOT_EACH_WINDOW,
        argv=argv,
    )
    return workflow.run_bayesian_position_evaluation(
        data_file=DATA_FILE,
        experiment=EXPERIMENT,
        priors=PRIORS,
        vi_config=VI_CONFIG,
        mcmc_config=MCMC_CONFIG,
        rbpf_config=RBPF_CONFIG,
        fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
        credible_interval=CREDIBLE_INTERVAL,
        sample_trajectories_per_forecast=SAMPLE_TRAJECTORIES_PER_FORECAST,
        options=options,
        show_time_labels=SHOW_TIME_LABELS,
    )


if __name__ == "__main__":
    main()
