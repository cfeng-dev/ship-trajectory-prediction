"""Evaluate Bayesian Position Model forecasts across rolling windows."""

import ship_trajectory_prediction.forecasting.bayesian_position_model as config
import ship_trajectory_prediction.forecasting.inference as inference
import ship_trajectory_prediction.models.bayesian_position_model as position_model
import ship_trajectory_prediction.observations.paths as paths
import ship_trajectory_prediction.validation.bayesian_position_model_workflow as workflow

DATA_FILE = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

EXPERIMENT = config.RollingExperimentConfig(
    run_id=102,
    window_mode="sliding",
    observation_count=20,
    prediction_count=3,
    history_position_count=10,  # Compare with 10 through --history-positions.
    additional_position_noise_std_m=5.0,
    position_noise_seed=2026,
    stride=None,
    inference_method="vi",
    inference_seed=42,
)
PRIORS = position_model.BayesianPositionModelPriors(
    log_displacement_scale_prior_scale=0.016354,
    rotation_angle_prior_scale=0.016980,
    sigma_displacement_residual_prior_scale=1.989083,
)
VI_CONFIG = inference.create_default_vi_config()
MCMC_CONFIG = inference.create_default_mcmc_config()
CREDIBLE_INTERVAL = 0.9
MAX_WINDOWS = None
PLOT_EACH_WINDOW = False
SAMPLE_TRAJECTORIES_PER_FORECAST = 15


def main(argv=None):
    """Run the configured rolling Bayesian Position Model evaluation."""
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
        fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
        credible_interval=CREDIBLE_INTERVAL,
        sample_trajectories_per_forecast=SAMPLE_TRAJECTORIES_PER_FORECAST,
        options=options,
    )


if __name__ == "__main__":
    main()
