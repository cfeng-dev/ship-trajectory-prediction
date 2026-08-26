"""Run one Bayesian latent-position measurement-error prediction."""

import bayestraj.forecasting.bayesian_position_model as config
import bayestraj.forecasting.bayesian_position_model_workflow as workflow
import bayestraj.forecasting.inference as inference
import bayestraj.models.bayesian_position_model as position_model
import bayestraj.observations.paths as paths

DATA_FILE = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

EXPERIMENT = config.ExperimentConfig(
    run_id=102,
    start_index=0,
    observation_count=10,
    prediction_count=3,
    additional_position_noise_std_m=5.0,
    position_noise_seed=2026,
    inference_method="vi",
    inference_seed=42,
)
PRIORS = position_model.BayesianPositionModelPriors(
    # The motion-residual value is provisional: its calibration used noisy
    # observed displacements rather than latent motion-only residuals.
    log_displacement_scale_prior_scale=0.016354,
    rotation_angle_prior_scale=0.016980,
    sigma_motion_residual_prior_scale=1.989083,
)
VI_CONFIG = inference.create_default_vi_config()
MCMC_CONFIG = inference.create_default_mcmc_config()
CREDIBLE_INTERVAL = 0.9
PLOT_COORDINATE_MODE = "m"


def main(argv=None):
    """Run the configured latent-position measurement-error experiment."""
    arguments = config.parse_prediction_arguments(
        description=__doc__,
        experiment=EXPERIMENT,
        vi_config=VI_CONFIG,
        plot_coordinate_mode=PLOT_COORDINATE_MODE,
        argv=argv,
    )
    return workflow.run_bayesian_position_prediction(
        data_file=DATA_FILE,
        experiment=EXPERIMENT,
        priors=PRIORS,
        vi_config=VI_CONFIG,
        mcmc_config=MCMC_CONFIG,
        fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
        credible_interval=CREDIBLE_INTERVAL,
        observation_count=arguments.observations,
        inference_method=arguments.inference,
        vi_algorithm=arguments.vi_algorithm,
        seed=arguments.seed,
        position_noise_std_m=arguments.position_noise_std_m,
        position_noise_seed=arguments.position_noise_seed,
        require_converged=arguments.require_converged,
        plot_coordinate_mode=arguments.plot_coordinates,
    )


if __name__ == "__main__":
    main()
