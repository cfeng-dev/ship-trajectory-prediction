"""Run one parametric Bayesian CTRV trajectory prediction."""

import ship_trajectory_prediction.forecasting.bayesian_ctrv as config
import ship_trajectory_prediction.forecasting.bayesian_ctrv_workflow as workflow
import ship_trajectory_prediction.forecasting.cli as cli
import ship_trajectory_prediction.forecasting.inference as inference
import ship_trajectory_prediction.models.bayesian_ctrv as bayesian_model
import ship_trajectory_prediction.observations.paths as paths

DATA_FILE = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

EXPERIMENT = config.ExperimentConfig(
    run_id=102,
    start_index=0,
    observation_count=20,
    prediction_count=3,
    history_position_count=20,  # Compare with K=10 through --history-positions.
    additional_position_noise_std_m=5.0,
    position_noise_seed=2026,
    inference_method="vi",
    inference_seed=42,
)
PRIORS = bayesian_model.BayesianCTRVPriors(
    # Provisional transfer; validate specifically for the parametric CTRV model.
    speed_prior_mean=3.524,
    speed_prior_scale=0.365,
    turn_rate_prior_mean=0.0,
    turn_rate_prior_scale=0.001698,
    # Scenario assumption, not derived from run IDs: P(sigma_obs > 20 m) = 5%.
    sigma_position_observation_prior_upper_m=20.0,
    sigma_position_observation_prior_tail_probability=0.05,
)
VI_CONFIG = inference.create_default_vi_config()
MCMC_CONFIG = inference.create_default_mcmc_config()
CREDIBLE_INTERVAL = 0.9
PLOT_COORDINATE_MODE = "m"


def main(argv=None):
    """Run the configured parametric Bayesian CTRV experiment."""
    arguments = cli.parse_bayesian_ctrv_prediction_arguments(
        description=__doc__,
        experiment=EXPERIMENT,
        vi_config=VI_CONFIG,
        plot_coordinate_mode=PLOT_COORDINATE_MODE,
        argv=argv,
    )
    return workflow.run_bayesian_ctrv_prediction(
        data_file=DATA_FILE,
        experiment=EXPERIMENT,
        priors=PRIORS,
        vi_config=VI_CONFIG,
        mcmc_config=MCMC_CONFIG,
        fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
        credible_interval=CREDIBLE_INTERVAL,
        history_position_count=arguments.history_positions,
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
