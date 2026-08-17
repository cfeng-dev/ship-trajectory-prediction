"""Run one fully Bayesian CTRV trajectory prediction."""

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
    run_id=1,  # Trajectory run to fit.
    start_index=0,  # First point of the selected window.
    observation_count=20,  # Position points used for fitting.
    prediction_count=3,  # Held-out future position points.
    additional_position_noise_std_m=2.0,  # Per x/y axis [m]; 0 disables.
    position_noise_seed=2026,  # Reproduces the added position noise.
    inference_method="vi",  # Fast "vi" or reference "mcmc".
    inference_seed=42,  # Reproduces VI or MCMC.
)
PRIORS = bayesian_model.BayesianCTRVPriors(
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
VI_CONFIG = inference.create_default_vi_config()
MCMC_CONFIG = inference.create_default_mcmc_config()
CREDIBLE_INTERVAL = 0.9  # Central 90% posterior interval.
PLOT_COORDINATE_MODE = "m"  # Display as local "m", "km", or absolute "gps".


def main(argv=None):
    """Run the configured fully Bayesian CTRV experiment."""
    arguments = cli.parse_bayesian_ctrv_prediction_arguments(
        description=__doc__,
        experiment=EXPERIMENT,
        vi_config=VI_CONFIG,
        plot_coordinate_mode=PLOT_COORDINATE_MODE,
        argv=argv,
    )
    return workflow.run_fully_bayesian_ctrv_prediction(
        data_file=DATA_FILE,
        experiment=EXPERIMENT,
        priors=PRIORS,
        vi_config=VI_CONFIG,
        mcmc_config=MCMC_CONFIG,
        fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
        credible_interval=CREDIBLE_INTERVAL,
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
