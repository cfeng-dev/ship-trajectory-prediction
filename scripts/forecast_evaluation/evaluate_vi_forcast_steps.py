"""Export Bayesian CTRV VI forecasts for 1, 2 and 3 future positions."""

import bayestraj.forecasting.bayesian_ctrv as ctrv_config
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.validation.prediction_horizons as evaluation
from bayestraj.paths import project_path

DATA_FILE = project_path("data/processed/BR24_imm_tracked_trajectory.csv")
OUTPUT_DIR = project_path("results/forecast_evaluation")  # Forecast-step folders.
PREDICTION_COUNTS = (1, 2, 3)  # Number of future positions, not seconds.
MAX_WINDOWS = None

EXPERIMENT = ctrv_config.RollingExperimentConfig(
    observation_count=4,  # Fixed VI observation window throughout the comparison.
    prediction_count=max(PREDICTION_COUNTS),
    position_noise_std_m=0.0,
    position_noise_seed=2026,
    stride=1,
    inference_method="vi",
    inference_seed=42,
)
PRIORS = bayesian_model.BayesianCTRVPriors(
    speed_prior_upper_mps=10.0,
    speed_prior_tail_probability=0.05,
    turn_rate_prior_abs_rate_deg_s=10.0,
    turn_rate_prior_tail_probability=0.05,
    sigma_position_observation_prior_upper_m=20.0,
    sigma_position_observation_prior_tail_probability=0.05,
    sigma_speed_process_prior_upper_mps=2.0,
    sigma_speed_process_prior_tail_probability=0.05,
    sigma_turn_rate_process_prior_upper_deg_s=2.0,
    sigma_turn_rate_process_prior_tail_probability=0.05,
)


def main(argv=None):
    """Run VI once at the longest horizon and export each requested prefix."""
    return evaluation.run_prediction_horizon_evaluation(
        data_file=DATA_FILE,
        output_dir=OUTPUT_DIR,
        prediction_counts=PREDICTION_COUNTS,
        experiment=EXPERIMENT,
        priors=PRIORS,
        max_windows=MAX_WINDOWS,
        argv=argv,
    )


if __name__ == "__main__":
    main()
