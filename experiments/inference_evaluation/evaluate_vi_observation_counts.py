"""Evaluate BR24 VI window sizes and save one CSV per observation count.

Run directly with the header settings, or limit a run with
--min-observations 2 --max-observations 20 --max-windows 1.
All sizes are evaluated from N=2 while each training window grows to its limit.
"""

import bayestraj.forecasting.bayesian_ctrv as ctrv_config
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.validation.bayesian_ctrv_evaluation as evaluation
from bayestraj.paths import project_path

DATA_FILE = project_path("data/processed/BR24Dataset20131107_with_reference.csv")
OUTPUT_DIR = project_path("results/bayesian_ctrv_vi_evaluation")
MIN_OBSERVATION_COUNT = 2
MAX_OBSERVATION_COUNT = 20
MAX_WINDOWS = None  # Optional limit on evaluated origins after initialization.

EXPERIMENT = ctrv_config.RollingExperimentConfig(
    prediction_count=3,
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
    """Evaluate each configured window size using the shared BR24 workflow."""
    return evaluation.run_observation_count_comparison(
        data_file=DATA_FILE,
        output_dir=OUTPUT_DIR,
        experiment=EXPERIMENT,
        priors=PRIORS,
        min_observation_count=MIN_OBSERVATION_COUNT,
        max_observation_count=MAX_OBSERVATION_COUNT,
        max_windows=MAX_WINDOWS,
        argv=argv,
    )


if __name__ == "__main__":
    main()
