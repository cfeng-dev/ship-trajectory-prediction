"""Evaluate Bayesian CTRV forecasts across rolling windows (SMC by default).

Set OUTPUT_CSV to None to skip CSV export.
"""

import bayestraj.forecasting.bayesian_ctrv as ctrv_config
import bayestraj.inference.configuration as inference_config
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.validation.bayesian_ctrv_evaluation as evaluation
from bayestraj.paths import project_path

DATA_FILE = project_path("data/processed/BR24Dataset20131107_with_reference.csv")
OUTPUT_CSV = project_path("results/bayesian_ctrv_{inference_method}_evaluation.csv")

EXPERIMENT = ctrv_config.RollingExperimentConfig(
    observation_count=10,
    prediction_count=3,
    position_noise_std_m=0.0,
    position_noise_seed=2026,
    stride=1,  # Forecast-origin step: one new forecast per radar cycle.
    inference_method="sequential_vi",  # Methods: "smc", "vi", "mcmc", or "sequential_vi".
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
CREDIBLE_INTERVAL = 0.9
MAX_WINDOWS = None
PLOT_EACH_WINDOW = False
SAMPLE_TRAJECTORIES_PER_FORECAST = 15
SHOW_TIME_LABELS = False  # Avoid repeated labels across all rolling windows.
SHOW_ROLLING_DIAGNOSTICS_PLOT = True


def main(argv=None):
    """Run rolling evaluation using this script's header settings."""
    return evaluation.run_evaluation(
        data_file=DATA_FILE,
        output_csv=OUTPUT_CSV,
        experiment=EXPERIMENT,
        priors=PRIORS,
        fullrank_grad_samples=inference_config.DEFAULT_FULLRANK_GRAD_SAMPLES,
        credible_interval=CREDIBLE_INTERVAL,
        max_windows=MAX_WINDOWS,
        plot_each_window=PLOT_EACH_WINDOW,
        sample_trajectories_per_forecast=SAMPLE_TRAJECTORIES_PER_FORECAST,
        show_time_labels=SHOW_TIME_LABELS,
        show_rolling_diagnostics_plot=SHOW_ROLLING_DIAGNOSTICS_PLOT,
        description=__doc__,
        argv=argv,
    )


if __name__ == "__main__":
    main()
