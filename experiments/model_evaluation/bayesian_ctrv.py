"""Evaluate parametric Bayesian CTRV forecasts across rolling windows."""

import bayestraj.forecasting.bayesian_ctrv as config
import bayestraj.inference.configuration as inference
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.validation.bayesian_ctrv_workflow as workflow
import bayestraj.validation.cli as cli
import bayestraj.validation.runtime_plotting as runtime_plotting
from bayestraj.paths import project_path

# Raw detections are tracked online; ground_truth_* is used only for evaluation.
DATA_FILE = project_path("data/processed/BR24Dataset20131107_with_reference.csv")
OUTPUT_CSV = project_path("results/BR24_bayesian_ctrv_forecast_evaluation.csv")

# Rolling inference:
# - Batch: "vi" or "mcmc" with a fixed sliding window.
# - Online: "smc" or "sequential_vi".
EXPERIMENT = config.RollingExperimentConfig(
    prediction_count=2,
    position_noise_std_m=0.0,
    position_noise_seed=2026,
    stride=1,  # Forecast-origin step: one new forecast per radar cycle.
    inference_method="smc",
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
    """Run the configured parametric Bayesian CTRV rolling evaluation."""
    options = cli.parse_bayesian_ctrv_evaluation_arguments(
        description=__doc__,
        experiment=EXPERIMENT,
        priors=PRIORS,
        max_windows=MAX_WINDOWS,
        plot_each_window=PLOT_EACH_WINDOW,
        argv=argv,
    )
    predictions, summary = workflow.run_bayesian_ctrv_evaluation(
        data_file=DATA_FILE,
        experiment=EXPERIMENT,
        priors=PRIORS,
        fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
        credible_interval=CREDIBLE_INTERVAL,
        sample_trajectories_per_forecast=SAMPLE_TRAJECTORIES_PER_FORECAST,
        options=options,
        show_time_labels=SHOW_TIME_LABELS,
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved per-forecast Bayesian CTRV metrics: {OUTPUT_CSV}")
    if SHOW_ROLLING_DIAGNOSTICS_PLOT:
        runtime_plotting.plot_bayesian_ctrv_rolling_diagnostics(
            predictions,
            inference_selection=options.inference_method,
        )
    return predictions, summary


if __name__ == "__main__":
    main()
