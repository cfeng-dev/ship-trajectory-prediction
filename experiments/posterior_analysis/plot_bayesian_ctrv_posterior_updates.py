"""Show Bayesian CTRV posterior updates alongside the recorded trajectory."""

import argparse

import bayestraj.inference.configuration as inference
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.observations.paths as paths
import bayestraj.validation.bayesian_ctrv_posterior_dashboard as dashboard

DATA_FILE = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

# Posterior-update inference:
# - Batch: "vi" or "mcmc".
# - Online: "rbpf" or "smc".
# None uses the largest supported observation count. Prefer a small value for
# VI/MCMC because every new posterior stage requires a separate batch fit.
EXPERIMENT = dashboard.PosteriorDashboardConfig(
    run_id=102,
    start_index=0,
    maximum_observation_count=None,
    position_noise_std_m=5.0,
    position_noise_seed=2026,
    inference_method="rbpf",
    inference_seed=42,
    # Future timestamps of this recording; the horizon shortens near its end.
    prediction_count=3,  # 30 seconds at 10-second intervals; 0 disables forecasts.
    prediction_sample_count=20,  # 0 hides sample paths; the median remains visible.
)
PRIORS = bayesian_model.BayesianCTRVPriors(
    speed_prior_upper_mps=20.0,
    speed_prior_tail_probability=0.05,
    turn_rate_prior_abs_heading_change_deg=45.0,
    turn_rate_prior_reference_interval_seconds=10.0,
    turn_rate_prior_tail_probability=0.05,
    sigma_position_observation_prior_upper_m=20.0,
    sigma_position_observation_prior_tail_probability=0.05,
    sigma_speed_process_prior_upper_mps=5.0,
    sigma_speed_process_prior_tail_probability=0.05,
    sigma_turn_rate_process_prior_upper_deg_s=4.5,
    sigma_turn_rate_process_prior_tail_probability=0.05,
)
VI_CONFIG = inference.create_default_vi_config()
MCMC_CONFIG = inference.create_default_mcmc_config()
RBPF_CONFIG = inference.create_default_ctrv_rbpf_config()
SMC_CONFIG = inference.create_default_ctrv_smc_config()
PLAYBACK_INTERVAL_MS = dashboard.DEFAULT_PLAYBACK_INTERVAL_MS
COORDINATE_DISPLAY_MODE = "m"  # "m", "km" or "gps" (longitude/latitude)
SHOW_LEGEND = True


def main(argv=None):
    """Run the combined trajectory and posterior update dashboard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    arguments = parser.parse_args(argv)
    return dashboard.run_bayesian_ctrv_posterior_dashboard(
        data_file=DATA_FILE,
        experiment=EXPERIMENT,
        priors=PRIORS,
        vi_config=VI_CONFIG,
        mcmc_config=MCMC_CONFIG,
        rbpf_config=RBPF_CONFIG,
        smc_config=SMC_CONFIG,
        playback_interval_ms=PLAYBACK_INTERVAL_MS,
        coordinate_display_mode=COORDINATE_DISPLAY_MODE,
        show_legend=SHOW_LEGEND,
        show=not arguments.no_show,
    )


if __name__ == "__main__":
    main()
