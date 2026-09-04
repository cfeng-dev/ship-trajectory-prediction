"""Show Bayesian CTRV posterior updates alongside the recorded trajectory."""

import argparse

import bayestraj.inference.configuration as inference
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.observations.paths as paths
import bayestraj.validation.bayesian_ctrv_posterior_dashboard as dashboard

DATA_FILE = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)
RUN_ID = 102
START_INDEX = 0
POSITION_NOISE_STD_M = 5.0
POSITION_NOISE_SEED = 2026
RBPF_SEED = 42
PLAYBACK_INTERVAL_MS = dashboard.DEFAULT_PLAYBACK_INTERVAL_MS
SHOW_LEGEND = True
PRIORS = bayesian_model.BayesianCTRVPriors()
RBPF_CONFIG = inference.create_default_ctrv_rbpf_config()


def main(argv=None):
    """Run the combined trajectory and posterior update dashboard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    arguments = parser.parse_args(argv)
    return dashboard.run_bayesian_ctrv_posterior_dashboard(
        data_file=DATA_FILE,
        run_id=RUN_ID,
        start_index=START_INDEX,
        position_noise_std_m=POSITION_NOISE_STD_M,
        position_noise_seed=POSITION_NOISE_SEED,
        priors=PRIORS,
        rbpf_config=RBPF_CONFIG,
        rbpf_seed=RBPF_SEED,
        playback_interval_ms=PLAYBACK_INTERVAL_MS,
        show_legend=SHOW_LEGEND,
        show=not arguments.no_show,
    )


if __name__ == "__main__":
    main()
