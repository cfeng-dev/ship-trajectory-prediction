"""Show how observations update the Bayesian CTRV initial-heading prior."""

import argparse

import bayestraj.forecasting.inference as inference
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.observations.paths as paths
import bayestraj.validation.bayesian_ctrv_prior_posterior as prior_posterior

DATA_FILE = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)
RUN_ID = 102
START_INDEX = 0
POSITION_NOISE_STD_M = 5.0
POSITION_NOISE_SEED = 2026
INFERENCE_METHOD = "vi"  # Choose "vi" or "mcmc".
INFERENCE_SEED = 42
CREDIBLE_INTERVAL = 0.90
SHOW_LEGEND = True
PRIORS = bayesian_model.BayesianCTRVPriors()
VI_CONFIG = inference.create_default_vi_config()
MCMC_CONFIG = inference.create_default_mcmc_config()
PARAMETER_NAME = "initial_heading"


def main(argv=None):
    """Fit consecutive prefixes on demand and show one interactive plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    arguments = parser.parse_args(argv)
    return prior_posterior.run_bayesian_ctrv_prior_posterior_analysis(
        data_file=DATA_FILE,
        run_id=RUN_ID,
        parameter_name=PARAMETER_NAME,
        start_index=START_INDEX,
        position_noise_std_m=POSITION_NOISE_STD_M,
        position_noise_seed=POSITION_NOISE_SEED,
        priors=PRIORS,
        inference_method=INFERENCE_METHOD,
        vi_config=VI_CONFIG,
        mcmc_config=MCMC_CONFIG,
        inference_seed=INFERENCE_SEED,
        credible_interval=CREDIBLE_INTERVAL,
        show_legend=SHOW_LEGEND,
        show=not arguments.no_show,
    )


if __name__ == "__main__":
    main()
