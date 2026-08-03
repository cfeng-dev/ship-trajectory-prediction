"""Fit the Bayesian CTRV model to one recorded trajectory window."""

import argparse
from time import perf_counter

import numpy as np

from ship_trajectory_prediction.evaluation.metrics import (
    evaluate_position_predictions,
    print_position_evaluation,
)
from ship_trajectory_prediction.evaluation.plotting import plot_prediction
from ship_trajectory_prediction.evaluation.reporting import (
    posterior_parameter_summary,
    posterior_variable_samples,
    print_prediction_setup,
    print_variational_diagnostics,
)
from ship_trajectory_prediction.models.bayesian_ctrv import (
    NOISE_PARAMETER_NAMES,
    BayesianCTRVPriors,
    build_stan_data,
    fit_bayesian_ctrv_model,
    simulate_position_observations,
    variational_converged,
)
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import prepare_trajectory_window
from ship_trajectory_prediction.trajectory.io import read_ship_data

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

RUN_ID = 1
START_INDEX = 0
OBSERVATION_COUNT = 20
PREDICTION_COUNT = 5
# Set to 0.0 to fit the converted GPS positions without extra perturbation.
# A positive value adds independent Normal(0, std) noise to each local x/y axis.
ADDITIONAL_POSITION_NOISE_STD_M = 2.0
POSITION_NOISE_SEED = 2026
# Select "vi" for fast approximation or "mcmc" for reference NUTS sampling.
INFERENCE_METHOD = "vi"
SEED = 42
PRIORS = BayesianCTRVPriors(
    position_initial_prior_scale=5.0,
    speed_initial_prior_scale=0.75,
    heading_initial_prior_scale=0.35,
    turn_rate_state_prior_scale=None,
    sigma_position_gps_prior_scale=5.0,
    sigma_position_process_prior_scale=0.5,
    sigma_speed_process_prior_scale=0.05,
    sigma_turn_rate_process_prior_scale=0.001,
)
VI_ALGORITHM = "meanfield"
VI_ITER = 20_000
VI_GRAD_SAMPLES = 1
VI_ELBO_SAMPLES = 100
VI_ETA = 1.0
VI_ADAPT_ITER = 50
VI_TOL_REL_OBJ = 0.01
VI_EVAL_ELBO = 100
VI_DRAWS = 1_000
VI_REQUIRE_CONVERGED = False
MCMC_CHAINS = 4
MCMC_PARALLEL_CHAINS = 4
MCMC_ITER_WARMUP = 1_000
MCMC_ITER_SAMPLING = 1_000
MCMC_ADAPT_DELTA = 0.9
MCMC_MAX_TREEDEPTH = 10
CREDIBLE_INTERVAL = 0.9


def main(
    *,
    inference_method=INFERENCE_METHOD,
    vi_algorithm=VI_ALGORITHM,
    seed=SEED,
    additional_position_noise_std_m=ADDITIONAL_POSITION_NOISE_STD_M,
    position_noise_seed=POSITION_NOISE_SEED,
    require_converged=VI_REQUIRE_CONVERGED,
    mcmc_chains=MCMC_CHAINS,
    mcmc_parallel_chains=MCMC_PARALLEL_CHAINS,
    mcmc_iter_warmup=MCMC_ITER_WARMUP,
    mcmc_iter_sampling=MCMC_ITER_SAMPLING,
    mcmc_adapt_delta=MCMC_ADAPT_DELTA,
    mcmc_max_treedepth=MCMC_MAX_TREEDEPTH,
):
    """Fit one recorded run with selected inference and evaluate predictions."""
    if not isinstance(inference_method, str):
        raise ValueError("inference_method must be 'vi' or 'mcmc'.")
    inference_method = inference_method.strip().lower()
    if inference_method not in {"vi", "mcmc"}:
        raise ValueError("inference_method must be 'vi' or 'mcmc'.")
    trajectory_data = read_ship_data(DATA_FILE, run_id=RUN_ID)
    window = prepare_trajectory_window(
        trajectory_data,
        observation_count=OBSERVATION_COUNT,
        prediction_count=PREDICTION_COUNT,
        start_index=START_INDEX,
    )
    position_observations = simulate_position_observations(
        window,
        additional_noise_std_m=additional_position_noise_std_m,
        seed=position_noise_seed,
    )
    additional_noise_enabled = position_observations.additional_noise_std_m > 0
    stan_data = build_stan_data(
        window,
        priors=PRIORS,
        position_observations=position_observations,
    )
    forecast_horizon_seconds = float(
        stan_data["time_prediction"][-1] - stan_data["time_observed"][-1]
    )
    inference_rows = [("Inference method", inference_method.upper())]
    if inference_method == "vi":
        inference_rows.append(("VI algorithm", vi_algorithm))
    else:
        inference_rows.extend(
            [
                ("MCMC chains", mcmc_chains),
                ("MCMC parallel chains", mcmc_parallel_chains),
                ("MCMC warmup per chain", mcmc_iter_warmup),
                ("MCMC samples per chain", mcmc_iter_sampling),
            ]
        )
    _print_ctrv_setup(
        window=window,
        inference_rows=inference_rows,
        inference_seed=seed,
        position_observations=position_observations,
        forecast_horizon_seconds=forecast_horizon_seconds,
    )

    fit_started = perf_counter()
    fit = fit_bayesian_ctrv_model(
        window,
        priors=PRIORS,
        position_observations=position_observations,
        inference_method=inference_method,
        algorithm=vi_algorithm,
        iter=VI_ITER,
        grad_samples=VI_GRAD_SAMPLES,
        elbo_samples=VI_ELBO_SAMPLES,
        eta=VI_ETA,
        adapt_iter=VI_ADAPT_ITER,
        tol_rel_obj=VI_TOL_REL_OBJ,
        eval_elbo=VI_EVAL_ELBO,
        draws=VI_DRAWS,
        chains=mcmc_chains,
        parallel_chains=mcmc_parallel_chains,
        iter_warmup=mcmc_iter_warmup,
        iter_sampling=mcmc_iter_sampling,
        adapt_delta=mcmc_adapt_delta,
        max_treedepth=mcmc_max_treedepth,
        seed=seed,
        require_converged=require_converged,
    )
    fit_and_forecast_runtime_seconds = perf_counter() - fit_started
    print(f"\nModel fit and forecast runtime: {fit_and_forecast_runtime_seconds:.2f} s")
    if inference_method == "vi":
        converged = variational_converged(fit)
        print_variational_diagnostics(fit)
        print(f"CmdStan convergence criterion met: {converged}")
        if not converged:
            print(
                "WARNING: Treat this posterior and its plot as preliminary; "
                "the VI convergence criterion was not met."
            )
    else:
        print("\nMCMC diagnostics:")
        print(fit.diagnose())

    print("\nPosterior parameter summary:")
    print(
        posterior_parameter_summary(
            fit,
            ["heading_initial", *NOISE_PARAMETER_NAMES],
        )
    )

    speed_state = posterior_variable_samples(fit, "speed_state")
    heading_state = posterior_variable_samples(fit, "heading_state")
    turn_rate_state = posterior_variable_samples(fit, "turn_rate_state")
    print("\nPosterior state medians:")
    print(
        "Speed [m/s]       : "
        f"{np.median(speed_state[:, 0]):.3f} -> "
        f"{np.median(speed_state[:, -1]):.3f}"
    )
    print(
        "Heading [rad]     : "
        f"{np.median(heading_state[:, 0]):.4f} -> "
        f"{np.median(heading_state[:, -1]):.4f}"
    )
    print(
        "Turn rate [rad/s] : "
        f"{np.median(turn_rate_state[:, 0]):.5f} -> "
        f"{np.median(turn_rate_state[:, -1]):.5f}"
    )
    print(
        "GPS speed was not used for fitting; it is retained only as an "
        "external post-fit plausibility reference."
    )

    evaluation = evaluate_position_predictions(
        fit,
        window,
        credible_interval=CREDIBLE_INTERVAL,
        position_variable_names=(
            "x_state_prediction",
            "y_state_prediction",
        ),
    )
    print_position_evaluation(evaluation)
    plot_prediction(
        window,
        fit,
        model_name="CTRV State-Space",
        state_prediction_variable_names=(
            "x_state_prediction",
            "y_state_prediction",
        ),
        observed_position_values=(
            position_observations.x_meters,
            position_observations.y_meters,
        ),
        observed_trajectory_label=(
            "Noise-augmented observations"
            if additional_noise_enabled
            else "Observed trajectory"
        ),
    )


def _print_ctrv_setup(
    *,
    window,
    inference_rows,
    inference_seed,
    position_observations,
    forecast_horizon_seconds,
):
    """Print the concise, reproducible setup for one Bayesian CTRV run."""
    noise_std_m = position_observations.additional_noise_std_m
    noise_description = (
        f"{noise_std_m:g} m (seed={position_observations.noise_seed})"
        if noise_std_m > 0
        else "disabled"
    )
    print_prediction_setup(
        "Bayesian CTRV State-Space Prediction",
        data_file=DATA_FILE,
        run_id=RUN_ID,
        window=window,
        extra_rows=[
            *inference_rows,
            ("Inference seed", inference_seed),
            ("Additional position noise", noise_description),
            ("Forecast horizon", f"{forecast_horizon_seconds:g} s"),
        ],
    )


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference",
        choices=("vi", "mcmc"),
        default=INFERENCE_METHOD,
    )
    parser.add_argument(
        "--vi-algorithm",
        choices=("meanfield", "fullrank"),
        default=VI_ALGORITHM,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--position-noise-std-m",
        type=float,
        default=ADDITIONAL_POSITION_NOISE_STD_M,
        help="Extra Gaussian standard deviation per local x/y axis; 0 disables it.",
    )
    parser.add_argument(
        "--position-noise-seed",
        type=int,
        default=POSITION_NOISE_SEED,
        help="Seed used only to generate the in-memory position perturbation.",
    )
    parser.add_argument(
        "--require-converged",
        action="store_true",
        default=VI_REQUIRE_CONVERGED,
        help="Abort instead of plotting if CmdStan reports non-converged VI.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(
        inference_method=arguments.inference,
        vi_algorithm=arguments.vi_algorithm,
        seed=arguments.seed,
        additional_position_noise_std_m=arguments.position_noise_std_m,
        position_noise_seed=arguments.position_noise_seed,
        require_converged=arguments.require_converged,
    )
