"""Run one fully Bayesian CTRV trajectory prediction."""

import argparse
from time import perf_counter

import numpy as np

from ship_trajectory_prediction.evaluation.bayesian_ctrv import (
    ExperimentConfig,
    normalize_bayesian_ctrv_model_variant,
    select_bayesian_ctrv_inference_config,
)
from ship_trajectory_prediction.evaluation.metrics import (
    evaluate_position_predictions,
    print_position_evaluation,
)
from ship_trajectory_prediction.evaluation.prediction_plotting import (
    PLOT_COORDINATE_MODES,
    normalize_plot_coordinate_mode,
    plot_prediction,
)
from ship_trajectory_prediction.evaluation.reporting import (
    posterior_parameter_summary,
    posterior_variable_samples,
    print_prediction_setup,
    print_variational_diagnostics,
)
from ship_trajectory_prediction.models.bayesian_ctrv import (
    DEFAULT_VI_ADAPT_ITER,
    NOISE_PARAMETER_NAMES,
    BayesianCTRVPriors,
    build_stan_data,
    fit_bayesian_ctrv_model,
    simulate_position_observations,
    variational_converged,
)
from ship_trajectory_prediction.models.hybrid_bayesian_ctrv import (
    FINAL_MOTION_HISTORY_SECONDS,
    fit_hybrid_bayesian_ctrv_model,
)
from ship_trajectory_prediction.models.hybrid_bayesian_ctrv import (
    build_stan_data as build_hybrid_stan_data,
)
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import prepare_trajectory_window
from ship_trajectory_prediction.trajectory.io import read_ship_data

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)


EXPERIMENT = ExperimentConfig(
    run_id=1,  # Trajectory run to fit.
    start_index=0,  # First point of the selected window.
    observation_count=20,  # Position points used for fitting.
    prediction_count=3,  # Held-out future position points.
    additional_position_noise_std_m=2.0,  # Per x/y axis [m]; 0 disables.
    position_noise_seed=2026,  # Reproduces the added position noise.
    inference_method="vi",  # Fast "vi" or reference "mcmc".
    inference_seed=42,  # Reproduces VI or MCMC.
)
PRIORS = BayesianCTRVPriors(
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
VI_CONFIG = {
    "algorithm": "meanfield",  # "meanfield" or "fullrank".
    "iter": 20_000,  # Maximum optimization iterations.
    "grad_samples": 1,  # Samples per gradient estimate.
    "elbo_samples": 100,  # Samples per ELBO estimate.
    "eta": 1.0,  # Initial step size.
    "adapt_iter": DEFAULT_VI_ADAPT_ITER,  # Step-size adaptation iterations.
    "tol_rel_obj": 0.01,  # Relative ELBO stopping tolerance.
    "eval_elbo": 100,  # ELBO evaluation interval.
    "draws": 1_000,  # Posterior draws to save.
    "require_converged": False,  # Allow preliminary non-converged VI.
}
FULLRANK_GRAD_SAMPLES = 10
MCMC_CONFIG = {
    "chains": 4,  # Independent NUTS chains.
    "parallel_chains": 4,  # Chains run concurrently.
    "iter_warmup": 1_000,  # Warmup iterations per chain.
    "iter_sampling": 1_000,  # Saved draws per chain.
    "adapt_delta": 0.9,  # Target acceptance probability.
    "max_treedepth": 10,  # Maximum NUTS tree depth.
}
CREDIBLE_INTERVAL = 0.9  # Central 90% posterior interval.
PLOT_COORDINATE_MODE = "m"  # Display as local "m", "km", or absolute "gps".
MODEL_VARIANT = "bayesian"


def main(
    *,
    model_variant=MODEL_VARIANT,
    inference_method=EXPERIMENT.inference_method,
    vi_algorithm=VI_CONFIG["algorithm"],
    seed=EXPERIMENT.inference_seed,
    position_noise_std_m=EXPERIMENT.additional_position_noise_std_m,
    position_noise_seed=EXPERIMENT.position_noise_seed,
    require_converged=VI_CONFIG["require_converged"],
    plot_coordinate_mode=PLOT_COORDINATE_MODE,
):
    """Fit one recorded run with selected inference and evaluate predictions."""
    model_variant = normalize_bayesian_ctrv_model_variant(model_variant)
    if model_variant == "hybrid":
        stan_data_builder = build_hybrid_stan_data
        fit_model = fit_hybrid_bayesian_ctrv_model
        model_label = "Hybrid Bayesian CTRV"
    else:
        stan_data_builder = build_stan_data
        fit_model = fit_bayesian_ctrv_model
        model_label = "Fully Bayesian CTRV"
    inference_method, inference_config = select_bayesian_ctrv_inference_config(
        inference_method,
        vi_algorithm=vi_algorithm,
        require_converged=require_converged,
        vi_config=VI_CONFIG,
        mcmc_config=MCMC_CONFIG,
        fullrank_grad_samples=FULLRANK_GRAD_SAMPLES,
    )
    plot_coordinate_mode = normalize_plot_coordinate_mode(plot_coordinate_mode)
    trajectory_data = read_ship_data(DATA_FILE, run_id=EXPERIMENT.run_id)
    window = prepare_trajectory_window(
        trajectory_data,
        observation_count=EXPERIMENT.observation_count,
        prediction_count=EXPERIMENT.prediction_count,
        start_index=EXPERIMENT.start_index,
    )
    position_observations = simulate_position_observations(
        window,
        additional_noise_std_m=position_noise_std_m,
        seed=position_noise_seed,
    )
    stan_data = stan_data_builder(
        window,
        priors=PRIORS,
        position_observations=position_observations,
    )
    forecast_horizon_seconds = float(
        stan_data["time_prediction"][-1] - stan_data["time_observed"][-1]
    )
    inference_rows = [
        ("Model variant", model_variant),
        ("Inference method", inference_method.upper()),
    ]
    if model_variant == "hybrid":
        inference_rows.extend(
            [
                (
                    "Deterministic terminal heading",
                    f"{stan_data['heading_final']:.4f} rad "
                    f"({np.degrees(stan_data['heading_final']):.1f} deg)",
                ),
                (
                    "Deterministic terminal turn rate",
                    f"{stan_data['turn_rate_final']:.5f} rad/s",
                ),
                (
                    "Terminal-motion history",
                    f"{FINAL_MOTION_HISTORY_SECONDS:g} s",
                ),
            ]
        )
    else:
        inference_rows.append(("Terminal heading and turn rate", "latent posterior"))
    if inference_method == "vi":
        inference_rows.append(("VI algorithm", inference_config["algorithm"]))
    else:
        inference_rows.extend(
            [
                ("MCMC chains", inference_config["chains"]),
                ("MCMC parallel chains", inference_config["parallel_chains"]),
                ("MCMC warmup per chain", inference_config["iter_warmup"]),
                ("MCMC samples per chain", inference_config["iter_sampling"]),
            ]
        )
    _print_ctrv_setup(
        model_label=model_label,
        window=window,
        inference_rows=inference_rows,
        inference_seed=seed,
        position_observations=position_observations,
        forecast_horizon_seconds=forecast_horizon_seconds,
        plot_coordinate_mode=plot_coordinate_mode,
    )

    fit_started = perf_counter()
    fit = fit_model(
        window,
        priors=PRIORS,
        position_observations=position_observations,
        inference_method=inference_method,
        seed=seed,
        **inference_config,
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
            NOISE_PARAMETER_NAMES,
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
    observed_trajectory_label = (
        "Verrauschte Beobachtungen"
        if position_observations.additional_noise_std_m > 0
        else "Beobachtungen"
    )
    plot_prediction(
        window,
        fit,
        state_prediction_variable_names=(
            "x_state_prediction",
            "y_state_prediction",
        ),
        observed_position_values=(
            position_observations.x_meters,
            position_observations.y_meters,
        ),
        observed_trajectory_label=observed_trajectory_label,
        additional_position_noise_std_m=(position_observations.additional_noise_std_m),
        coordinate_mode=plot_coordinate_mode,
    )


def _print_ctrv_setup(
    *,
    model_label,
    window,
    inference_rows,
    inference_seed,
    position_observations,
    forecast_horizon_seconds,
    plot_coordinate_mode,
):
    """Print the concise, reproducible setup for one Bayesian CTRV run."""
    noise_std_m = position_observations.additional_noise_std_m
    noise_description = (
        f"{noise_std_m:g} m (seed={position_observations.noise_seed})"
        if noise_std_m > 0
        else "disabled"
    )
    print_prediction_setup(
        f"{model_label} State-Space Prediction",
        data_file=DATA_FILE,
        run_id=EXPERIMENT.run_id,
        window=window,
        extra_rows=[
            *inference_rows,
            ("Inference seed", inference_seed),
            ("Additional position noise", noise_description),
            ("Forecast horizon", f"{forecast_horizon_seconds:g} s"),
            ("Plot coordinates", plot_coordinate_mode),
        ],
    )


def _parse_arguments(*, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--inference",
        choices=("vi", "mcmc"),
        default=EXPERIMENT.inference_method,
    )
    parser.add_argument(
        "--vi-algorithm",
        choices=("meanfield", "fullrank"),
        default=VI_CONFIG["algorithm"],
    )
    parser.add_argument("--seed", type=int, default=EXPERIMENT.inference_seed)
    parser.add_argument(
        "--position-noise-std-m",
        type=float,
        default=EXPERIMENT.additional_position_noise_std_m,
        help="Extra Gaussian standard deviation per local x/y axis; 0 disables it.",
    )
    parser.add_argument(
        "--position-noise-seed",
        type=int,
        default=EXPERIMENT.position_noise_seed,
        help="Seed used only to generate the in-memory position perturbation.",
    )
    parser.add_argument(
        "--require-converged",
        action="store_true",
        default=VI_CONFIG["require_converged"],
        help="Abort instead of plotting if CmdStan reports non-converged VI.",
    )
    parser.add_argument(
        "--plot-coordinates",
        metavar="{" + ",".join(PLOT_COORDINATE_MODES) + "}",
        default=PLOT_COORDINATE_MODE,
        help=(
            "Display local meters, local kilometers, or GPS coordinates; "
            "invalid values fall back to meters."
        ),
    )
    return parser.parse_args()


def run_cli(*, model_variant=MODEL_VARIANT, description=__doc__):
    """Parse shared options and run one fixed Bayesian CTRV variant."""
    arguments = _parse_arguments(description=description)
    main(
        model_variant=model_variant,
        inference_method=arguments.inference,
        vi_algorithm=arguments.vi_algorithm,
        seed=arguments.seed,
        position_noise_std_m=arguments.position_noise_std_m,
        position_noise_seed=arguments.position_noise_seed,
        require_converged=arguments.require_converged,
        plot_coordinate_mode=arguments.plot_coordinates,
    )


if __name__ == "__main__":
    run_cli()
