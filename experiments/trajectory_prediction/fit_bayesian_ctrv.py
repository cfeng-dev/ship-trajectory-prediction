"""Fit the Bayesian CTRV model to one recorded trajectory window."""

import argparse

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
VI_ITER = 20_000
VI_GRAD_SAMPLES = 1
VI_ELBO_SAMPLES = 100
VI_ETA = 1.0
VI_ADAPT_ITER = 50
VI_TOL_REL_OBJ = 0.01
VI_EVAL_ELBO = 100
VI_DRAWS = 1_000
CREDIBLE_INTERVAL = 0.9


def main(*, vi_algorithm="meanfield", seed=42, require_converged=False):
    """Fit one recorded run and plot predictions against held-out GPS data."""
    trajectory_data = read_ship_data(DATA_FILE, run_id=RUN_ID)
    window = prepare_trajectory_window(
        trajectory_data,
        observation_count=OBSERVATION_COUNT,
        prediction_count=PREDICTION_COUNT,
        start_index=START_INDEX,
    )
    stan_data = build_stan_data(window, priors=PRIORS)
    print_prediction_setup(
        "Bayesian CTRV State-Space Prediction",
        data_file=DATA_FILE,
        run_id=RUN_ID,
        window=window,
        extra_rows=[
            ("Inference method", "VI"),
            ("VI algorithm", vi_algorithm),
            ("Seed", seed),
            ("Observation model", "position only"),
            (
                "Initial speed center",
                f"{stan_data['speed_initial_prior_mean']:.3f} m/s (from positions)",
            ),
            (
                "Initial turn-rate center",
                f"{stan_data['turn_rate_initial_prior_mean']:.5f} rad/s",
            ),
            (
                "Turn-rate state prior scale",
                f"{stan_data['turn_rate_state_prior_scale']:.5f} rad/s",
            ),
            ("Turn-rate limit", f"{stan_data['turn_rate_limit']:.5f} rad/s"),
        ],
    )

    fit = fit_bayesian_ctrv_model(
        window,
        priors=PRIORS,
        algorithm=vi_algorithm,
        iter=VI_ITER,
        grad_samples=VI_GRAD_SAMPLES,
        elbo_samples=VI_ELBO_SAMPLES,
        eta=VI_ETA,
        adapt_iter=VI_ADAPT_ITER,
        tol_rel_obj=VI_TOL_REL_OBJ,
        eval_elbo=VI_EVAL_ELBO,
        draws=VI_DRAWS,
        seed=seed,
        require_converged=require_converged,
    )
    converged = variational_converged(fit)
    print_variational_diagnostics(fit)
    print(f"CmdStan convergence criterion met: {converged}")
    if not converged:
        print(
            "WARNING: Treat this posterior and its plot as preliminary; "
            "the VI convergence criterion was not met."
        )

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
            "x_observation_prediction",
            "y_observation_prediction",
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
    )


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vi-algorithm",
        choices=("meanfield", "fullrank"),
        default="meanfield",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-converged",
        action="store_true",
        help="Abort instead of plotting if CmdStan reports non-converged VI.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(
        vi_algorithm=arguments.vi_algorithm,
        seed=arguments.seed,
        require_converged=arguments.require_converged,
    )
