"""Fit the Bayesian constant-turn-rate model to one trajectory window."""

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
from ship_trajectory_prediction.models.constant_turn_rate import (
    build_stan_data,
    fit_constant_turn_rate_model,
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
SPEED_PRIOR_LOG_SD = 0.5
HEADING_PRIOR_SCALE = 0.5
TURN_RATE_PRIOR_SCALE = 0.01
SIGMA_PRIOR_SCALE = 20.0
VI_ITER = 20_000
VI_DRAWS = 1_000
VI_TOL_REL_OBJ = 0.05
VI_EVAL_ELBO = 100


def main(*, inference_method="mcmc", vi_algorithm="meanfield"):
    """Fit the model and plot posterior trajectories against held-out data."""
    trajectory_data = read_ship_data(DATA_FILE, run_id=RUN_ID)
    window = prepare_trajectory_window(
        trajectory_data,
        observation_count=OBSERVATION_COUNT,
        prediction_count=PREDICTION_COUNT,
        start_index=START_INDEX,
    )
    model_kwargs = {
        "speed_prior_log_sd": SPEED_PRIOR_LOG_SD,
        "heading_prior_scale": HEADING_PRIOR_SCALE,
        "turn_rate_prior_scale": TURN_RATE_PRIOR_SCALE,
        "sigma_prior_scale": SIGMA_PRIOR_SCALE,
    }
    variational_options = {
        "algorithm": vi_algorithm,
        "iter": VI_ITER,
        "draws": VI_DRAWS,
        "tol_rel_obj": VI_TOL_REL_OBJ,
        "eval_elbo": VI_EVAL_ELBO,
    }
    stan_data = build_stan_data(window, **model_kwargs)

    print_prediction_setup(
        "Bayesian Constant-Turn-Rate Trajectory Prediction",
        data_file=DATA_FILE,
        run_id=RUN_ID,
        window=window,
        extra_rows=[
            ("Inference method", inference_method.upper()),
            ("VI algorithm", vi_algorithm if inference_method == "vi" else "-"),
            (
                "Speed prior median",
                f"{stan_data['speed_prior_median']:.2f} m/s",
            ),
        ],
    )

    fit = fit_constant_turn_rate_model(
        window,
        **model_kwargs,
        inference_method=inference_method,
        variational_options=variational_options,
    )

    if inference_method == "vi":
        print_variational_diagnostics(fit)

    print("\nPosterior parameter summary:")
    print(
        posterior_parameter_summary(
            fit,
            ["speed", "heading_initial", "turn_rate", "radius", "sigma"],
        )
    )

    turn_rate_samples = posterior_variable_samples(fit, "turn_rate")
    radius_samples = posterior_variable_samples(fit, "radius")
    radius_lower, radius_median, radius_upper = np.quantile(
        radius_samples,
        [0.05, 0.5, 0.95],
    )
    print(f"\nLeft-turn probability : {np.mean(turn_rate_samples > 0):.1%}")
    print(f"Right-turn probability: {np.mean(turn_rate_samples < 0):.1%}")
    print(
        "Derived radius       : "
        f"{radius_median:.1f} m "
        f"(90% interval: {radius_lower:.1f} to {radius_upper:.1f} m)"
    )

    evaluation = evaluate_position_predictions(fit, window)
    print_position_evaluation(evaluation)

    plot_prediction(window, fit, model_name="Constant-Turn-Rate")


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", choices=("mcmc", "vi"), default="mcmc")
    parser.add_argument(
        "--vi-algorithm",
        choices=("meanfield", "fullrank"),
        default="meanfield",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(
        inference_method=arguments.inference,
        vi_algorithm=arguments.vi_algorithm,
    )
