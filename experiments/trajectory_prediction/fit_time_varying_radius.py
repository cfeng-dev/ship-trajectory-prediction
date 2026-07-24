"""Fit the Bayesian time-varying-radius model to one trajectory window."""

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
from ship_trajectory_prediction.models.time_varying_radius import (
    build_stan_data,
    fit_time_varying_radius_model,
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

CURVATURE_INITIAL_PRIOR_SCALE = 0.002
CURVATURE_RATE_PRIOR_SCALE = 5e-6
SIGMA_PRIOR_SCALE = 20.0
INTEGRATION_SUBSTEPS = 4
VI_ITER = 20_000
VI_DRAWS = 1_000
VI_TOL_REL_OBJ = 0.05
VI_EVAL_ELBO = 100


def main(*, inference_method="mcmc", vi_algorithm="meanfield"):
    """Fit the model and evaluate posterior predictions on held-out data."""
    trajectory_data = read_ship_data(DATA_FILE, run_id=RUN_ID)
    window = prepare_trajectory_window(
        trajectory_data,
        observation_count=OBSERVATION_COUNT,
        prediction_count=PREDICTION_COUNT,
        start_index=START_INDEX,
    )
    model_kwargs = {
        "curvature_initial_prior_scale": CURVATURE_INITIAL_PRIOR_SCALE,
        "curvature_rate_prior_scale": CURVATURE_RATE_PRIOR_SCALE,
        "sigma_prior_scale": SIGMA_PRIOR_SCALE,
        "integration_substeps": INTEGRATION_SUBSTEPS,
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
        "Bayesian Time-Varying-Radius Prediction",
        data_file=DATA_FILE,
        run_id=RUN_ID,
        window=window,
        extra_rows=[
            ("Inference method", inference_method.upper()),
            ("VI algorithm", vi_algorithm if inference_method == "vi" else "-"),
            ("Fixed speed", f"{stan_data['speed']:.2f} m/s"),
        ],
    )

    fit = fit_time_varying_radius_model(
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
            [
                "curvature_initial",
                "curvature_rate",
                "radius_initial",
                "radius_horizon",
                "sigma",
            ],
        )
    )

    curvature_initial = posterior_variable_samples(fit, "curvature_initial")
    curvature_prediction = posterior_variable_samples(fit, "curvature_prediction")
    radius_prediction = posterior_variable_samples(fit, "radius_prediction")
    initial_left_probability = float(np.mean(curvature_initial > 0))
    horizon_left_probability = float(np.mean(curvature_prediction[:, -1] > 0))
    print("\nPosterior turn direction:")
    print(
        "Initial left / right : "
        f"{initial_left_probability:.1%} / {1 - initial_left_probability:.1%}"
    )
    print(
        "Horizon left / right : "
        f"{horizon_left_probability:.1%} / {1 - horizon_left_probability:.1%}"
    )

    print("\nPosterior median development:")
    print(
        "Curvature [1/m]: "
        f"{np.median(curvature_prediction[:, 0]):.6f} -> "
        f"{np.median(curvature_prediction[:, -1]):.6f}"
    )
    print(
        "Radius [m]     : "
        f"{np.median(radius_prediction[:, 0]):.1f} -> "
        f"{np.median(radius_prediction[:, -1]):.1f}"
    )

    evaluation = evaluate_position_predictions(fit, window)
    print_position_evaluation(evaluation)

    plot_prediction(window, fit, model_name="Time-Varying-Radius")


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
