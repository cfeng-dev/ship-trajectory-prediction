"""Fit the Bayesian time-varying-radius model to one trajectory window."""

import numpy as np

from ship_trajectory_prediction.evaluation.metrics import (
    evaluate_position_predictions,
    print_position_evaluation,
)
from ship_trajectory_prediction.evaluation.plotting import plot_prediction
from ship_trajectory_prediction.evaluation.reporting import print_prediction_setup
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

RADIUS_PRIOR_MEDIAN = 500.0
CURVATURE_INITIAL_PRIOR_SCALE = 0.002
CURVATURE_RATE_PRIOR_SCALE = 5e-6
SIGMA_PRIOR_SCALE = 20.0
INTEGRATION_SUBSTEPS = 4


def main():
    """Fit the model and evaluate posterior predictions on held-out data."""
    trajectory_data = read_ship_data(DATA_FILE, run_id=RUN_ID)
    window = prepare_trajectory_window(
        trajectory_data,
        observation_count=OBSERVATION_COUNT,
        prediction_count=PREDICTION_COUNT,
        start_index=START_INDEX,
    )
    model_kwargs = {
        "radius_prior_median": RADIUS_PRIOR_MEDIAN,
        "curvature_initial_prior_scale": CURVATURE_INITIAL_PRIOR_SCALE,
        "curvature_rate_prior_scale": CURVATURE_RATE_PRIOR_SCALE,
        "sigma_prior_scale": SIGMA_PRIOR_SCALE,
        "integration_substeps": INTEGRATION_SUBSTEPS,
    }
    stan_data = build_stan_data(window, **model_kwargs)

    direction = np.sign(stan_data["curvature_prior_mean"])
    print_prediction_setup(
        "Bayesian Time-Varying-Radius Prediction",
        data_file=DATA_FILE,
        run_id=RUN_ID,
        window=window,
        extra_rows=[
            ("Fixed speed", f"{stan_data['speed']:.2f} m/s"),
            ("Turn direction", f"{direction:+.0f}"),
        ],
    )

    fit = fit_time_varying_radius_model(window, **model_kwargs)

    print("\nPosterior parameter summary:")
    print(
        fit.summary().loc[
            [
                "curvature_initial",
                "curvature_rate",
                "radius_initial",
                "radius_horizon",
                "sigma",
            ]
        ]
    )

    curvature_prediction = fit.stan_variable("curvature_prediction")
    radius_prediction = fit.stan_variable("radius_prediction")
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


if __name__ == "__main__":
    main()
