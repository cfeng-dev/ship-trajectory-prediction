"""Fit the Bayesian constant-radius model to one recorded trajectory window."""

from ship_trajectory_prediction.evaluation.metrics import (
    evaluate_position_predictions,
    print_position_evaluation,
)
from ship_trajectory_prediction.evaluation.plotting import plot_prediction
from ship_trajectory_prediction.evaluation.reporting import print_prediction_setup
from ship_trajectory_prediction.models.constant_radius import (
    build_stan_data,
    fit_constant_radius_model,
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
RADIUS_PRIOR_LOG_SD = 1.0
SIGMA_PRIOR_SCALE = 20.0


def main():
    """Fit the model and plot posterior trajectories against held-out data."""
    trajectory_data = read_ship_data(DATA_FILE, run_id=RUN_ID)
    window = prepare_trajectory_window(
        trajectory_data,
        observation_count=OBSERVATION_COUNT,
        prediction_count=PREDICTION_COUNT,
        start_index=START_INDEX,
    )
    model_kwargs = {
        "radius_prior_median": RADIUS_PRIOR_MEDIAN,
        "radius_prior_log_sd": RADIUS_PRIOR_LOG_SD,
        "sigma_prior_scale": SIGMA_PRIOR_SCALE,
    }
    stan_data = build_stan_data(window, **model_kwargs)

    print_prediction_setup(
        "Bayesian Constant-Radius Trajectory Prediction",
        data_file=DATA_FILE,
        run_id=RUN_ID,
        window=window,
        extra_rows=[
            ("Estimated speed", f"{stan_data['speed']:.2f} m/s"),
            ("Turn direction", f"{stan_data['turn_direction']:+d}"),
        ],
    )

    fit = fit_constant_radius_model(window, **model_kwargs)

    print("\nPosterior parameter summary:")
    print(fit.summary().loc[["radius", "sigma"]])

    evaluation = evaluate_position_predictions(fit, window)
    print_position_evaluation(evaluation)

    plot_prediction(window, fit, model_name="Constant-Radius")


if __name__ == "__main__":
    main()
