"""Fit the Bayesian constant-radius model to one recorded trajectory window."""

from ship_trajectory_prediction.evaluation.metrics import (
    evaluate_position_predictions,
    print_position_evaluation,
)
from ship_trajectory_prediction.evaluation.plotting import (
    plot_constant_radius_prediction,
)
from ship_trajectory_prediction.models.constant_radius import (
    fit_constant_radius_model,
    prepare_trajectory_window,
)
from ship_trajectory_prediction.paths import project_path
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

    print("=" * 60)
    print("Bayesian Constant-Radius Trajectory Prediction")
    print("=" * 60)
    print(f"Data file          : {DATA_FILE}")
    print(f"Run ID             : {RUN_ID}")
    print(f"Observed positions : {window.observation_count}")
    print(f"Predicted positions: {window.prediction_count}")
    print(f"Estimated speed    : {window.speed_mps:.2f} m/s")
    print(f"Turn direction     : {window.turn_direction:+d}")

    fit = fit_constant_radius_model(
        window,
        radius_prior_median=RADIUS_PRIOR_MEDIAN,
        radius_prior_log_sd=RADIUS_PRIOR_LOG_SD,
        sigma_prior_scale=SIGMA_PRIOR_SCALE,
    )

    print("\nPosterior parameter summary:")
    print(fit.summary().loc[["radius", "sigma"]])

    evaluation = evaluate_position_predictions(fit, window)
    print_position_evaluation(evaluation)

    plot_constant_radius_prediction(window, fit)


if __name__ == "__main__":
    main()
