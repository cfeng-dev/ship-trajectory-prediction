"""Fit the Bayesian time-varying motion model to one trajectory window."""

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
from ship_trajectory_prediction.models.time_varying_motion import (
    build_stan_data,
    fit_time_varying_motion_model,
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
ACCELERATION_INITIAL_SCALE = 0.1
ACCELERATION_STATE_SCALE = 0.02
ACCELERATION_DECAY_TIME = 60.0
TURN_RATE_INITIAL_SCALE = 0.01
TURN_RATE_STATE_SCALE = 0.003
TURN_RATE_DECAY_TIME = 600.0
SIGMA_POSITION = 5.0
SIGMA_SPEED = 0.2
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
        "speed_prior_log_sd": SPEED_PRIOR_LOG_SD,
        "heading_prior_scale": HEADING_PRIOR_SCALE,
        "acceleration_initial_scale": ACCELERATION_INITIAL_SCALE,
        "acceleration_state_scale": ACCELERATION_STATE_SCALE,
        "acceleration_decay_time": ACCELERATION_DECAY_TIME,
        "turn_rate_initial_scale": TURN_RATE_INITIAL_SCALE,
        "turn_rate_state_scale": TURN_RATE_STATE_SCALE,
        "turn_rate_decay_time": TURN_RATE_DECAY_TIME,
        "sigma_position": SIGMA_POSITION,
        "sigma_speed": SIGMA_SPEED,
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
        "Bayesian Time-Varying Motion Prediction",
        data_file=DATA_FILE,
        run_id=RUN_ID,
        window=window,
        extra_rows=[
            ("Inference method", inference_method.upper()),
            ("VI algorithm", vi_algorithm if inference_method == "vi" else "-"),
            ("Turn-rate level", f"{stan_data['turn_rate_level']:.5f} rad/s"),
            ("Position noise", f"{SIGMA_POSITION:.2f} m"),
            ("Speed noise", f"{SIGMA_SPEED:.2f} m/s"),
        ],
    )

    fit = fit_time_varying_motion_model(
        window,
        **model_kwargs,
        inference_method=inference_method,
        variational_options=variational_options,
    )

    if inference_method == "vi":
        print_variational_diagnostics(fit)

    print("\nPosterior initial-state summary:")
    print(
        posterior_parameter_summary(
            fit,
            [
                "speed_initial",
                "heading_initial",
                "acceleration_initial",
                "turn_rate_initial",
            ],
        )
    )

    acceleration_state = posterior_variable_samples(fit, "acceleration_state")
    turn_rate_state = posterior_variable_samples(fit, "turn_rate_state")
    speed_state = posterior_variable_samples(fit, "speed_state")
    speed_prediction = posterior_variable_samples(fit, "speed_prediction_mean")
    print("\nPosterior state medians:")
    print(
        "Acceleration [m/s^2]: "
        f"{np.median(acceleration_state[:, 0]):.4f} -> "
        f"{np.median(acceleration_state[:, -1]):.4f}"
    )
    print(
        "Turn rate [rad/s]   : "
        f"{np.median(turn_rate_state[:, 0]):.5f} -> "
        f"{np.median(turn_rate_state[:, -1]):.5f}"
    )
    print(
        "Speed [m/s]         : "
        f"{np.median(speed_state[:, 0]):.3f} -> "
        f"{np.median(speed_state[:, -1]):.3f}"
    )
    physically_valid_speed = np.all(
        (speed_state > 0.001) & (speed_state <= 100),
        axis=1,
    ) & np.all((speed_prediction > 0.001) & (speed_prediction <= 100), axis=1)
    print(f"Physically valid speed paths: {np.mean(physically_valid_speed):.1%}")

    evaluation = evaluate_position_predictions(fit, window)
    print_position_evaluation(evaluation)

    plot_prediction(window, fit, model_name="Time-Varying Motion")


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
