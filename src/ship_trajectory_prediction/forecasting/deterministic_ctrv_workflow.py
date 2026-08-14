"""Single-window deterministic CTRV prediction workflow."""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ship_trajectory_prediction.forecasting.deterministic_ctrv import (
    DeterministicExperimentConfig,
    build_prediction_table,
    estimate_ctrv_state,
)
from ship_trajectory_prediction.observations import (
    TrajectoryWindowData,
    prepare_trajectory_window,
    read_ship_data,
)
from ship_trajectory_prediction.validation.prediction_plotting import (
    plot_trajectory_paths,
)
from ship_trajectory_prediction.validation.reporting import print_prediction_setup

DETERMINISTIC_PLOT_TITLE = (
    "Deterministisches CTRV-Modell auf Basis von Positionsdaten:\n"
    "Trajektorienvorhersage"
)


def run_deterministic_ctrv_prediction(
    *,
    data_file,
    experiment: DeterministicExperimentConfig,
    speed_estimation_points: int,
    heading_estimation_segments: int,
    position_noise_std_m: float,
    position_noise_seed: int,
    show_plot: bool,
) -> pd.DataFrame:
    """Estimate and evaluate one deterministic CTRV prediction."""
    trajectory_data = read_ship_data(data_file, run_id=experiment.run_id)
    window = prepare_trajectory_window(
        trajectory_data,
        observation_count=experiment.observation_count,
        prediction_count=experiment.prediction_count,
        start_index=experiment.start_index,
    )
    window = _add_position_observation_noise(
        window,
        additional_noise_std_m=position_noise_std_m,
        seed=position_noise_seed,
    )
    initial_state = estimate_ctrv_state(
        window,
        speed_estimation_points=speed_estimation_points,
        heading_estimation_segments=heading_estimation_segments,
    )
    prediction_table = build_prediction_table(window, initial_state)

    print_prediction_setup(
        "Deterministic CTRV Trajectory Prediction",
        data_file=data_file,
        run_id=experiment.run_id,
        window=window,
        extra_rows=[
            (
                "Additional position noise",
                (
                    f"{position_noise_std_m:g} m (seed={position_noise_seed})"
                    if position_noise_std_m > 0
                    else "disabled"
                ),
            ),
            ("Estimated speed", f"{initial_state.speed:.3f} m/s"),
            ("Estimated heading", f"{initial_state.heading:.5f} rad"),
            ("Estimated turn rate", f"{initial_state.turn_rate:.6f} rad/s"),
        ],
    )
    print("\nHeld-out prediction:")
    print(prediction_table.round(3).to_string(index=False))
    print(f"\nADE: {prediction_table['position_error_m'].mean():.2f} m")
    print(f"FDE: {prediction_table['position_error_m'].iloc[-1]:.2f} m")

    if show_plot:
        plot_deterministic_ctrv_prediction(
            window,
            prediction_table,
            additional_position_noise_std_m=position_noise_std_m,
        )
    return prediction_table


def plot_deterministic_ctrv_prediction(
    window: TrajectoryWindowData,
    prediction_table: pd.DataFrame,
    *,
    additional_position_noise_std_m=0.0,
):
    """Plot observed, held-out, and deterministic CTRV trajectories."""
    observed = window.observed_slice
    prediction_start_index = window.observation_count - 1
    start_x = window.x_meters[prediction_start_index]
    start_y = window.y_meters[prediction_start_index]

    held_out_x = np.concatenate(([start_x], prediction_table["x_actual"]))
    held_out_y = np.concatenate(([start_y], prediction_table["y_actual"]))
    predicted_x = np.concatenate(([start_x], prediction_table["x_predicted"]))
    predicted_y = np.concatenate(([start_y], prediction_table["y_predicted"]))

    observed_label = (
        "Verrauschte Beobachtungen"
        if additional_position_noise_std_m > 0
        else "Beobachtungen"
    )
    figure, axis = plot_trajectory_paths(
        observed_path=(
            window.x_meters[observed],
            window.y_meters[observed],
        ),
        reference_path=(held_out_x, held_out_y),
        forecast_paths=((predicted_x, predicted_y),),
        prediction_origins=([start_x], [start_y]),
        title=DETERMINISTIC_PLOT_TITLE,
        observed_label=observed_label,
        reference_label="Referenztrajektorie",
        forecast_label="Deterministische CTRV-Vorhersage",
        prediction_origin_label="Prognosebeginn",
    )
    plt.show()
    return figure, axis


def _add_position_observation_noise(
    window: TrajectoryWindowData,
    *,
    additional_noise_std_m: float,
    seed: int,
) -> TrajectoryWindowData:
    """Return a window with reproducible noise on observed positions only."""
    if (
        isinstance(additional_noise_std_m, bool)
        or not np.isfinite(additional_noise_std_m)
        or additional_noise_std_m < 0
    ):
        raise ValueError("additional_noise_std_m must be finite and non-negative.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    x_meters = np.asarray(window.x_meters, dtype=float).copy()
    y_meters = np.asarray(window.y_meters, dtype=float).copy()
    if additional_noise_std_m > 0:
        generator = np.random.default_rng(int(seed))
        observed_count = window.observation_count
        x_meters[window.observed_slice] += generator.normal(
            0.0,
            additional_noise_std_m,
            observed_count,
        )
        y_meters[window.observed_slice] += generator.normal(
            0.0,
            additional_noise_std_m,
            observed_count,
        )
    return replace(window, x_meters=x_meters, y_meters=y_meters)
