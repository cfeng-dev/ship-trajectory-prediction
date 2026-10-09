"""Tests for deterministic CTRV rolling evaluation input handling."""

import numpy as np
import pandas as pd

import bayestraj.forecasting.deterministic_ctrv as forecasting
import bayestraj.forecasting.deterministic_ctrv_workflow as single_workflow
import bayestraj.validation.deterministic_ctrv_workflow as workflow


def test_deterministic_rolling_configuration_uses_processed_trajectory_contract():
    config = forecasting.DeterministicRollingExperimentConfig(
        window_mode="sliding",
        observation_count=3,
        prediction_count=1,
        position_noise_std_m=0.0,
        position_noise_seed=2026,
        stride=None,
    )

    assert not hasattr(config, "run_id")


def test_deterministic_processed_input_keeps_local_positions_and_numeric_time(
    tmp_path,
):
    data_file = tmp_path / "processed_trajectory.csv"
    pd.DataFrame(
        {
            "time": [0.0, 2.5, 7.0, 11.0],
            "x": [100.0, 103.0, 107.0, 110.0],
            "y": [-20.0, -18.0, -15.0, -13.0],
        }
    ).to_csv(data_file, index=False)

    trajectory_data = workflow._load_evaluation_trajectory(data_file)
    window = workflow._prepare_evaluation_window(
        trajectory_data,
        observation_count=3,
        prediction_count=1,
        start_index=0,
    )
    route_x, route_y = workflow._prepare_route_coordinates(trajectory_data)

    np.testing.assert_allclose(window.x_meters, [100.0, 103.0, 107.0, 110.0])
    np.testing.assert_allclose(window.y_meters, [-20.0, -18.0, -15.0, -13.0])
    np.testing.assert_allclose(route_x, [100.0, 103.0, 107.0, 110.0])
    np.testing.assert_allclose(route_y, [-20.0, -18.0, -15.0, -13.0])


def test_deterministic_single_run_uses_processed_trajectory(tmp_path):
    data_file = tmp_path / "processed_trajectory.csv"
    pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    ).to_csv(data_file, index=False)
    experiment = forecasting.DeterministicExperimentConfig(
        start_index=0,
        observation_count=3,
        prediction_count=2,
        position_noise_std_m=0.0,
        position_noise_seed=2026,
    )

    prediction_table = single_workflow.run_deterministic_ctrv_prediction(
        data_file=data_file,
        experiment=experiment,
        position_noise_std_m=0.0,
        position_noise_seed=2026,
        show_plot=False,
        show_time_labels=False,
    )

    np.testing.assert_allclose(prediction_table["x_actual"], [3.0, 4.0])
    np.testing.assert_allclose(prediction_table["y_actual"], [0.0, 0.0])
