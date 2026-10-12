"""Tests for BR24 radar preprocessing without reference-trajectory input."""

import numpy as np
import pandas as pd
import pytest

from bayestraj.observations.br24 import (
    ASSOCIATION_GATE_NIS,
    associate_br24_detections,
    br24_measurement_covariance,
    br24_measurement_position,
    export_br24_target_radar_trajectory,
    export_br24_target_radar_with_ground_truth,
    export_br24_tracking_input,
    export_br24_tracking_input_with_reference,
    polar_to_cartesian,
    position_measurement_noise,
    read_br24_tracking_input,
)


def test_br24_measurement_position_and_covariance_use_only_sensor_and_detection():
    position = br24_measurement_position(
        sensor_position=np.array([100.0, -20.0]),
        range_m=10.0,
        azimuth_rad=0.0,
    )
    covariance = br24_measurement_covariance(
        range_m=10.0,
        azimuth_rad=0.0,
        std_range_m=2.0,
        std_azimuth_rad=0.0,
    )

    np.testing.assert_allclose(position, [110.0, -20.0])
    np.testing.assert_allclose(
        covariance,
        [[4.0, 0.0], [0.0, (10.0 * np.deg2rad(0.5)) ** 2]],
    )


def test_br24_tracking_input_flattens_sensor_cycles_and_raw_detections(tmp_path):
    radar_csv = tmp_path / "BR24_radar_detections.csv"
    sensor_csv = tmp_path / "BR24_sensor_state.csv"
    output_csv = tmp_path / "BR24Dataset20131107.csv"
    pd.DataFrame(
        {
            "Time": [11.0, 10.0, 11.0],
            "CycleCounter": [2, 1, 2],
            "DetectionID": [2, 1, 1],
            "Range": [20.0, 10.0, 15.0],
            "Azimuth": [0.2, 0.1, 0.15],
            "StdRange": [2.3, 2.3, 2.3],
            "StdAzimuth": [0.03, 0.03, 0.03],
            "XRelative": [19.6, 9.9, 14.8],
            "YRelative": [4.0, 1.0, 2.2],
        }
    ).to_csv(radar_csv, index=False)
    pd.DataFrame(
        {
            "Time": [12.0, 10.0, 11.0],
            "CycleCounter": [3, 1, 2],
            "X": [102.0, 100.0, 101.0],
            "Y": [-2.0, 0.0, -1.0],
            "Yaw": [0.3, 0.1, 0.2],
        }
    ).to_csv(sensor_csv, index=False)

    output_path = export_br24_tracking_input(
        radar_csv=radar_csv,
        sensor_csv=sensor_csv,
        output_csv=output_csv,
    )

    exported = pd.read_csv(output_path)

    assert output_path == output_csv
    assert exported.columns.tolist() == [
        "time",
        "cycle_counter",
        "sensor_x",
        "sensor_y",
        "sensor_yaw",
        "detection_id",
        "range",
        "azimuth",
        "std_range",
        "std_azimuth",
        "x_relative",
        "y_relative",
    ]
    assert exported["cycle_counter"].tolist() == [1, 2, 2, 3]
    assert exported["detection_id"].tolist()[:3] == [1.0, 1.0, 2.0]
    assert np.isnan(exported["detection_id"].iloc[3])
    np.testing.assert_allclose(
        exported.loc[:2, ["sensor_x", "sensor_y", "sensor_yaw"]],
        [[100.0, 0.0, 0.1], [101.0, -1.0, 0.2], [101.0, -1.0, 0.2]],
    )
    assert exported.loc[3, ["range", "azimuth"]].isna().all()


def test_read_br24_tracking_input_groups_all_detections_by_sensor_cycle(tmp_path):
    csv_path = tmp_path / "BR24Dataset20131107.csv"
    pd.DataFrame(
        {
            "time": [10.0, 10.0, 11.0],
            "cycle_counter": [1, 1, 2],
            "sensor_x": [100.0, 100.0, 101.0],
            "sensor_y": [-20.0, -20.0, -21.0],
            "sensor_yaw": [0.1, 0.1, 0.2],
            "detection_id": [2, 4, np.nan],
            "range": [10.0, 20.0, np.nan],
            "azimuth": [0.0, 0.2, np.nan],
            "std_range": [2.3, 2.3, np.nan],
            "std_azimuth": [0.03, 0.03, np.nan],
            "x_relative": [10.0, 19.6, np.nan],
            "y_relative": [0.0, 4.0, np.nan],
        }
    ).to_csv(csv_path, index=False)

    cycles = read_br24_tracking_input(csv_path)

    assert [cycle.cycle_counter for cycle in cycles] == [1, 2]
    assert cycles[0].time_seconds == 10.0
    np.testing.assert_allclose(cycles[0].sensor_position, [100.0, -20.0])
    assert [detection.detection_id for detection in cycles[0].detections] == [2, 4]
    assert cycles[1].detections == ()


def test_polar_to_cartesian_matches_reuter_measurement_formation():
    positions = polar_to_cartesian(
        np.array([5.0, 4.0]),
        np.array([np.pi / 2.0, np.pi]),
    )

    np.testing.assert_allclose(positions, [[0.0, 5.0], [-4.0, 0.0]], atol=1e-12)


def test_position_measurement_noise_keeps_the_cartesian_cross_covariance():
    covariance = position_measurement_noise(
        predicted_position=np.array([3.0, 4.0]),
        range_variance=4.0,
        azimuth_variance=0.01,
    )

    np.testing.assert_allclose(covariance, [[1.6, 1.8], [1.8, 2.65]])
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance)[0] >= -1e-12
    assert covariance[0, 1] != 0.0


def test_associate_br24_detections_selects_the_nearest_detection_and_rejects_gate():
    detections = np.array([[3.0, 0.0], [0.5, 0.5]])
    result = associate_br24_detections(
        predicted_position=np.zeros(2),
        innovation_covariance=np.eye(2),
        detections=detections,
    )

    assert result is not None
    assert result.detection_index == 1
    assert result.nis == pytest.approx(0.5)
    assert result.nis < ASSOCIATION_GATE_NIS
    assert (
        associate_br24_detections(
            predicted_position=np.zeros(2),
            innovation_covariance=np.eye(2),
            detections=np.array([[4.0, 0.0]]),
        )
        is None
    )


def test_br24_export_uses_radar_and_sensor_csvs_without_target_reference(tmp_path):
    radar_csv = tmp_path / "BR24_radar_detections.csv"
    sensor_csv = tmp_path / "BR24_sensor_state.csv"
    output_csv = tmp_path / "BR24_target_radar_trajectory.csv"
    pd.DataFrame(
        {
            "Time": [0.0, 1.0, 1.0, 2.0],
            "CycleCounter": [1, 2, 2, 3],
            "DetectionID": [1, 1, 2, 1],
            "Range": [10.0, 100.0, 9.0, 1000.0],
            "Azimuth": [0.0, 0.0, 0.0, 0.0],
            "StdRange": [2.3, 2.3, 2.3, 2.3],
            "StdAzimuth": [np.deg2rad(2.0)] * 4,
            "XRelative": [-999.0, -999.0, -999.0, -999.0],
            "YRelative": [-999.0, -999.0, -999.0, -999.0],
        }
    ).to_csv(radar_csv, index=False)
    pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0],
            "CycleCounter": [1, 2, 3],
            "X": [10.0, 11.0, 12.0],
            "Y": [0.0, 0.0, 0.0],
            "Yaw": [1.0, 2.0, 3.0],
        }
    ).to_csv(sensor_csv, index=False)

    output_path = export_br24_target_radar_trajectory(
        radar_csv=radar_csv,
        sensor_csv=sensor_csv,
        output_csv=output_csv,
    )

    exported = pd.read_csv(output_path)
    assert exported.columns.tolist() == [
        "time",
        "cycle_counter",
        "detection_id",
        "x",
        "y",
        "cov_xx",
        "cov_xy",
        "cov_yy",
    ]
    assert exported["cycle_counter"].tolist() == [2]
    assert exported["detection_id"].tolist() == [2]
    np.testing.assert_allclose(exported[["x", "y"]], [[20.0, 0.0]])


def test_br24_radar_export_adds_interpolated_ground_truth_without_changing_radar(
    tmp_path,
):
    radar_csv = tmp_path / "BR24_target_radar_trajectory.csv"
    reference_csv = tmp_path / "BR24_target_reference_trajectory.csv"
    output_csv = tmp_path / "BR24_target_radar_trajectory_with_ground_truth.csv"
    radar = pd.DataFrame(
        {
            "time": [0.5, 1.5],
            "cycle_counter": [2, 3],
            "detection_id": [4, 5],
            "x": [12.0, 24.0],
            "y": [-3.0, -9.0],
            "cov_xx": [1.0, 2.0],
            "cov_xy": [0.1, 0.2],
            "cov_yy": [3.0, 4.0],
        }
    )
    radar.to_csv(radar_csv, index=False)
    pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "x": [0.0, 10.0, 20.0],
            "y": [0.0, -4.0, -8.0],
            "yaw": [2.8, -3.0, -2.5],
            "velocity": [1.0, 3.0, 5.0],
        }
    ).to_csv(reference_csv, index=False)

    output_path = export_br24_target_radar_with_ground_truth(
        radar_csv=radar_csv,
        reference_csv=reference_csv,
        output_csv=output_csv,
    )

    exported = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(exported.loc[:, radar.columns], radar)
    assert exported.columns.tolist() == [
        *radar.columns,
        "ground_truth_x",
        "ground_truth_y",
        "ground_truth_yaw",
        "ground_truth_velocity",
    ]
    np.testing.assert_allclose(exported["ground_truth_x"], [5.0, 15.0])
    np.testing.assert_allclose(exported["ground_truth_y"], [-2.0, -6.0])
    np.testing.assert_allclose(exported["ground_truth_velocity"], [2.0, 4.0])
    assert exported.loc[0, "ground_truth_yaw"] == pytest.approx(3.041592653589793)


def test_br24_tracking_input_export_adds_evaluation_reference_without_changing_radar(
    tmp_path,
):
    tracking_csv = tmp_path / "BR24Dataset20131107.csv"
    reference_csv = tmp_path / "BR24_reference_trajectory.csv"
    output_csv = tmp_path / "BR24Dataset20131107_with_reference.csv"
    tracking_input = pd.DataFrame(
        {
            "time": [0.5, 0.5, 1.5],
            "cycle_counter": [4, 4, 5],
            "sensor_x": [100.0, 100.0, 101.0],
            "sensor_y": [-20.0, -20.0, -19.0],
            "sensor_yaw": [0.0, 0.0, 0.1],
            "detection_id": [1, 2, 1],
            "range": [10.0, 20.0, 11.0],
            "azimuth": [0.0, 0.1, 0.0],
            "std_range": [0.5, 0.5, 0.6],
            "std_azimuth": [0.01, 0.01, 0.02],
            "x_relative": [10.0, 19.9, 11.0],
            "y_relative": [0.0, 2.0, 0.0],
        }
    )
    tracking_input.to_csv(tracking_csv, index=False)
    pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "x": [0.0, 10.0, 20.0],
            "y": [0.0, -4.0, -8.0],
            "yaw": [2.8, -3.0, -2.5],
            "velocity": [1.0, 3.0, 5.0],
        }
    ).to_csv(reference_csv, index=False)

    output_path = export_br24_tracking_input_with_reference(
        tracking_csv=tracking_csv,
        reference_csv=reference_csv,
        output_csv=output_csv,
    )

    exported = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(
        exported.loc[:, tracking_input.columns], tracking_input
    )
    assert exported.columns.tolist() == [
        *tracking_input.columns,
        "ground_truth_x",
        "ground_truth_y",
        "ground_truth_yaw",
        "ground_truth_velocity",
    ]
    np.testing.assert_allclose(exported["ground_truth_x"], [5.0, 5.0, 15.0])
    np.testing.assert_allclose(exported["ground_truth_y"], [-2.0, -2.0, -6.0])
    np.testing.assert_allclose(exported["ground_truth_velocity"], [2.0, 2.0, 4.0])
    assert exported.loc[0, "ground_truth_yaw"] == pytest.approx(3.041592653589793)

    cycles = read_br24_tracking_input(output_path)

    assert [cycle.cycle_counter for cycle in cycles] == [4, 5]
    assert [detection.detection_id for detection in cycles[0].detections] == [1, 2]


def test_br24_radar_export_rejects_ground_truth_extrapolation(tmp_path):
    radar_csv = tmp_path / "BR24_target_radar_trajectory.csv"
    reference_csv = tmp_path / "BR24_target_reference_trajectory.csv"
    pd.DataFrame(
        {
            "time": [2.5],
            "cycle_counter": [2],
            "detection_id": [4],
            "x": [12.0],
            "y": [-3.0],
            "cov_xx": [1.0],
            "cov_xy": [0.1],
            "cov_yy": [3.0],
        }
    ).to_csv(radar_csv, index=False)
    pd.DataFrame(
        {
            "time": [0.0, 2.0],
            "x": [0.0, 20.0],
            "y": [0.0, -8.0],
            "yaw": [0.0, 0.0],
            "velocity": [1.0, 5.0],
        }
    ).to_csv(reference_csv, index=False)

    with pytest.raises(ValueError, match="outside the ground-truth time range"):
        export_br24_target_radar_with_ground_truth(
            radar_csv=radar_csv,
            reference_csv=reference_csv,
            output_csv=tmp_path / "output.csv",
        )


def test_br24_radar_export_ignores_incomplete_ground_truth_outside_radar_times(
    tmp_path,
):
    radar_csv = tmp_path / "BR24_target_radar_trajectory.csv"
    reference_csv = tmp_path / "BR24_target_reference_trajectory.csv"
    pd.DataFrame(
        {
            "time": [10.5],
            "cycle_counter": [2],
            "detection_id": [4],
            "x": [12.0],
            "y": [-3.0],
            "cov_xx": [1.0],
            "cov_xy": [0.1],
            "cov_yy": [3.0],
        }
    ).to_csv(radar_csv, index=False)
    pd.DataFrame(
        {
            "time": [0.0, 1.0, 10.0, 11.0],
            "x": [np.nan, 1.0, 10.0, 20.0],
            "y": [np.nan, -1.0, -10.0, -20.0],
            "yaw": [np.nan, 0.1, 0.2, 0.3],
            "velocity": [np.nan, 1.0, 2.0, 3.0],
        }
    ).to_csv(reference_csv, index=False)

    exported = pd.read_csv(
        export_br24_target_radar_with_ground_truth(
            radar_csv=radar_csv,
            reference_csv=reference_csv,
            output_csv=tmp_path / "output.csv",
        )
    )

    np.testing.assert_allclose(exported["ground_truth_x"], [15.0])


def test_br24_radar_export_rejects_interpolation_through_incomplete_ground_truth(
    tmp_path,
):
    radar_csv = tmp_path / "BR24_target_radar_trajectory.csv"
    reference_csv = tmp_path / "BR24_target_reference_trajectory.csv"
    pd.DataFrame(
        {
            "time": [0.5],
            "cycle_counter": [2],
            "detection_id": [4],
            "x": [12.0],
            "y": [-3.0],
            "cov_xx": [1.0],
            "cov_xy": [0.1],
            "cov_yy": [3.0],
        }
    ).to_csv(radar_csv, index=False)
    pd.DataFrame(
        {
            "time": [0.0, 0.25, 1.0],
            "x": [0.0, np.nan, 10.0],
            "y": [0.0, np.nan, -4.0],
            "yaw": [0.0, np.nan, 0.2],
            "velocity": [1.0, np.nan, 3.0],
        }
    ).to_csv(reference_csv, index=False)

    with pytest.raises(ValueError, match="incomplete ground-truth interval"):
        export_br24_target_radar_with_ground_truth(
            radar_csv=radar_csv,
            reference_csv=reference_csv,
            output_csv=tmp_path / "output.csv",
        )


def test_br24_radar_export_preserves_precise_radar_values(tmp_path):
    radar_csv = tmp_path / "BR24_target_radar_trajectory.csv"
    reference_csv = tmp_path / "BR24_target_reference_trajectory.csv"
    output_csv = tmp_path / "output.csv"
    radar_csv.write_text(
        "time,cycle_counter,detection_id,x,y,cov_xx,cov_xy,cov_yy\n"
        "0.5,2,4,12.0,-3.0,1.7363816676561725,0.1,3.0\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "x": [0.0, 10.0],
            "y": [0.0, -4.0],
            "yaw": [0.0, 0.2],
            "velocity": [1.0, 3.0],
        }
    ).to_csv(reference_csv, index=False)

    export_br24_target_radar_with_ground_truth(
        radar_csv=radar_csv,
        reference_csv=reference_csv,
        output_csv=output_csv,
    )

    assert output_csv.read_text(encoding="utf-8").splitlines()[1].split(",")[5] == (
        "1.7363816676561725"
    )
