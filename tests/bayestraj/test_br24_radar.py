"""Tests for BR24 radar preprocessing without reference-trajectory input."""

import numpy as np
import pandas as pd
import pytest

from bayestraj.observations.br24 import (
    ASSOCIATION_GATE_NIS,
    associate_br24_detections,
    export_br24_target_radar_trajectory,
    export_br24_target_radar_with_ground_truth,
    polar_to_cartesian,
    position_measurement_noise,
)


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
