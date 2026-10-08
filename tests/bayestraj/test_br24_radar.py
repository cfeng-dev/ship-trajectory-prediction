"""Tests for BR24 radar preprocessing without reference-trajectory input."""

import numpy as np
import pandas as pd
import pytest

from bayestraj.observations.br24 import (
    ASSOCIATION_GATE_NIS,
    associate_br24_detections,
    export_br24_target_radar_trajectory,
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
