"""Tests for the position-only Bayesian Position Model prior calibration."""

import numpy as np
import pandas as pd
import pytest

import experiments.data_exploration.calibrate_bayesian_position_model_priors as calibration
import ship_trajectory_prediction.observations.coordinates as coordinates


def test_calibration_uses_only_positions_and_returns_positive_scales():
    """No speed column should be required by the position-only calibration."""
    rows = []
    for run_id, angle_step in ((0, 0.01), (1, 0.03)):
        x_values = [0.0]
        y_values = [0.0]
        for step in range(1, 9):
            distance = 8.0 + 0.2 * (step % 3)
            angle = angle_step * step + 0.003 * (step % 2)
            x_values.append(x_values[-1] + distance * np.cos(angle))
            y_values.append(y_values[-1] + distance * np.sin(angle))
        longitude, latitude = coordinates.local_to_gps_coordinates(
            np.asarray(x_values),
            np.asarray(y_values),
            reference_longitude=8.0,
            reference_latitude=53.0,
            unit="m",
        )
        for index in range(len(x_values)):
            rows.append(
                {
                    "time": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.to_timedelta(10 * index, unit="s"),
                    "run_id": run_id,
                    "gps_longitude": longitude[index],
                    "gps_latitude": latitude[index],
                }
            )

    result = calibration.calibrate_position_model_priors(pd.DataFrame(rows))

    assert result.sample_count == 14
    assert result.log_displacement_scale_prior_scale > 0
    assert result.rotation_angle_prior_scale > 0
    assert result.displacement_residual_empirical_scale_m > 0
    assert result.sigma_displacement_residual_prior_scale_m == pytest.approx(
        result.displacement_residual_empirical_scale_m
        / calibration.HALF_NORMAL_MEDIAN_FACTOR
    )


def test_robust_scale_matches_gaussian_consistent_mad():
    """The calibration scale should use one transparent robust formula."""
    values = np.asarray([-2.0, -1.0, 0.0, 1.0, 5.0])

    assert calibration._robust_scale(values) == pytest.approx(1.4826)
