"""Raw-cycle covariance and missing-measurement contracts for batch CTRV."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bayestraj.inference.ctrv_cmdstan import _default_initial_values
from bayestraj.models.bayesian_ctrv import STAN_FILE, build_stan_data
from bayestraj.observations.position import PositionObservations
from bayestraj.observations.window import TrajectoryWindowData


def _observations():
    return PositionObservations(
        time_seconds=np.array([0.0, 1.5, 3.0, 4.0, 6.0]),
        x_meters=np.array([0.0, 999.0, 6.0, -999.0, 12.0]),
        y_meters=np.zeros(5),
        position_noise_std_m=0.0,
        noise_seed=42,
        measurement_covariances=np.broadcast_to([[4.0, 1.5], [1.5, 9.0]], (5, 2, 2)),
        observed_mask=np.array([True, False, True, False, True]),
    )


def _window(observations):
    times = np.append(observations.time_seconds, 8.0)
    return TrajectoryWindowData(
        timestamps=pd.to_datetime(times, unit="s", utc=True),
        time_seconds=times,
        x_meters=np.append(observations.x_meters, 0.0),
        y_meters=np.append(observations.y_meters, 0.0),
        reference_longitude=0.0,
        reference_latitude=0.0,
        gps_speed_mps=np.full(6, np.nan),
        observation_count=5,
    )


def test_covariances_and_mask_are_copied_and_immutable():
    observations = _observations()
    covariances = observations.measurement_covariances.copy()
    mask = observations.observed_mask.copy()
    observations = replace(
        observations, measurement_covariances=covariances, observed_mask=mask
    )
    covariances[:] = 0.0
    mask[:] = False
    assert not observations.measurement_covariances.flags.writeable
    assert not observations.observed_mask.flags.writeable
    assert observations.measurement_covariances[0, 0, 1] == 1.5
    assert np.count_nonzero(observations.observed_mask) == 3


@pytest.mark.parametrize(
    "mask", [np.ones(5), np.ones(4, dtype=bool), np.ones((5, 1), dtype=bool)]
)
def test_invalid_observed_masks_are_rejected(mask):
    with pytest.raises(ValueError, match="observed_mask"):
        replace(_observations(), observed_mask=mask)


def test_zero_and_rank_one_covariances_are_valid():
    covariance = np.broadcast_to([[4.0, 2.0], [2.0, 1.0]], (5, 2, 2))
    replace(_observations(), measurement_covariances=covariance)
    replace(_observations(), measurement_covariances=np.zeros((5, 2, 2)))


def test_missing_positions_require_finite_placeholders():
    with pytest.raises(ValueError, match="x_meters"):
        replace(_observations(), x_meters=np.array([0.0, np.nan, 6.0, 0.0, 12.0]))


@pytest.mark.parametrize(
    "covariance",
    [
        np.zeros((5, 2)),
        np.broadcast_to([[1.0, 2.0], [0.0, 1.0]], (5, 2, 2)),
        np.broadcast_to([[1.0, 2.0], [2.0, 1.0]], (5, 2, 2)),
        np.full((5, 2, 2), np.nan),
    ],
)
def test_invalid_measurement_covariances_are_rejected(covariance):
    with pytest.raises(ValueError, match="measurement_covariances"):
        replace(_observations(), measurement_covariances=covariance)


def test_batch_data_preserves_full_grid_mask_and_correlated_covariance():
    observations = _observations()
    data = build_stan_data(_window(observations), position_observations=observations)
    assert data["N_history"] == 5
    np.testing.assert_array_equal(data["time_observed"], observations.time_seconds)
    np.testing.assert_array_equal(data["observed_mask"], observations.observed_mask)
    np.testing.assert_array_equal(
        data["measurement_covariances"], observations.measurement_covariances
    )


def test_batch_initializers_ignore_missing_position_placeholders():
    observations = _observations()
    changed = replace(observations, x_meters=np.array([0.0, -1e9, 6.0, 1e9, 12.0]))
    first = _default_initial_values(
        build_stan_data(_window(observations), position_observations=observations),
        seed=7,
    )
    second = _default_initial_values(
        build_stan_data(_window(changed), position_observations=changed), seed=7
    )
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_batch_accepts_two_measurements_on_a_grid_with_missing_cycles():
    observations = replace(
        _observations(), observed_mask=np.array([True, False, False, False, True])
    )
    data = build_stan_data(_window(observations), position_observations=observations)

    assert data["N_history"] == 5
    np.testing.assert_array_equal(data["observed_mask"], [1, 0, 0, 0, 1])


def test_batch_rejects_a_single_accepted_measurement():
    observations = replace(
        _observations(), observed_mask=np.array([True, False, False, False, False])
    )
    with pytest.raises(ValueError, match="at least 2"):
        build_stan_data(_window(observations), position_observations=observations)


def test_zero_covariance_complete_data_retains_original_vectorized_likelihood():
    source = STAN_FILE.read_text(encoding="utf-8")
    assert "if (use_isotropic_observations == 1)" in source
    assert "x_observed ~ normal(x_state, sigma_position_observation);" in source
    assert "y_observed ~ normal(y_state, sigma_position_observation);" in source
    assert "x_observed[n] | x_state[n], sigma_position_observation" in source
    assert "y_observed[n] | y_state[n], sigma_position_observation" in source
