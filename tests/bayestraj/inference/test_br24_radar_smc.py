"""Tests for the BR24-specific Bayesian online radar tracker."""

import numpy as np
import pandas as pd

from bayestraj.inference.br24_radar_smc import (
    BR24BayesianRadarTracker,
    track_br24_csv,
)
from bayestraj.inference.ctrv_smc import SequentialMonteCarloCTRVConfig
from bayestraj.models.bayesian_ctrv import BayesianCTRVPriors
from bayestraj.observations.br24 import BR24Detection, BR24RadarCycle


def _cycle(time_seconds, cycle_counter, detections):
    return BR24RadarCycle(
        time_seconds=time_seconds,
        cycle_counter=cycle_counter,
        sensor_position=np.array([100.0, -20.0]),
        sensor_yaw_rad=0.0,
        detections=tuple(detections),
    )


def _detection(detection_id, range_m):
    return BR24Detection(
        detection_id=detection_id,
        range_m=range_m,
        azimuth_rad=0.0,
        std_range_m=0.5,
        std_azimuth_rad=0.0,
    )


def test_bayesian_br24_tracker_associates_online_and_continues_after_a_miss():
    first_cycle = _cycle(0.0, 1, [_detection(4, 10.0)])
    tracker = BR24BayesianRadarTracker.initialize(
        first_cycle,
        initial_detection_id=4,
        priors=BayesianCTRVPriors(),
        config=SequentialMonteCarloCTRVConfig(
            particle_count=128,
            posterior_draw_count=16,
            predictive_log_density_threshold=None,
        ),
        seed=42,
    )

    accepted = tracker.process_cycle(
        _cycle(1.0, 2, [_detection(5, 10.0), _detection(6, 200.0)])
    )
    missed = tracker.process_cycle(_cycle(2.0, 3, []))

    assert accepted.detection_id == 5
    assert accepted.nis is not None
    np.testing.assert_allclose(
        accepted.measurement_covariance,
        [[0.25, 0.0], [0.0, 100.0 * np.deg2rad(0.5) ** 2]],
    )
    assert missed.detection_id is None
    assert missed.nis is None
    assert missed.measurement_covariance is None
    assert tracker.filter.last_observation_time_seconds == 2.0
    assert tracker.filter.position_observation_noise_floor_m == 1e-6


def test_bayesian_br24_tracker_disables_smc_reinitialization_by_default():
    tracker = BR24BayesianRadarTracker.initialize(
        _cycle(0.0, 1, [_detection(4, 10.0)]),
        initial_detection_id=4,
        priors=BayesianCTRVPriors(),
    )

    assert tracker.filter.config.predictive_log_density_threshold is None


def test_track_br24_csv_starts_at_the_manual_seed_and_never_reads_a_reference(
    tmp_path,
):
    input_csv = tmp_path / "BR24Dataset20131107.csv"
    pd.DataFrame(
        {
            "time": [0.0, 1.0, 1.0],
            "cycle_counter": [4, 5, 5],
            "sensor_x": [100.0, 100.0, 100.0],
            "sensor_y": [-20.0, -20.0, -20.0],
            "sensor_yaw": [0.0, 0.0, 0.0],
            "detection_id": [9, 10, 11],
            "range": [10.0, 10.0, 200.0],
            "azimuth": [0.0, 0.0, 0.0],
            "std_range": [0.5, 0.5, 0.5],
            "std_azimuth": [0.0, 0.0, 0.0],
            "x_relative": [10.0, 10.0, 200.0],
            "y_relative": [0.0, 0.0, 0.0],
        }
    ).to_csv(input_csv, index=False)

    tracker, steps = track_br24_csv(
        input_csv,
        initial_cycle_counter=4,
        initial_detection_id=9,
        priors=BayesianCTRVPriors(),
        config=SequentialMonteCarloCTRVConfig(
            particle_count=128,
            posterior_draw_count=16,
            predictive_log_density_threshold=None,
        ),
        seed=42,
    )

    assert [step.cycle_counter for step in steps] == [4, 5]
    assert [step.detection_id for step in steps] == [9, 10]
    assert tracker.filter.last_observation_time_seconds == 1.0
