"""Background inference remains serial, cancellable between fits and isolated."""

from dataclasses import replace
from threading import Event, get_ident
from time import monotonic

import numpy as np
import pandas as pd
import pytest
from trajectory_predictor import dashboard
from trajectory_predictor.dashboard import PARAMETER_NAMES, PosteriorDashboardUpdate
from trajectory_predictor.session import PosteriorAnalysisWorker, prepare_analysis
from trajectory_predictor.settings import default_form_values, parse_settings


def _update(observation_count):
    return PosteriorDashboardUpdate(
        observation_count=observation_count,
        samples_by_parameter={name: np.array([0.0, 0.0]) for name in PARAMETER_NAMES},
    )


@pytest.mark.parametrize("method", ["smc", "sequential_vi", "vi", "mcmc"])
@pytest.mark.parametrize("prediction_count", [0, 1])
def test_br24_runtime_counts_update_and_forecast_and_survives_cache(
    tmp_path, monkeypatch, method, prediction_count
):
    from types import SimpleNamespace

    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr(dashboard, "perf_counter", lambda: clock.value, raising=False)

    class Filter:
        effective_sample_size = 10.0
        resample_count = 0

        def sample_current_posterior(self, **kwargs):
            clock.value += 11.0
            return object()

        def forecast(self, *args, **kwargs):
            clock.value += 3.0
            return object()

    class Tracker:
        filter = Filter()
        is_prior_predictive = False

        @property
        def estimated_position(self):
            clock.value += 19.0
            return np.array([10.0, 0.0])

        def process_measurement(self, measurement):
            clock.value += 2.0
            return SimpleNamespace(
                detection_id=measurement.detection_id,
                nis=None,
                measurement_covariance=measurement.measurement_covariance,
            )

    def initialize(*args, **kwargs):
        clock.value += 7.0
        return Tracker()

    def samples(*args):
        clock.value += 13.0
        return _update(1).samples_by_parameter

    def forecast(fit, offsets, sample_count):
        clock.value += 17.0
        return dashboard.PosteriorDashboardForecast(
            time_offsets_seconds=offsets,
            median_positions=np.zeros((len(offsets), 2)),
            sample_positions=np.zeros((0, len(offsets), 2)),
        )

    monkeypatch.setattr(
        dashboard.BR24BayesianRadarTracker, "initialize_position", initialize
    )
    monkeypatch.setattr(dashboard, "_extract_dashboard_samples", samples)
    monkeypatch.setattr(dashboard, "_extract_dashboard_forecast", forecast)
    path = tmp_path / "associated.csv"
    pd.DataFrame(
        {
            "time": [10.0, 12.0, 17.0],
            "cycle_counter": [1, 2, 4],
            "detection_id": 1,
            "x": 10.0,
            "y": 0.0,
            "cov_xx": 2.0,
            "cov_xy": 0.0,
            "cov_yy": 1.0,
        }
    ).to_csv(path, index=False)
    values = default_form_values()
    values["data"].update(
        data_file=str(path),
        inference_method=method,
        prediction_count=str(prediction_count),
    )
    _, _, _, load = prepare_analysis(parse_settings(values).analysis)
    update = load(2)
    assert update.fit_runtime_seconds == pytest.approx(2.0 + 3.0 * prediction_count)
    completed_at = clock.value
    assert load(2) is update
    assert clock.value == completed_at
    assert load(2).fit_runtime_seconds == pytest.approx(2.0 + 3.0 * prediction_count)


def test_worker_preserves_runtime_supplied_by_loader():
    expected = replace(_update(1), fit_runtime_seconds=3.25)
    worker = PosteriorAnalysisWorker(prepare=lambda _: (None, 1, 1, lambda _: expected))
    try:
        worker.start(None)
        worker.request(1)
        events = wait_for_event(worker, "update")
        assert (
            next(event.payload for event in events if event.kind == "update")
            is expected
        )
    finally:
        worker.close()
        worker.join(5)


@pytest.mark.parametrize("tool", ["bayesian", "imm"])
def test_associated_br24_preserves_covariances_and_ignores_reference(
    tmp_path, monkeypatch, tool
):
    from imm_predictor import player

    from bayestraj.inference.br24_radar_tracker import BR24BayesianRadarTracker

    def forbidden(*args, **kwargs):
        pytest.fail("Associated fixes must bypass raw detection association.")

    monkeypatch.setattr(BR24BayesianRadarTracker, "initialize", forbidden)
    monkeypatch.setattr(BR24BayesianRadarTracker, "process_cycle", forbidden)
    calls = []

    class Tracker:
        initialize = forbidden
        process_cycle = forbidden

        @classmethod
        def initialize_position(cls, time_seconds, position, covariance):
            tracker = cls()
            tracker.update_position(time_seconds, position, covariance)
            return tracker

        def update_position(self, time_seconds, position, covariance):
            calls.append((time_seconds, position.copy(), covariance.copy()))
            self.last_observation_time_seconds = time_seconds
            self.estimated_position = position.copy()

    monkeypatch.setattr(player, "_load_optional_imm_tracker", lambda: Tracker)
    data = pd.DataFrame(
        {
            "time": [100.0, 102.5, 109.0],
            "cycle_counter": [4, 5, 8],
            "detection_id": [17, 22, 31],
            "x": [10.0, 11.0, 12.0],
            "y": [-4.0, -3.0, -2.0],
            "cov_xx": [2.0, 3.0, 4.0],
            "cov_xy": [0.3, 0.4, 0.5],
            "cov_yy": [1.0, 2.0, 3.0],
        }
    )
    results = []
    for reference_offset in [0.0, 1000.0]:
        path = tmp_path / f"associated_{reference_offset}.csv"
        data.assign(
            ground_truth_x=data.x + reference_offset,
            ground_truth_y=data.y - reference_offset,
        ).to_csv(path, index=False)
        if tool == "bayesian":
            values = default_form_values()
            values["data"].update(
                data_file=str(path), inference_method="smc", prediction_count="1"
            )
            values["smc"].update(particle_count="32", posterior_draw_count="20")
            trajectory, maximum, minimum, load = prepare_analysis(
                parse_settings(values).analysis
            )
            first, second = load(1), load(2)
            assert (maximum, minimum) == (3, 1)
            assert (first.radar_detection_id, second.radar_detection_id) == (17, 22)
            assert second.radar_nis is None
            np.testing.assert_array_equal(
                first.radar_measurement_covariance, [[2.0, 0.3], [0.3, 1.0]]
            )
            np.testing.assert_array_equal(
                second.radar_measurement_covariance, [[3.0, 0.4], [0.4, 2.0]]
            )
            np.testing.assert_array_equal(trajectory.reference_time_seconds, data.time)
            np.testing.assert_array_equal(
                trajectory.reference_x, data.x + reference_offset
            )
            assert second.forecast.time_offsets_seconds.tolist() == [6.5]
            results.append(second.tracked_positions)
        else:
            store = player.IMMPlaybackStore(
                player.IMMPlayerConfig(data_file=path, prediction_count=0)
            )
            first, second = store.stage_for(1), store.stage_for(2)
            assert (first.detection_id, second.detection_id) == (17, 22)
            assert second.nis is None
            np.testing.assert_array_equal(
                first.measurement_covariance, [[2.0, 0.3], [0.3, 1.0]]
            )
            np.testing.assert_array_equal(
                second.measurement_covariance, [[3.0, 0.4], [0.4, 2.0]]
            )
            np.testing.assert_array_equal(store.times, data.time)
            np.testing.assert_array_equal(
                store.reference.x_meters, data.x + reference_offset
            )
            np.testing.assert_array_equal(calls[-1][1], [11.0, -3.0])
            assert calls[-1][0] == 102.5
            results.append(second.tracked_positions)
    np.testing.assert_array_equal(*results)


def test_imm_associated_position_noise_preserves_nominal_covariance():
    from imm_predictor import player

    measurement = player.br24.BR24AssociatedMeasurement(
        100.0, 4, 17, np.array([10.0, -4.0]), np.array([[2.0, 0.3], [0.3, 1.0]])
    )
    config = player.IMMPlayerConfig(position_noise_std_m=2.0)
    first = player._add_position_noise((measurement,), config)[0]
    second = player._add_position_noise((measurement,), config)[0]
    np.testing.assert_array_equal(first.position, second.position)
    assert not np.array_equal(first.position, measurement.position)
    np.testing.assert_array_equal(
        first.measurement_covariance, measurement.measurement_covariance
    )
    assert first.time_seconds == measurement.time_seconds
    assert first.detection_id == measurement.detection_id


@pytest.mark.parametrize("supplied", [False, True])
def test_processed_csv_reference_motion_preserves_native_values_or_derives(
    tmp_path, supplied
):
    data = pd.DataFrame(
        {
            "time": [2.0, 0.5, 0.0, 3.0],
            "x": [1.0, 1.0, 0.0, 1.0],
            "y": [2.0, 0.0, 0.0, 2.0],
        }
    )
    if supplied:
        data["speed"] = 0.0
        data["theta"] = 0.0
        data["omega"] = 0.0
    path = tmp_path / "processed.csv"
    data.to_csv(path, index=False)
    values = default_form_values()
    values["data"].update(
        data_file=str(path), inference_method="smc", prediction_count="0"
    )
    trajectory, _, _, _ = prepare_analysis(parse_settings(values).analysis)
    assert trajectory.reference_speed_is_derived is not supplied
    assert trajectory.reference_heading_is_derived is not supplied
    assert trajectory.reference_turn_rate_is_derived is not supplied
    np.testing.assert_allclose(
        trajectory.reference_speed_mps,
        np.zeros(4) if supplied else [np.nan, 2.0, 4 / 3, 0.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        trajectory.reference_heading_degrees,
        np.zeros(4) if supplied else [np.nan, 0.0, 90.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        trajectory.reference_turn_rate_degrees_per_second,
        np.zeros(4) if supplied else [np.nan, np.nan, 60.0, np.nan],
        equal_nan=True,
    )
    if supplied:
        assert trajectory.reference_state_at(1)[2:5] == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("method", ["vi", "mcmc", "smc", "sequential_vi"])
def test_processed_ground_truth_drives_reference_and_scores_without_replacing_inputs(
    tmp_path, method
):
    path = tmp_path / "tracked_with_reference.csv"
    pd.DataFrame(
        {
            "time": [105.0, 100.0, 109.0, 102.0, 114.0, 120.0],
            "x": [12.0, 10.0, 13.0, 11.0, 14.0, 15.0],
            "y": [-2.0, -4.0, -1.0, -3.0, 0.0, 1.0],
            "ground_truth_x": [110.0, 100.0, 118.0, 104.0, 128.0, 140.0],
            "ground_truth_y": [0.0] * 6,
        }
    ).to_csv(path, index=False)
    values = default_form_values()
    values["data"].update(data_file=str(path), inference_method=method)
    settings = parse_settings(values).analysis
    settings = replace(
        settings,
        experiment=replace(
            settings.experiment,
            start_index=1,
            observation_interval_seconds=5.0,
            position_noise_std_m=0.0,
        ),
    )

    trajectory, _, _, _ = prepare_analysis(settings)

    np.testing.assert_array_equal(trajectory.reference_x, [104.0, 118.0, 128.0, 140.0])
    np.testing.assert_array_equal(trajectory.reference_y, [0.0] * 4)
    np.testing.assert_array_equal(trajectory.observed_x, [11.0, 13.0, 14.0, 15.0])
    np.testing.assert_array_equal(trajectory.observed_y, [-3.0, -1.0, 0.0, 1.0])
    np.testing.assert_array_equal(
        trajectory.reference_time_seconds, [0.0, 7.0, 12.0, 18.0]
    )
    np.testing.assert_allclose(
        trajectory.reference_speed_mps, [np.nan, 2.0, 2.0, 2.0], equal_nan=True
    )
    np.testing.assert_allclose(
        trajectory.reference_heading_degrees, [np.nan, 0.0, 0.0, 0.0], equal_nan=True
    )
    positions = np.array([[128.0, 0.0], [140.0, 0.0]])
    update = PosteriorDashboardUpdate(
        observation_count=2,
        samples_by_parameter={name: np.array([0.0, 0.0]) for name in PARAMETER_NAMES},
        forecast=dashboard.PosteriorDashboardForecast(
            time_offsets_seconds=[5.0, 11.0],
            median_positions=positions,
            sample_positions=positions[None, :, :],
        ),
    )
    metrics = dashboard.analysis_metrics_at(trajectory, {2: update}, 2)
    assert metrics.ade_m == pytest.approx(0.0)
    assert metrics.fde_m == pytest.approx(0.0)


def test_processed_ground_truth_changes_do_not_change_smc_posterior(tmp_path):
    path = tmp_path / "tracked_with_reference.csv"
    data = pd.DataFrame(
        {
            "time": [100.0, 102.0, 105.0, 109.0],
            "x": [10.0, 11.0, 12.0, 13.0],
            "y": [-4.0, -3.0, -2.0, -1.0],
            "ground_truth_x": [100.0, 104.0, 110.0, 118.0],
            "ground_truth_y": [0.0] * 4,
        }
    )
    values = default_form_values()
    values["data"].update(data_file=str(path), inference_method="smc")
    values["smc"].update(particle_count="32", posterior_draw_count="20")

    def prepare():
        data.to_csv(path, index=False)
        settings = parse_settings(values).analysis
        settings = replace(
            settings,
            experiment=replace(settings.experiment, position_noise_std_m=2.0),
        )
        trajectory, _, _, load = prepare_analysis(settings)
        return trajectory, load(3)

    before_trajectory, before = prepare()
    data["ground_truth_x"] += 1000.0
    data["ground_truth_y"] -= 500.0
    after_trajectory, after = prepare()

    np.testing.assert_allclose(
        after_trajectory.reference_x - before_trajectory.reference_x, 1000.0
    )
    np.testing.assert_array_equal(
        before_trajectory.observed_x, after_trajectory.observed_x
    )
    np.testing.assert_array_equal(
        before_trajectory.observed_y, after_trajectory.observed_y
    )
    for name in PARAMETER_NAMES:
        np.testing.assert_array_equal(
            before.samples_by_parameter[name], after.samples_by_parameter[name]
        )
    np.testing.assert_array_equal(
        before.forecast.posterior_positions, after.forecast.posterior_positions
    )


@pytest.mark.parametrize("invalid", ["missing_x", "missing_y", "nan", "inf"])
def test_processed_ground_truth_rejects_incomplete_or_nonfinite_positions(
    tmp_path, invalid
):
    path = tmp_path / "tracked_with_reference.csv"
    data = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0] * 4,
            "ground_truth_x": [10.0, 11.0, 12.0, 13.0],
            "ground_truth_y": [1.0] * 4,
        }
    )
    if invalid.startswith("missing_"):
        data = data.drop(columns=["ground_truth_" + invalid.removeprefix("missing_")])
    else:
        data.loc[1, "ground_truth_x"] = np.nan if invalid == "nan" else np.inf
    data.to_csv(path, index=False)
    values = default_form_values()
    values["data"].update(data_file=str(path), inference_method="mcmc")

    with pytest.raises(ValueError, match="ground_truth"):
        prepare_analysis(parse_settings(values).analysis)


@pytest.fixture
def analysis(tmp_path):
    values = default_form_values()
    data_file = tmp_path / "route.csv"
    data_file.touch()
    values["data"]["data_file"] = str(data_file)
    return parse_settings(values).analysis


def wait_for_event(worker, kind, timeout=5, *, match=lambda _: True):
    deadline = monotonic() + timeout
    collected = []
    while monotonic() < deadline:
        collected.extend(worker.drain())
        if any(event.kind == kind and match(event) for event in collected):
            return collected
        Event().wait(0.005)
    pytest.fail(f"No {kind} event; received {collected}")


@pytest.mark.parametrize("method", ("vi", "mcmc", "smc", "sequential_vi"))
def test_br24_session_forwards_each_method_and_its_configs(
    analysis, method, monkeypatch
):
    captured = {}
    sentinel = object()
    selected = replace(
        analysis,
        experiment=replace(analysis.experiment, inference_method=method),
    )

    def create_loader(data_file, **options):
        captured["data_file"] = data_file
        captured.update(options)
        return sentinel

    monkeypatch.setattr(
        "trajectory_predictor.session.is_br24_tracking_input", lambda _path: True
    )
    monkeypatch.setattr(dashboard, "create_br24_radar_dashboard_loader", create_loader)

    assert prepare_analysis(selected) is sentinel
    assert captured["data_file"] == selected.data_file
    assert captured["experiment"] is selected.experiment
    assert captured["experiment"].inference_method == method
    assert captured["priors"] is selected.priors
    assert captured["vi_config"] is selected.vi_config
    assert captured["mcmc_config"] is selected.mcmc_config
    assert captured["smc_config"] is selected.smc_config
    assert captured["sequential_vi_config"] is selected.sequential_vi_config
    if method == "vi":
        assert captured["vi_config"]["iter"] == 10_000
        assert captured["vi_config"]["tol_rel_obj"] == 0.01
        assert captured["vi_config"]["adapt_iter"] == 250
        assert captured["vi_config"]["grad_samples"] == 20
    elif method == "sequential_vi":
        config = captured["sequential_vi_config"]
        assert config.iter == 10_000
        assert config.tol_rel_obj == 0.01
        assert config.adapt_iter == 250
        assert config.grad_samples == 20


@pytest.mark.parametrize("method", ["smc"])
def test_default_worker_reads_csv_and_runs_online_inference(tmp_path, method):
    data_file = tmp_path / "route.csv"
    pd.DataFrame(
        {
            "time": [0.0, 10.0, 20.0, 30.0],
            "x": [0.0, 20.0, 40.0, 60.0],
            "y": [0.0, 5.0, 10.0, 15.0],
        }
    ).to_csv(data_file, index=False)
    values = default_form_values()
    values["data"]["data_file"] = str(data_file)
    values["data"]["inference_method"] = method
    values[method]["particle_count"] = "32"
    values[method]["posterior_draw_count"] = "20"
    values["data"]["prediction_count"] = "2"
    values["data"]["prediction_sample_count"] = "5"
    worker = PosteriorAnalysisWorker()
    try:
        worker.start(parse_settings(values).analysis)
        worker.request(4)
        events = wait_for_event(
            worker, "update", match=lambda event: event.payload.observation_count == 4
        )
        updates = [event.payload for event in events if event.kind == "update"]
        assert [update.observation_count for update in updates] == [1, 2, 3, 4]
        assert all(update.particle_count == 32 for update in updates)
        assert [
            len(update.forecast.time_offsets_seconds) for update in updates[:-1]
        ] == [2, 2, 1]
        assert updates[-1].forecast is None
        assert updates[0].forecast.sample_positions.shape == (5, 2, 2)
        assert not updates[0].forecast.sample_positions.flags.writeable
        assert all(
            samples.size == 20 for samples in updates[-1].samples_by_parameter.values()
        )
    finally:
        worker.close()
        worker.join(5)


def test_br24_input_uses_manual_radar_seed_and_never_exposes_reference(tmp_path):
    data_file = tmp_path / "BR24Dataset20131107.csv"
    pd.DataFrame(
        {
            "time": [0.0, 0.0, 1.0, 1.0, 2.0],
            "cycle_counter": [4, 4, 5, 5, 6],
            "sensor_x": [100.0] * 5,
            "sensor_y": [-20.0] * 5,
            "sensor_yaw": [0.0] * 5,
            "detection_id": [10, 9, 10, 11, np.nan],
            "range": [200.0, 10.0, 10.0, 200.0, np.nan],
            "azimuth": [0.0, 0.0, 0.0, 0.0, np.nan],
            "std_range": [0.5, 0.5, 0.5, 0.5, np.nan],
            "std_azimuth": [0.0, 0.0, 0.0, 0.0, np.nan],
            "x_relative": [200.0, 10.0, 10.0, 200.0, np.nan],
            "y_relative": [0.0, 0.0, 0.0, 0.0, np.nan],
        }
    ).to_csv(data_file, index=False)
    values = default_form_values()
    values["data"].update(
        {
            "data_file": str(data_file),
            "inference_method": "smc",
            "prediction_count": "1",
            "prediction_sample_count": "5",
        }
    )
    values["smc"].update({"particle_count": "32", "posterior_draw_count": "20"})

    trajectory, maximum, minimum, load = prepare_analysis(
        parse_settings(values).analysis
    )
    first_update = load(1)
    accepted_update = load(2)
    missed_update = load(3)

    assert maximum == 3
    assert minimum == 1
    assert trajectory.has_reference is False
    assert first_update.radar_tracking_active is True
    assert first_update.radar_detection_id == 9
    assert first_update.radar_nis is None
    np.testing.assert_allclose(
        first_update.radar_measurement_covariance,
        [[0.25, 0.0], [0.0, 100.0 * np.deg2rad(0.5) ** 2]],
    )
    assert accepted_update.radar_detection_id == 10
    assert accepted_update.radar_nis is not None
    np.testing.assert_allclose(
        accepted_update.radar_measurement_covariance,
        [[0.25, 0.0], [0.0, 100.0 * np.deg2rad(0.5) ** 2]],
    )
    assert missed_update.radar_detection_id is None
    assert missed_update.radar_measurement_covariance is None
    assert first_update.tracked_positions.shape == (1, 2)
    assert missed_update.tracked_positions.shape == (3, 2)
    assert first_update.forecast is not None


def test_br24_reference_labels_are_dashboard_only_and_do_not_change_tracking(tmp_path):
    raw_file = tmp_path / "BR24Dataset20131107.csv"
    labelled_file = tmp_path / "labelled_radar.csv"
    raw = pd.DataFrame(
        {
            "time": [0.0, 0.0, 1.0, 1.0, 2.0],
            "cycle_counter": [4, 4, 5, 5, 6],
            "sensor_x": [100.0] * 5,
            "sensor_y": [-20.0] * 5,
            "sensor_yaw": [0.0] * 5,
            "detection_id": [10, 9, 10, 11, np.nan],
            "range": [200.0, 10.0, 10.0, 200.0, np.nan],
            "azimuth": [0.0, 0.0, 0.0, 0.0, np.nan],
            "std_range": [0.5, 0.5, 0.5, 0.5, np.nan],
            "std_azimuth": [0.0, 0.0, 0.0, 0.0, np.nan],
            "x_relative": [200.0, 10.0, 10.0, 200.0, np.nan],
            "y_relative": [0.0, 0.0, 0.0, 0.0, np.nan],
        }
    )
    raw.to_csv(raw_file, index=False)
    labelled = raw.assign(
        ground_truth_x=[110.0, 110.0, 111.0, 111.0, 112.0],
        ground_truth_y=[-20.0, -20.0, -20.0, -20.0, -20.0],
        ground_truth_yaw=[0.1] * 5,
        ground_truth_velocity=[2.0] * 5,
    )
    labelled.to_csv(labelled_file, index=False)

    values = default_form_values()
    values["data"].update(
        {
            "inference_method": "smc",
            "prediction_count": "1",
            "prediction_sample_count": "5",
        }
    )
    values["smc"].update({"particle_count": "32", "posterior_draw_count": "20"})
    values["data"]["data_file"] = str(raw_file)
    raw_trajectory, _, _, raw_load = prepare_analysis(parse_settings(values).analysis)
    values["data"]["data_file"] = str(labelled_file)
    labelled_trajectory, _, _, labelled_load = prepare_analysis(
        parse_settings(values).analysis
    )

    raw_update = raw_load(2)
    labelled_update = labelled_load(2)

    assert raw_trajectory.has_reference is False
    assert labelled_trajectory.has_reference is True
    np.testing.assert_allclose(
        raw_update.tracked_positions, labelled_update.tracked_positions
    )
    assert raw_update.radar_detection_id == labelled_update.radar_detection_id
    assert raw_update.radar_nis == labelled_update.radar_nis
    np.testing.assert_allclose(
        raw_update.forecast.median_positions,
        labelled_update.forecast.median_positions,
    )
    reference_state = labelled_trajectory.reference_state_at(2)
    np.testing.assert_allclose(
        reference_state[:4], [111.0, -20.0, 0.1 * 180 / np.pi, 2.0]
    )
    assert reference_state[-1] == 1.0
    assert (
        dashboard.analysis_metrics_at(raw_trajectory, {2: raw_update}, 2).ade_m is None
    )
    assert (
        dashboard.analysis_metrics_at(
            labelled_trajectory, {2: labelled_update}, 2
        ).ade_m
        is not None
    )


def test_pause_lowers_target_without_repeating_an_inflight_fit(analysis):
    entered, release = Event(), Event()
    calls = []

    def load(count):
        calls.append(count)
        entered.set()
        assert release.wait(5)
        return _update(count)

    worker = PosteriorAnalysisWorker(lambda _: (None, 4, 1, load))
    try:
        worker.start(analysis)
        worker.request(4)
        assert entered.wait(5)
        worker.request(0)
        release.set()
        wait_for_event(worker, "update")
        worker.request(2)
        wait_for_event(
            worker, "update", match=lambda e: e.payload.observation_count == 2
        )
        assert calls == [1, 2]
    finally:
        release.set()
        worker.close()
        worker.join(5)


def test_worker_owns_loader_and_computes_prefix_only_once(analysis):
    calls, thread_ids = [], []
    completed = Event()

    def prepare(settings):
        thread_ids.append(get_ident())

        def load(count):
            calls.append(count)
            thread_ids.append(get_ident())
            if count == 4:
                completed.set()
            return _update(count)

        return None, 4, 1, load

    worker = PosteriorAnalysisWorker(prepare)
    try:
        worker.start(analysis)
        worker.request(3)
        wait_for_event(worker, "ready")
        worker.request(2)
        worker.request(4)
        assert completed.wait(5)
        worker.request(1)
    finally:
        worker.close()
        worker.join(5)
    assert calls == [1, 2, 3, 4]
    assert len(set(thread_ids)) == 1
    assert thread_ids[0] != get_ident()
    assert not worker.is_alive


def test_replacement_discards_old_fit_and_snapshots_mutable_options(analysis):
    entered, release, prepared_new = Event(), Event(), Event()
    snapshots = []

    def prepare(settings):
        snapshots.append(settings)
        if len(snapshots) == 2:
            prepared_new.set()

        def load(count):
            entered.set()
            assert release.wait(5)
            return _update(count)

        return None, 4, 1, load

    worker = PosteriorAnalysisWorker(prepare)
    try:
        worker.start(analysis)
        worker.request(4)
        assert entered.wait(5)
        replacement = replace(analysis, vi_config={"iter": 123})
        generation = worker.start(replacement)
        replacement.vi_config["iter"] = 999
        release.set()
        assert prepared_new.wait(5)
        events = wait_for_event(worker, "ready")
        assert all(event.generation == generation for event in events)
        assert not any(event.kind == "update" for event in events)
        assert snapshots[1].vi_config == {"iter": 123}
    finally:
        release.set()
        worker.close()
        worker.join(5)


def test_resize_suspension_finishes_current_fit_then_resumes_without_replay(analysis):
    entered, release, second = Event(), Event(), Event()
    calls, preparations = [], []

    def load(count):
        calls.append(count)
        if count == 1:
            entered.set()
            assert release.wait(5)
        else:
            second.set()
        return _update(count)

    def prepare(settings):
        preparations.append(settings)
        return None, 3, 1, load

    worker = PosteriorAnalysisWorker(prepare)
    try:
        worker.start(analysis)
        worker.request(3)
        assert entered.wait(5)
        worker.set_suspended(True)
        release.set()
        wait_for_event(
            worker, "update", match=lambda event: event.payload.observation_count == 1
        )
        assert not second.wait(0.1)
        assert calls == [1]
        worker.set_suspended(False)
        wait_for_event(
            worker, "update", match=lambda event: event.payload.observation_count == 3
        )
        assert calls == [1, 2, 3]
        assert len(preparations) == 1
    finally:
        release.set()
        worker.close()
        worker.join(5)


def test_suspended_worker_can_close_with_a_pending_analysis(analysis):
    entered = Event()

    def prepare(_settings):
        entered.set()
        return None, 2, 1, _update

    worker = PosteriorAnalysisWorker(prepare)
    try:
        worker.set_suspended(True)
        worker.start(analysis)
        worker.request(2)
        assert not entered.wait(0.1)
        worker.close()
        worker.join(5)
        assert not worker.is_alive
        assert not entered.is_set()
    finally:
        worker.close()
        worker.join(5)


def test_close_is_nonblocking_and_finishes_only_inflight_fit(analysis):
    entered, release = Event(), Event()
    calls = []

    def load(count):
        calls.append(count)
        entered.set()
        assert release.wait(5)
        return _update(count)

    worker = PosteriorAnalysisWorker(lambda _: (None, 4, 1, load))
    try:
        worker.start(analysis)
        worker.request(4)
        assert entered.wait(5)
        worker.close()
        assert worker.is_alive
        release.set()
        worker.join(5)
        assert calls == [1]
        assert worker.drain() == []
        with pytest.raises(RuntimeError, match="closed"):
            worker.start(analysis)
    finally:
        release.set()
        worker.close()
        worker.join(5)


def test_error_stops_generation_but_allows_new_analysis(analysis):
    calls = []

    def prepare(_):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("bad CSV")
        return None, 4, 3, _update

    worker = PosteriorAnalysisWorker(prepare)
    try:
        worker.start(analysis)
        events = wait_for_event(worker, "error")
        assert "bad CSV" in events[-1].payload
        worker.request(4)
        worker.start(analysis)
        worker.request(3)
        events = wait_for_event(worker, "update")
        assert [e.payload.observation_count for e in events if e.kind == "update"] == [
            3
        ]
        assert len(calls) == 2
    finally:
        worker.close()
        worker.join(5)


def test_worker_surfaces_sequential_vi_update_failure(analysis):
    def prepare(_):
        def load(_count):
            raise RuntimeError("Sequential ADVI failed")

        return None, 8, 5, load

    worker = PosteriorAnalysisWorker(prepare)
    try:
        worker.start(analysis)
        worker.request(5)
        events = wait_for_event(worker, "error")
        assert "Sequential ADVI failed" in events[-1].payload
    finally:
        worker.close()
        worker.join(5)


def test_worker_logs_complete_failure_traceback_and_analysis_context(analysis, caplog):
    def prepare(_):
        raise ValueError("diagnostic sentinel")

    worker = PosteriorAnalysisWorker(prepare)
    try:
        worker.start(analysis)
        wait_for_event(worker, "error")
        records = [
            record
            for record in caplog.records
            if "diagnostic sentinel" in record.getMessage() or record.exc_info
        ]
        assert records and records[-1].exc_info
        assert str(analysis.data_file) in records[-1].getMessage()
        assert "generation=" in records[-1].getMessage()
    finally:
        worker.close()
        worker.join(5)
