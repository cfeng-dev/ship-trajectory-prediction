"""Background inference remains serial, cancellable between fits and isolated."""

from dataclasses import replace
from threading import Event, get_ident
from time import monotonic

import numpy as np
import pandas as pd
import pytest
from trajectory_predictor.dashboard import PARAMETER_NAMES, PosteriorDashboardUpdate
from trajectory_predictor.session import PosteriorAnalysisWorker
from trajectory_predictor.settings import default_form_values, parse_settings


def _update(observation_count):
    return PosteriorDashboardUpdate(
        observation_count=observation_count,
        samples_by_parameter={name: np.array([0.0, 0.0]) for name in PARAMETER_NAMES},
    )


@pytest.fixture
def analysis(tmp_path):
    values = default_form_values()
    data_file = tmp_path / "route.csv"
    data_file.touch()
    values["data"]["data_file"] = str(data_file)
    values["data"]["run_id"] = "102"
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


@pytest.mark.parametrize("method", ["rbpf", "smc"])
def test_default_worker_reads_csv_and_runs_online_inference(tmp_path, method):
    data_file = tmp_path / "route.csv"
    pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=4, freq="10s", tz="UTC"),
            "run_id": 102,
            "gps_latitude": 54.0 + np.arange(4) * 1e-5,
            "gps_longitude": 10.0 + np.arange(4) * 2e-5,
            "gps_speed": 18.0,
        }
    ).to_csv(data_file, index=False)
    values = default_form_values()
    values["data"]["data_file"] = str(data_file)
    values["data"]["run_id"] = "102"
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
