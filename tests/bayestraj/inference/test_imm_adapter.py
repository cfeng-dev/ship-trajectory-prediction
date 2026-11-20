"""Shared IMM computations must not depend on the desktop application."""

import importlib
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_imm_evaluation_imports_without_the_predictor_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "imm_predictor", None)
    monkeypatch.setitem(sys.modules, "imm_predictor.player", None)
    module = runpy.run_path(ROOT / "scripts/model_evaluation/imm.py")
    assert callable(module["main"])


def test_shared_mixture_medians_preserve_atoms_and_immutable_snapshots():
    adapter = importlib.import_module("bayestraj.inference.imm_adapter")
    means = np.array(
        [[2, 0, 0, 0, 1], [34, 12, 0, 0, 1], [99, 99, 0, 0, 1]], dtype=float
    )
    weights = np.array([0.75, 0.25, 0])
    mixture = adapter.IMMStateMixture(
        weights,
        tuple(adapter.IMMStatePosterior(mean, np.zeros((5, 5))) for mean in means),
    )
    tracker = SimpleNamespace(forecast_mixtures=lambda times: (mixture,))
    forecast = SimpleNamespace(
        time_seconds=np.array([1.0]), position_means=np.array([[10.0, 3.0]])
    )
    medians = adapter.forecast_position_medians(tracker, forecast)
    np.testing.assert_array_equal(medians, [[2, 0]])
    means[:] = 100
    weights[:] = [0, 1, 0]
    np.testing.assert_array_equal(
        adapter.forecast_position_medians(tracker, forecast), [[2, 0]]
    )
    assert not medians.flags.writeable
    assert not mixture.weights.flags.writeable
    assert not mixture.components[0].mean.flags.writeable


def test_shared_loader_accepts_explicit_path_and_preserves_gaussian_fallback(tmp_path):
    adapter = importlib.import_module("bayestraj.inference.imm_adapter")
    backend = tmp_path / "minimal_imm.py"
    backend.write_text("class BR24IMMTracker:\n    pass\n", encoding="utf-8")
    tracker = adapter.load_imm_tracker(backend)()
    means = np.array([[1.0, 2.0], [3.0, 4.0]])
    forecast = SimpleNamespace(time_seconds=np.array([1, 2]), position_means=means)
    medians = adapter.forecast_position_medians(tracker, forecast)
    np.testing.assert_array_equal(medians, [[1, 2], [3, 4]])
    means[:] = 99
    np.testing.assert_array_equal(medians, [[1, 2], [3, 4]])
    assert not medians.flags.writeable
    assert adapter.forecast_position_medians(tracker, None) is None


@pytest.mark.parametrize("problem", ["missing_file", "missing_tracker"])
def test_shared_loader_reports_unavailable_backend(tmp_path, problem):
    adapter = importlib.import_module("bayestraj.inference.imm_adapter")
    path = tmp_path / "unavailable.py"
    if problem == "missing_file":
        with pytest.raises(FileNotFoundError, match="IMM implementation not found"):
            adapter.load_imm_tracker(path)
    else:
        path.write_text("other_value = 42\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="must expose BR24IMMTracker"):
            adapter.load_imm_tracker(path)


def test_predictor_uses_the_shared_snapshot_types_and_median_function():
    adapter = importlib.import_module("bayestraj.inference.imm_adapter")
    from imm_predictor import player

    # Type identity matters because mixtures validate their component types.
    assert player.IMMStatePosterior is adapter.IMMStatePosterior
    assert player.IMMStateMixture is adapter.IMMStateMixture
    assert player._forecast_position_medians is adapter.forecast_position_medians


@pytest.mark.skipif(
    not (ROOT / "external/IMM/br24_imm.py").is_file(),
    reason="Optional IMM backend is unavailable",
)
def test_associated_imm_consumes_known_positions_without_gating_and_uses_covariance():
    adapter = importlib.import_module("bayestraj.inference.imm_adapter")
    tracker_type = adapter.load_imm_tracker(ROOT / "external/IMM/br24_imm.py")
    initial = np.array([100.0, -20.0])
    covariance = np.array([[2.0, 0.25], [0.25, 1.0]])
    tracker = tracker_type.initialize_position(49073.2, initial, covariance)
    np.testing.assert_array_equal(tracker.estimated_position, initial)
    np.testing.assert_allclose(tracker.estimated_position_covariance, covariance)
    # A supplied association is assimilated even far outside the raw-radar gate.
    tracker.update_position(49075.8, np.array([1100.0, -20.0]), covariance)
    assert tracker.estimated_position[0] > 500.0
    uncertain = tracker_type.initialize_position(49073.2, initial, covariance)
    uncertain.update_position(49075.8, np.array([1100.0, -20.0]), covariance * 1e8)
    assert uncertain.estimated_position[0] < tracker.estimated_position[0]
    before = tracker.estimated_position.copy()
    forecast = tracker.forecast([49081.0, 49084.1], seed=42)
    assert forecast.position_means.shape == (2, 2)
    assert np.isfinite(adapter.forecast_position_medians(tracker, forecast)).all()
    np.testing.assert_array_equal(tracker.estimated_position, before)
    with pytest.raises(ValueError):
        tracker.update_position(49075.8, initial, covariance)


@pytest.mark.skipif(
    not (ROOT / "external/IMM/br24_imm.py").is_file(),
    reason="Optional IMM backend is unavailable",
)
@pytest.mark.parametrize("forecast_method", ["forecast", "sample_forecast"])
@pytest.mark.parametrize(
    ("mode", "predictor", "expected_parameters"),
    [
        (0, "_predict_cv", (0.2**2, 0.2**2)),
        (1, "_predict_ctrv", (0.4**2, 0.06**2)),
        (2, "_predict_ca", (8.0, 0.7**2, 0.7**2)),
    ],
)
def test_imm_forecasts_use_configured_process_noise(
    monkeypatch, forecast_method, mode, predictor, expected_parameters
):
    module = runpy.run_path(ROOT / "external/IMM/br24_imm.py")
    backend = module["BR24IMMTracker"].forecast.__globals__
    settings = {
        "CV_ACCELERATION_NOISE_STD": 0.2,
        "CTRV_SPEED_NOISE_STD": 0.4,
        "CTRV_TURN_RATE_NOISE_STD_RAD": 0.06,
        "CA_ACCELERATION_NOISE_STD": 0.7,
        "CA_ACCELERATION_TIME_CONSTANT_SECONDS": 8.0,
    }
    for name, value in settings.items():
        monkeypatch.setitem(backend, name, value)
    if forecast_method == "sample_forecast":
        monkeypatch.setitem(backend, "_TRANSITION_PROBABILITIES", np.eye(3))
    prediction = Mock(wraps=backend[predictor])
    monkeypatch.setitem(backend, predictor, prediction)
    state = module["_initialize_imm_state"](np.zeros(2))
    state.mode_probabilities = np.eye(3)[mode]
    tracker = module["BR24IMMTracker"](
        _state=state,
        _last_observation_time_seconds=0.0,
        _sensor_position=np.zeros(2),
    )

    getattr(tracker, forecast_method)([1.0], seed=42)

    assert prediction.call_args_list
    for call in prediction.call_args_list:
        np.testing.assert_allclose(call.args[3:], expected_parameters)
