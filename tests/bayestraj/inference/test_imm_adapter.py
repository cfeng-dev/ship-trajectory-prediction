"""Shared IMM computations must not depend on the desktop application."""

import importlib
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_imm_evaluation_imports_without_the_predictor_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "imm_predictor", None)
    monkeypatch.setitem(sys.modules, "imm_predictor.player", None)
    module = runpy.run_path(ROOT / "experiments/model_evaluation/imm.py")
    assert module["IMMOptions"](10, 2, 1, 3, False).prediction_count == 2


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
