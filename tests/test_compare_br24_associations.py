"""Tests for the BR24 IMM-versus-Bayesian association audit script."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
SCRIPT_PATH = (
    PROJECT_ROOT / "experiments/tracking_validation/compare_br24_associations.py"
)


def test_default_association_output_is_in_results_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    module = runpy.run_path(SCRIPT_PATH)

    assert module["OUTPUT_DIR"] == PROJECT_ROOT / "results/validation"


def _input_csvs(tmp_path):
    raw_csv = tmp_path / "BR24Dataset20131107.csv"
    imm_csv = tmp_path / "BR24_imm_tracked_trajectory.csv"
    output_csv = tmp_path / "BR24_association_comparison.csv"
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
    ).to_csv(raw_csv, index=False)
    # The MATLAB output starts recording associations after its first fix.
    pd.DataFrame({"cycle_counter": [5], "detection_id": [10]}).to_csv(
        imm_csv, index=False
    )
    return raw_csv, imm_csv, output_csv


def test_association_audit_uses_the_imm_initialization_and_records_misses(tmp_path):
    module = runpy.run_path(SCRIPT_PATH)
    raw_csv, imm_csv, output_csv = _input_csvs(tmp_path)

    comparison = module["compare_associations"](
        raw_csv=raw_csv,
        imm_csv=imm_csv,
        output_csv=output_csv,
        seed=42,
    )

    assert comparison["cycle_counter"].tolist() == [4, 5, 6]
    assert comparison["imm_detection_id"].tolist() == [9, 10, pd.NA]
    assert comparison["bayesian_detection_id"].tolist() == [9, 10, pd.NA]
    assert comparison["association_status"].tolist() == [
        "same_detection",
        "same_detection",
        "both_missed",
    ]
    persisted = pd.read_csv(
        output_csv,
        dtype={
            "imm_detection_id": "Int64",
            "bayesian_detection_id": "Int64",
        },
    )
    pd.testing.assert_frame_equal(persisted, comparison)


@pytest.mark.parametrize("method", ["smc", "mcmc", "vi", "seq_vi"])
def test_audit_dispatches_selected_method_and_reports_both_misses(
    tmp_path, monkeypatch, method
):
    module = runpy.run_path(SCRIPT_PATH)
    raw_csv, imm_csv, output_csv = _input_csvs(tmp_path)
    calls = []

    class Tracker:
        def process_cycle(self, cycle):
            return SimpleNamespace(detection_id=10 if cycle.detections else None)

    def initialize(cycle, **kwargs):
        calls.append((cycle, kwargs))
        return Tracker()

    monkeypatch.setattr(module["BR24BayesianRadarTracker"], "initialize", initialize)
    result = module["compare_associations"](
        raw_csv=raw_csv,
        imm_csv=imm_csv,
        output_csv=output_csv,
        inference_method=method,
        observation_count=4,
        seed=13,
    )
    canonical = "sequential_vi" if method == "seq_vi" else method
    assert calls[0][1]["inference_method"] == canonical
    assert calls[0][1]["initial_detection_id"] == 9
    assert calls[0][1]["observation_count"] == 4
    assert calls[0][1]["seed"] == 13
    assert result["inference_method"].eq(canonical).all()
    if method in {"vi", "mcmc"}:
        assert result["configured_observation_count"].eq(4).all()
    else:
        assert result["configured_observation_count"].isna().all()
    assert result["association_status"].tolist() == [
        "same_detection",
        "same_detection",
        "both_missed",
    ]


@pytest.mark.parametrize(
    ("imm_id", "bayesian_id", "status"),
    [
        (10, 10, "same_detection"),
        (10, 11, "different_detection"),
        (None, None, "both_missed"),
        (10, None, "imm_only"),
        (None, 10, "bayesian_only"),
    ],
)
def test_all_association_outcomes_are_distinguished(imm_id, bayesian_id, status):
    module = runpy.run_path(SCRIPT_PATH)
    assert module["_association_status"](imm_id, bayesian_id) == status


@pytest.mark.parametrize("different", [False, True])
def test_terminal_reports_verdict_and_first_differing_cycle(capsys, different):
    module = runpy.run_path(SCRIPT_PATH)
    table = pd.DataFrame(
        {
            "cycle_counter": [4, 5, 6],
            "association_status": [
                "same_detection",
                "imm_only" if different else "same_detection",
                "both_missed",
            ],
        }
    )
    assert "print_association_summary" in module
    module["print_association_summary"](table, inference_method="mcmc")
    output = capsys.readouterr().out
    assert (
        f"IMM vs MCMC associations: {'DIFFERENT' if different else 'MATCH'}" in output
    )
    assert "Compared 3 cycles" in output
    assert "both_missed: 1" in output
    if different:
        assert "First differing cycles: 5" in output


@pytest.mark.parametrize(
    "method", [None, "smc", "mcmc", "vi", "seq_vi", "sequential_vi"]
)
def test_main_only_runs_selected_method_and_supports_cycle_limit(
    tmp_path, monkeypatch, capsys, method
):
    module = runpy.run_path(SCRIPT_PATH)
    raw_csv, imm_csv, _ = _input_csvs(tmp_path)
    main = module["main"]
    monkeypatch.setitem(main.__globals__, "RAW_INPUT_CSV", raw_csv)
    monkeypatch.setitem(main.__globals__, "IMM_TRACKED_TRAJECTORY_CSV", imm_csv)
    output_dir = tmp_path / "validation"
    monkeypatch.setitem(main.__globals__, "OUTPUT_DIR", output_dir)
    monkeypatch.setitem(main.__globals__, "INFERENCE_METHOD", "mcmc")
    calls = []

    class Tracker:
        def process_cycle(self, cycle):
            return SimpleNamespace(detection_id=10 if cycle.detections else None)

    def initialize(*args, **kwargs):
        calls.append(kwargs["inference_method"])
        return Tracker()

    monkeypatch.setattr(module["BR24BayesianRadarTracker"], "initialize", initialize)
    arguments = ["--max-cycles", "2", "--observations", "4"]
    if method is not None:
        arguments += ["--method", method]
    result = main(arguments)
    selected = method or "mcmc"
    canonical = "sequential_vi" if selected == "seq_vi" else selected
    assert calls == [canonical]
    name = "seq_vi" if canonical == "sequential_vi" else canonical
    path = output_dir / f"BR24_association_comparison_{name}.csv"
    assert list(output_dir.glob("*.csv")) == [path]
    assert result["cycle_counter"].tolist() == [4, 5]
    assert pd.read_csv(path)["inference_method"].eq(canonical).all()
    text = capsys.readouterr().out
    assert text.count("associations: MATCH") == 1


@pytest.mark.parametrize("bad_cycle", [4, 5])
def test_audit_rejects_invalid_imm_reference_detections(tmp_path, bad_cycle):
    module = runpy.run_path(SCRIPT_PATH)
    raw_csv, imm_csv, output_csv = _input_csvs(tmp_path)
    pd.DataFrame({"cycle_counter": [bad_cycle], "detection_id": [999]}).to_csv(
        imm_csv, index=False
    )
    with pytest.raises(ValueError, match="detection"):
        module["compare_associations"](
            raw_csv=raw_csv,
            imm_csv=imm_csv,
            output_csv=output_csv,
        )


def test_audit_rejects_a_different_imm_initial_detection(tmp_path):
    module = runpy.run_path(SCRIPT_PATH)
    raw_csv, imm_csv, output_csv = _input_csvs(tmp_path)
    pd.DataFrame({"cycle_counter": [4], "detection_id": [10]}).to_csv(
        imm_csv, index=False
    )
    with pytest.raises(ValueError, match="initial detection"):
        module["compare_associations"](
            raw_csv=raw_csv, imm_csv=imm_csv, output_csv=output_csv
        )


@pytest.mark.parametrize(
    "options", [{"max_cycles": 0}, {"max_cycles": -1}, {"observation_count": 1}]
)
def test_audit_rejects_invalid_cycle_and_window_limits(tmp_path, options):
    module = runpy.run_path(SCRIPT_PATH)
    raw_csv, imm_csv, output_csv = _input_csvs(tmp_path)
    with pytest.raises(ValueError):
        module["compare_associations"](
            raw_csv=raw_csv, imm_csv=imm_csv, output_csv=output_csv, **options
        )


def test_cli_rejects_multiple_methods_before_tracking(monkeypatch):
    module = runpy.run_path(SCRIPT_PATH)

    def forbidden_tracking(**kwargs):
        pytest.fail("Multiple methods must be rejected before tracking starts")

    monkeypatch.setitem(
        module["main"].__globals__, "compare_associations", forbidden_tracking
    )
    with pytest.raises(SystemExit) as error:
        module["main"](["--method", "smc", "vi"])
    assert error.value.code == 2
