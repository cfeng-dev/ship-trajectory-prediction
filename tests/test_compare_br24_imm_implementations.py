"""Behavioral checks for the MATLAB-versus-Python BR24 IMM audit."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from bayestraj.observations.br24 import (
    BR24Detection,
    BR24RadarCycle,
    read_br24_tracking_input,
)

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/data_exploration/compare_br24_imm_implementations.py"
)


def _trace():
    return pd.DataFrame(
        {
            "cycle_counter": [1, 2, 3],
            "time": [10.0, 11.0, 12.5],
            "association_status": ["matched", "matched", "missed"],
            "detection_id": [1.0, 3.0, np.nan],
            "x": [100.0, 101.0, 102.0],
            "y": [20.0, 21.0, 22.0],
            "cov_xx": [25.0, 2.0, 3.0],
            "cov_xy": [0.0, 0.1, 0.2],
            "cov_yy": [25.0, 4.0, 5.0],
            "mode_cv": [1.0, 0.8, 0.7],
            "mode_ctrv": [0.0, 0.1, 0.2],
            "mode_ca": [0.0, 0.1, 0.1],
        }
    )


def _matlab_radar_inputs(cycles):
    def cells(attribute):
        result = np.empty((1, len(cycles)), dtype=object)
        for index, cycle in enumerate(cycles):
            result[0, index] = np.array(
                [getattr(detection, attribute) for detection in cycle.detections]
            )
        return result

    return {
        "sensor_position": np.stack(
            [cycle.sensor_position for cycle in cycles], axis=1
        ),
        "detection_range": cells("range_m"),
        "detection_azimuth": cells("azimuth_rad"),
        "detection_std_range": cells("std_range_m"),
        "detection_std_azimuth": cells("std_azimuth_rad"),
    }


def test_comparison_identifies_numeric_and_association_differences():
    assert SCRIPT.exists()
    module = runpy.run_path(SCRIPT)
    matlab = _trace()
    python = _trace()
    python.loc[1, "x"] += 0.5
    python.loc[1, "detection_id"] = 4
    matlab.loc[2, "association_status"] = "ambiguous"

    comparison = module["compare_traces"](matlab, python)

    assert comparison["cycle_counter"].tolist() == [1, 2, 3]
    np.testing.assert_allclose(comparison["delta_x"], [0.0, 0.5, 0.0])
    assert comparison["association_comparison"].tolist() == [
        "same",
        "different",
        "ambiguous",
    ]


def test_comparison_rejects_missing_cycles_and_mismatched_times():
    assert SCRIPT.exists()
    module = runpy.run_path(SCRIPT)
    matlab = _trace()
    with pytest.raises(ValueError, match="cycles"):
        module["compare_traces"](matlab, _trace().iloc[:2])
    shifted = _trace()
    shifted.loc[1, "time"] += 0.1
    with pytest.raises(ValueError, match="time"):
        module["compare_traces"](matlab, shifted)


def test_comparison_rejects_nonfinite_states_and_invalid_associations():
    module = runpy.run_path(SCRIPT)
    incomplete = _trace()
    incomplete.loc[1, "x"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        module["compare_traces"](_trace(), incomplete)
    invalid = _trace()
    invalid.loc[1, "association_status"] = "unknown"
    with pytest.raises(ValueError, match="association"):
        module["compare_traces"](_trace(), invalid)


def test_cycle_times_use_unrounded_matlab_values_without_changing_input(tmp_path):
    module = runpy.run_path(SCRIPT)
    cycles = (
        BR24RadarCycle(10.0, 1, np.array([100.0, 50.0]), 0.0, ()),
        BR24RadarCycle(11.0, 2, np.array([100.0, 50.0]), 0.0, ()),
    )
    source = tmp_path / "times.mat"
    savemat(source, {"time": [10.012, 10.971], "cycle_counter": [1, 2]})

    aligned = module["align_cycle_times"](source, cycles)

    assert [cycle.time_seconds for cycle in aligned] == [10.012, 10.971]
    assert [cycle.time_seconds for cycle in cycles] == [10.0, 11.0]
    assert aligned[0].detections == cycles[0].detections


@pytest.mark.parametrize(
    ("times", "counters", "message"),
    [
        ([10.051, 11.0], [1, 2], "rounding"),
        ([10.0, 11.0], [1, 3], "cycles"),
        ([10.0, float("nan")], [1, 2], "finite"),
    ],
)
def test_cycle_times_reject_invalid_or_unrelated_radar_times(
    tmp_path, times, counters, message
):
    module = runpy.run_path(SCRIPT)
    cycles = (
        BR24RadarCycle(10.0, 1, np.array([100.0, 50.0]), 0.0, ()),
        BR24RadarCycle(11.0, 2, np.array([100.0, 50.0]), 0.0, ()),
    )
    source = tmp_path / "times.mat"
    savemat(source, {"time": times, "cycle_counter": counters})
    with pytest.raises(ValueError, match=message):
        module["align_cycle_times"](source, cycles)


def test_matlab_trace_flags_ambiguous_and_missing_detections(tmp_path):
    module = runpy.run_path(SCRIPT)

    def detection(identity):
        return BR24Detection(identity, 10.0, 0.0, 0.5, 0.01)

    cycles = (
        BR24RadarCycle(
            10.0, 1, np.array([100.0, 50.0]), 0.0, (detection(1), detection(2))
        ),
        BR24RadarCycle(11.0, 2, np.array([100.0, 50.0]), 0.0, (detection(3),)),
        BR24RadarCycle(12.5, 3, np.array([100.0, 50.0]), 0.0, ()),
    )
    state = np.zeros((5, 3))
    state[:2] = [[110, 111, 112], [50, 51, 52]]
    covariance = np.repeat(np.eye(5)[:, :, None], 3, axis=2)
    trace_path = tmp_path / "matlab.mat"
    savemat(
        trace_path,
        {
            "time": [10.0, 11.0, 12.5],
            "cycle_counter": [1, 2, 3],
            "state": state,
            "covariance": covariance,
            "mode": [[1, 0.8, 0.7], [0, 0.1, 0.2], [0, 0.1, 0.1]],
            "measurement": [[110, 110, np.nan], [50, 50, np.nan]],
            **_matlab_radar_inputs(cycles),
        },
    )

    result = module["read_matlab_trace"](trace_path, cycles)

    assert result["association_status"].tolist() == ["ambiguous", "matched", "missed"]
    assert np.isnan(result.loc[0, "detection_id"])
    assert result.loc[1, "detection_id"] == 3
    np.testing.assert_allclose(result["x"], [110, 111, 112])
    np.testing.assert_allclose(result["cov_xx"], [1, 1, 1])

    changed = list(cycles)
    changed[1] = BR24RadarCycle(11.0, 2, np.array([101.0, 50.0]), 0.0, (detection(3),))
    with pytest.raises(ValueError, match="sensor"):
        module["read_matlab_trace"](trace_path, changed)
    changed[1] = BR24RadarCycle(
        11.0,
        2,
        np.array([100.0, 50.0]),
        0.0,
        (BR24Detection(3, 11.0, 0.0, 0.5, 0.01),),
    )
    with pytest.raises(ValueError, match="detection_range"):
        module["read_matlab_trace"](trace_path, changed)


def test_python_imm_trace_preserves_initial_state_and_missed_cycles():
    module = runpy.run_path(SCRIPT)
    if not module["IMM_PYTHON"].is_file():
        pytest.skip("Optional external IMM implementation is not available")
    detection = BR24Detection(1, 10.0, 0.0, 0.5, 0.01)
    cycles = (
        BR24RadarCycle(10.0, 1, np.array([100.0, -20.0]), 0.0, (detection,)),
        BR24RadarCycle(11.0, 2, np.array([100.0, -20.0]), 0.0, ()),
    )

    result = module["track_python_trace"](cycles)

    assert result["association_status"].tolist() == ["matched", "missed"]
    assert result["cycle_counter"].tolist() == [1, 2]
    np.testing.assert_allclose(
        result.loc[0, ["x", "y"]].to_numpy(dtype=float), [110, -20]
    )
    np.testing.assert_allclose(
        result.loc[0, ["cov_xx", "cov_xy", "cov_yy"]].to_numpy(dtype=float), [25, 0, 25]
    )
    np.testing.assert_allclose(
        result[["mode_cv", "mode_ctrv", "mode_ca"]].sum(axis=1), [1, 1]
    )


def test_main_replaces_cyclewise_comparison_at_fixed_output_path(
    tmp_path, monkeypatch, capsys
):
    module = runpy.run_path(SCRIPT)
    assert module["OUTPUT_CSV"] == (
        SCRIPT.parents[2] / "results/BR24_imm_implementation_comparison.csv"
    )
    if not module["IMM_PYTHON"].is_file():
        pytest.skip("Optional external IMM implementation is not available")
    input_csv = tmp_path / "radar.csv"
    pd.DataFrame(
        {
            "time": [10.0, 11.0],
            "cycle_counter": [1, 2],
            "sensor_x": [100.0, 100.0],
            "sensor_y": [-20.0, -20.0],
            "sensor_yaw": [0.0, 0.0],
            "detection_id": [1, np.nan],
            "range": [10.0, np.nan],
            "azimuth": [0.0, np.nan],
            "std_range": [0.5, np.nan],
            "std_azimuth": [0.01, np.nan],
            "x_relative": [10.0, np.nan],
            "y_relative": [0.0, np.nan],
        }
    ).to_csv(input_csv, index=False)
    state = np.zeros((5, 2))
    state[:2] = [[110, 110], [-20, -20]]
    matlab = tmp_path / "matlab.mat"
    savemat(
        matlab,
        {
            "time": [10.012, 10.971],
            "cycle_counter": [1, 2],
            "state": state,
            "covariance": np.repeat(np.eye(5)[:, :, None], 2, axis=2),
            "mode": [[1, 1], [0, 0], [0, 0]],
            "measurement": [[110, np.nan], [-20, np.nan]],
            **_matlab_radar_inputs(read_br24_tracking_input(input_csv)),
        },
    )
    output = tmp_path / "comparison.csv"
    main = module["main"]
    monkeypatch.setitem(main.__globals__, "INPUT_CSV", input_csv)
    monkeypatch.setitem(main.__globals__, "MATLAB_TRACE", matlab)
    monkeypatch.setitem(main.__globals__, "OUTPUT_CSV", output)

    main()
    table = pd.read_csv(output)
    assert table["association_comparison"].tolist() == ["same", "same"]
    assert table["cycle_counter"].tolist() == [1, 2]
    np.testing.assert_allclose(table["time_python"], [10.012, 10.971])
    output.write_text("outdated result", encoding="utf-8")
    main()
    pd.testing.assert_frame_equal(pd.read_csv(output), table)
    assert not (tmp_path / "comparison_1.csv").exists()
    assert str(output) in capsys.readouterr().out


def test_python_trace_processes_real_br24_cycles_when_available():
    module = runpy.run_path(SCRIPT)
    if not module["IMM_PYTHON"].is_file() or not module["INPUT_CSV"].is_file():
        pytest.skip("Optional IMM implementation or BR24 dataset is not available")
    cycles = read_br24_tracking_input(module["INPUT_CSV"])

    trace = module["track_python_trace"](cycles)

    assert len(trace) == len(cycles)
    assert trace["cycle_counter"].tolist() == [cycle.cycle_counter for cycle in cycles]
    assert (trace["association_status"] == "matched").any()
    np.testing.assert_allclose(
        trace[["mode_cv", "mode_ctrv", "mode_ca"]].sum(axis=1), 1.0, atol=1e-12
    )
