"""Compare MATLAB and Python BR24 IMM diagnostics cycle by cycle."""

import argparse
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from bayestraj.observations.br24 import polar_to_cartesian, read_br24_tracking_input
from bayestraj.paths import project_path

IMM_PYTHON = project_path("external/IMM/br24_imm.py")
INPUT_CSV = project_path("data/processed/BR24Dataset20131107_with_reference.csv")
MATLAB_TRACE = project_path("data/processed/BR24_imm_matlab_trace.mat")
OUTPUT_CSV = project_path("results/BR24_imm_implementation_comparison.csv")


NUMERIC_FIELDS = (
    "x",
    "y",
    "cov_xx",
    "cov_xy",
    "cov_yy",
    "mode_cv",
    "mode_ctrv",
    "mode_ca",
)


def align_cycle_times(path, cycles):
    """Use the original MATLAB radar times after checking the CSV rounding."""
    data = loadmat(path, variable_names=["time", "cycle_counter"])
    times = np.asarray(data["time"], dtype=float).reshape(-1)
    counters = np.asarray(data["cycle_counter"]).reshape(-1)
    if times.shape != (len(cycles),) or not np.array_equal(
        counters, [cycle.cycle_counter for cycle in cycles]
    ):
        raise ValueError("MATLAB trace cycles do not match radar input.")
    if not np.all(np.isfinite(times)):
        raise ValueError("MATLAB radar times must be finite.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("MATLAB radar times must be strictly increasing.")
    csv_times = np.array([cycle.time_seconds for cycle in cycles])
    if not np.all(np.abs(times - csv_times) <= 0.05 + 1e-8):
        raise ValueError("MATLAB and CSV radar times differ beyond rounding tolerance.")
    return tuple(
        replace(cycle, time_seconds=float(time))
        for cycle, time in zip(cycles, times, strict=True)
    )


def read_matlab_trace(path, cycles) -> pd.DataFrame:
    """Read MATLAB's world-frame output and recover only unambiguous IDs."""
    data = loadmat(path)
    count = len(cycles)
    times = np.asarray(data["time"], dtype=float).reshape(-1)
    counters = np.asarray(data["cycle_counter"]).reshape(-1)
    state = np.asarray(data["state"], dtype=float)
    covariance = np.asarray(data["covariance"], dtype=float)
    mode = np.asarray(data["mode"], dtype=float)
    measurement = np.asarray(data["measurement"], dtype=float)
    sensor_position = np.asarray(data["sensor_position"], dtype=float)
    if (
        times.shape != (count,)
        or counters.shape != (count,)
        or state.shape != (5, count)
        or covariance.shape != (5, 5, count)
        or mode.shape != (3, count)
        or measurement.shape != (2, count)
        or sensor_position.shape != (2, count)
    ):
        raise ValueError("MATLAB trace dimensions do not match the radar cycles.")
    if not np.array_equal(counters, [cycle.cycle_counter for cycle in cycles]):
        raise ValueError("MATLAB trace cycles do not match radar input.")
    if not np.allclose(
        times, [cycle.time_seconds for cycle in cycles], rtol=0.0, atol=1e-6
    ):
        raise ValueError("MATLAB trace time values do not match radar input.")
    if not np.allclose(
        sensor_position,
        np.array([cycle.sensor_position for cycle in cycles]).T,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("MATLAB sensor positions do not match radar input.")
    for field, attribute in (
        ("detection_range", "range_m"),
        ("detection_azimuth", "azimuth_rad"),
        ("detection_std_range", "std_range_m"),
        ("detection_std_azimuth", "std_azimuth_rad"),
    ):
        matlab_values = np.asarray(data[field], dtype=object).reshape(-1)
        if len(matlab_values) != count:
            raise ValueError(f"MATLAB {field} cycles do not match radar input.")
        for index, cycle in enumerate(cycles):
            actual = np.asarray(matlab_values[index], dtype=float).reshape(-1)
            expected = np.array(
                [getattr(detection, attribute) for detection in cycle.detections]
            )
            if actual.shape != expected.shape or not np.allclose(
                actual, expected, rtol=0.0, atol=1e-8
            ):
                raise ValueError(
                    f"MATLAB {field} does not match radar input in cycle {cycle.cycle_counter}."
                )
    if not all(
        np.isfinite(values).all() for values in (times, state, covariance, mode)
    ):
        raise ValueError("MATLAB state, covariance, mode, and time must be finite.")

    rows = []
    for index, cycle in enumerate(cycles):
        observed = measurement[:, index]
        if np.isnan(observed).all():
            status, detection_id = "missed", None
        elif not np.isfinite(observed).all():
            raise ValueError("MATLAB measurement must be finite or entirely missing.")
        else:
            matches = [
                detection.detection_id
                for detection in cycle.detections
                if np.allclose(
                    cycle.sensor_position
                    + polar_to_cartesian(detection.range_m, detection.azimuth_rad)[0],
                    observed,
                    rtol=0.0,
                    atol=1e-6,
                )
            ]
            status = (
                "matched"
                if len(matches) == 1
                else ("ambiguous" if matches else "unmatched")
            )
            detection_id = matches[0] if len(matches) == 1 else None
        rows.append(
            _trace_row(
                cycle,
                state[:, index],
                covariance[:, :, index],
                mode[:, index],
                status,
                detection_id,
            )
        )
    return pd.DataFrame(rows)


def _trace_row(cycle, state, covariance, mode, status, detection_id):
    return {
        "cycle_counter": cycle.cycle_counter,
        "time": cycle.time_seconds,
        "association_status": status,
        "detection_id": detection_id,
        "x": state[0],
        "y": state[1],
        "cov_xx": covariance[0, 0],
        "cov_xy": covariance[0, 1],
        "cov_yy": covariance[1, 1],
        "mode_cv": mode[0],
        "mode_ctrv": mode[1],
        "mode_ca": mode[2],
    }


def track_python_trace(cycles) -> pd.DataFrame:
    """Record the local Python IMM state once per cycle, including misses."""
    if not IMM_PYTHON.is_file():
        raise FileNotFoundError(
            "Optional IMM implementation not found: external/IMM/br24_imm.py"
        )
    if not cycles or not cycles[0].detections:
        raise ValueError("The earliest BR24 radar cycle requires a detection.")
    imm = runpy.run_path(str(IMM_PYTHON))
    initial = cycles[0].detections[0]
    state = imm["_initialize_imm_state"](
        polar_to_cartesian(initial.range_m, initial.azimuth_rad)[0]
    )
    initial_state = state.ctrv_state.copy()
    initial_state[:2] += cycles[0].sensor_position
    rows = [
        _trace_row(
            cycles[0],
            initial_state,
            state.ctrv_covariance,
            state.mode_probabilities,
            "matched",
            initial.detection_id,
        )
    ]
    for previous, cycle in zip(cycles, cycles[1:], strict=False):
        predicted, mixing, predicted_modes = imm["_predict_imm"](
            state,
            cycle.time_seconds - previous.time_seconds,
            cycle.sensor_position - previous.sensor_position,
        )
        detections = pd.DataFrame(
            [
                {
                    "detection_id": d.detection_id,
                    "range": d.range_m,
                    "azimuth": d.azimuth_rad,
                    "std_range": d.std_range_m,
                    "std_azimuth": d.std_azimuth_rad,
                }
                for d in cycle.detections
            ]
        )
        state, record = imm["_associate_and_update_imm"](
            predicted=predicted,
            mixing_probabilities=mixing,
            predicted_mode_probabilities=predicted_modes,
            detections=detections,
            time_seconds=cycle.time_seconds,
            cycle_counter=cycle.cycle_counter,
            sensor_position=cycle.sensor_position,
        )
        combined, covariance = imm["_mix_to_ctrv"](state, state.mode_probabilities)
        combined = combined.copy()
        combined[:2] += cycle.sensor_position
        rows.append(
            _trace_row(
                cycle,
                combined,
                covariance,
                state.mode_probabilities,
                "matched" if record is not None else "missed",
                record["detection_id"] if record is not None else None,
            )
        )
    return pd.DataFrame(rows)


def compare_traces(matlab: pd.DataFrame, python: pd.DataFrame) -> pd.DataFrame:
    """Compare complete cycle-aligned diagnostics without hiding ambiguities."""
    required = {
        "cycle_counter",
        "time",
        "association_status",
        "detection_id",
        *NUMERIC_FIELDS,
    }
    for name, frame in (("MATLAB", matlab), ("Python", python)):
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} trace is missing required columns.")
        if frame["cycle_counter"].duplicated().any():
            raise ValueError(f"{name} trace contains duplicate cycles.")
        if not np.isfinite(
            frame[["cycle_counter", "time", *NUMERIC_FIELDS]].to_numpy(dtype=float)
        ).all():
            raise ValueError(f"{name} trace contains non-finite numeric values.")
        if (
            not frame["association_status"]
            .isin({"matched", "missed", "ambiguous", "unmatched"})
            .all()
        ):
            raise ValueError(f"{name} trace contains an invalid association status.")
        ids_present = frame["detection_id"].notna()
        if not (ids_present == (frame["association_status"] == "matched")).all():
            raise ValueError(f"{name} trace has inconsistent association IDs.")
    merged = matlab.merge(
        python,
        on="cycle_counter",
        how="outer",
        suffixes=("_matlab", "_python"),
        indicator=True,
        validate="one_to_one",
    ).sort_values("cycle_counter", ignore_index=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("MATLAB and Python cycles must match exactly.")
    if not np.allclose(
        merged["time_matlab"], merged["time_python"], rtol=0.0, atol=1e-6
    ):
        raise ValueError("MATLAB and Python cycle time values must match.")
    for field in NUMERIC_FIELDS:
        merged[f"delta_{field}"] = merged[f"{field}_python"] - merged[f"{field}_matlab"]

    def compare_association(row):
        left, right = row["association_status_matlab"], row["association_status_python"]
        if "ambiguous" in (left, right) or "unmatched" in (left, right):
            return "ambiguous"
        if left == right == "missed":
            return "same"
        if (
            left == right == "matched"
            and row["detection_id_matlab"] == row["detection_id_python"]
        ):
            return "same"
        return "different"

    merged["association_comparison"] = merged.apply(compare_association, axis=1)
    return merged.drop(columns="_merge")


def main(matlab_trace=None, output_csv=None) -> pd.DataFrame:
    """Compare configured traces and replace the cyclewise results CSV."""
    matlab_trace = MATLAB_TRACE if matlab_trace is None else Path(matlab_trace)
    output_csv = OUTPUT_CSV if output_csv is None else Path(output_csv)
    cycles = read_br24_tracking_input(INPUT_CSV)
    cycles = align_cycle_times(matlab_trace, cycles)
    matlab = read_matlab_trace(matlab_trace, cycles)
    python = track_python_trace(cycles)
    comparison = compare_traces(matlab, python)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_csv, index=False)
    print(f"Compared {len(comparison)} cycles; output: {output_csv}")
    print("Association counts:")
    print(comparison["association_comparison"].value_counts().to_string())
    for field in NUMERIC_FIELDS:
        print(
            f"Maximum absolute {field} difference: {comparison[f'delta_{field}'].abs().max():.9g}"
        )
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab-trace", type=Path, default=MATLAB_TRACE)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    arguments = parser.parse_args()
    main(matlab_trace=arguments.matlab_trace, output_csv=arguments.output)
