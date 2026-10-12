"""Tests for the BR24 IMM-versus-Bayesian association audit script."""

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
SCRIPT_PATH = PROJECT_ROOT / "experiments/data_exploration/compare_br24_associations.py"


def test_association_audit_uses_the_imm_initialization_and_records_misses(tmp_path):
    assert SCRIPT_PATH.exists()
    module = runpy.run_path(SCRIPT_PATH)
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
