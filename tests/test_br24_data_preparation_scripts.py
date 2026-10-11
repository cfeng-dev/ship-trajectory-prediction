"""Tests for BR24 data-preparation script configuration."""

import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_br24_tracking_input_export_writes_to_processed_data():
    module = runpy.run_path(
        PROJECT_ROOT
        / "experiments"
        / "data_preparation"
        / "export_br24_dataset20131107.py"
    )

    assert (
        str(module["OUTPUT_CSV"])
        .replace("\\", "/")
        .endswith("data/processed/BR24Dataset20131107.csv")
    )
