"""Tests for the processed-trajectory plotting entrypoints."""

import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("script_name", "title", "expected_x_coordinates"),
    (
        (
            "plot_shiptech_trajectory.py",
            "Shiptech-Trajektorie mit Fahrtrichtung",
            [0.0, 3.0, 6.0],
        ),
        (
            "plot_br24_target_reference_trajectory.py",
            "BR24-Ziel-Referenztrajektorie mit Fahrtrichtung",
            [0.0, 3.0, 6.0],
        ),
        (
            "plot_br24_target_radar_trajectory.py",
            "BR24-Radar-Zieltrajektorie mit Fahrtrichtung",
            [0.0, 3.0, 6.0],
        ),
    ),
)
def test_processed_trajectory_plot_script_loads_its_configured_csv(
    tmp_path,
    monkeypatch,
    script_name,
    title,
    expected_x_coordinates,
):
    """Each entrypoint plots an interchangeable processed position CSV."""
    input_csv = tmp_path / "processed_trajectory.csv"
    pd.DataFrame(
        {
            "time": [0.0, 10.0, 20.0],
            "x": [0.0, 3.0, 6.0],
            "y": [0.0, 4.0, 8.0],
        }
    ).to_csv(input_csv, index=False)

    module = runpy.run_path(
        PROJECT_ROOT / "experiments" / "data_exploration" / script_name
    )
    monkeypatch.setitem(module["main"].__globals__, "INPUT_CSV", input_csv)
    monkeypatch.setattr(module["plotting"].plt, "show", lambda: None)
    try:
        module["main"]()

        trajectory_axis = plt.gcf().axes[0]
        assert trajectory_axis.get_title() == title
        np.testing.assert_allclose(
            trajectory_axis.lines[0].get_xdata(),
            expected_x_coordinates,
        )
    finally:
        plt.close("all")
