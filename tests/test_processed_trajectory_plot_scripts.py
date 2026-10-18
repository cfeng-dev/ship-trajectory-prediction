"""Tests for the processed-trajectory plotting entrypoints."""

import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bayestraj.inference.ctrv_smc import SequentialMonteCarloCTRVConfig

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
            "plot_br24_reference_trajectory.py",
            "BR24-Ziel-Referenztrajektorie mit Fahrtrichtung",
            [0.0, 3.0, 6.0],
        ),
        (
            "plot_br24_imm_tracked_trajectory.py",
            "IMM-getrackte BR24-Zieltrajektorie mit Fahrtrichtung",
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


def test_br24_bayesian_plot_tracks_every_cycle_without_reference_input(
    tmp_path, monkeypatch
):
    """The displayed track is online SMC output, including missed cycles."""
    script = (
        PROJECT_ROOT
        / "experiments/data_exploration/plot_br24_bayesian_ctrv_tracked_trajectory.py"
    )
    assert script.exists()
    module = runpy.run_path(script)
    input_csv = tmp_path / "radar.csv"
    radar = pd.DataFrame(
        {
            "time": [10.0, 10.0, 11.5, 14.0],
            "cycle_counter": [4, 4, 5, 6],
            "sensor_x": [100.0] * 4,
            "sensor_y": [-20.0] * 4,
            "sensor_yaw": [0.0] * 4,
            "detection_id": [10, 9, 10, np.nan],
            "range": [200.0, 10.0, 10.0, np.nan],
            "azimuth": [0.0, 0.0, 0.0, np.nan],
            "std_range": [0.5, 0.5, 0.5, np.nan],
            "std_azimuth": [0.0, 0.0, 0.0, np.nan],
            "x_relative": [200.0, 10.0, 10.0, np.nan],
            "y_relative": [0.0, 0.0, 0.0, np.nan],
            "ground_truth_x": [9999.0] * 4,
            "ground_truth_y": [-9999.0] * 4,
            "ground_truth_yaw": [0.0] * 4,
            "ground_truth_velocity": [0.0] * 4,
        }
    )
    radar.to_csv(input_csv, index=False)
    globals_ = module["main"].__globals__
    monkeypatch.setitem(globals_, "INPUT_CSV", input_csv)
    monkeypatch.setitem(
        globals_,
        "SMC_CONFIG",
        SequentialMonteCarloCTRVConfig(particle_count=128, posterior_draw_count=16),
    )
    monkeypatch.setattr(module["plotting"].plt, "show", lambda: None)
    plotted_data = []
    plot_trajectory = module["plotting"].plot_processed_trajectory

    def record_and_plot(data, **kwargs):
        plotted_data.append(data.copy())
        return plot_trajectory(data, **kwargs)

    monkeypatch.setattr(
        module["plotting"], "plot_processed_trajectory", record_and_plot
    )
    try:
        module["main"]()
        axis = plt.gcf().axes[0]
        first_x = np.asarray(axis.lines[0].get_xdata()).copy()
        first_y = np.asarray(axis.lines[0].get_ydata()).copy()
        assert len(first_x) == 3
        assert abs(first_x[0] - 110.0) < 2.0
        assert abs(first_y[0] + 20.0) < 2.0
        assert plotted_data[0]["time"].tolist() == [10.0, 11.5, 14.0]
        assert not np.allclose(first_x[1:], 110.0)
        assert (
            axis.get_title()
            == "Bayesian-CTRV-getrackte BR24-Zieltrajektorie mit Fahrtrichtung"
        )
        plt.close("all")

        radar["ground_truth_x"] = -1e6
        radar["ground_truth_y"] = 1e6
        radar["ground_truth_yaw"] = 2.0
        radar["ground_truth_velocity"] = 1000.0
        radar.to_csv(input_csv, index=False)
        module["main"]()
        axis = plt.gcf().axes[0]
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), first_x)
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), first_y)
    finally:
        plt.close("all")
