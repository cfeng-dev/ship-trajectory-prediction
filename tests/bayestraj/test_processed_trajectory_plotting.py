"""Tests for plotting processed local position trajectories."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bayestraj.observations.plotting as plotting


def test_plot_processed_trajectory_uses_local_positions_and_time_based_arrows(
    monkeypatch,
):
    """Processed x/y data is plotted with readable time-spaced directions."""
    trajectory_data = pd.DataFrame(
        {
            "time": [0.0, 90.0, 180.0, 270.0, 360.0],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )

    monkeypatch.setattr(plotting.plt, "show", lambda: None)
    figure, axis = plotting.plot_processed_trajectory(
        trajectory_data,
        trajectory_label="Test trajectory",
        title="Processed trajectory",
        direction_arrow_interval_seconds=180.0,
    )
    try:
        trajectory_line = axis.lines[0]
        np.testing.assert_allclose(
            trajectory_line.get_xdata(), [0.0, 1.0, 2.0, 3.0, 4.0]
        )
        np.testing.assert_allclose(
            trajectory_line.get_ydata(), [0.0, 1.0, 0.0, 1.0, 0.0]
        )
        assert axis.get_xlabel() == "Ostposition x [m]"
        assert axis.get_ylabel() == "Nordposition y [m]"
        assert axis.get_title() == "Processed trajectory"
        assert len(axis.texts) == 2
        assert [text.get_text() for text in axis.get_legend().get_texts()] == [
            "Test trajectory",
            "Start",
            "Ende",
            "Fahrtrichtung",
        ]
    finally:
        plt.close(figure)


def test_plot_processed_trajectory_adds_reproducible_noise_without_mutating_data(
    monkeypatch,
):
    """Configured plotting noise changes only the rendered local positions."""
    trajectory_data = pd.DataFrame(
        {
            "time": [0.0, 10.0, 20.0],
            "x": [0.0, 3.0, 6.0],
            "y": [0.0, 4.0, 8.0],
        }
    )
    original_data = trajectory_data.copy()

    monkeypatch.setattr(plotting.plt, "show", lambda: None)
    figure, axis = plotting.plot_processed_trajectory(
        trajectory_data,
        position_noise_std_m=5.0,
        position_noise_seed=2026,
    )
    try:
        trajectory_line = axis.lines[0]
        np.testing.assert_allclose(
            trajectory_line.get_xdata(),
            [-3.96561238, 4.20285642, -3.48163175],
        )
        np.testing.assert_allclose(
            trajectory_line.get_ydata(),
            [6.97885855, 7.1914737, 6.53976257],
        )
        pd.testing.assert_frame_equal(trajectory_data, original_data)
    finally:
        plt.close(figure)
