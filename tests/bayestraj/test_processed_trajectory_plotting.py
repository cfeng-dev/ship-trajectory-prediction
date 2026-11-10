"""Tests for plotting processed local position trajectories."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import bayestraj.observations.plotting as plotting


@pytest.mark.parametrize(
    "times",
    [np.arange(7) * 90.0, 49073.0 + np.arange(7) * 90.0, [10, 11, 15, 50, 51, 52, 100]],
)
@pytest.mark.parametrize("arrow_step", [2, None, 100])
def test_plot_processed_trajectory_spaces_arrows_by_points_after_start(
    monkeypatch, times, arrow_step
):
    """Arrow positions depend on point count, including for offset/irregular times."""
    trajectory_data = pd.DataFrame(
        {
            "time": times,
            "x": np.arange(7.0),
            "y": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )

    monkeypatch.setattr(plotting.plt, "show", lambda: None)
    figure, axis = plotting.plot_processed_trajectory(
        trajectory_data,
        trajectory_label="Test trajectory",
        title="Processed trajectory",
        direction_arrow_step=arrow_step,
    )
    try:
        trajectory_line = axis.lines[0]
        assert trajectory_line.get_marker() == "None"
        np.testing.assert_allclose(trajectory_line.get_xdata(), np.arange(7.0))
        np.testing.assert_allclose(
            trajectory_line.get_ydata(), [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        )
        assert axis.get_xlabel() == "Ostposition x [m]"
        assert axis.get_ylabel() == "Nordposition y [m]"
        assert axis.get_title() == "Processed trajectory"
        expected_starts = [(2.0, 0.0), (4.0, 0.0)] if arrow_step == 2 else []
        assert [arrow.get_position() for arrow in axis.texts] == expected_starts
        assert [arrow.xy for arrow in axis.texts] == (
            [(3.0, 1.0), (5.0, 1.0)] if arrow_step == 2 else []
        )
        expected_labels = [
            "Test trajectory",
            "Start",
            "Ende",
        ]
        if expected_starts:
            expected_labels.append("Fahrtrichtung")
        assert [
            text.get_text() for text in axis.get_legend().get_texts()
        ] == expected_labels
        np.testing.assert_array_equal(trajectory_data["time"], times)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("arrow_step", [0, -1, True, 1.5, "2"])
def test_plot_processed_trajectory_rejects_invalid_arrow_step(arrow_step):
    data = pd.DataFrame({"time": [0, 1], "x": [0, 1], "y": [0, 1]})
    with pytest.raises(ValueError, match="direction_arrow_step"):
        plotting.plot_processed_trajectory(data, direction_arrow_step=arrow_step)


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
