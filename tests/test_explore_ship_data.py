"""Tests for the recorded ship-data exploration entry point."""

from unittest.mock import Mock

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import experiments.data_exploration.explore_ship_data as experiment  # noqa: E402
import ship_trajectory_prediction.observations.coordinates as coordinates  # noqa: E402
import ship_trajectory_prediction.observations.plotting as plotting  # noqa: E402


@pytest.fixture(autouse=True)
def _prevent_test_windows(monkeypatch):
    """Avoid GUI windows while inspecting the generated figures."""
    monkeypatch.setattr(plt, "show", Mock())


def _create_ship_data():
    """Create one short curved run with all columns used by the plots."""
    x_meters = np.array([0.0, 12.0, 25.0, 39.0, 54.0])
    y_meters = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    longitude, latitude = coordinates.local_to_gps_coordinates(
        x_meters,
        y_meters,
        reference_longitude=8.0,
        reference_latitude=54.0,
    )
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=5, freq="10s", tz="UTC"),
            "run_id": np.ones(5, dtype=int),
            "gps_longitude": longitude,
            "gps_latitude": latitude,
            "gps_speed": np.full(5, 5.0),
            "shaft_speed": np.linspace(100.0, 120.0, 5),
            "thruster_speed": np.linspace(200.0, 220.0, 5),
        }
    )


def test_position_noise_is_reproducible_and_does_not_change_source_data():
    """A fixed seed should perturb copied GPS positions reproducibly."""
    source = _create_ship_data()
    original = source.copy(deep=True)

    first = experiment._add_position_noise(
        source,
        additional_noise_std_m=5.0,
        seed=2026,
    )
    second = experiment._add_position_noise(
        source,
        additional_noise_std_m=5.0,
        seed=2026,
    )

    pd.testing.assert_frame_equal(source, original)
    pd.testing.assert_frame_equal(first, second)
    assert not np.array_equal(first["gps_longitude"], source["gps_longitude"])
    assert not np.array_equal(first["gps_latitude"], source["gps_latitude"])
    pd.testing.assert_frame_equal(
        first.drop(columns=["gps_longitude", "gps_latitude"]),
        source.drop(columns=["gps_longitude", "gps_latitude"]),
    )


def test_zero_position_noise_returns_an_unchanged_copy():
    """Zero noise should preserve every recorded value without aliasing data."""
    source = _create_ship_data()

    result = experiment._add_position_noise(
        source,
        additional_noise_std_m=0.0,
        seed=2026,
    )

    assert result is not source
    pd.testing.assert_frame_equal(result, source)


@pytest.mark.parametrize(
    ("standard_deviation", "seed", "message"),
    [
        (-1.0, 2026, "additional_noise_std_m"),
        (np.inf, 2026, "additional_noise_std_m"),
        (1.0, -1, "seed"),
        (1.0, True, "seed"),
    ],
)
def test_position_noise_rejects_invalid_configuration(
    standard_deviation,
    seed,
    message,
):
    """Invalid noise settings should fail with a focused message."""
    with pytest.raises(ValueError, match=message):
        experiment._add_position_noise(
            _create_ship_data(),
            additional_noise_std_m=standard_deviation,
            seed=seed,
        )


def test_ship_data_plots_can_hide_all_legends():
    """One shared style setting should suppress every ship-data legend."""
    data = _create_ship_data()
    plot_style = plotting.ShipDataPlotStyle(show_legend=False)

    plotting.plot_ship_trajectory(data, coordinate_unit="m", plot_style=plot_style)
    trajectory_axis = plt.gca()
    speed_figures, speed_axes = plotting.plot_ship_speeds(
        data,
        plot_style=plot_style,
    )
    curvature_figure, curvature_axis = plotting.plot_ship_curvature(
        data,
        plot_style=plot_style,
    )

    assert trajectory_axis.get_legend() is None
    assert all(axis.get_legend() is None for axis in speed_axes)
    assert curvature_axis.get_legend() is None
    plt.close(trajectory_axis.figure)
    for figure in speed_figures:
        plt.close(figure)
    plt.close(curvature_figure)


def test_ship_data_plot_style_shows_legends_by_default():
    """Existing callers should retain legends unless they disable them."""
    assert plotting.ShipDataPlotStyle().show_legend is True
