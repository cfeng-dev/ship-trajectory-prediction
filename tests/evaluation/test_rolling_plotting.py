"""Tests for shared rolling-evaluation trajectory plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ship_trajectory_prediction.evaluation.plotting import (
    ROLLING_FIGURE_SIZE,
    RollingPosteriorPlotData,
    plot_bayesian_rolling_predictions,
    plot_deterministic_rolling_predictions,
)
from ship_trajectory_prediction.evaluation.prediction_plotting import (
    AXIS_LABEL_FONT_SIZE,
    AXIS_TICK_FONT_SIZE,
    PLOT_TITLE_FONT_SIZE,
    PLOT_TITLE_FONT_WEIGHT,
)


def _assert_shared_rolling_style(axis):
    assert axis.get_xlabel() == "Ostposition x [m]"
    assert axis.get_ylabel() == "Nordposition y [m]"
    assert axis.title.get_fontsize() == PLOT_TITLE_FONT_SIZE
    assert axis.title.get_fontweight() == PLOT_TITLE_FONT_WEIGHT
    assert axis.title.get_position()[1] == pytest.approx(1.0)
    assert axis.xaxis.label.get_fontsize() == AXIS_LABEL_FONT_SIZE
    assert axis.yaxis.label.get_fontsize() == AXIS_LABEL_FONT_SIZE
    assert all(
        label.get_fontsize() == AXIS_TICK_FONT_SIZE
        for label in (*axis.get_xticklabels(), *axis.get_yticklabels())
    )
    assert axis.get_aspect() == pytest.approx(1.0)
    assert axis.get_legend()._loc == 1


def test_deterministic_rolling_plot_uses_shared_german_style(monkeypatch):
    predictions = pd.DataFrame(
        {
            "window_index": [0, 0],
            "forecast_origin_x_route": [1.0, 1.0],
            "forecast_origin_y_route": [2.0, 2.0],
            "x_predicted_route": [3.0, 5.0],
            "y_predicted_route": [4.0, 6.0],
        }
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    figure, axis = plot_deterministic_rolling_predictions(
        np.array([0.0, 1.0, 3.0, 5.0]),
        np.array([0.0, 2.0, 4.0, 6.0]),
        predictions,
        initial_observation_count=2,
        window_mode="sliding",
        observed_route_x=np.array([0.0, 1.0, 3.0, 5.0]),
        observed_route_y=np.array([0.0, 2.0, 4.0, 6.0]),
        position_noise_std_m=0.0,
    )

    assert axis.get_title() == (
        "Rollierende deterministische CTRV-Prognose (gleitendes Fenster)"
    )
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Anfängliche Beobachtungen",
        "Aufgezeichnete Trajektorie",
        "Rollierende deterministische CTRV-Prognosen",
        "Startpunkte der Prognosen",
    ]
    assert figure.get_size_inches() == pytest.approx(ROLLING_FIGURE_SIZE)
    _assert_shared_rolling_style(axis)
    plt.close(figure)


def test_bayesian_rolling_plot_uses_the_same_typography(monkeypatch):
    group = RollingPosteriorPlotData(
        forecast_origin_x=1.0,
        forecast_origin_y=2.0,
        x_samples=np.array([[3.0, 5.0], [3.2, 5.2]]),
        y_samples=np.array([[4.0, 6.0], [4.2, 6.2]]),
        forecast_time_seconds=np.array([0.0, 10.0, 20.0]),
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    figure, axis = plot_bayesian_rolling_predictions(
        np.array([0.0, 1.0, 3.0, 5.0]),
        np.array([0.0, 2.0, 4.0, 6.0]),
        [group],
        initial_observation_count=2,
        window_mode="sliding",
        sample_trajectories_per_forecast=1,
        observed_route_x=np.array([0.0, 1.0, 3.0, 5.0]),
        observed_route_y=np.array([0.0, 2.0, 4.0, 6.0]),
    )

    assert axis.get_title() == (
        "Rollierende bayessche CTRV-Prognose (gleitendes Fenster)"
    )
    assert figure.get_size_inches() == pytest.approx(ROLLING_FIGURE_SIZE)
    _assert_shared_rolling_style(axis)
    plt.close(figure)
