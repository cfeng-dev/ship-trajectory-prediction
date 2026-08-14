"""Tests for the deterministic CTRV single-window prediction workflow."""

import numpy as np
import pandas as pd
import pytest

import ship_trajectory_prediction.forecasting.deterministic_ctrv_workflow as workflow
from ship_trajectory_prediction.forecasting.deterministic_ctrv_workflow import (
    DETERMINISTIC_PLOT_TITLE,
    _add_position_observation_noise,
    plot_deterministic_ctrv_prediction,
)
from ship_trajectory_prediction.observations import TrajectoryWindowData
from ship_trajectory_prediction.validation.prediction_plotting import (
    AXIS_LABEL_FONT_SIZE,
    AXIS_TICK_FONT_SIZE,
    PLOT_FIGURE_SIZE,
    PLOT_TITLE_FONT_SIZE,
    PLOT_TITLE_FONT_WEIGHT,
)


def _window():
    return TrajectoryWindowData(
        timestamps=pd.date_range("2026-01-01", periods=5, freq="s", tz="UTC"),
        time_seconds=np.arange(5, dtype=float),
        x_meters=np.arange(5, dtype=float),
        y_meters=np.arange(5, dtype=float) * 2,
        reference_longitude=10.0,
        reference_latitude=54.0,
        gps_speed_mps=np.ones(5),
        observation_count=3,
    )


def test_position_observation_noise_is_reproducible_and_keeps_targets():
    """Noise should affect observations without changing held-out truth."""
    window = _window()

    first = _add_position_observation_noise(
        window,
        additional_noise_std_m=2.0,
        seed=2026,
    )
    second = _add_position_observation_noise(
        window,
        additional_noise_std_m=2.0,
        seed=2026,
    )

    np.testing.assert_array_equal(first.x_meters, second.x_meters)
    np.testing.assert_array_equal(first.y_meters, second.y_meters)
    assert not np.array_equal(
        first.x_meters[first.observed_slice],
        window.x_meters[window.observed_slice],
    )
    np.testing.assert_array_equal(
        first.x_meters[first.prediction_slice],
        window.x_meters[window.prediction_slice],
    )
    np.testing.assert_array_equal(
        first.y_meters[first.prediction_slice],
        window.y_meters[window.prediction_slice],
    )


def test_deterministic_plot_uses_german_bayesian_plot_style(monkeypatch):
    """The deterministic plot should share German labels and typography."""
    monkeypatch.setattr(workflow.plt, "show", lambda: None)
    prediction_table = pd.DataFrame(
        {
            "x_actual": [3.0, 4.0],
            "y_actual": [6.0, 8.0],
            "x_predicted": [3.1, 4.2],
            "y_predicted": [5.9, 7.8],
        }
    )

    figure, axis = plot_deterministic_ctrv_prediction(
        _window(),
        prediction_table,
        additional_position_noise_std_m=2.0,
    )

    assert axis.get_title() == DETERMINISTIC_PLOT_TITLE
    assert axis.get_xlabel() == "Ostposition x [m]"
    assert axis.get_ylabel() == "Nordposition y [m]"
    assert axis.title.get_fontsize() == PLOT_TITLE_FONT_SIZE
    assert axis.title.get_fontweight() == PLOT_TITLE_FONT_WEIGHT
    assert axis.xaxis.label.get_fontsize() == AXIS_LABEL_FONT_SIZE
    assert axis.yaxis.label.get_fontsize() == AXIS_LABEL_FONT_SIZE
    assert all(
        label.get_fontsize() == AXIS_TICK_FONT_SIZE
        for label in (*axis.get_xticklabels(), *axis.get_yticklabels())
    )
    assert figure.get_size_inches() == pytest.approx(PLOT_FIGURE_SIZE)
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Verrauschte Beobachtungen",
        "Referenztrajektorie",
        "Deterministische CTRV-Vorhersage",
        "Prognosebeginn",
    ]
    workflow.plt.close(figure)
