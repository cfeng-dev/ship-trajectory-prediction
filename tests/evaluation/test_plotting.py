"""Tests for the shared posterior trajectory plot."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ship_trajectory_prediction.evaluation.plotting import plot_prediction

plt.switch_backend("Agg")


def test_plot_prediction_uses_the_requested_model_name(monkeypatch):
    """One public plot function should generate every model-specific title."""
    monkeypatch.setattr(plt, "show", lambda: None)

    figure, axis = plot_prediction(
        FakeWindow(),
        FakeFit(),
        model_name="Time-Varying Radius",
        max_posterior_trajectories=2,
    )

    assert axis.get_title() == "Bayesian Time-Varying Radius Prediction"
    assert len(axis.lines) == 5
    plt.close(figure)


def test_plot_prediction_connects_future_trajectories_to_prediction_start(
    monkeypatch,
):
    """Held-out and posterior paths should start at the final observation."""
    monkeypatch.setattr(plt, "show", lambda: None)

    figure, axis = plot_prediction(
        FakeWindow(),
        FakeFit(),
        model_name="Constant Radius",
        max_posterior_trajectories=2,
    )

    expected_start = (1.0, 0.5)
    for line in axis.lines[1:]:
        assert line.get_xdata()[0] == expected_start[0]
        assert line.get_ydata()[0] == expected_start[1]
    plt.close(figure)


@pytest.mark.parametrize("model_name", [None, "", "   "])
def test_plot_prediction_rejects_empty_model_name(model_name):
    """A model name is required to create a meaningful title."""
    with pytest.raises(ValueError, match="model_name"):
        plot_prediction(FakeWindow(), FakeFit(), model_name=model_name)


class FakeWindow:
    """Minimal shared trajectory-window interface for plotting."""

    def __init__(self):
        self.x_meters = np.array([0.0, 1.0, 2.0, 3.0])
        self.y_meters = np.array([0.0, 0.5, 1.0, 1.5])
        self.observation_count = 2

    @property
    def observed_slice(self):
        """Return the observed part of the trajectory."""
        return slice(0, self.observation_count)

    @property
    def prediction_slice(self):
        """Return the held-out part of the trajectory."""
        return slice(self.observation_count, None)


class FakeFit:
    """Minimal fit object exposing posterior mean trajectories."""

    def __init__(self):
        self.variables = {
            "x_prediction_mean": np.array([[2.0, 3.0], [2.1, 3.1]]),
            "y_prediction_mean": np.array([[1.0, 1.5], [1.1, 1.6]]),
        }

    def stan_variable(self, name):
        """Return one stored posterior variable."""
        return self.variables[name]
