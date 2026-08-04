"""Tests for the shared posterior trajectory plot."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ship_trajectory_prediction.evaluation.plotting import (
    _joint_prediction_density,
    plot_operational_prediction,
    plot_prediction,
    plot_trajectory_paths,
)

plt.switch_backend("Agg")


def test_plot_trajectory_paths_supports_multiple_forecast_origins():
    """The shared renderer should accept variable-length rolling paths."""
    figure, axis = plot_trajectory_paths(
        observed_path=([0.0, 1.0], [0.0, 0.5]),
        reference_path=([0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 1.0, 1.5]),
        forecast_paths=(
            ([1.0, 1.8, 2.6], [0.5, 0.9, 1.3]),
            ([2.0, 2.7], [1.0, 1.4]),
        ),
        prediction_origins=([1.0, 2.0], [0.5, 1.0]),
        title="Bayesian CTRV Rolling Prediction",
        observed_label="Initial observations",
        reference_label="Recorded trajectory",
        forecast_label="Rolling posterior medians",
        prediction_origin_label="Forecast origins",
    )

    labels = [line.get_label() for line in axis.lines]
    assert labels[:3] == [
        "Initial observations",
        "Recorded trajectory",
        "Rolling posterior medians",
    ]
    assert labels[3].startswith("_")
    assert axis.collections[0].get_label() == "Forecast origins"
    assert axis.get_title() == "Bayesian CTRV Rolling Prediction"
    plt.close(figure)


@pytest.mark.parametrize(
    "forecast_paths",
    [(), (([1.0, 2.0], [1.0]),), (([1.0, np.nan], [1.0, 2.0]),)],
)
def test_plot_trajectory_paths_rejects_invalid_forecasts(forecast_paths):
    """Every shared forecast path must be non-empty, finite, and aligned."""
    with pytest.raises(ValueError, match="forecast_paths"):
        plot_trajectory_paths(
            observed_path=([0.0], [0.0]),
            reference_path=([0.0, 1.0], [0.0, 1.0]),
            forecast_paths=forecast_paths,
            title="Prediction",
        )


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


def test_plot_prediction_uses_equal_spatial_scale_and_professional_labels(
    monkeypatch,
):
    """The spatial plot should preserve geometry and use physical axis names."""
    monkeypatch.setattr(plt, "show", lambda: None)

    figure, axis = plot_prediction(
        FakeWindow(),
        FakeFit(),
        model_name="CTRV",
    )

    assert axis.get_aspect() == pytest.approx(1.0)
    assert axis.get_xlabel() == "East position [m]"
    assert axis.get_ylabel() == "North position [m]"
    legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert legend_labels == [
        "Observed history",
        "Held-out trajectory",
        "50% prediction region",
        "90% prediction region",
        "Posterior median",
        "Prediction start",
    ]
    plt.close(figure)


def test_plot_prediction_draws_at_most_30_sample_trajectories(monkeypatch):
    """A large posterior should still produce a readable number of paths."""
    monkeypatch.setattr(plt, "show", lambda: None)

    figure, axis = plot_prediction(
        FakeWindow(),
        ManyDrawFit(),
        model_name="CTRV",
        max_posterior_trajectories=100,
    )

    sample_lines = [line for line in axis.lines if line.get_alpha() == 0.06]
    assert len(sample_lines) == 30
    plt.close(figure)


def test_joint_prediction_regions_contain_valid_empirical_mass():
    """The 50% joint region should be nested in the joint 90% region."""
    fit = ManyDrawFit()
    _, _, density, thresholds = _joint_prediction_density(
        (
            fit.variables["x_prediction_mean"],
            fit.variables["y_prediction_mean"],
        )
    )

    region_50 = density >= thresholds[0.5]
    region_90 = density >= thresholds[0.9]
    assert thresholds[0.5] >= thresholds[0.9] > 0
    assert np.all(region_50 <= region_90)
    assert density[region_50].sum() >= 0.5
    assert density[region_90].sum() >= 0.9


def test_posterior_trajectory_selection_is_reproducible(monkeypatch):
    """The same plotting seed should select the same posterior trajectories."""
    monkeypatch.setattr(plt, "show", lambda: None)
    figures_and_axes = [
        plot_prediction(
            FakeWindow(),
            ManyDrawFit(),
            model_name="CTRV",
            max_posterior_trajectories=10,
            trajectory_sample_seed=seed,
        )
        for seed in (17, 17, 23)
    ]

    selected_paths = [
        [
            np.column_stack((line.get_xdata(), line.get_ydata()))
            for line in axis.lines
            if line.get_alpha() == 0.06
        ]
        for _, axis in figures_and_axes
    ]
    assert all(
        np.array_equal(first, second)
        for first, second in zip(selected_paths[0], selected_paths[1], strict=True)
    )
    assert any(
        not np.array_equal(first, third)
        for first, third in zip(selected_paths[0], selected_paths[2], strict=True)
    )
    for figure, _ in figures_and_axes:
        plt.close(figure)


def test_plot_prediction_adds_time_markers_and_evaluation_inset(monkeypatch):
    """Available elapsed times and held-out metrics should be visible."""
    monkeypatch.setattr(plt, "show", lambda: None)

    figure, axis = plot_prediction(
        FakeWindow(),
        FakeFit(),
        model_name="CTRV",
    )

    text_values = [text.get_text() for text in axis.texts]
    assert "+30 s" in text_values
    assert "+60 s" in text_values
    inset = next(text for text in text_values if "Observation duration" in text)
    assert "Prediction horizon: 60 s" in inset
    assert "ADE:" in inset
    assert "FDE:" in inset
    assert "90% coverage:" in inset
    plt.close(figure)


def test_operational_plot_omits_unknown_future_and_evaluation_metrics(monkeypatch):
    """The operational view should not expose held-out data or its metrics."""
    monkeypatch.setattr(plt, "show", lambda: None)

    figure, axis = plot_operational_prediction(
        FakeWindow(),
        FakeFit(),
        model_name="CTRV",
    )

    legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert "Held-out trajectory" not in legend_labels
    inset = next(
        text.get_text()
        for text in axis.texts
        if "Observation duration" in text.get_text()
    )
    assert "ADE:" not in inset
    assert "FDE:" not in inset
    assert "coverage" not in inset.lower()
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


def test_plot_prediction_accepts_position_only_state_variable_names(monkeypatch):
    """Bayesian CTRV plots should read the explicitly named latent states."""
    monkeypatch.setattr(plt, "show", lambda: None)
    fit = FakeFit()
    fit.variables = {
        "x_state_prediction": np.array([[2.0, 3.0], [2.1, 3.1]]),
        "y_state_prediction": np.array([[1.0, 1.5], [1.1, 1.6]]),
    }

    figure, axis = plot_prediction(
        FakeWindow(),
        fit,
        model_name="CTRV State-Space",
        state_prediction_variable_names=(
            "x_state_prediction",
            "y_state_prediction",
        ),
    )

    assert axis.lines[-1].get_xdata()[-1] == pytest.approx(3.05)
    plt.close(figure)


def test_plot_prediction_uses_supplied_observations_and_their_final_position(
    monkeypatch,
):
    """A noise experiment should plot and connect from its actual fit input."""
    monkeypatch.setattr(plt, "show", lambda: None)
    observed_x = np.array([10.0, 11.0])
    observed_y = np.array([20.0, 20.5])

    figure, axis = plot_prediction(
        FakeWindow(),
        FakeFit(),
        model_name="CTRV State-Space",
        max_posterior_trajectories=1,
        observed_position_values=(observed_x, observed_y),
        observed_trajectory_label="Noise-augmented observations",
    )

    assert axis.lines[0].get_xdata() == pytest.approx(observed_x)
    assert axis.lines[0].get_ydata() == pytest.approx(observed_y)
    assert axis.lines[0].get_label() == "Noise-augmented observations"
    for line in axis.lines[1:]:
        assert line.get_xdata()[0] == pytest.approx(observed_x[-1])
        assert line.get_ydata()[0] == pytest.approx(observed_y[-1])
    plt.close(figure)


@pytest.mark.parametrize(
    "observed_position_values",
    [([0.0], [1.0]), ([0.0, np.nan], [1.0, 2.0]), [0.0, 1.0]],
)
def test_plot_prediction_rejects_invalid_observed_position_values(
    observed_position_values,
):
    """Plot overrides must be finite x/y vectors aligned to the fit window."""
    with pytest.raises(ValueError, match="observed_position_values"):
        plot_prediction(
            FakeWindow(),
            FakeFit(),
            model_name="CTRV",
            observed_position_values=observed_position_values,
        )


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
        self.time_seconds = np.array([0.0, 10.0, 40.0, 70.0])
        self.timestamps = np.array(
            [
                "2026-01-01T00:00:00",
                "2026-01-01T00:00:10",
                "2026-01-01T00:00:40",
                "2026-01-01T00:01:10",
            ],
            dtype="datetime64[s]",
        )
        self.observation_count = 2

    @property
    def prediction_count(self):
        """Return the number of held-out positions."""
        return len(self.x_meters) - self.observation_count

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


class ManyDrawFit(FakeFit):
    """Posterior fixture with distinguishable trajectories."""

    def __init__(self, draw_count=100):
        offsets = np.arange(draw_count, dtype=float)
        self.variables = {
            "x_prediction_mean": np.column_stack(
                (2.0 + 0.01 * offsets, 3.0 + 0.02 * offsets)
            ),
            "y_prediction_mean": np.column_stack(
                (1.0 - 0.01 * offsets, 1.5 - 0.015 * offsets)
            ),
        }
