"""Tests for the shared posterior trajectory plot."""

from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Ellipse

from ship_trajectory_prediction.coordinates import local_to_gps_coordinates
from ship_trajectory_prediction.evaluation.metrics import (
    empirical_covariance_regions,
)
from ship_trajectory_prediction.evaluation.prediction_plotting import (
    AXIS_LABEL_FONT_SIZE,
    AXIS_TICK_FONT_SIZE,
    PLOT_FIGURE_SIZE,
    PLOT_MAX_WIDTH_TO_HEIGHT_RATIO,
    PLOT_TITLE_FONT_SIZE,
    PLOT_TITLE_FONT_WEIGHT,
    POSTERIOR_SAMPLE_ALPHA,
    PREDICTION_PLOT_TITLE,
    plot_operational_prediction,
    plot_prediction,
    plot_trajectory_paths,
)

plt.switch_backend("Agg")


@pytest.fixture(autouse=True)
def _prevent_test_windows(monkeypatch):
    """Avoid GUI windows while individual tests verify the display call."""
    monkeypatch.setattr(plt, "show", Mock())


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


def test_plot_trajectory_paths_draws_regions_for_multiple_forecasts():
    """A rolling plot should show both uncertainty regions for every window."""
    phase = np.linspace(0.0, 2 * np.pi, 20, endpoint=False)
    posterior_draw_groups = (
        (
            np.column_stack((1.0 + 0.1 * np.cos(phase), 2.0 + 0.2 * np.cos(phase))),
            np.column_stack((0.5 + 0.1 * np.sin(phase), 1.0 + 0.2 * np.sin(phase))),
        ),
        (
            np.column_stack((3.0 + 0.1 * np.cos(phase), 4.0 + 0.2 * np.cos(phase))),
            np.column_stack((1.5 + 0.1 * np.sin(phase), 2.0 + 0.2 * np.sin(phase))),
        ),
    )

    figure, axis = plot_trajectory_paths(
        observed_path=([0.0, 1.0], [0.0, 0.5]),
        reference_path=([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 0.5, 1.0, 1.5, 2.0]),
        forecast_paths=(
            ([1.0, 1.5, 2.0], [0.5, 0.75, 1.0]),
            ([3.0, 3.5, 4.0], [1.5, 1.75, 2.0]),
        ),
        posterior_draw_groups=posterior_draw_groups,
        forecast_time_groups=(
            np.array([0.0, 10.0, 20.0]),
            np.array([0.0, 10.0, 20.0]),
        ),
        annotate_prediction_regions=False,
        title="Bayesian CTRV Rolling Prediction",
    )

    assert len(_region_patches(axis)) == 8
    assert not axis.texts
    legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert legend_labels.count("Posterior-prädiktiver Bereich (50 %)") == 1
    assert legend_labels.count("Posterior-prädiktiver Bereich (90 %)") == 1
    plt.close(figure)


def test_plot_trajectory_paths_rejects_misaligned_region_groups():
    """Every rolling posterior group should have one matching time vector."""
    with pytest.raises(ValueError, match="equal length"):
        plot_trajectory_paths(
            observed_path=([0.0], [0.0]),
            reference_path=([0.0, 1.0], [0.0, 1.0]),
            forecast_paths=(([0.0, 1.0], [0.0, 1.0]),),
            posterior_draw_groups=((np.ones((2, 1)), np.ones((2, 1))),),
            forecast_time_groups=(),
            title="Prediction",
        )


def test_plot_trajectory_paths_reserves_upper_space_for_wide_routes():
    """A wide route should gain mostly upper space without distorting meters."""
    figure, axis = plot_trajectory_paths(
        observed_path=([0.0, 4.0], [0.0, -1.0]),
        reference_path=([4.0, 8.0], [-1.0, -2.0]),
        forecast_paths=(([4.0, 8.0], [-1.0, -1.8]),),
        title="Prediction",
    )

    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    assert (x_max - x_min) / (y_max - y_min) == pytest.approx(
        PLOT_MAX_WIDTH_TO_HEIGHT_RATIO
    )
    assert y_max - 0.0 > -2.0 - y_min
    assert axis.get_aspect() == pytest.approx(1.0)
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


def test_plot_prediction_uses_fixed_title_and_displays_plot(monkeypatch):
    """The thesis plot should have a stable title and display automatically."""
    show_mock = Mock()
    monkeypatch.setattr(plt, "show", show_mock)

    figure, axis = plot_prediction(
        FakeWindow(),
        FakeFit(),
        max_sample_trajectories=2,
    )

    assert axis.get_title() == PREDICTION_PLOT_TITLE
    assert len(axis.lines) == 5
    show_mock.assert_called_once_with()
    plt.close(figure)


def test_plot_prediction_uses_equal_spatial_scale_and_professional_labels():
    """The spatial plot should preserve geometry and use physical axis names."""
    figure, axis = plot_prediction(
        FakeWindow(),
        FakeFit(),
    )

    assert axis.get_aspect() == pytest.approx(1.0)
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
    assert PLOT_FIGURE_SIZE == (11, 9)
    assert figure.get_size_inches() == pytest.approx(PLOT_FIGURE_SIZE)
    assert axis.get_legend()._loc == 1
    legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert legend_labels == [
        "Beobachtungen",
        "Referenztrajektorie",
        "Posterior-prädiktive Trajektorien (n = 2)",
        "Posterior-prädiktiver Bereich (50 %)",
        "Posterior-prädiktiver Bereich (90 %)",
        "Posterior-prädiktiver Median",
        "Prognosebeginn",
    ]
    plt.close(figure)


def test_plot_prediction_can_display_local_kilometers():
    """The kilometer view should scale every plotted coordinate consistently."""
    figure, axis = plot_prediction(
        FakeWindow(),
        FakeFit(),
        coordinate_mode="km",
    )

    assert axis.get_xlabel() == "Ostposition x [km]"
    assert axis.get_ylabel() == "Nordposition y [km]"
    assert axis.get_aspect() == pytest.approx(1.0)
    assert axis.lines[0].get_xdata() == pytest.approx([0.0, 0.001])
    assert axis.lines[0].get_ydata() == pytest.approx([0.0, 0.0005])
    assert axis.lines[1].get_xdata() == pytest.approx(
        np.array([1.0, 2.0, 3.0, 4.0]) / 1000
    )
    plt.close(figure)


def test_plot_prediction_can_display_longitude_and_latitude():
    """The GPS view should transform all paths from the stored local origin."""
    window = FakeWindow()
    figure, axis = plot_prediction(
        window,
        FakeFit(),
        coordinate_mode="gps",
    )
    expected_longitude, expected_latitude = local_to_gps_coordinates(
        window.x_meters[window.observed_slice],
        window.y_meters[window.observed_slice],
        window.reference_longitude,
        window.reference_latitude,
    )

    assert axis.get_xlabel() == "Längengrad [°]"
    assert axis.get_ylabel() == "Breitengrad [°]"
    assert axis.get_aspect() == pytest.approx(
        1 / np.cos(np.radians(window.reference_latitude))
    )
    assert axis.lines[0].get_xdata() == pytest.approx(expected_longitude)
    assert axis.lines[0].get_ydata() == pytest.approx(expected_latitude)
    assert axis.lines[0].get_xdata()[0] == pytest.approx(window.reference_longitude)
    assert axis.lines[0].get_ydata()[0] == pytest.approx(window.reference_latitude)
    plt.close(figure)


def test_plot_prediction_draws_at_most_15_sample_trajectories():
    """A large posterior should still produce a readable number of paths."""
    figure, axis = plot_prediction(
        FakeWindow(),
        ManyDrawFit(),
        max_sample_trajectories=100,
    )

    assert len(_sample_lines(axis)) == 15
    legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert "Posterior-prädiktive Trajektorien (n = 15)" in legend_labels
    plt.close(figure)


def test_plot_prediction_can_hide_sample_trajectories():
    """Posterior path lines should be optional without removing uncertainty."""
    figure, axis = plot_prediction(
        FakeWindow(),
        ManyDrawFit(),
        show_sample_trajectories=False,
    )

    assert not _sample_lines(axis)
    assert len(_region_patches(axis)) == 2 * FakeWindow().prediction_count
    legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert not any("Trajektorien (n =" in label for label in legend_labels)
    plt.close(figure)


def test_posterior_trajectory_selection_is_reproducible():
    """The same plotting seed should select the same posterior trajectories."""
    figures_and_axes = [
        plot_prediction(
            FakeWindow(),
            ManyDrawFit(),
            max_sample_trajectories=10,
            sample_seed=seed,
        )
        for seed in (17, 17, 23)
    ]

    selected_paths = [
        [
            np.column_stack((line.get_xdata(), line.get_ydata()))
            for line in _sample_lines(axis)
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


def test_prediction_regions_are_per_time_and_nested():
    """Every horizon should have a nested joint 50% and 90% ellipse pair."""
    window = FakeWindow()
    figure, axis = plot_prediction(
        window,
        ManyDrawFit(),
        show_sample_trajectories=False,
    )

    regions = _region_patches(axis)
    assert len(regions) == 2 * window.prediction_count
    for time_index in range(window.prediction_count):
        region_50 = _region(axis, probability=0.5, time_index=time_index)
        region_90 = _region(axis, probability=0.9, time_index=time_index)
        assert region_50.center == pytest.approx(region_90.center)
        assert region_50.angle == pytest.approx(region_90.angle)
        assert region_50.width <= region_90.width
        assert region_50.height <= region_90.height
        assert region_50.get_alpha() > region_90.get_alpha()
    plt.close(figure)


def test_prediction_region_geometry_uses_shared_metric_definition():
    """The plotted ellipse must use the same geometry as coverage evaluation."""
    fit = ManyDrawFit()
    figure, axis = plot_prediction(
        FakeWindow(),
        fit,
        show_sample_trajectories=False,
    )
    region = empirical_covariance_regions(
        fit.stan_variable("x_prediction_mean")[:, 0],
        fit.stan_variable("y_prediction_mean")[:, 0],
        probabilities=(0.9,),
    )[0.9]
    patch = _region(axis, probability=0.9, time_index=0)

    assert patch.center == pytest.approx(region.center)
    assert patch.width == pytest.approx(region.width)
    assert patch.height == pytest.approx(region.height)
    assert patch.angle == pytest.approx(region.angle_degrees)
    plt.close(figure)


def test_plot_prediction_labels_selected_prediction_regions():
    """Future regions should carry unique and correct elapsed-time labels."""
    figure, axis = plot_prediction(
        FakeWindow(),
        FakeFit(),
    )

    time_labels = [
        text.get_text() for text in axis.texts if text.get_text().startswith("+")
    ]
    assert time_labels == ["+10 s", "+20 s", "+30 s"]
    assert len(time_labels) == len(set(time_labels))
    time_annotations = {
        text.get_text(): text for text in axis.texts if text.get_text().startswith("+")
    }
    assert {annotation.get_position() for annotation in time_annotations.values()} == {
        (6, 8)
    }
    plt.close(figure)


def test_evaluation_mode_contains_ground_truth_and_window_metrics():
    """Evaluation mode should show held-out positions and window-level metrics."""
    figure, axis = plot_prediction(
        FakeWindow(),
        FakeFit(),
        plot_mode="evaluation",
    )

    legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert "Referenztrajektorie" in legend_labels
    footer_artist = next(
        text for text in figure.texts if "Beobachtungsdauer" in text.get_text()
    )
    footer = footer_artist.get_text()
    assert footer_artist.get_position()[1] < axis.get_position().y0
    assert not any("Beobachtungsdauer" in text.get_text() for text in axis.texts)
    assert "Beobachtungsdauer: 10 s" in footer
    assert "Prognosehorizont: 30 s" in footer
    assert "ADE: 0,07 m" in footer
    assert "FDE: 0,07 m" in footer
    assert "Empirische 2D-Abdeckung (90 %):" in footer
    assert "/3 Punkte)" in footer
    assert "Kalibrierung" not in footer
    plt.close(figure)


def test_operational_mode_contains_no_ground_truth_or_unknown_metrics():
    """Operational mode should not access or display unknown future positions."""
    figure, axis = plot_prediction(
        OperationalWindow(),
        FakeFit(),
        plot_mode="operational",
    )

    legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert "Referenztrajektorie" not in legend_labels
    footer = next(
        text.get_text()
        for text in figure.texts
        if "Beobachtungsdauer" in text.get_text()
    )
    assert "ADE:" not in footer
    assert "FDE:" not in footer
    assert "Abdeckung" not in footer
    plt.close(figure)


def test_operational_wrapper_uses_the_shared_plot_mode():
    """The convenience function should produce the same ground-truth-free view."""
    figure, axis = plot_operational_prediction(
        OperationalWindow(),
        FakeFit(),
    )

    legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert "Referenztrajektorie" not in legend_labels
    plt.close(figure)


def test_plot_prediction_connects_future_trajectories_to_prediction_start():
    """Held-out and posterior paths should start at the final observation."""
    figure, axis = plot_prediction(
        FakeWindow(),
        FakeFit(),
        max_sample_trajectories=2,
    )

    expected_start = (1.0, 0.5)
    for line in axis.lines[1:]:
        assert line.get_xdata()[0] == expected_start[0]
        assert line.get_ydata()[0] == expected_start[1]
    plt.close(figure)


def test_plot_prediction_accepts_position_only_state_variable_names():
    """Bayesian CTRV plots should read the explicitly named latent states."""
    fit = FakeFit()
    fit.variables = {
        "x_state_prediction": np.array([[2.0, 3.0, 4.0], [2.1, 3.1, 4.1]]),
        "y_state_prediction": np.array([[1.0, 1.5, 2.0], [1.1, 1.6, 2.1]]),
    }

    figure, axis = plot_prediction(
        FakeWindow(),
        fit,
        state_prediction_variable_names=(
            "x_state_prediction",
            "y_state_prediction",
        ),
    )

    assert axis.lines[-1].get_xdata()[-1] == pytest.approx(4.05)
    plt.close(figure)


def test_plot_prediction_uses_supplied_observations_and_their_final_position():
    """A noise experiment should plot and connect from its actual fit input."""
    observed_x = np.array([10.0, 11.0])
    observed_y = np.array([20.0, 20.5])

    figure, axis = plot_prediction(
        FakeWindow(),
        FakeFit(),
        max_sample_trajectories=1,
        observed_position_values=(observed_x, observed_y),
        observed_trajectory_label="Noise-augmented observations",
        additional_position_noise_std_m=2.5,
    )

    assert axis.lines[0].get_xdata() == pytest.approx(observed_x)
    assert axis.lines[0].get_ydata() == pytest.approx(observed_y)
    assert axis.lines[0].get_label() == "Noise-augmented observations"
    footer = next(
        text.get_text()
        for text in figure.texts
        if "Beobachtungsdauer" in text.get_text()
    )
    assert "Zusatzrauschen: σ_add = 2,5 m je Achse" in footer
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
            observed_position_values=observed_position_values,
        )


@pytest.mark.parametrize("plot_mode", [None, "", "analysis"])
def test_plot_prediction_rejects_invalid_plot_mode(plot_mode):
    """Only evaluation and operational views should be accepted."""
    with pytest.raises(ValueError, match="plot_mode"):
        plot_prediction(
            FakeWindow(),
            FakeFit(),
            plot_mode=plot_mode,
        )


@pytest.mark.parametrize(
    "coordinate_mode",
    [None, "", "   ", "meters", "latlon", 123],
)
def test_plot_prediction_warns_and_defaults_invalid_coordinate_mode_to_meters(
    coordinate_mode,
):
    """Every invalid coordinate value should warn and retain the meter plot."""
    with pytest.warns(UserWarning, match="Use 'm', 'km', or 'gps'"):
        figure, axis = plot_prediction(
            FakeWindow(),
            FakeFit(),
            coordinate_mode=coordinate_mode,
        )

    assert axis.get_xlabel() == "Ostposition x [m]"
    assert axis.get_ylabel() == "Nordposition y [m]"
    assert axis.lines[0].get_xdata() == pytest.approx([0.0, 1.0])
    plt.close(figure)


def test_gps_plot_requires_a_finite_window_reference():
    """Absolute GPS display must not silently invent a coordinate origin."""
    window = FakeWindow()
    window.reference_latitude = np.nan

    with pytest.raises(ValueError, match="reference_longitude"):
        plot_prediction(
            window,
            FakeFit(),
            coordinate_mode="gps",
        )


@pytest.mark.parametrize(
    "additional_position_noise_std_m",
    [-1.0, np.nan, True, "2"],
)
def test_plot_prediction_rejects_invalid_additional_noise(
    additional_position_noise_std_m,
):
    """The optional footer noise must be a finite non-negative number."""
    with pytest.raises(ValueError, match="additional_position_noise_std_m"):
        plot_prediction(
            FakeWindow(),
            FakeFit(),
            additional_position_noise_std_m=additional_position_noise_std_m,
        )


def _sample_lines(axis):
    """Return the deliberately faint posterior sample paths."""
    return [line for line in axis.lines if line.get_alpha() == POSTERIOR_SAMPLE_ALPHA]


def _region_patches(axis):
    """Return posterior-predictive region ellipses."""
    return [
        patch
        for patch in axis.patches
        if isinstance(patch, Ellipse)
        and str(patch.get_gid()).startswith("posterior-predictive-region-")
    ]


def _region(axis, *, probability, time_index):
    """Return one identified probability region at one future time."""
    expected_gid = f"posterior-predictive-region-{probability:g}-t{time_index}"
    return next(
        patch for patch in _region_patches(axis) if patch.get_gid() == expected_gid
    )


class FakeWindow:
    """Minimal shared trajectory-window interface for plotting."""

    def __init__(self):
        self.x_meters = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        self.y_meters = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        self.reference_longitude = 8.3122
        self.reference_latitude = 47.0515
        self.time_seconds = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        self.timestamps = np.array(
            [
                "2026-01-01T00:00:00",
                "2026-01-01T00:00:10",
                "2026-01-01T00:00:20",
                "2026-01-01T00:00:30",
                "2026-01-01T00:00:40",
            ],
            dtype="datetime64[s]",
        )
        self.observation_count = 2

    @property
    def prediction_count(self):
        """Return the number of held-out positions."""
        return len(self.time_seconds) - self.observation_count

    @property
    def observed_slice(self):
        """Return the observed part of the trajectory."""
        return slice(0, self.observation_count)

    @property
    def prediction_slice(self):
        """Return the held-out part of the trajectory."""
        return slice(self.observation_count, None)


class OperationalWindow(FakeWindow):
    """Forecast window without future ground-truth positions."""

    def __init__(self):
        super().__init__()
        self.x_meters = self.x_meters[: self.observation_count]
        self.y_meters = self.y_meters[: self.observation_count]


class FakeFit:
    """Minimal fit object exposing posterior mean trajectories."""

    def __init__(self):
        self.variables = {
            "x_prediction_mean": np.array([[2.0, 3.0, 4.0], [2.1, 3.1, 4.1]]),
            "y_prediction_mean": np.array([[1.0, 1.5, 2.0], [1.1, 1.6, 2.1]]),
        }

    def stan_variable(self, name):
        """Return one stored posterior variable."""
        return self.variables[name]


class ManyDrawFit(FakeFit):
    """Posterior fixture with distinguishable two-dimensional trajectories."""

    def __init__(self, draw_count=100):
        phase = np.linspace(0.0, 2 * np.pi, draw_count, endpoint=False)
        self.variables = {
            "x_prediction_mean": np.column_stack(
                (
                    2.0 + 0.2 * np.cos(phase),
                    3.0 + 0.3 * np.cos(phase),
                    4.0 + 0.4 * np.cos(phase),
                )
            ),
            "y_prediction_mean": np.column_stack(
                (
                    1.0 + 0.1 * np.sin(phase),
                    1.5 + 0.2 * np.sin(phase),
                    2.0 + 0.3 * np.sin(phase),
                )
            ),
        }
