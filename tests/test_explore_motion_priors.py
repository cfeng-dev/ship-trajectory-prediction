"""Tests for exploratory motion-prior calculations."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from experiments.data_exploration.explore_motion_priors import (  # noqa: E402
    AXIS_LABEL_FONT_SIZE,
    AXIS_TICK_FONT_SIZE,
    HISTOGRAM_ALPHA,
    HISTOGRAM_COLOR,
    HISTOGRAM_EDGE_COLOR,
    HISTOGRAM_EDGE_LINE_WIDTH,
    PLOT_TITLE_FONT_SIZE,
    PLOT_TITLE_FONT_WEIGHT,
    collect_motion_prior_samples,
    plot_motion_prior_distributions,
    suggest_prior_scales,
)
from ship_trajectory_prediction.models.ctrv import CTRVState  # noqa: E402
from ship_trajectory_prediction.simulation.synthetic_ctrv import (  # noqa: E402
    SyntheticCTRVNoise,
    simulate_synthetic_ctrv_data,
)


def create_noise_free_data():
    """Return one exact synthetic CTRV run for aligned residual checks."""
    return simulate_synthetic_ctrv_data(
        count=12,
        dt_seconds=5.0,
        initial_state=CTRVState(
            x=0.0,
            y=0.0,
            speed=3.0,
            heading=0.3,
            turn_rate=0.012,
        ),
        noise=SyntheticCTRVNoise(
            sigma_position_gps=0.0,
            sigma_speed_gps=0.0,
            sigma_position_process=0.0,
            sigma_speed_process=0.0,
            sigma_turn_rate_process=0.0,
        ),
        seed=7,
    )


def test_collect_motion_samples_recovers_noise_free_ctrv_quantities():
    """Exact synthetic motion should yield its turn rate and zero innovations."""
    samples = collect_motion_prior_samples(
        create_noise_free_data(),
        run_ids=(0,),
    )

    assert samples.speed_mps == pytest.approx(np.full(12, 3.0))
    assert samples.turn_rate_rad_s == pytest.approx(np.full(10, 0.012), abs=1e-7)
    assert samples.turn_rate_innovation == pytest.approx(
        np.zeros(9),
        abs=1e-8,
    )
    assert samples.position_innovation == pytest.approx(
        np.zeros(18),
        abs=1e-5,
    )
    assert samples.per_run_summary.loc[0, "run_id"] == 0


def test_prior_suggestions_use_robust_turn_rate_scale_floor():
    """Constant synthetic turning should retain a non-degenerate state prior."""
    samples = collect_motion_prior_samples(create_noise_free_data(), run_ids=(0,))

    suggestions = suggest_prior_scales(samples)

    assert suggestions.speed_median_mps == pytest.approx(3.0)
    assert suggestions.turn_rate_center_rad_s == pytest.approx(0.012, abs=1e-7)
    assert suggestions.turn_rate_state_prior_scale_rad_s == pytest.approx(0.002)
    assert suggestions.turn_rate_process_scale == pytest.approx(0.0, abs=1e-8)
    assert suggestions.position_process_scale == pytest.approx(0.0, abs=1e-5)


def test_plot_motion_prior_distributions_returns_four_separate_figures():
    """Each prior evidence source should have its own labelled figure."""
    samples = collect_motion_prior_samples(create_noise_free_data(), run_ids=(0,))
    suggestions = suggest_prior_scales(samples)

    figures, axes = plot_motion_prior_distributions(samples, suggestions)

    assert len(figures) == len(axes) == 4
    assert all(
        axis.figure is figure for figure, axis in zip(figures, axes, strict=True)
    )
    assert all(figure._suptitle is None for figure in figures)
    assert len({id(figure) for figure in figures}) == 4
    assert {axis.get_title() for axis in axes} == {
        "GPS-Geschwindigkeit des Schiffs",
        "Drehrate",
        "Prozessinnovationen der Drehrate",
        "CTRV-Positionsresiduen (Ein-Schritt)",
    }
    assert [axis.get_xlabel() for axis in axes] == [
        "Geschwindigkeit [m/s]",
        "Drehrate [rad/s]",
        "Drehratenänderung / √Δt [rad/s/√s]",
        "Positionsresiduum / √Δt [m/√s]",
    ]
    assert all(axis.get_ylabel() == "Dichte" for axis in axes)
    assert axes[1].child_axes[0].get_xlabel() == "Drehrate [°/s]"
    assert all(axis.title.get_fontsize() == PLOT_TITLE_FONT_SIZE for axis in axes)
    assert all(axis.title.get_fontweight() == PLOT_TITLE_FONT_WEIGHT for axis in axes)
    assert all(axis.xaxis.label.get_fontsize() == AXIS_LABEL_FONT_SIZE for axis in axes)
    assert all(axis.yaxis.label.get_fontsize() == AXIS_LABEL_FONT_SIZE for axis in axes)
    assert all(
        tick.get_fontsize() == AXIS_TICK_FONT_SIZE
        for axis in axes
        for tick in (*axis.get_xticklabels(), *axis.get_yticklabels())
    )
    assert axes[1].child_axes[0].xaxis.label.get_fontsize() == AXIS_LABEL_FONT_SIZE
    assert len(axes[0].lines) == 0
    assert all(
        patch.get_linewidth() == HISTOGRAM_EDGE_LINE_WIDTH
        for axis in axes
        for patch in axis.patches
    )
    assert all(
        patch.get_facecolor()
        == pytest.approx(matplotlib.colors.to_rgba(HISTOGRAM_COLOR, HISTOGRAM_ALPHA))
        for axis in axes
        for patch in axis.patches
    )
    assert all(
        patch.get_edgecolor()
        == pytest.approx(
            matplotlib.colors.to_rgba(HISTOGRAM_EDGE_COLOR, HISTOGRAM_ALPHA)
        )
        for axis in axes
        for patch in axis.patches
    )
    assert [text.get_text() for text in axes[0].get_legend().get_texts()] == [
        "Empirische Dichte"
    ]
    legend_labels = [
        text.get_text() for axis in axes for text in axis.get_legend().get_texts()
    ]
    assert legend_labels == ["Empirische Dichte"] * 4
    assert all(
        [line.get_label() for line in axis.lines] == ["_nolegend_"] for axis in axes[1:]
    )
    assert all(axis.lines[0].get_color() == "black" for axis in axes[1:])
    for figure in figures:
        plt.close(figure)


def test_plot_motion_prior_distributions_can_show_one_figure_at_a_time(monkeypatch):
    """Sequential display should never register more than one open figure."""
    samples = collect_motion_prior_samples(create_noise_free_data(), run_ids=(0,))
    suggestions = suggest_prior_scales(samples)
    open_figure_counts = []
    blocking_values = []
    plt.close("all")

    def record_show(*, block):
        open_figure_counts.append(len(plt.get_fignums()))
        blocking_values.append(block)

    monkeypatch.setattr(plt, "show", record_show)

    figures, axes = plot_motion_prior_distributions(
        samples,
        suggestions,
        show_sequentially=True,
    )

    assert len(figures) == len(axes) == 4
    assert open_figure_counts == [1, 1, 1, 1]
    assert blocking_values == [True, True, True, True]
    assert plt.get_fignums() == []


def test_plot_motion_prior_distributions_can_show_absolute_frequencies():
    """Frequency mode should use counts and label every y-axis accordingly."""
    samples = collect_motion_prior_samples(create_noise_free_data(), run_ids=(0,))
    suggestions = suggest_prior_scales(samples)

    figures, axes = plot_motion_prior_distributions(
        samples,
        suggestions,
        histogram_mode="frequency",
    )

    assert all(axis.get_ylabel() == "Häufigkeit" for axis in axes)
    assert sum(patch.get_height() for patch in axes[0].patches) == pytest.approx(
        len(samples.speed_mps)
    )
    for figure in figures:
        plt.close(figure)


def test_plot_motion_prior_distributions_rejects_invalid_histogram_mode():
    """Unknown histogram modes should fail with a focused message."""
    samples = collect_motion_prior_samples(create_noise_free_data(), run_ids=(0,))
    suggestions = suggest_prior_scales(samples)

    with pytest.raises(ValueError, match="histogram_mode"):
        plot_motion_prior_distributions(
            samples,
            suggestions,
            histogram_mode="unknown",
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"run_ids": ()}, "run_ids"),
        ({"min_course_displacement_m": 0.0}, "min_course_displacement_m"),
        ({"max_time_gap_s": 0.0}, "max_time_gap_s"),
    ],
)
def test_collect_motion_samples_rejects_invalid_configuration(options, message):
    """Invalid calibration selections should fail before plotting."""
    with pytest.raises(ValueError, match=message):
        collect_motion_prior_samples(create_noise_free_data(), **options)
