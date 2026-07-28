"""Tests for exploratory motion-prior calculations."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from experiments.data_exploration.explore_motion_priors import (  # noqa: E402
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
        "GPS speed",
        "Signed turn rate",
        "Turn-rate process innovations",
        "One-step CTRV position innovations",
    }
    assert all(axis.title.get_fontsize() == 16 for axis in axes)
    assert all(axis.title.get_fontweight() == "bold" for axis in axes)
    assert all(axis.xaxis.label.get_fontsize() == 13 for axis in axes)
    assert all(axis.yaxis.label.get_fontsize() == 13 for axis in axes)
    assert all(
        tick.get_fontsize() == 11
        for axis in axes
        for tick in (*axis.get_xticklabels(), *axis.get_yticklabels())
    )
    assert axes[1].child_axes[0].xaxis.label.get_fontsize() == 13
    assert len(axes[0].lines) == 0
    assert [text.get_text() for text in axes[0].get_legend().get_texts()] == [
        "Empirical density"
    ]
    legend_labels = [
        text.get_text() for axis in axes for text in axis.get_legend().get_texts()
    ]
    assert "Candidate state prior" in legend_labels
    assert "Normal innovation model" in legend_labels
    assert "Normal residual model" in legend_labels
    assert "Current turn-rate limits" not in legend_labels
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
