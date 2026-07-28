"""Tests for exploratory motion-prior calculations."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

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


def test_plot_motion_prior_distributions_returns_four_labelled_axes():
    """The exploration figure should expose every prior evidence source."""
    samples = collect_motion_prior_samples(create_noise_free_data(), run_ids=(0,))
    suggestions = suggest_prior_scales(samples)

    figure, axes = plot_motion_prior_distributions(samples, suggestions)

    assert axes.shape == (2, 2)
    assert {axis.get_title() for axis in axes.flat} == {
        "GPS speed",
        "Signed turn rate",
        "Turn-rate process innovations",
        "One-step CTRV position innovations",
    }
    legend_labels = [
        text.get_text() for axis in axes.flat for text in axis.get_legend().get_texts()
    ]
    assert "Candidate state prior" in legend_labels
    assert "Normal innovation model" in legend_labels
    assert "Normal residual model" in legend_labels
    figure.clear()


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
