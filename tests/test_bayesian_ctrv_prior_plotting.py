"""Tests for the standalone Bayesian CTRV prior visualization."""

import importlib.util
import runpy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_prior_plotting_module():
    """Load the repository script without treating experiments as a package."""
    module_name = "plot_bayesian_ctrv_priors_for_tests"
    script_path = (
        PROJECT_ROOT
        / "experiments"
        / "data_exploration"
        / "plot_bayesian_ctrv_priors.py"
    )
    specification = importlib.util.spec_from_file_location(module_name, script_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load prior plotting script: {script_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


prior_plotting = _load_prior_plotting_module()
EXPERIMENT_PRIORS = runpy.run_path(
    PROJECT_ROOT / "experiments" / "trajectory_prediction" / "bayesian_ctrv.py"
)["PRIORS"]


def test_visualization_uses_current_experiment_prior_configuration():
    assert prior_plotting.PRIORS == EXPERIMENT_PRIORS


def test_prior_curves_use_configured_thresholds_and_derived_parameters():
    priors = prior_plotting.PRIORS
    curves = {
        curve.filename_stem: curve
        for curve in prior_plotting.build_prior_curves(priors)
    }

    assert len(curves) == 6
    assert curves["prior_initial_speed"].central_upper == pytest.approx(
        priors.speed_prior_upper_mps
    )
    assert curves["prior_initial_turn_rate"].central_upper == pytest.approx(
        priors.turn_rate_prior_abs_heading_change_deg
        / priors.turn_rate_prior_reference_interval_seconds
    )
    assert curves["prior_position_observation_noise"].density[0] == pytest.approx(
        priors.sigma_position_observation_prior_rate
    )
    assert curves["prior_speed_process_noise"].density[0] == pytest.approx(
        priors.sigma_speed_process_prior_rate
    )
    assert curves["prior_initial_heading"].title == "Prior: Anfangskurswinkel"
    assert "Prior-Wahrscheinlichkeit" in curves["prior_initial_speed"].annotation


def test_prior_curves_are_normalized_over_the_presentation_range():
    for curve in prior_plotting.build_prior_curves(prior_plotting.PRIORS):
        integral = np.trapezoid(curve.density, curve.x_values)
        assert 0.998 < integral <= 1.001


def test_terminal_report_distinguishes_configured_and_derived_values(capsys):
    prior_plotting.print_prior_report(prior_plotting.PRIORS)

    report = capsys.readouterr().out
    assert "configured statement" in report
    assert "derived scale" in report
    assert "derived rate" in report
    assert f"{prior_plotting.PRIORS.speed_prior_scale:.6g} m/s" in report
    assert (
        f"{prior_plotting.PRIORS.sigma_position_observation_prior_rate:.6g} 1/m"
        in report
    )


def test_header_option_controls_prior_annotation_boxes():
    curves = prior_plotting.build_prior_curves(prior_plotting.PRIORS)
    without_annotations = prior_plotting.create_individual_figures(
        curves,
        show_annotations=False,
    )
    with_annotations = prior_plotting.create_individual_figures(
        curves,
        show_annotations=True,
    )

    assert all(not figure.axes[0].texts for figure in without_annotations.values())
    assert all(len(figure.axes[0].texts) == 1 for figure in with_annotations.values())
    assert (
        "Prior-Wahrscheinlichkeit"
        in with_annotations["prior_initial_speed"].axes[0].texts[0].get_text()
    )
    for figure in (*without_annotations.values(), *with_annotations.values()):
        plt.close(figure)


def test_main_shows_individual_priors_sequentially(monkeypatch):
    open_figure_counts = []
    blocking_options = []

    def record_show(*, block):
        open_figure_counts.append(len(plt.get_fignums()))
        blocking_options.append(block)

    monkeypatch.setattr(plt, "show", record_show)

    figures = prior_plotting.main([])

    assert set(figures) == {
        "prior_initial_speed",
        "prior_initial_heading",
        "prior_initial_turn_rate",
        "prior_position_observation_noise",
        "prior_speed_process_noise",
        "prior_turn_rate_process_noise",
    }
    assert open_figure_counts == [1] * 6
    assert blocking_options == [True] * 6
    assert not plt.get_fignums()
