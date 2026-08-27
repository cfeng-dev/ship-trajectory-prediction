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


def test_saves_each_individual_prior_and_optional_overview_as_png_and_pdf(tmp_path):
    curves = prior_plotting.build_prior_curves(prior_plotting.PRIORS)
    figures = prior_plotting.create_individual_figures(curves)
    figures["prior_overview"] = prior_plotting.create_overview_figure(curves)

    saved_paths = prior_plotting.save_figures(figures, tmp_path)
    for figure in figures.values():
        plt.close(figure)

    assert len(saved_paths) == 14
    assert all(path.is_file() for path in saved_paths)
    assert {path.suffix for path in saved_paths} == {".png", ".pdf"}
    assert {path.parent for path in saved_paths} == {Path(tmp_path).resolve()}
