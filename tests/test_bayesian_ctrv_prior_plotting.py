"""Tests for the standalone Bayesian CTRV prior visualization."""

import importlib.util
import runpy
import sys
from dataclasses import replace
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
    assert curves["prior_initial_speed"].central_probability == pytest.approx(
        1.0 - priors.speed_prior_tail_probability
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


def test_header_option_controls_prior_legends():
    curves = prior_plotting.build_prior_curves(prior_plotting.PRIORS)
    without_legends = prior_plotting.create_individual_figures(
        curves,
        show_legend=False,
    )
    with_legends = prior_plotting.create_individual_figures(
        curves,
        show_legend=True,
    )

    assert all(
        figure.axes[0].get_legend() is None for figure in without_legends.values()
    )
    assert all(figure.axes[0].get_legend() for figure in with_legends.values())
    assert all(not figure.axes[0].texts for figure in with_legends.values())
    speed_legend = with_legends["prior_initial_speed"].axes[0].get_legend()
    assert [text.get_text() for text in speed_legend.get_texts()] == [
        "Prior-Dichte",
        "95 % innerhalb der Grenze",
        "5 % außerhalb der Grenze",
        "Konfigurierter Grenzwert",
    ]
    heading_legend = with_legends["prior_initial_heading"].axes[0].get_legend()
    assert [text.get_text() for text in heading_legend.get_texts()] == [
        "Prior-Dichte",
        "Gleichverteilte Anfangsrichtungen",
        "Konfigurierter Winkelbereich",
    ]
    for figure in (*without_legends.values(), *with_legends.values()):
        plt.close(figure)


def test_prior_legend_percentages_follow_changed_configuration():
    changed_priors = replace(
        prior_plotting.PRIORS,
        speed_prior_tail_probability=0.10,
    )
    speed_curve = next(
        curve
        for curve in prior_plotting.build_prior_curves(changed_priors)
        if curve.filename_stem == "prior_initial_speed"
    )

    figure = prior_plotting.create_prior_figure(speed_curve, show_legend=True)

    try:
        legend = figure.axes[0].get_legend()
        assert [text.get_text() for text in legend.get_texts()] == [
            "Prior-Dichte",
            "90 % innerhalb der Grenze",
            "10 % außerhalb der Grenze",
            "Konfigurierter Grenzwert",
        ]
    finally:
        plt.close(figure)


def test_all_prior_titles_have_visible_spacing_from_axes():
    figures = prior_plotting.create_individual_figures(
        prior_plotting.build_prior_curves(prior_plotting.PRIORS)
    )

    try:
        for figure in figures.values():
            figure.canvas.draw()
            axis = figure.axes[0]
            renderer = figure.canvas.get_renderer()
            title_bounds = axis.title.get_window_extent(renderer)
            axis_bounds = axis.get_window_extent(renderer)
            gap_points = (title_bounds.y0 - axis_bounds.y1) * 72.0 / figure.dpi
            assert gap_points >= 8.0
    finally:
        for figure in figures.values():
            plt.close(figure)


def test_initial_heading_plot_shows_boundaries_inside_extended_degree_range():
    heading_curve = next(
        curve
        for curve in prior_plotting.build_prior_curves(prior_plotting.PRIORS)
        if curve.filename_stem == "prior_initial_heading"
    )

    figure = prior_plotting.create_prior_figure(heading_curve)

    try:
        axis = figure.axes[0]
        assert axis.get_xticks().tolist() == [
            -270.0,
            -180.0,
            -90.0,
            0.0,
            90.0,
            180.0,
            270.0,
        ]
        assert axis.get_xlim() == pytest.approx((-270.0, 270.0))
        assert axis.lines[0].get_xdata()[[0, -1]].tolist() == [-180.0, 180.0]
        assert [line.get_xdata()[0] for line in axis.lines[1:]] == [
            -180.0,
            180.0,
        ]
    finally:
        plt.close(figure)


def test_all_prior_thresholds_are_shown_as_x_axis_ticks():
    threshold_curves = [
        curve
        for curve in prior_plotting.build_prior_curves(prior_plotting.PRIORS)
        if curve.thresholds
    ]

    assert len(threshold_curves) == 6
    for curve in threshold_curves:
        figure = prior_plotting.create_prior_figure(curve)
        try:
            ticks = figure.axes[0].get_xticks()
            assert all(
                np.any(np.isclose(ticks, threshold)) for threshold in curve.thresholds
            )
        finally:
            plt.close(figure)


def test_changed_prior_threshold_is_used_without_fixed_tick_values():
    changed_priors = replace(
        prior_plotting.PRIORS,
        turn_rate_prior_abs_heading_change_deg=65.0,
    )
    changed_curve = next(
        curve
        for curve in prior_plotting.build_prior_curves(changed_priors)
        if curve.filename_stem == "prior_initial_turn_rate"
    )

    figure = prior_plotting.create_prior_figure(changed_curve)

    try:
        ticks = figure.axes[0].get_xticks()
        assert np.any(np.isclose(ticks, -6.5))
        assert np.any(np.isclose(ticks, 6.5))
        assert not np.any(np.isclose(ticks, -4.5))
        assert not np.any(np.isclose(ticks, 4.5))
    finally:
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
