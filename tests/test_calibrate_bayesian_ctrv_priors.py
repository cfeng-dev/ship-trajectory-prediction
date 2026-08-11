"""Tests for the position-only Bayesian CTRV prior calibration script."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from experiments.data_exploration.calibrate_bayesian_ctrv_priors import (  # noqa: E402
    _derive_heading_residuals,
    _print_compact_summary,
    _summarize_circular_distribution,
    calibrate_position_only_priors,
    create_calibration_figures,
)
from ship_trajectory_prediction.coordinates import (  # noqa: E402
    local_to_gps_coordinates,
)


def create_position_only_runs():
    """Create two finite curved GPS runs without external speed columns."""
    rows = []
    x_meters = np.arange(8, dtype=float) * 12.0
    y_meters = np.array([0.0, 0.2, 0.8, 1.8, 3.2, 5.0, 7.2, 9.8])
    for run_id in range(2):
        longitude, latitude = local_to_gps_coordinates(
            x_meters,
            y_meters + run_id,
            reference_longitude=8.0,
            reference_latitude=54.0,
        )
        timestamps = pd.date_range(
            "2026-01-01",
            periods=len(x_meters),
            freq="5s",
            tz="UTC",
        )
        rows.extend(
            {
                "time": timestamp,
                "run_id": run_id,
                "gps_latitude": lat,
                "gps_longitude": lon,
            }
            for timestamp, lat, lon in zip(
                timestamps,
                latitude,
                longitude,
                strict=True,
            )
        )
    return pd.DataFrame(rows)


def test_heading_residual_wraps_across_the_angle_boundary():
    """Courses near minus/plus pi should produce a small residual."""
    segment_headings = np.radians([179.0, 179.0, 179.0, -179.0])
    x_meters = np.concatenate(([0.0], np.cumsum(10 * np.cos(segment_headings))))
    y_meters = np.concatenate(([0.0], np.cumsum(10 * np.sin(segment_headings))))

    residual = _derive_heading_residuals(
        np.arange(5, dtype=float) * 5.0,
        x_meters,
        y_meters,
    )

    assert residual.shape == (1,)
    assert abs(residual[0]) < np.radians(3.0)


def test_circular_summary_keeps_boundary_angles_together():
    """Circular statistics should not average 179 and -179 degrees to zero."""
    summary = _summarize_circular_distribution(np.radians([179.0, -179.0, 180.0]))

    assert abs(abs(summary.median) - np.pi) < np.radians(1.0)
    assert summary.standard_deviation < np.radians(2.0)


def test_heading_distribution_is_plotted_and_reported(capsys):
    """Heading evidence should appear after speed in plots and the report."""
    result = calibrate_position_only_priors(
        create_position_only_runs(),
        run_start=0,
        run_stop=2,
    )

    figures = create_calibration_figures(result, show_prior_density=True)
    figures_without_prior = create_calibration_figures(
        result,
        show_prior_density=False,
    )
    _print_compact_summary(result)

    assert list(figures)[:3] == [
        "initial_speed_prior",
        "forecast_origin_heading_prior",
        "turn_rate_prior",
    ]
    heading_axis = figures["forecast_origin_heading_prior"].axes[0]
    assert heading_axis.get_title().startswith(
        "Historische Heading-Abweichungen am Prognosebeginn"
    )
    assert heading_axis.get_xlabel() == "Heading-Abweichung [rad]"
    legend_labels = [text.get_text() for text in heading_axis.get_legend().get_texts()]
    assert legend_labels[:3] == [
        "Historische Heading-Abweichungen\n(zentrale 99 %)",
        "Empfohlene Fehlerverteilung\n(Normal, robuste Skala)",
        "Null",
    ]
    assert legend_labels[3].startswith("Median =")
    assert len(heading_axis.lines) == 3
    assert (
        len(figures_without_prior["forecast_origin_heading_prior"].axes[0].lines) == 2
    )
    report = capsys.readouterr().out
    assert "Forecast-origin heading" in report
    assert "Suggested prior (robust)" in report
    assert "Sensitivity alt. (circ. SD)" in report
    for figure in (*figures.values(), *figures_without_prior.values()):
        plt.close(figure)
