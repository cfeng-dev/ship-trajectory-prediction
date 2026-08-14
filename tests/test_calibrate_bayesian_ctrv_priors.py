"""Tests for the position-only Bayesian CTRV prior calibration script."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from experiments.data_exploration.calibrate_bayesian_ctrv_priors import (  # noqa: E402
    _print_compact_summary,
    calibrate_position_only_priors,
    create_calibration_figures,
)
from ship_trajectory_prediction.observations.coordinates import (  # noqa: E402
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


def test_heading_prior_is_absent_from_calibration_plots_and_report(capsys):
    """Calibration should report speed and turn rate but no heading prior."""
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

    assert list(figures)[:2] == [
        "initial_speed_prior",
        "turn_rate_prior",
    ]
    assert all("heading" not in name for name in figures)
    assert all("heading" not in name for name in figures_without_prior)
    report = capsys.readouterr().out
    assert "Forecast-origin heading" not in report
    assert "heading_final_prior" not in report
    assert "Initial speed" in report
    assert "Turn rate" in report
    assert "Suggested prior (robust)" in report
    for figure in (*figures.values(), *figures_without_prior.values()):
        plt.close(figure)
