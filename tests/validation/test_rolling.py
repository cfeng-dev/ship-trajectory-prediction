"""Tests for rolling-origin trajectory evaluation helpers."""

import pandas as pd
import pytest

from ship_trajectory_prediction.validation.rolling import (
    build_rolling_window_specs,
    summarize_rolling_predictions,
)


def test_sliding_windows_cover_complete_future_with_fixed_history():
    """Sliding windows should move a fixed observation history along the run."""
    windows = build_rolling_window_specs(
        11,
        initial_observation_count=4,
        prediction_count=3,
        stride=3,
        window_mode="sliding",
    )

    assert [window.start_index for window in windows] == [0, 3, 6]
    assert [window.forecast_start_index for window in windows] == [4, 7, 10]
    assert [window.observation_count for window in windows] == [4, 4, 4]
    assert [window.prediction_count for window in windows] == [3, 3, 1]


def test_expanding_windows_cover_complete_future_with_growing_history():
    """Expanding windows should retain every observation from the run start."""
    windows = build_rolling_window_specs(
        11,
        initial_observation_count=4,
        prediction_count=3,
        stride=3,
        window_mode="expanding",
    )

    assert [window.start_index for window in windows] == [0, 0, 0]
    assert [window.forecast_start_index for window in windows] == [4, 7, 10]
    assert [window.observation_count for window in windows] == [4, 7, 10]
    assert [window.prediction_count for window in windows] == [3, 3, 1]


def test_default_stride_uses_non_overlapping_prediction_horizons():
    """Omitting stride should advance by one complete prediction horizon."""
    windows = build_rolling_window_specs(
        10,
        initial_observation_count=4,
        prediction_count=2,
    )

    assert [window.forecast_start_index for window in windows] == [4, 6, 8]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"row_count": 0}, "row_count"),
        ({"initial_observation_count": 0}, "initial_observation_count"),
        ({"prediction_count": 0}, "prediction_count"),
        ({"stride": 0}, "stride"),
        ({"stride": 4}, "avoid gaps"),
        ({"window_mode": "unknown"}, "window_mode"),
        ({"row_count": 4}, "must exceed"),
    ],
)
def test_rolling_windows_reject_invalid_configuration(arguments, message):
    """Invalid sizes, modes, and coverage gaps should fail clearly."""
    options = {
        "row_count": 10,
        "initial_observation_count": 4,
        "prediction_count": 3,
        "stride": 3,
        "window_mode": "sliding",
    }
    options.update(arguments)

    with pytest.raises(ValueError, match=message):
        build_rolling_window_specs(**options)


def test_summarize_rolling_predictions_aggregates_windows_and_horizons():
    """Overall and horizon metrics should retain their intended weighting."""
    predictions = pd.DataFrame(
        {
            "window_index": [0, 0, 1],
            "horizon_step": [1, 2, 1],
            "horizon_seconds": [10.0, 20.0, 10.0],
            "position_error_m": [1.0, 2.0, 3.0],
            "prediction_radius_m": [2.0, 3.0, 4.0],
            "radial_covered": [True, True, False],
            "mean_marginal_interval_width_m": [4.0, 5.0, 6.0],
            "inference_method": ["vi", "vi", "vi"],
            "converged": [True, True, False],
            "mcmc_diagnostics_ok": [None, None, None],
            "window_runtime_seconds": [1.5, 1.5, 2.5],
        }
    )

    summary = summarize_rolling_predictions(predictions)

    assert summary.inference_method == "vi"
    assert summary.window_count == 2
    assert summary.forecast_count == 3
    assert summary.ade_m == pytest.approx(2.0)
    assert summary.fde_m == pytest.approx(2.0)
    assert summary.radial_coverage == pytest.approx(2 / 3)
    assert summary.mean_prediction_radius_m == pytest.approx(3.0)
    assert summary.mean_marginal_interval_width_m == pytest.approx(5.0)
    assert summary.mean_window_runtime_seconds == pytest.approx(2.0)
    assert summary.median_window_runtime_seconds == pytest.approx(2.0)
    assert summary.total_computation_time_seconds == pytest.approx(4.0)
    assert summary.vi_convergence_rate == pytest.approx(0.5)
    assert summary.mcmc_diagnostics_pass_rate is None
    assert summary.per_horizon_table["forecast_count"].tolist() == [2, 1]
    assert summary.per_horizon_table["ade_m"].tolist() == pytest.approx([2.0, 2.0])


def test_summarize_rolling_predictions_reports_mcmc_diagnostics_separately():
    """MCMC sampler diagnostics must not be labelled as VI convergence."""
    predictions = pd.DataFrame(
        {
            "window_index": [0, 1],
            "horizon_step": [1, 1],
            "horizon_seconds": [10.0, 10.0],
            "position_error_m": [1.0, 3.0],
            "prediction_radius_m": [2.0, 4.0],
            "radial_covered": [True, False],
            "mean_marginal_interval_width_m": [4.0, 6.0],
            "inference_method": ["mcmc", "mcmc"],
            "converged": [None, None],
            "mcmc_diagnostics_ok": [True, False],
            "window_runtime_seconds": [8.0, 12.0],
        }
    )

    summary = summarize_rolling_predictions(predictions)

    assert summary.inference_method == "mcmc"
    assert summary.vi_convergence_rate is None
    assert summary.mcmc_diagnostics_pass_rate == pytest.approx(0.5)


def test_summarize_rolling_predictions_rejects_missing_or_empty_data():
    """Incomplete prediction tables cannot produce reliable aggregate metrics."""
    with pytest.raises(ValueError, match="Missing rolling prediction columns"):
        summarize_rolling_predictions(pd.DataFrame({"window_index": [0]}))

    empty = pd.DataFrame(
        columns=[
            "window_index",
            "horizon_step",
            "horizon_seconds",
            "position_error_m",
            "prediction_radius_m",
            "radial_covered",
            "mean_marginal_interval_width_m",
            "inference_method",
            "converged",
            "mcmc_diagnostics_ok",
            "window_runtime_seconds",
        ]
    )
    with pytest.raises(ValueError, match="at least one forecast"):
        summarize_rolling_predictions(empty)


def test_summarize_rolling_predictions_rejects_inconsistent_window_runtimes():
    """One rolling window must not contain multiple computation times."""
    predictions = pd.DataFrame(
        {
            "window_index": [0, 0],
            "horizon_step": [1, 2],
            "horizon_seconds": [10.0, 20.0],
            "position_error_m": [1.0, 2.0],
            "prediction_radius_m": [2.0, 3.0],
            "radial_covered": [True, True],
            "mean_marginal_interval_width_m": [4.0, 5.0],
            "inference_method": ["vi", "vi"],
            "converged": [True, True],
            "mcmc_diagnostics_ok": [None, None],
            "window_runtime_seconds": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="constant within each rolling window"):
        summarize_rolling_predictions(predictions)
