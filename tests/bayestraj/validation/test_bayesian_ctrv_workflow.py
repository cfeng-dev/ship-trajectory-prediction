"""Tests for Bayesian CTRV rolling-evaluation presentation rules."""

import bayestraj.validation.bayesian_ctrv_workflow as workflow


def test_rolling_plot_is_hidden_only_for_overlapping_forecast_origins():
    assert workflow._should_plot_rolling_predictions(prediction_count=3, stride=None)
    assert workflow._should_plot_rolling_predictions(prediction_count=3, stride=3)
    assert not workflow._should_plot_rolling_predictions(prediction_count=3, stride=1)
    assert not workflow._should_plot_rolling_predictions(prediction_count=3, stride=2)
