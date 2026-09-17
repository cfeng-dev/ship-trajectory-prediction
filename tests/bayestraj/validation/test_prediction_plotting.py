"""Tests for posterior-predictive trajectory plot sampling."""

import bayestraj.validation.prediction_plotting as prediction_plotting


def test_sample_trajectory_indices_honors_requests_above_default_count():
    """Allow an experiment to show more than the default number of paths."""
    indices = prediction_plotting._sample_trajectory_indices(120, 100, seed=42)

    assert len(indices) == 100
