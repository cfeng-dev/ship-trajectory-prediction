"""Tests for the Bayesian Position Model single-window entry point."""

import experiments.trajectory_prediction.bayesian_position_model as experiment


def test_main_delegates_k20_default_and_k10_override(monkeypatch):
    """Both approved local history choices must reach the independent workflow."""
    calls = []
    expected_result = object()
    monkeypatch.setattr(
        experiment.workflow,
        "run_bayesian_position_prediction",
        lambda **kwargs: calls.append(kwargs) or expected_result,
    )

    first_result = experiment.main([])
    second_result = experiment.main(["--history-positions", "10"])

    assert first_result is expected_result
    assert second_result is expected_result
    assert calls[0]["history_position_count"] == 20
    assert calls[1]["history_position_count"] == 10
    assert calls[0]["priors"] is experiment.PRIORS
    assert calls[0]["position_noise_std_m"] == 5.0
    assert calls[0]["credible_interval"] == 0.9
