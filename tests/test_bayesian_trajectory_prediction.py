"""Tests for the parametric Bayesian CTRV single-window entry point."""

import experiments.trajectory_prediction.bayesian_ctrv as experiment


def test_main_delegates_history_and_reproducibility_options(monkeypatch):
    """The entry point should expose K, seeds, and the common window config."""
    captured = {}
    expected_result = object()
    monkeypatch.setattr(
        experiment.workflow,
        "run_bayesian_ctrv_prediction",
        lambda **kwargs: captured.update(kwargs) or expected_result,
    )

    result = experiment.main(["--history-positions", "10"])

    assert result is expected_result
    assert captured["experiment"] is experiment.EXPERIMENT
    assert captured["history_position_count"] == 10
    assert captured["seed"] == experiment.EXPERIMENT.inference_seed
    assert captured["position_noise_seed"] == (
        experiment.EXPERIMENT.position_noise_seed
    )
    assert captured["position_noise_std_m"] == (
        experiment.EXPERIMENT.additional_position_noise_std_m
    )
    assert captured["priors"] is experiment.PRIORS


def test_default_and_comparison_k_share_all_other_cli_settings(monkeypatch):
    """K=20 and K=10 should differ only in the selected trailing history."""
    calls = []
    monkeypatch.setattr(
        experiment.workflow,
        "run_bayesian_ctrv_prediction",
        lambda **kwargs: calls.append(kwargs),
    )

    experiment.main([])
    experiment.main(["--history-positions", "10"])

    assert [call["history_position_count"] for call in calls] == [20, 10]
    for name in (
        "experiment",
        "seed",
        "position_noise_seed",
        "position_noise_std_m",
        "credible_interval",
    ):
        assert calls[0][name] == calls[1][name]
