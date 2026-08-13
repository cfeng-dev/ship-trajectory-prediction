"""Tests for the hybrid Bayesian CTRV rolling entry point."""

import experiments.model_evaluation.rolling.hybrid_bayesian_ctrv as experiment


def test_main_selects_hybrid_model(monkeypatch):
    """The dedicated entry point should not require a model-selection flag."""
    captured = {}
    monkeypatch.setattr(
        experiment,
        "run_cli",
        lambda **kwargs: captured.update(kwargs),
    )

    experiment.main()

    assert captured == {
        "model_variant": "hybrid",
        "description": "Evaluate hybrid Bayesian CTRV forecasts across rolling windows.",
    }
