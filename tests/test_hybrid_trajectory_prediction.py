"""Tests for the hybrid Bayesian CTRV single-window entry point."""

import sys

import experiments.trajectory_prediction.bayesian_ctrv as bayesian_experiment
import experiments.trajectory_prediction.hybrid_bayesian_ctrv as experiment


def test_main_passes_independent_hybrid_configuration(monkeypatch):
    """The hybrid entry point should pass its own visible configuration."""
    captured = {}
    monkeypatch.setattr(
        experiment,
        "run_cli",
        lambda **kwargs: captured.update(kwargs),
    )

    experiment.main()

    assert captured == {
        "model_variant": "hybrid",
        "description": "Run one hybrid Bayesian CTRV trajectory prediction.",
        "data_file": experiment.DATA_FILE,
        "experiment": experiment.EXPERIMENT,
        "priors": experiment.PRIORS,
        "vi_config": experiment.VI_CONFIG,
        "mcmc_config": experiment.MCMC_CONFIG,
        "fullrank_grad_samples": experiment.DEFAULT_FULLRANK_GRAD_SAMPLES,
        "credible_interval": experiment.CREDIBLE_INTERVAL,
        "plot_coordinate_mode": experiment.PLOT_COORDINATE_MODE,
        "hybrid_config": experiment.HYBRID_CONFIG,
    }


def test_hybrid_configuration_does_not_alias_bayesian_defaults():
    """Editing hybrid settings should not mutate the Bayesian experiment."""
    assert experiment.EXPERIMENT is not bayesian_experiment.EXPERIMENT
    assert experiment.PRIORS is not bayesian_experiment.PRIORS
    assert experiment.VI_CONFIG is not bayesian_experiment.VI_CONFIG
    assert experiment.MCMC_CONFIG is not bayesian_experiment.MCMC_CONFIG


def test_shared_cli_forwards_hybrid_configuration(monkeypatch):
    """The shared CLI should use the selected entry point's defaults."""
    captured = {}
    monkeypatch.setattr(sys, "argv", ["hybrid_bayesian_ctrv.py"])
    monkeypatch.setattr(
        bayesian_experiment,
        "main",
        lambda **kwargs: captured.update(kwargs),
    )

    experiment.main()

    assert captured["model_variant"] == "hybrid"
    assert captured["data_file"] == experiment.DATA_FILE
    assert captured["experiment"] is experiment.EXPERIMENT
    assert captured["priors"] is experiment.PRIORS
    assert captured["vi_config"] is experiment.VI_CONFIG
    assert captured["mcmc_config"] is experiment.MCMC_CONFIG
    assert captured["hybrid_config"] is experiment.HYBRID_CONFIG
    assert captured["inference_method"] == experiment.EXPERIMENT.inference_method
    assert captured["vi_algorithm"] == experiment.VI_CONFIG["algorithm"]
