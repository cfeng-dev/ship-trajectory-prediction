"""Tests for the hybrid Bayesian CTRV rolling entry point."""

import sys

import experiments.model_evaluation.bayesian_ctrv as bayesian_experiment
import experiments.model_evaluation.hybrid_bayesian_ctrv as experiment


def test_main_selects_hybrid_model(monkeypatch):
    """The dedicated entry point should not require a model-selection flag."""
    captured = {}
    monkeypatch.setattr(
        experiment.rolling_bayesian,
        "run_cli",
        lambda **kwargs: captured.update(kwargs),
    )

    experiment.main()

    assert captured == {
        "model_variant": "hybrid",
        "description": "Evaluate hybrid Bayesian CTRV forecasts across rolling windows.",
        "data_file": experiment.DATA_FILE,
        "experiment": experiment.EXPERIMENT,
        "priors": experiment.PRIORS,
        "vi_config": experiment.VI_CONFIG,
        "mcmc_config": experiment.MCMC_CONFIG,
        "fullrank_grad_samples": experiment.forecasting.DEFAULT_FULLRANK_GRAD_SAMPLES,
        "credible_interval": experiment.CREDIBLE_INTERVAL,
        "max_windows": experiment.MAX_WINDOWS,
        "plot_each_window": experiment.PLOT_EACH_WINDOW,
        "sample_trajectories_per_forecast": (
            experiment.SAMPLE_TRAJECTORIES_PER_FORECAST
        ),
        "hybrid_config": experiment.HYBRID_CONFIG,
    }


def test_hybrid_configuration_does_not_alias_bayesian_defaults():
    """Editing rolling hybrid settings should not mutate Bayesian defaults."""
    assert experiment.EXPERIMENT is not bayesian_experiment.EXPERIMENT
    assert experiment.PRIORS is not bayesian_experiment.PRIORS
    assert experiment.VI_CONFIG is not bayesian_experiment.VI_CONFIG
    assert experiment.MCMC_CONFIG is not bayesian_experiment.MCMC_CONFIG


def test_shared_cli_forwards_hybrid_configuration(monkeypatch):
    """The shared rolling CLI should use the hybrid entry point's defaults."""
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
    assert captured["priors"] is not experiment.PRIORS
    assert captured["priors"].turn_rate_state_prior_scale == (
        experiment.PRIORS.turn_rate_state_prior_scale
    )
    assert captured["vi_config"] is experiment.VI_CONFIG
    assert captured["mcmc_config"] is experiment.MCMC_CONFIG
    assert captured["hybrid_config"] is experiment.HYBRID_CONFIG
    assert captured["inference_method"] == experiment.EXPERIMENT.inference_method
    assert captured["vi_algorithm"] == experiment.VI_CONFIG["algorithm"]
