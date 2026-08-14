"""Tests for the hybrid Bayesian CTRV rolling entry point."""

import experiments.model_evaluation.bayesian_ctrv as bayesian_experiment
import experiments.model_evaluation.hybrid_bayesian_ctrv as experiment


def test_main_calls_hybrid_workflow_without_model_variant(monkeypatch):
    """The hybrid entry point should call its dedicated rolling workflow."""
    captured = {}
    expected_result = object()
    monkeypatch.setattr(
        experiment.workflow,
        "_run_bayesian_ctrv_evaluation",
        lambda **kwargs: captured.update(kwargs) or expected_result,
    )

    result = experiment.main([])

    assert result is expected_result
    assert not hasattr(experiment, "MODEL_VARIANT")
    assert captured["model_name"] == "hybrid"
    assert captured["model_label"] == "Hybrid Bayesian CTRV"
    assert (
        captured["fit_model"].func
        is experiment.hybrid_model.fit_hybrid_bayesian_ctrv_model
    )
    assert captured["fit_model"].keywords == {"hybrid_config": experiment.HYBRID_CONFIG}
    assert captured["data_file"] == experiment.DATA_FILE
    assert captured["experiment"] is not experiment.EXPERIMENT
    assert captured["experiment"] == experiment.EXPERIMENT
    assert captured["priors"] is not experiment.PRIORS
    assert captured["priors"].turn_rate_state_prior_scale == (
        experiment.PRIORS.turn_rate_state_prior_scale
    )
    assert captured["vi_config"] is experiment.VI_CONFIG
    assert captured["mcmc_config"] is experiment.MCMC_CONFIG
    assert captured["experiment"].inference_method == (
        experiment.EXPERIMENT.inference_method
    )
    assert captured["vi_algorithm"] == experiment.VI_CONFIG["algorithm"]


def test_hybrid_configuration_does_not_alias_bayesian_defaults():
    """Editing rolling hybrid settings should not mutate Bayesian defaults."""
    assert experiment.EXPERIMENT is not bayesian_experiment.EXPERIMENT
    assert experiment.PRIORS is not bayesian_experiment.PRIORS
    assert experiment.VI_CONFIG is not bayesian_experiment.VI_CONFIG
    assert experiment.MCMC_CONFIG is not bayesian_experiment.MCMC_CONFIG
