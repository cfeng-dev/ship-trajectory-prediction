"""Tests for the hybrid Bayesian CTRV rolling entry point."""

import numpy as np
import pytest

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
    assert isinstance(
        captured["priors"],
        experiment.hybrid_model.HybridBayesianCTRVPriors,
    )
    assert not hasattr(captured["priors"], "turn_rate_state_prior_scale")
    assert captured["vi_config"] is experiment.VI_CONFIG
    assert captured["mcmc_config"] is experiment.MCMC_CONFIG
    assert captured["experiment"].inference_method == (
        experiment.EXPERIMENT.inference_method
    )
    assert captured["vi_algorithm"] == experiment.VI_CONFIG["algorithm"]
    assert captured["noise_parameter_names"] == (
        experiment.hybrid_model.NOISE_PARAMETER_NAMES
    )
    assert captured["has_latent_turn_rate"] is False


def test_hybrid_configuration_does_not_alias_bayesian_defaults():
    """Editing rolling hybrid settings should not mutate Bayesian defaults."""
    assert experiment.EXPERIMENT is not bayesian_experiment.EXPERIMENT
    assert experiment.PRIORS is not bayesian_experiment.PRIORS
    assert not hasattr(experiment.PRIORS, "sigma_turn_rate_process_prior_scale")
    assert experiment.VI_CONFIG is not bayesian_experiment.VI_CONFIG
    assert experiment.MCMC_CONFIG is not bayesian_experiment.MCMC_CONFIG


def test_hybrid_cli_has_no_latent_turn_rate_prior_option():
    """The hybrid CLI should not expose a prior for a removed latent state."""
    with pytest.raises(SystemExit):
        experiment.main(["--turn-rate-prior-scale", "0.002"])


def test_hybrid_vi_guard_checks_only_existing_noise_parameters():
    """Rolling hybrid validation should not request turn-rate process draws."""

    class FakeFit:
        def stan_variable(self, name, mean=False):
            del mean
            return np.array([getattr(experiment.PRIORS, f"{name}_prior_scale")])

    reason = experiment.workflow._vi_numerical_instability_reason(
        FakeFit(),
        experiment.PRIORS,
        noise_parameter_names=experiment.hybrid_model.NOISE_PARAMETER_NAMES,
    )

    assert reason is None


def test_hybrid_diagnostics_use_the_fixed_prediction_turn_rate():
    """Hybrid diagnostics should not require a latent forecast-origin output."""

    class FakeFit:
        variables = {
            "turn_rate_prediction": np.array(
                [[-0.012, -0.012], [-0.012, -0.012]],
            ),
            "heading_state": np.array([[0.3, 0.4], [0.3, 0.4]]),
            "heading_state_prediction": np.array(
                [[0.28, 0.16], [0.28, 0.16]],
            ),
            "sigma_position_process": np.array([0.5, 0.5]),
            "sigma_speed_process": np.array([0.05, 0.05]),
        }

        def stan_variable(self, name, mean=False):
            del mean
            return self.variables[name]

    diagnostics = experiment.workflow._posterior_window_diagnostics(
        FakeFit(),
        noise_parameter_names=experiment.hybrid_model.NOISE_PARAMETER_NAMES,
        has_latent_turn_rate=False,
    )

    assert diagnostics["forecast_origin_turn_rate_rad_s"] == pytest.approx(-0.012)
    assert diagnostics["forecast_heading_change_rad"] == pytest.approx(-0.24)
