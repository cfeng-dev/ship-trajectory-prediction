"""Tests for the hybrid Bayesian CTRV single-window entry point."""

import experiments.trajectory_prediction.bayesian_ctrv as bayesian_experiment
import experiments.trajectory_prediction.hybrid_bayesian_ctrv as experiment


def test_main_delegates_visible_hybrid_configuration(monkeypatch):
    """The hybrid entry point should pass its own visible configuration."""
    captured = {}
    expected_result = object()

    def fake_run(**kwargs):
        captured.update(kwargs)
        return expected_result

    monkeypatch.setattr(
        experiment.workflow,
        "run_hybrid_bayesian_ctrv_prediction",
        fake_run,
    )

    result = experiment.main([])

    assert result is expected_result
    assert captured == {
        "data_file": experiment.DATA_FILE,
        "experiment": experiment.EXPERIMENT,
        "priors": experiment.PRIORS,
        "vi_config": experiment.VI_CONFIG,
        "mcmc_config": experiment.MCMC_CONFIG,
        "fullrank_grad_samples": experiment.config.DEFAULT_FULLRANK_GRAD_SAMPLES,
        "credible_interval": experiment.CREDIBLE_INTERVAL,
        "hybrid_config": experiment.HYBRID_CONFIG,
        "inference_method": experiment.EXPERIMENT.inference_method,
        "vi_algorithm": experiment.VI_CONFIG["algorithm"],
        "seed": experiment.EXPERIMENT.inference_seed,
        "position_noise_std_m": (experiment.EXPERIMENT.additional_position_noise_std_m),
        "position_noise_seed": experiment.EXPERIMENT.position_noise_seed,
        "require_converged": experiment.VI_CONFIG["require_converged"],
        "plot_coordinate_mode": experiment.PLOT_COORDINATE_MODE,
    }


def test_hybrid_configuration_does_not_alias_bayesian_defaults():
    """Editing hybrid settings should not mutate the Bayesian experiment."""
    assert experiment.workflow is not bayesian_experiment.workflow
    assert experiment.EXPERIMENT is not bayesian_experiment.EXPERIMENT
    assert experiment.PRIORS is not bayesian_experiment.PRIORS
    assert isinstance(
        experiment.PRIORS,
        experiment.hybrid_model.HybridBayesianCTRVPriors,
    )
    assert not hasattr(experiment.PRIORS, "turn_rate_state_prior_scale")
    assert not hasattr(experiment.PRIORS, "sigma_turn_rate_process_prior_scale")
    assert experiment.VI_CONFIG is not bayesian_experiment.VI_CONFIG
    assert experiment.MCMC_CONFIG is not bayesian_experiment.MCMC_CONFIG


def test_main_forwards_hybrid_cli_overrides(monkeypatch):
    """The hybrid entry point should forward explicitly selected CLI values."""
    captured = {}
    monkeypatch.setattr(
        experiment.workflow,
        "run_hybrid_bayesian_ctrv_prediction",
        lambda **kwargs: captured.update(kwargs),
    )

    experiment.main(
        [
            "--inference",
            "mcmc",
            "--seed",
            "17",
            "--position-noise-std-m",
            "0",
            "--plot-coordinates",
            "km",
        ]
    )

    assert captured["inference_method"] == "mcmc"
    assert captured["seed"] == 17
    assert captured["position_noise_std_m"] == 0.0
    assert captured["plot_coordinate_mode"] == "km"
