"""Tests for the fully Bayesian CTRV single-window entry point."""

import experiments.trajectory_prediction.bayesian_ctrv as experiment


def test_main_delegates_visible_experiment_configuration(monkeypatch):
    """The experiment entry point should delegate all configured values."""
    captured = {}
    expected_result = object()

    def fake_run(**kwargs):
        captured.update(kwargs)
        return expected_result

    monkeypatch.setattr(
        experiment.workflow,
        "run_fully_bayesian_ctrv_prediction",
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
        "inference_method": experiment.EXPERIMENT.inference_method,
        "vi_algorithm": experiment.VI_CONFIG["algorithm"],
        "seed": experiment.EXPERIMENT.inference_seed,
        "position_noise_std_m": (experiment.EXPERIMENT.additional_position_noise_std_m),
        "position_noise_seed": experiment.EXPERIMENT.position_noise_seed,
        "require_converged": experiment.VI_CONFIG["require_converged"],
        "plot_coordinate_mode": experiment.PLOT_COORDINATE_MODE,
    }
