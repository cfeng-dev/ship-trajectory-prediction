"""Tests for explicit batch/online inference configuration."""

import pytest

import bayestraj.forecasting.bayesian_ctrv as ctrv_config
import bayestraj.forecasting.bayesian_position_model as position_config
import bayestraj.forecasting.inference as inference
import bayestraj.models.bayesian_ctrv as ctrv_model
import bayestraj.validation.cli as validation_cli

ROLLING_CONFIG_TYPES = (
    ctrv_config.RollingExperimentConfig,
    position_config.RollingExperimentConfig,
)
SINGLE_CONFIG_TYPES = (
    ctrv_config.ExperimentConfig,
    position_config.ExperimentConfig,
)


def _rolling_config(config_type, **overrides):
    values = {
        "run_id": 102,
        "observation_count": 5,
        "prediction_count": 3,
        "position_noise_std_m": 5.0,
        "position_noise_seed": 2026,
        "stride": None,
        "inference_mode": "batch",
        "inference_method": "vi",
        "inference_seed": 42,
        "window_mode": "sliding",
    }
    values.update(overrides)
    return config_type(**values)


@pytest.mark.parametrize("config_type", ROLLING_CONFIG_TYPES)
@pytest.mark.parametrize(
    ("inference_method", "window_mode"),
    (
        ("vi", "sliding"),
        ("vi", "expanding"),
        ("mcmc", "sliding"),
        ("mcmc", "expanding"),
    ),
)
def test_batch_inference_combinations_are_valid(
    config_type,
    inference_method,
    window_mode,
):
    config = _rolling_config(
        config_type,
        inference_method=inference_method,
        window_mode=window_mode,
    )

    assert config.inference_mode == "batch"
    assert config.inference_method == inference_method
    assert config.window_mode == window_mode


@pytest.mark.parametrize("config_type", ROLLING_CONFIG_TYPES)
def test_online_rbpf_is_valid_without_a_window_mode(config_type):
    config = _rolling_config(
        config_type,
        inference_mode="online",
        inference_method="rbpf",
        window_mode=None,
    )

    assert config.inference_mode == "online"
    assert config.inference_method == "rbpf"
    assert config.window_mode is None


@pytest.mark.parametrize("config_type", ROLLING_CONFIG_TYPES)
@pytest.mark.parametrize(
    ("inference_mode", "inference_method", "window_mode", "message"),
    (
        ("batch", "rbpf", "sliding", "Batch inference requires"),
        ("online", "vi", None, "Online inference requires"),
        ("online", "mcmc", None, "Online inference requires"),
        ("batch", "vi", "sequential", "Batch inference requires window_mode"),
        ("online", "rbpf", "sliding", "does not use window_mode"),
    ),
)
def test_invalid_inference_combinations_fail_early(
    config_type,
    inference_mode,
    inference_method,
    window_mode,
    message,
):
    with pytest.raises(ValueError, match=message):
        _rolling_config(
            config_type,
            inference_mode=inference_mode,
            inference_method=inference_method,
            window_mode=window_mode,
        )


def test_cmdstan_configuration_rejects_online_rbpf():
    with pytest.raises(ValueError, match="only used in batch mode"):
        inference.select_inference_config(
            "online",
            "rbpf",
            vi_algorithm="meanfield",
            require_converged=False,
            vi_config=inference.create_default_vi_config(),
            mcmc_config=inference.create_default_mcmc_config(),
            fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
        )


@pytest.mark.parametrize("config_type", SINGLE_CONFIG_TYPES)
def test_single_window_predictions_require_batch_inference(config_type):
    with pytest.raises(ValueError, match="batch inference only"):
        config_type(
            run_id=102,
            start_index=0,
            observation_count=5,
            prediction_count=3,
            position_noise_std_m=5.0,
            position_noise_seed=2026,
            inference_mode="online",
            inference_method="rbpf",
            inference_seed=42,
        )


def test_ctrv_cli_exposes_online_rbpf_without_a_window_mode():
    experiment = _rolling_config(
        ctrv_config.RollingExperimentConfig,
        inference_mode="online",
        inference_method="rbpf",
        window_mode=None,
    )

    options = validation_cli.parse_bayesian_ctrv_evaluation_arguments(
        description=None,
        experiment=experiment,
        priors=ctrv_model.BayesianCTRVPriors(),
        vi_config=inference.create_default_vi_config(),
        max_windows=None,
        plot_each_window=False,
        argv=[],
    )

    assert options.inference_mode == "online"
    assert options.inference_method == "rbpf"
    assert options.window_mode is None


def test_position_cli_can_select_batch_mcmc_with_an_expanding_window():
    experiment = _rolling_config(
        position_config.RollingExperimentConfig,
        inference_mode="online",
        inference_method="rbpf",
        window_mode=None,
    )

    options = position_config.parse_evaluation_arguments(
        description=None,
        experiment=experiment,
        vi_config=inference.create_default_vi_config(),
        max_windows=None,
        plot_each_window=False,
        argv=[
            "--inference-mode",
            "batch",
            "--inference-method",
            "mcmc",
            "--window-mode",
            "expanding",
        ],
    )

    assert options.inference_mode == "batch"
    assert options.inference_method == "mcmc"
    assert options.window_mode == "expanding"


@pytest.mark.parametrize(
    ("config_type", "model_family"),
    (
        (ctrv_config.RollingExperimentConfig, "ctrv"),
        (position_config.RollingExperimentConfig, "position"),
    ),
)
def test_cli_rejects_sequential_as_a_window_mode(config_type, model_family):
    experiment = _rolling_config(
        config_type,
        inference_mode="online",
        inference_method="rbpf",
        window_mode=None,
    )

    with pytest.raises(SystemExit):
        if model_family == "ctrv":
            validation_cli.parse_bayesian_ctrv_evaluation_arguments(
                description=None,
                experiment=experiment,
                priors=ctrv_model.BayesianCTRVPriors(),
                vi_config=inference.create_default_vi_config(),
                max_windows=None,
                plot_each_window=False,
                argv=["--window-mode", "sequential"],
            )
        else:
            position_config.parse_evaluation_arguments(
                description=None,
                experiment=experiment,
                vi_config=inference.create_default_vi_config(),
                max_windows=None,
                plot_each_window=False,
                argv=["--window-mode", "sequential"],
            )
