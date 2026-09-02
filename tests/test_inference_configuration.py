"""Tests for method-based batch and online inference configuration."""

import pytest

import bayestraj.forecasting.bayesian_ctrv as ctrv_config
import bayestraj.forecasting.bayesian_position_model as position_config
import bayestraj.forecasting.cli as forecasting_cli
import bayestraj.forecasting.inference as inference
import bayestraj.models.bayesian_ctrv as ctrv_model
import bayestraj.models.bayesian_position_model as position_model
import bayestraj.models.sequential_monte_carlo_ctrv as smc_model
import bayestraj.validation.cli as validation_cli


def _ctrv_rolling_config(**overrides):
    values = {
        "run_id": 102,
        "observation_count": 5,
        "prediction_count": 3,
        "position_noise_std_m": 5.0,
        "position_noise_seed": 2026,
        "stride": None,
        "inference_method": "vi_sliding",
        "inference_seed": 42,
    }
    values.update(overrides)
    return ctrv_config.RollingExperimentConfig(**values)


def _position_rolling_config(**overrides):
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
    return position_config.RollingExperimentConfig(**values)


@pytest.mark.parametrize(
    ("inference_method", "window_mode"),
    (
        ("vi", "sliding"),
        ("vi", "expanding"),
        ("mcmc", "sliding"),
        ("mcmc", "expanding"),
    ),
)
def test_position_batch_inference_combinations_are_valid(
    inference_method,
    window_mode,
):
    config = _position_rolling_config(
        inference_method=inference_method,
        window_mode=window_mode,
    )

    assert config.inference_mode == "batch"
    assert config.inference_method == inference_method
    assert config.window_mode == window_mode


@pytest.mark.parametrize(
    ("selection", "inference_method", "window_mode"),
    (
        ("vi_sliding", "vi", "sliding"),
        ("vi_expanding", "vi", "expanding"),
        ("mcmc_sliding", "mcmc", "sliding"),
        ("mcmc_expanding", "mcmc", "expanding"),
    ),
)
def test_ctrv_rolling_batch_selection_derives_method_and_window(
    selection,
    inference_method,
    window_mode,
):
    config = _ctrv_rolling_config(inference_method=selection)

    normalized = inference.normalize_ctrv_rolling_inference_method(
        config.inference_method
    )

    assert config.inference_method == selection
    assert not hasattr(config, "inference_mode")
    assert not hasattr(config, "window_mode")
    assert normalized == ("batch", inference_method, window_mode)


def test_position_online_rbpf_is_valid_without_a_window_mode():
    config = _position_rolling_config(
        inference_mode="online",
        inference_method="rbpf",
        window_mode=None,
    )

    assert config.inference_mode == "online"
    assert config.inference_method == "rbpf"
    assert config.window_mode is None


def test_ctrv_online_rbpf_derives_online_mode_without_a_window():
    config = _ctrv_rolling_config(inference_method="rbpf")

    normalized = inference.normalize_ctrv_rolling_inference_method(
        config.inference_method
    )

    assert config.inference_method == "rbpf"
    assert not hasattr(config, "inference_mode")
    assert not hasattr(config, "window_mode")
    assert normalized == ("online", "rbpf", None)


def test_ctrv_online_smc_is_valid_without_a_window_mode():
    config = _ctrv_rolling_config(inference_method="smc")

    normalized = inference.normalize_ctrv_rolling_inference_method(
        config.inference_method
    )

    assert config.inference_method == "smc"
    assert not hasattr(config, "inference_mode")
    assert not hasattr(config, "window_mode")
    assert normalized == ("online", "smc", None)


def test_position_online_smc_is_rejected():
    with pytest.raises(ValueError, match="Online inference requires"):
        _position_rolling_config(
            inference_mode="online",
            inference_method="smc",
            window_mode=None,
        )


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
    inference_mode,
    inference_method,
    window_mode,
    message,
):
    with pytest.raises(ValueError, match=message):
        _position_rolling_config(
            inference_mode=inference_mode,
            inference_method=inference_method,
            window_mode=window_mode,
        )


@pytest.mark.parametrize(
    "inference_method",
    (
        "vi",
        "mcmc",
        "sliding_vi",
        "expanding_vi",
        "unsupported",
    ),
)
def test_invalid_ctrv_rolling_inference_selection_fails_early(inference_method):
    with pytest.raises(ValueError, match="rolling inference_method must be one of"):
        _ctrv_rolling_config(inference_method=inference_method)


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


def test_default_rbpf_factories_return_model_specific_configs():
    ctrv_rbpf_config = inference.create_default_ctrv_rbpf_config()
    position_rbpf_config = inference.create_default_position_rbpf_config()

    assert isinstance(ctrv_rbpf_config, ctrv_model.SequentialCTRVFilterConfig)
    assert isinstance(
        position_rbpf_config,
        position_model.SequentialPositionFilterConfig,
    )


def test_default_ctrv_smc_factory_returns_an_independent_config():
    first_config = inference.create_default_ctrv_smc_config()
    second_config = inference.create_default_ctrv_smc_config()

    assert isinstance(first_config, smc_model.SequentialMonteCarloCTRVConfig)
    assert first_config == second_config
    assert first_config is not second_config


@pytest.mark.parametrize("inference_method", ("vi", "mcmc", "rbpf", "smc"))
def test_single_window_ctrv_derives_mode_from_inference_method(inference_method):
    config = ctrv_config.ExperimentConfig(
        run_id=102,
        start_index=0,
        observation_count=5,
        prediction_count=3,
        position_noise_std_m=5.0,
        position_noise_seed=2026,
        inference_method=inference_method,
        inference_seed=42,
    )

    assert config.inference_method == inference_method
    assert not hasattr(config, "inference_mode")


@pytest.mark.parametrize(
    "inference_method",
    (
        "vi_sliding",
        "vi_expanding",
        "mcmc_sliding",
        "mcmc_expanding",
        "rbpf",
        "smc",
    ),
)
def test_rolling_ctrv_accepts_only_one_inference_selection(inference_method):
    config = ctrv_config.RollingExperimentConfig(
        run_id=102,
        observation_count=5,
        prediction_count=3,
        position_noise_std_m=5.0,
        position_noise_seed=2026,
        stride=None,
        inference_method=inference_method,
        inference_seed=42,
    )

    assert config.inference_method == inference_method
    assert not hasattr(config, "inference_mode")
    assert not hasattr(config, "window_mode")


def test_single_window_position_prediction_requires_batch_inference():
    with pytest.raises(ValueError, match="batch inference only"):
        position_config.ExperimentConfig(
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


@pytest.mark.parametrize("online_method", ("rbpf", "smc"))
def test_single_window_ctrv_cli_accepts_online_particle_filters(online_method):
    experiment = ctrv_config.ExperimentConfig(
        run_id=102,
        start_index=0,
        observation_count=5,
        prediction_count=3,
        position_noise_std_m=5.0,
        position_noise_seed=2026,
        inference_method="vi",
        inference_seed=42,
    )

    arguments = forecasting_cli.parse_bayesian_ctrv_prediction_arguments(
        description=None,
        experiment=experiment,
        vi_config=inference.create_default_vi_config(),
        plot_coordinate_mode="m",
        argv=["--inference-method", online_method],
    )

    assert not hasattr(arguments, "inference_mode")
    assert arguments.inference_method == online_method


@pytest.mark.parametrize(
    "inference_method",
    (
        "vi_sliding",
        "vi_expanding",
        "mcmc_sliding",
        "mcmc_expanding",
        "rbpf",
        "smc",
    ),
)
def test_ctrv_rolling_cli_accepts_one_inference_selection(inference_method):
    experiment = _ctrv_rolling_config(inference_method="rbpf")

    options = validation_cli.parse_bayesian_ctrv_evaluation_arguments(
        description=None,
        experiment=experiment,
        priors=ctrv_model.BayesianCTRVPriors(),
        vi_config=inference.create_default_vi_config(),
        max_windows=None,
        plot_each_window=False,
        argv=["--inference-method", inference_method],
    )

    assert not hasattr(options, "inference_mode")
    assert not hasattr(options, "window_mode")
    assert options.inference_method == inference_method


def test_position_cli_can_select_batch_mcmc_with_an_expanding_window():
    experiment = _position_rolling_config(
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


def test_ctrv_rolling_cli_rejects_removed_window_mode_option():
    experiment = _ctrv_rolling_config(inference_method="rbpf")

    with pytest.raises(SystemExit):
        validation_cli.parse_bayesian_ctrv_evaluation_arguments(
            description=None,
            experiment=experiment,
            priors=ctrv_model.BayesianCTRVPriors(),
            vi_config=inference.create_default_vi_config(),
            max_windows=None,
            plot_each_window=False,
            argv=["--window-mode", "sliding"],
        )


def test_position_cli_rejects_sequential_as_a_window_mode():
    experiment = _position_rolling_config(
        inference_mode="online",
        inference_method="rbpf",
        window_mode=None,
    )

    with pytest.raises(SystemExit):
        position_config.parse_evaluation_arguments(
            description=None,
            experiment=experiment,
            vi_config=inference.create_default_vi_config(),
            max_windows=None,
            plot_each_window=False,
            argv=["--window-mode", "sequential"],
        )
