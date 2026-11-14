"""Tests for method-based batch and online inference configuration."""

from types import SimpleNamespace

import numpy as np
import pytest

import bayestraj.forecasting.bayesian_ctrv as ctrv_config
import bayestraj.inference.cmdstan as cmdstan_inference
import bayestraj.inference.configuration as inference
import bayestraj.inference.ctrv_smc as smc_model
import bayestraj.models.bayesian_ctrv as ctrv_model
import bayestraj.validation.cli as validation_cli
from bayestraj.inference.ctrv_sequential_vi import SequentialVIConfig


@pytest.mark.parametrize(
    ("note", "messages", "termination"),
    [
        ("MEAN ELBO CONVERGED", "", "elbo_converged"),
        (
            "MEDIAN ELBO CONVERGED",
            "This variational approximation may not have converged to a good optimum.",
            "elbo_converged",
        ),
        (
            "MAY BE DIVERGING... INSPECT ELBO",
            "Informational Message: The maximum number of iterations is reached! "
            "The algorithm may not have converged.",
            "iteration_limit",
        ),
        ("", "", "completed"),
    ],
)
def test_vi_diagnostics_keep_last_elbo_changes_and_stopping_reason(
    tmp_path, note, messages, termination
):
    transcript = tmp_path / "vi.txt"
    transcript.write_text(
        "method = variational\n  iter = 250\n  adapt\n    iter = 50\n"
        "  tol_rel_obj = 0.01\n"
        "Begin stochastic gradient ascent.\n"
        "  iter             ELBO   delta_ELBO_mean   delta_ELBO_med   notes\n"
        "   100        -1200.000             1.000            1.000\n"
        f"   200        -1.105e3             0.080            0.009   {note}\n"
        f"{messages}\nDrawing a sample of size 32 from the approximate posterior...\n"
        "COMPLETED.\n",
        encoding="utf-8",
    )
    fit = SimpleNamespace(
        runset=SimpleNamespace(stdout_files=[str(transcript)]),
    )

    values = cmdstan_inference.variational_diagnostics(fit)

    assert values == {
        "vi_termination": termination,
        "vi_elbo_delta_mean": 0.08,
        "vi_elbo_delta_median": 0.009,
    }


@pytest.mark.parametrize("problem", ["no_fit", "no_runset", "no_files", "removed_file"])
def test_vi_diagnostics_leave_missing_evidence_unavailable(tmp_path, problem):
    fit = None if problem == "no_fit" else SimpleNamespace()
    if problem in {"no_files", "removed_file"}:
        fit.runset = SimpleNamespace(
            stdout_files=[] if problem == "no_files" else [tmp_path / "missing.txt"]
        )

    values = cmdstan_inference.variational_diagnostics(fit)

    assert np.isnan(values["vi_elbo_delta_mean"])
    assert np.isnan(values["vi_elbo_delta_median"])
    assert values["vi_termination"] == (None if fit is None else "unavailable")


def test_vi_diagnostics_retain_iteration_limit_without_an_elbo_evaluation(tmp_path):
    transcript = tmp_path / "vi.txt"
    transcript.write_text(
        "method = variational\n  iter = 50\n  tol_rel_obj = 0.01\n"
        "Begin stochastic gradient ascent.\n"
        "  iter             ELBO   delta_ELBO_mean   delta_ELBO_med   notes\n"
        "The maximum number of iterations is reached! "
        "The algorithm may not have converged.\nCOMPLETED.\n",
        encoding="utf-8",
    )
    fit = SimpleNamespace(
        runset=SimpleNamespace(stdout_files=[transcript]),
    )
    values = cmdstan_inference.variational_diagnostics(fit)
    assert values["vi_termination"] == "iteration_limit"
    assert np.isnan(values["vi_elbo_delta_mean"])
    assert np.isnan(values["vi_elbo_delta_median"])


def test_vi_diagnostics_do_not_assume_completion_from_an_elbo_table(tmp_path):
    transcript = tmp_path / "vi.txt"
    transcript.write_text(
        "Begin stochastic gradient ascent.\n"
        "  iter             ELBO   delta_ELBO_mean   delta_ELBO_med   notes\n"
        "100 -10.0 0.1 0.2\n",
        encoding="utf-8",
    )
    fit = SimpleNamespace(runset=SimpleNamespace(stdout_files=[transcript]))
    values = cmdstan_inference.variational_diagnostics(fit)
    assert values["vi_termination"] == "incomplete"
    assert values["vi_elbo_delta_mean"] == 0.1
    assert values["vi_elbo_delta_median"] == 0.2


def _ctrv_rolling_config(**overrides):
    values = {
        "observation_count": 5,
        "prediction_count": 3,
        "position_noise_std_m": 5.0,
        "position_noise_seed": 2026,
        "stride": None,
        "inference_method": "vi",
        "inference_seed": 42,
    }
    values.update(overrides)
    return ctrv_config.RollingExperimentConfig(**values)


@pytest.mark.parametrize(
    ("inference_method", "window_mode"),
    (
        ("vi", "sliding"),
        ("vi", "expanding"),
        ("mcmc", "sliding"),
        ("mcmc", "expanding"),
    ),
)
def test_batch_inference_combinations_are_valid(inference_method, window_mode):
    normalized = inference.normalize_rolling_inference_configuration(
        "batch",
        inference_method,
        window_mode,
    )

    assert normalized == ("batch", inference_method, window_mode)


@pytest.mark.parametrize(
    ("selection", "inference_method", "window_mode"),
    (
        ("vi", "vi", "sliding"),
        ("mcmc", "mcmc", "sliding"),
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


def test_ctrv_rolling_observation_count_defaults_to_source_owned_value():
    config = ctrv_config.RollingExperimentConfig(
        prediction_count=3,
        position_noise_std_m=5.0,
        position_noise_seed=2026,
        stride=None,
        inference_method="vi",
        inference_seed=42,
    )

    assert inference.DEFAULT_CTRV_ROLLING_OBSERVATION_COUNT == 10
    assert config.observation_count == 10


def test_ctrv_rolling_configuration_uses_processed_trajectory_contract():
    config = ctrv_config.RollingExperimentConfig(
        prediction_count=3,
        position_noise_std_m=0.0,
        position_noise_seed=2026,
        stride=None,
        inference_method="smc",
        inference_seed=42,
    )

    assert not hasattr(config, "run_id")
    assert not hasattr(config, "trajectory_format")


def test_default_vi_config_uses_stable_fullrank_optimization():
    config = inference.create_default_vi_config()

    assert config["algorithm"] == "fullrank"
    assert config["iter"] == 10_000
    assert config["grad_samples"] == 20
    assert config["eta"] == 0.5
    assert config["adapt_iter"] == 250
    assert config["tol_rel_obj"] == 0.01
    assert config["require_converged"] is False
    assert inference.DEFAULT_FULLRANK_GRAD_SAMPLES == 20


@pytest.mark.parametrize("method", ("smc", "sequential_vi"))
def test_online_inference_is_valid_without_a_window_mode(method):
    normalized = inference.normalize_rolling_inference_configuration(
        "online",
        method,
        None,
    )

    assert normalized == ("online", method, None)


def test_ctrv_online_smc_is_valid_without_a_window_mode():
    config = _ctrv_rolling_config(inference_method="smc")

    normalized = inference.normalize_ctrv_rolling_inference_method(
        config.inference_method
    )

    assert config.inference_method == "smc"
    assert not hasattr(config, "inference_mode")
    assert not hasattr(config, "window_mode")
    assert normalized == ("online", "smc", None)


def test_supported_inference_methods_exclude_removed_rbpf():
    assert inference.ONLINE_INFERENCE_METHODS == ("smc", "sequential_vi")
    assert inference.CTRV_ROLLING_INFERENCE_METHODS == (
        "vi",
        "mcmc",
        "smc",
        "sequential_vi",
    )
    with pytest.raises(ValueError, match="inference_method must be one of"):
        inference.normalize_inference_method("rbpf")
    assert not hasattr(inference, "create_default_ctrv_rbpf_config")


@pytest.mark.parametrize(
    ("inference_mode", "inference_method", "window_mode", "message"),
    (
        ("batch", "smc", "sliding", "Batch inference requires"),
        ("online", "vi", None, "Online inference requires"),
        ("online", "mcmc", None, "Online inference requires"),
        ("batch", "vi", "sequential", "Batch inference requires window_mode"),
        ("online", "smc", "sliding", "does not use window_mode"),
    ),
)
def test_invalid_inference_combinations_fail_early(
    inference_mode,
    inference_method,
    window_mode,
    message,
):
    with pytest.raises(ValueError, match=message):
        inference.normalize_rolling_inference_configuration(
            inference_mode,
            inference_method,
            window_mode,
        )


@pytest.mark.parametrize(
    "inference_method",
    (
        "vi_sliding",
        "vi_expanding",
        "mcmc_sliding",
        "mcmc_expanding",
        "sliding_vi",
        "expanding_vi",
        "unsupported",
    ),
)
def test_invalid_ctrv_rolling_inference_selection_fails_early(inference_method):
    with pytest.raises(ValueError, match="rolling inference_method must be one of"):
        _ctrv_rolling_config(inference_method=inference_method)


def test_cmdstan_configuration_rejects_online_smc():
    with pytest.raises(ValueError, match="only used in batch mode"):
        inference.select_inference_config(
            "online",
            "smc",
            vi_algorithm="meanfield",
            require_converged=False,
            vi_config=inference.create_default_vi_config(),
            mcmc_config=inference.create_default_mcmc_config(),
            fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
        )


def test_variational_runner_uses_default_initials_and_forwards_options():
    class RecordingModel:
        def variational(self, **arguments):
            self.arguments = arguments
            return "variational-fit"

    model = RecordingModel()
    default_initials = {"speed_at_origin": 4.0}

    fit = cmdstan_inference.run_variational_inference(
        model,
        {"N": 3},
        algorithm="meanfield",
        iter=200,
        grad_samples=2,
        elbo_samples=10,
        eta=1.0,
        adapt_iter=20,
        tol_rel_obj=0.01,
        eval_elbo=10,
        draws=50,
        seed=42,
        inits=None,
        default_inits_factory=lambda: default_initials,
        require_converged=False,
        show_console=False,
        options={"refresh": 0},
    )

    assert fit == "variational-fit"
    assert model.arguments["data"] == {"N": 3}
    assert model.arguments["inits"] is default_initials
    assert model.arguments["algorithm"] == "meanfield"
    assert model.arguments["draws"] == 50
    assert model.arguments["refresh"] == 0


def test_mcmc_runner_uses_explicit_initials_and_defaults_parallel_chains():
    class RecordingModel:
        def sample(self, **arguments):
            self.arguments = arguments
            return "mcmc-fit"

    model = RecordingModel()
    explicit_initials = [{"speed_at_origin": 4.0}]

    fit = cmdstan_inference.run_mcmc_inference(
        model,
        {"N": 3},
        chains=1,
        parallel_chains=None,
        iter_warmup=100,
        iter_sampling=50,
        adapt_delta=0.9,
        max_treedepth=10,
        seed=42,
        inits=explicit_initials,
        default_inits_factory=lambda: pytest.fail(
            "default initials must not replace explicit initials"
        ),
        show_console=False,
        options={"refresh": 0},
    )

    assert fit == "mcmc-fit"
    assert model.arguments["data"] == {"N": 3}
    assert model.arguments["inits"] is explicit_initials
    assert model.arguments["chains"] == 1
    assert model.arguments["parallel_chains"] == 1
    assert model.arguments["refresh"] == 0


def test_default_ctrv_smc_factory_returns_an_independent_config():
    first_config = inference.create_default_ctrv_smc_config()
    second_config = inference.create_default_ctrv_smc_config()

    assert isinstance(first_config, smc_model.SequentialMonteCarloCTRVConfig)
    assert first_config == second_config
    assert first_config is not second_config


def test_sequential_vi_is_registered_for_online_inference():
    assert "sequential_vi" in inference.CTRV_ONLINE_INFERENCE_METHODS
    assert "sequential_vi" in inference.ONLINE_INFERENCE_METHODS
    assert inference.normalize_inference_method(
        "sequential_vi",
        online_inference_methods=inference.CTRV_ONLINE_INFERENCE_METHODS,
    ) == ("online", "sequential_vi")
    assert inference.normalize_ctrv_rolling_inference_method("sequential_vi") == (
        "online",
        "sequential_vi",
        None,
    )


@pytest.mark.parametrize("selection", ["seq_vi", " SEQ_VI "])
def test_short_sequential_vi_name_selects_the_existing_online_method(selection):
    assert inference.normalize_inference_method(selection) == (
        "online",
        "sequential_vi",
    )
    assert inference.normalize_inference_configuration("online", selection) == (
        "online",
        "sequential_vi",
    )
    assert inference.normalize_ctrv_rolling_inference_method(selection) == (
        "online",
        "sequential_vi",
        None,
    )
    assert _ctrv_rolling_config(inference_method=selection).inference_method == (
        "sequential_vi"
    )
    with pytest.raises(ValueError, match="Batch inference requires"):
        inference.normalize_inference_configuration("batch", selection)
    with pytest.raises(ValueError, match="inference_method must be one of"):
        inference.normalize_inference_method(
            selection, online_inference_methods=("smc",)
        )


def test_default_sequential_vi_factory_returns_independent_configs():
    first_config = inference.create_default_sequential_vi_config()
    second_config = inference.create_default_sequential_vi_config()

    assert isinstance(first_config, SequentialVIConfig)
    assert first_config == second_config
    assert first_config is not second_config


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("vi", ("batch", "vi")),
        ("mcmc", ("batch", "mcmc")),
        ("smc", ("online", "smc")),
    ],
)
def test_existing_ctrv_method_normalization_is_unchanged(method, expected):
    assert (
        inference.normalize_inference_method(
            method,
            online_inference_methods=inference.CTRV_ONLINE_INFERENCE_METHODS,
        )
        == expected
    )


@pytest.mark.parametrize(
    "inference_method",
    (
        "vi",
        "mcmc",
        "smc",
        "sequential_vi",
    ),
)
def test_rolling_ctrv_accepts_only_one_inference_selection(inference_method):
    config = ctrv_config.RollingExperimentConfig(
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


@pytest.mark.parametrize(
    "inference_method",
    (
        "vi",
        "mcmc",
        "smc",
        "sequential_vi",
    ),
)
def test_ctrv_rolling_cli_accepts_one_inference_selection(inference_method):
    experiment = _ctrv_rolling_config(inference_method="smc")

    options = validation_cli.parse_bayesian_ctrv_evaluation_arguments(
        description=None,
        experiment=experiment,
        priors=ctrv_model.BayesianCTRVPriors(),
        max_windows=None,
        argv=["--inference-method", inference_method],
    )

    assert not hasattr(options, "inference_mode")
    assert not hasattr(options, "window_mode")
    assert options.inference_method == inference_method
    assert options.vi_algorithm == inference.create_default_vi_config()["algorithm"]


def test_ctrv_rolling_cli_observations_override_source_owned_default():
    experiment = ctrv_config.RollingExperimentConfig(
        prediction_count=3,
        position_noise_std_m=5.0,
        position_noise_seed=2026,
        stride=None,
        inference_method="vi",
        inference_seed=42,
    )

    options = validation_cli.parse_bayesian_ctrv_evaluation_arguments(
        description=None,
        experiment=experiment,
        priors=ctrv_model.BayesianCTRVPriors(),
        max_windows=None,
        argv=["--observations", "12"],
    )

    assert experiment.observation_count == 10
    assert options.observation_count == 12


def test_ctrv_rolling_cli_accepts_turn_rate_prior_threshold_in_degrees_per_second():
    experiment = _ctrv_rolling_config(inference_method="smc")

    options = validation_cli.parse_bayesian_ctrv_evaluation_arguments(
        description=None,
        experiment=experiment,
        priors=ctrv_model.BayesianCTRVPriors(),
        vi_config=inference.create_default_vi_config(),
        max_windows=None,
        argv=["--turn-rate-prior-abs-rate-deg-s", "6.5"],
    )

    assert options.turn_rate_prior_abs_rate_deg_s == pytest.approx(6.5)


def test_ctrv_rolling_cli_rejects_removed_window_mode_option():
    experiment = _ctrv_rolling_config(inference_method="smc")

    with pytest.raises(SystemExit):
        validation_cli.parse_bayesian_ctrv_evaluation_arguments(
            description=None,
            experiment=experiment,
            priors=ctrv_model.BayesianCTRVPriors(),
            vi_config=inference.create_default_vi_config(),
            max_windows=None,
            argv=["--window-mode", "sliding"],
        )
