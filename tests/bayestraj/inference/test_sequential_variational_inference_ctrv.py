"""Tests for sequential variational inference of the Bayesian CTRV model."""

import os
from pathlib import Path

import numpy as np
import pytest

from bayestraj.inference import cmdstan
from bayestraj.inference import ctrv_sequential_vi as sequential_vi
from bayestraj.models.bayesian_ctrv import BayesianCTRVPriors

RUN_CMDSTAN_INTEGRATION = os.environ.get("RUN_CMDSTAN_INTEGRATION") == "1"


class FakeFit:
    def __init__(self, variables: dict[str, np.ndarray]) -> None:
        self.variables = variables

    def stan_variable(self, name: str) -> np.ndarray:
        if name not in self.variables:
            raise ValueError(f"Unknown variable: {name}")
        return self.variables[name]

    converged = True


class FakeVariationalFit(FakeFit):
    variational_sample = object()

    def stan_variable(self, name: str, *, mean: bool = True) -> np.ndarray:
        if mean:
            return np.asarray(super().stan_variable(name)).mean(axis=0)
        return super().stan_variable(name)


def _origin_variables() -> dict[str, np.ndarray]:
    return {
        "x_at_origin": np.array([1.0, 1.1, 0.9]),
        "y_at_origin": np.array([2.0, 2.1, 1.9]),
        "speed_at_origin": np.array([0.0, 2.0, 2.2]),
        "heading_at_origin": np.array([np.pi - 0.02, -np.pi + 0.01, np.pi - 0.01]),
        "turn_rate_at_origin": np.array([0.01, 0.02, 0.015]),
        "sigma_position_observation": np.array([4.0, 4.2, 3.8]),
        "sigma_speed_process": np.array([0.5, 0.6, 0.4]),
        "sigma_turn_rate_process": np.array([0.02, 0.03, 0.025]),
    }


def _carry() -> sequential_vi.GaussianCarry:
    return sequential_vi.GaussianCarry(
        mean=np.arange(8, dtype=float) / 10.0,
        cholesky=np.eye(8),
        heading_reference=0.25,
        draw_count=100,
    )


def test_gaussian_carry_rejects_invalid_covariance() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        sequential_vi.GaussianCarry(
            mean=np.zeros(8),
            cholesky=np.zeros((8, 8)),
            heading_reference=0.0,
            draw_count=100,
        )


def test_gaussian_carry_copies_and_freezes_arrays() -> None:
    mean = np.zeros(8)
    carry = sequential_vi.GaussianCarry(
        mean=mean,
        cholesky=np.eye(8),
        heading_reference=0.0,
        draw_count=100,
    )

    mean[0] = 99.0

    assert carry.mean[0] == 0.0
    assert not carry.mean.flags.writeable


def test_build_gaussian_carry_uses_local_heading_and_positive_transforms() -> None:
    fit = FakeFit(_origin_variables())

    carry = sequential_vi.build_gaussian_carry(fit)
    transformed, heading_reference = sequential_vi.transformed_draws(fit)

    assert abs(abs(carry.heading_reference) - np.pi) < 0.05
    assert heading_reference == pytest.approx(carry.heading_reference)
    assert np.std(transformed[:, 3]) < 0.05


def test_build_gaussian_carry_uses_variational_draws_instead_of_mean() -> None:
    fit = FakeVariationalFit(_origin_variables())

    carry = sequential_vi.build_gaussian_carry(fit)

    assert carry.draw_count == 3
    assert carry.mean.shape == (sequential_vi.CARRY_SIZE,)
    assert np.all(np.isfinite(carry.mean))
    assert np.all(np.diag(carry.cholesky) > 0.0)


def test_sample_gaussian_carry_keeps_stationary_speed_non_negative() -> None:
    carry = sequential_vi.build_gaussian_carry(FakeFit(_origin_variables()))

    draws = sequential_vi.sample_gaussian_carry(
        carry,
        draw_count=20,
        generator=np.random.default_rng(42),
    )

    assert np.all(np.isfinite(draws["speed_at_origin"]))
    assert np.all(draws["speed_at_origin"] >= 0.0)
    assert np.all(draws["sigma_position_observation"] > 0.0)
    assert np.all(np.abs(draws["heading_at_origin"]) <= np.pi)


def test_transformed_draws_rejects_non_positive_sigma() -> None:
    variables = _origin_variables()
    variables["sigma_speed_process"] = np.array([0.5, 0.0, 0.4])

    with pytest.raises(ValueError, match="sigma_speed_process.*strictly positive"):
        sequential_vi.transformed_draws(FakeFit(variables))


def test_transformed_draws_rejects_mismatched_draw_counts() -> None:
    variables = _origin_variables()
    variables["turn_rate_at_origin"] = np.array([0.01, 0.02])

    with pytest.raises(ValueError, match="matching draw counts"):
        sequential_vi.transformed_draws(FakeFit(variables))


def test_transformed_draws_rejects_fewer_than_two_draws() -> None:
    variables = {name: values[:1] for name, values in _origin_variables().items()}

    with pytest.raises(ValueError, match="at least two"):
        sequential_vi.transformed_draws(FakeFit(variables))


def test_batch_and_sequential_origin_positions_produce_same_carry() -> None:
    sequential_variables = _origin_variables()
    batch_variables = {
        name: values.copy() for name, values in sequential_variables.items()
    }
    batch_variables.pop("x_at_origin")
    batch_variables.pop("y_at_origin")
    batch_variables["x_state"] = np.column_stack(
        (np.zeros(3), sequential_variables["x_at_origin"])
    )
    batch_variables["y_state"] = np.column_stack(
        (np.zeros(3), sequential_variables["y_at_origin"])
    )

    sequential_carry = sequential_vi.build_gaussian_carry(FakeFit(sequential_variables))
    batch_carry = sequential_vi.build_gaussian_carry(FakeFit(batch_variables))

    np.testing.assert_allclose(batch_carry.mean, sequential_carry.mean)
    np.testing.assert_allclose(batch_carry.cholesky, sequential_carry.cholesky)


def test_summarize_draws_treats_heading_circularly() -> None:
    summary = sequential_vi.summarize_draws(
        np.array([np.pi - 0.02, -np.pi + 0.01, np.pi - 0.01]),
        circular=True,
    )

    assert abs(abs(summary.mean) - np.pi) < 0.05
    assert summary.standard_deviation < 0.05


def test_build_gaussian_carry_rejects_nan_draws() -> None:
    variables = _origin_variables()
    variables["x_at_origin"][1] = np.nan

    with pytest.raises(ValueError, match="x_at_origin.*finite"):
        sequential_vi.build_gaussian_carry(FakeFit(variables))


def test_regularized_cholesky_reports_unrecoverable_covariance() -> None:
    covariance = np.eye(8)
    covariance[0, 0] = -100.0

    with pytest.raises(ValueError, match="could not be regularized"):
        sequential_vi._regularized_cholesky(
            covariance,
            covariance_jitter=1e-9,
            regularization_attempts=1,
        )


def test_sequential_vi_config_uses_fullrank_cmdstan_defaults() -> None:
    config = sequential_vi.SequentialVIConfig()

    assert config.n_bootstrap == 10
    assert config.algorithm == "fullrank"
    assert config.iter == cmdstan.DEFAULT_VI_ITER
    assert config.grad_samples == cmdstan.DEFAULT_VI_GRAD_SAMPLES
    assert config.elbo_samples == cmdstan.DEFAULT_VI_ELBO_SAMPLES
    assert config.draws == cmdstan.DEFAULT_VI_DRAWS


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_bootstrap": 2}, "n_bootstrap"),
        ({"algorithm": "meanfield"}, "algorithm"),
        ({"draws": 1}, "draws"),
        ({"covariance_jitter": 0.0}, "covariance_jitter"),
        ({"regularization_attempts": 0}, "regularization_attempts"),
        ({"require_converged": 1}, "require_converged"),
        ({"show_console": 0}, "show_console"),
    ],
)
def test_sequential_vi_config_rejects_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sequential_vi.SequentialVIConfig(**overrides)


def test_sequential_update_data_contains_one_observation_and_two_intervals() -> None:
    data = sequential_vi.build_sequential_update_data(
        _carry(),
        process_interval_seconds=7.0,
        observation_interval_seconds=11.0,
        x_observed=12.5,
        y_observed=-3.0,
    )

    assert data["x_observed"] == 12.5
    assert data["y_observed"] == -3.0
    assert data["process_interval_seconds"] == 7.0
    assert data["observation_interval_seconds"] == 11.0
    assert data["process_reference_interval_seconds"] == 1.0
    assert "x_observed_history" not in data
    assert "y_observed_history" not in data
    assert not any(
        isinstance(data[key], np.ndarray) and data[key].ndim == 1
        for key in ("x_observed", "y_observed")
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"process_interval_seconds": 0.0}, "process_interval_seconds"),
        ({"observation_interval_seconds": -1.0}, "observation_interval_seconds"),
        ({"x_observed": [1.0]}, "x_observed"),
        ({"y_observed": np.inf}, "y_observed"),
    ],
)
def test_sequential_update_data_rejects_invalid_scalars(
    overrides: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "process_interval_seconds": 7.0,
        "observation_interval_seconds": 11.0,
        "x_observed": 12.5,
        "y_observed": -3.0,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        sequential_vi.build_sequential_update_data(_carry(), **arguments)


def test_sequential_stan_model_has_scalar_one_observation_contract() -> None:
    source = Path("src/stan/models/sequential_bayesian_ctrv.stan").read_text(
        encoding="utf-8"
    )

    assert "vector[8] carry_mean" in source
    assert "matrix[8, 8] carry_cholesky" in source
    assert "real x_observed;" in source
    assert "real y_observed;" in source
    assert "process_interval_seconds" in source
    assert "observation_interval_seconds" in source
    assert "log_sum_exp" in source
    assert source.count("x_observed ~ normal") == 1
    assert source.count("y_observed ~ normal") == 1
    for variable_name in (
        "x_at_origin",
        "y_at_origin",
        "speed_at_origin",
        "heading_at_origin",
        "turn_rate_at_origin",
        "sigma_position_observation",
        "sigma_speed_process",
        "sigma_turn_rate_process",
    ):
        assert variable_name in source
    for forbidden_name in (
        "N_history",
        "time_observed",
        "vector[1] x_observed",
        "vector[1] y_observed",
    ):
        assert forbidden_name not in source


def test_fit_sequential_update_runs_fullrank_vi_with_carry_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected_fit = object()

    def fake_run_variational_inference(model, data, **options):
        captured["model"] = model
        captured["data"] = data
        captured.update(options)
        return expected_fit

    monkeypatch.setattr(
        sequential_vi.cmdstan,
        "run_variational_inference",
        fake_run_variational_inference,
    )
    model = object()
    config = sequential_vi.SequentialVIConfig(
        iter=200,
        grad_samples=3,
        elbo_samples=5,
        draws=20,
    )

    fit = sequential_vi.fit_sequential_update(
        _carry(),
        process_interval_seconds=7.0,
        observation_interval_seconds=11.0,
        x_observed=12.5,
        y_observed=-3.0,
        config=config,
        seed=123,
        model=model,
    )

    assert fit is expected_fit
    assert captured["model"] is model
    assert captured["algorithm"] == "fullrank"
    assert captured["seed"] == 123
    assert captured["draws"] == 20
    assert captured["default_inits_factory"] is None
    expected_data = sequential_vi.build_sequential_update_data(
        _carry(),
        process_interval_seconds=7.0,
        observation_interval_seconds=11.0,
        x_observed=12.5,
        y_observed=-3.0,
    )
    assert captured["data"].keys() == expected_data.keys()
    np.testing.assert_allclose(
        captured["data"]["carry_mean"], expected_data["carry_mean"]
    )
    np.testing.assert_allclose(
        captured["data"]["carry_cholesky"], expected_data["carry_cholesky"]
    )
    for key in expected_data.keys() - {"carry_mean", "carry_cholesky"}:
        assert captured["data"][key] == pytest.approx(expected_data[key])
    inits = captured["inits"]
    np.testing.assert_array_equal(inits["previous_standardized"], np.zeros(8))
    assert inits["speed_for_interval"] == pytest.approx(np.exp(_carry().mean[2]))
    assert inits["turn_rate_for_interval"] == pytest.approx(_carry().mean[4])


def test_compile_sequential_model_reports_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.stan"

    with pytest.raises(FileNotFoundError, match="Stan model not found"):
        sequential_vi.compile_sequential_bayesian_ctrv_model(missing)


def _online_observations(count: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_seconds = np.arange(count, dtype=float) * 2.0
    return time_seconds, 3.0 * time_seconds, -1.5 * time_seconds


def _initialized_online_vi(
    *,
    draws: int = 20,
) -> sequential_vi.SequentialCTRVVI:
    def batch_fitter(window, **options):
        return FakeFit(_origin_variables())

    time_seconds, x_observed, y_observed = _online_observations(10)
    return sequential_vi.SequentialCTRVVI.initialize(
        time_seconds,
        x_observed,
        y_observed,
        priors=BayesianCTRVPriors(),
        config=sequential_vi.SequentialVIConfig(draws=draws),
        seed=42,
        batch_fitter=batch_fitter,
    )


def test_sequential_vi_bootstraps_batch_once_then_updates_unseen_positions() -> None:
    batch_calls: list[dict[str, object]] = []
    update_calls: list[dict[str, object]] = []

    def batch_fitter(window, **options):
        batch_calls.append({"window": window, **options})
        return FakeFit(_origin_variables())

    def update_fitter(carry, **options):
        update_calls.append({"carry": carry, **options})
        return FakeFit(_origin_variables())

    time_seconds = np.array(
        [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 21.0, 28.0, 39.0, 52.0]
    )
    x_observed = 3.0 * time_seconds
    y_observed = -1.5 * time_seconds

    online_vi = sequential_vi.SequentialCTRVVI.initialize(
        time_seconds,
        x_observed,
        y_observed,
        priors=BayesianCTRVPriors(),
        config=sequential_vi.SequentialVIConfig(draws=20),
        seed=42,
        batch_fitter=batch_fitter,
        update_fitter=update_fitter,
    )

    assert len(batch_calls) == 1
    batch_call = batch_calls[0]
    assert batch_call["inference_method"] == "vi"
    assert batch_call["algorithm"] == "fullrank"
    assert batch_call["position_observations"].x_meters.shape == (10,)
    assert batch_call["position_observations"].y_meters.shape == (10,)
    assert batch_call["window"].observation_count == 10
    assert batch_call["window"].prediction_count == 1
    assert np.all(np.isnan(batch_call["window"].gps_speed_mps))
    assert len(update_calls) == 2
    assert [call["x_observed"] for call in update_calls] == list(x_observed[10:])
    assert [call["y_observed"] for call in update_calls] == list(y_observed[10:])
    assert [call["seed"] for call in update_calls] == [43, 44]
    assert [call["process_interval_seconds"] for call in update_calls] == [7.0, 11.0]
    assert [call["observation_interval_seconds"] for call in update_calls] == [
        11.0,
        13.0,
    ]
    assert update_calls[1]["carry"] is not update_calls[0]["carry"]
    assert online_vi.processed_observation_count == 12
    assert len(batch_calls) == 1


def test_sequential_vi_rejects_repeated_timestamp_before_fitting() -> None:
    update_call_count = 0

    def batch_fitter(window, **options):
        return FakeFit(_origin_variables())

    def update_fitter(carry, **options):
        nonlocal update_call_count
        update_call_count += 1
        return FakeFit(_origin_variables())

    time_seconds, x_observed, y_observed = _online_observations(10)
    online_vi = sequential_vi.SequentialCTRVVI.initialize(
        time_seconds,
        x_observed,
        y_observed,
        priors=BayesianCTRVPriors(),
        config=sequential_vi.SequentialVIConfig(draws=20),
        seed=42,
        batch_fitter=batch_fitter,
        update_fitter=update_fitter,
    )

    with pytest.raises(ValueError, match="follow the previous observation"):
        online_vi.update(time_seconds[-1], x_observed[-1], y_observed[-1])

    assert update_call_count == 0


def test_sequential_vi_update_rolls_back_all_state_on_failure() -> None:
    update_calls: list[dict[str, object]] = []

    def batch_fitter(window, **options):
        return FakeFit(_origin_variables())

    def update_fitter(carry, **options):
        update_calls.append({"carry": carry, **options})
        if len(update_calls) == 2:
            raise RuntimeError("ADVI failed")
        return FakeFit(_origin_variables())

    time_seconds, x_observed, y_observed = _online_observations(10)
    online_vi = sequential_vi.SequentialCTRVVI.initialize(
        time_seconds,
        x_observed,
        y_observed,
        priors=BayesianCTRVPriors(),
        config=sequential_vi.SequentialVIConfig(draws=20),
        seed=42,
        batch_fitter=batch_fitter,
        update_fitter=update_fitter,
    )
    online_vi.update(20.0, 60.0, -30.0)
    carry_before = online_vi.carry
    mean_before = carry_before.mean.tobytes()
    cholesky_before = carry_before.cholesky.tobytes()
    time_before = online_vi.last_observation_time_seconds
    interval_before = online_vi.last_process_interval_seconds
    count_before = online_vi.processed_observation_count

    with pytest.raises(RuntimeError, match="ADVI failed"):
        online_vi.update(22.0, 66.0, -33.0)

    assert online_vi.carry is carry_before
    assert online_vi.carry.mean.tobytes() == mean_before
    assert online_vi.carry.cholesky.tobytes() == cholesky_before
    assert online_vi.last_observation_time_seconds == time_before
    assert online_vi.last_process_interval_seconds == interval_before
    assert online_vi.processed_observation_count == count_before


def test_sequential_vi_fit_records_terminal_failure_and_stops() -> None:
    update_call_count = 0

    def batch_fitter(window, **options):
        return FakeFit(_origin_variables())

    def update_fitter(carry, **options):
        nonlocal update_call_count
        update_call_count += 1
        if update_call_count == 2:
            raise RuntimeError("ADVI failed")
        return FakeFit(_origin_variables())

    time_seconds, x_observed, y_observed = _online_observations(10)
    online_vi = sequential_vi.SequentialCTRVVI.initialize(
        time_seconds,
        x_observed,
        y_observed,
        priors=BayesianCTRVPriors(),
        config=sequential_vi.SequentialVIConfig(draws=20),
        seed=42,
        batch_fitter=batch_fitter,
        update_fitter=update_fitter,
    )

    result = online_vi.fit(
        np.array([20.0, 22.0, 24.0]),
        np.array([60.0, 66.0, 72.0]),
        np.array([-30.0, -33.0, -36.0]),
    )

    assert update_call_count == 2
    assert len(result.updates) == 2
    assert result.updates[0].error_message is None
    assert result.updates[1].error_message == "ADVI failed"
    assert result.completed is False
    assert result.failure_index == 11
    assert result.failure_message == "ADVI failed"
    assert result.processed_observation_count == 11


def test_sample_current_posterior_matches_shared_contract_without_mutation() -> None:
    online_vi = _initialized_online_vi(draws=20)
    carry_before = online_vi.carry
    count_before = online_vi.processed_observation_count

    first = online_vi.sample_current_posterior(seed=123)
    second = online_vi.sample_current_posterior(seed=123)

    for name in (
        "speed_at_origin",
        "heading_at_origin",
        "turn_rate_at_origin",
        "sigma_position_observation",
        "sigma_speed_process",
        "sigma_turn_rate_process",
    ):
        first_values = first.stan_variable(name)
        second_values = second.stan_variable(name)
        assert first_values.shape == (20,)
        assert np.all(np.isfinite(first_values))
        np.testing.assert_array_equal(first_values, second_values)
    assert online_vi.carry is carry_before
    assert online_vi.processed_observation_count == count_before


def test_sequential_vi_forecast_matches_shared_prediction_contract() -> None:
    online_vi = _initialized_online_vi(draws=20)
    carry_before = online_vi.carry
    count_before = online_vi.processed_observation_count
    future_times = np.array([20.0, 22.0, 25.0])

    first = online_vi.forecast(future_times, seed=123)
    second = online_vi.forecast(future_times, seed=123)

    for name in (
        "x_prediction",
        "y_prediction",
        "x_observation_prediction",
        "y_observation_prediction",
    ):
        first_values = first.stan_variable(name)
        assert first_values.shape == (20, 3)
        assert np.all(np.isfinite(first_values))
        np.testing.assert_array_equal(first_values, second.stan_variable(name))
    assert not np.array_equal(
        first.stan_variable("x_prediction"),
        first.stan_variable("x_observation_prediction"),
    )
    assert not np.array_equal(
        first.stan_variable("y_prediction"),
        first.stan_variable("y_observation_prediction"),
    )
    assert np.all(first.stan_variable("speed_at_origin") >= 0.0)
    assert online_vi.carry is carry_before
    assert online_vi.processed_observation_count == count_before


@pytest.mark.parametrize(
    "future_times",
    [
        np.array([18.0, 20.0]),
        np.array([20.0, 20.0]),
        np.array([22.0, 20.0]),
    ],
)
def test_sequential_vi_forecast_rejects_non_future_increasing_times(
    future_times: np.ndarray,
) -> None:
    online_vi = _initialized_online_vi(draws=20)

    with pytest.raises(ValueError, match="future_time_seconds"):
        online_vi.forecast(future_times, seed=123)


@pytest.mark.skipif(
    not RUN_CMDSTAN_INTEGRATION,
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to run CmdStan Sequential VI.",
)
def test_sequential_vi_cmdstan_integration() -> None:
    time_seconds = np.arange(7, dtype=float) * 2.0
    x_observed = 4.0 * time_seconds
    y_observed = np.zeros_like(time_seconds)
    config = sequential_vi.SequentialVIConfig(
        n_bootstrap=5,
        iter=3_000,
        grad_samples=5,
        elbo_samples=20,
        adapt_iter=100,
        eval_elbo=100,
        draws=50,
        require_converged=False,
    )

    online_vi = sequential_vi.SequentialCTRVVI.initialize(
        time_seconds,
        x_observed,
        y_observed,
        priors=BayesianCTRVPriors(),
        config=config,
        seed=42,
    )

    assert online_vi.processed_observation_count == 7
    assert len(online_vi.updates) == 2
    for update in online_vi.updates:
        assert update.error_message is None
        assert update.filtered_state.speed.median >= 0.0
        assert update.filtered_parameters.sigma_position_observation.median > 0.0
        assert update.filtered_parameters.sigma_speed_process.median > 0.0
        assert update.filtered_parameters.sigma_turn_rate_process.median > 0.0

    forecast = online_vi.forecast(np.array([14.0, 16.0]), seed=123)
    for name in (
        "x_prediction",
        "y_prediction",
        "x_observation_prediction",
        "y_observation_prediction",
    ):
        values = forecast.stan_variable(name)
        assert values.shape == (config.draws, 2)
        assert np.all(np.isfinite(values))
