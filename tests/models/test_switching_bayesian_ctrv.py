"""Tests for the parallel Switching Bayesian CTRV implementation."""

from __future__ import annotations

import ast
import hashlib
import os
import re
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ship_trajectory_prediction.evaluation.reporting import (
    posterior_variable_samples,
)
from ship_trajectory_prediction.models import switching_bayesian_ctrv as model_module
from ship_trajectory_prediction.models.bayesian_ctrv import (
    PositionObservations,
    VIRunResult,
    simulate_position_observations,
)
from ship_trajectory_prediction.models.ctrv import CTRVState, ctrv_step
from ship_trajectory_prediction.models.switching_bayesian_ctrv import (
    MODE_COUNT,
    MODE_CRUISE,
    MODE_MANEUVER,
    MODE_NAMES,
    MODE_STOP,
    STAN_FILE,
    SwitchingCTRVConfig,
    build_stan_data,
    compare_switching_vi_runs,
    expected_mode_transition,
    fit_switching_bayesian_ctrv_model,
    forward_log_probabilities,
    summarize_mode_probabilities,
    summarize_switching_predictions,
    summarize_transition_probabilities,
)
from ship_trajectory_prediction.simulation.synthetic_switching_ctrv import (
    MODE_INITIAL,
    SyntheticSwitchingCTRVNoise,
    simulate_synthetic_switching_ctrv_data,
)
from ship_trajectory_prediction.trajectory import prepare_trajectory_window

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_BASELINE_SHA256 = {
    "src/ship_trajectory_prediction/models/bayesian_ctrv.py": (
        "2cacdd55638c3e0a6fe5c0aa115bfb05208c322a48043a4165456036b71321d4"
    ),
    "stan/models/bayesian_ctrv.stan": (
        "7ec2b079da7be6de5e13eb336d4a8454f057ea343d695dc02ad906d8a94adfd5"
    ),
    "experiments/trajectory_prediction/fit_bayesian_ctrv.py": (
        "fc7e6f18dbdfa1b3c1747ac2803151288ce58c570a0dacae0630a0dced448cde"
    ),
    "experiments/trajectory_prediction/evaluate_bayesian_ctrv_rolling.py": (
        "e8b7bda8487fa07e56328e7c873e48747ffab55c6a76dcecf2bedc71e882725c"
    ),
}


def create_synthetic_window():
    """Create a short four-phase window for fast interface tests."""
    data = simulate_synthetic_switching_ctrv_data(
        phase_transition_counts=(2, 2, 2, 3),
        seed=17,
    )
    return prepare_trajectory_window(
        data,
        observation_count=7,
        prediction_count=3,
    )


def test_protected_bayesian_ctrv_baseline_matches_the_original_snapshot():
    """The parallel extension must not alter any protected baseline source."""
    for relative_path, expected_digest in PROTECTED_BASELINE_SHA256.items():
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        normalized_source = source.replace("\r\n", "\n")
        actual_digest = hashlib.sha256(normalized_source.encode()).hexdigest()
        assert actual_digest == expected_digest, relative_path


def test_mode_constants_and_default_config_have_fixed_semantics():
    """The public one-based mode mapping and identifying scales stay explicit."""
    config = SwitchingCTRVConfig()

    assert MODE_COUNT == 3
    assert (MODE_STOP, MODE_CRUISE, MODE_MANEUVER) == (1, 2, 3)
    assert dict(MODE_NAMES) == {1: "stop", 2: "cruise", 3: "maneuver"}
    assert config.initial_mode_probability == pytest.approx([1 / 3] * 3)
    assert np.diag(config.alpha_transition) == pytest.approx([20.0] * 3)
    assert np.asarray(config.alpha_transition)[~np.eye(3, dtype=bool)] == pytest.approx(
        [1.0] * 6
    )
    assert config.position_process_multiplier == (0.5, 1.0, 3.0)
    assert config.speed_process_multiplier == (0.25, 1.0, 4.0)
    assert config.turn_rate_process_multiplier == (0.25, 1.0, 4.0)
    assert config.stop_speed_decay_time == 20.0
    assert config.stop_turn_decay_time == 10.0
    with pytest.raises(FrozenInstanceError):
        config.stop_speed_decay_time = 5.0


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"initial_mode_probability": (0.5, 0.5, 0.0)}, "simplex"),
        ({"initial_mode_probability": (0.4, 0.4, 0.4)}, "simplex"),
        (
            {
                "alpha_transition": (
                    (20.0, 1.0, 1.0),
                    (1.0, 0.0, 1.0),
                    (1.0, 1.0, 20.0),
                )
            },
            "alpha_transition",
        ),
        ({"position_process_multiplier": (1.0, 1.0, 3.0)}, "stop < cruise"),
        ({"speed_process_multiplier": (1.0, 0.5, 4.0)}, "stop < cruise"),
        ({"turn_rate_process_multiplier": (0.25, 4.0, 1.0)}, "maneuver"),
        ({"stop_speed_decay_time": 0.0}, "stop_speed_decay_time"),
        ({"stop_turn_decay_time": np.inf}, "stop_turn_decay_time"),
    ],
)
def test_switching_config_rejects_non_identifiable_or_invalid_values(
    options,
    message,
):
    """Mode hyperparameters should fail before reaching CmdStan."""
    with pytest.raises(ValueError, match=message):
        SwitchingCTRVConfig(**options)


def test_build_stan_data_contains_position_only_inputs_and_three_modes():
    """Switching data should extend the common position-only data schema."""
    window = create_synthetic_window()

    stan_data = build_stan_data(window)

    assert stan_data["K"] == 3
    assert np.asarray(stan_data["initial_mode_probability"]).shape == (3,)
    assert np.asarray(stan_data["alpha_transition"]).shape == (3, 3)
    for name in (
        "position_process_multiplier",
        "speed_process_multiplier",
        "turn_rate_process_multiplier",
    ):
        assert np.asarray(stan_data[name]).shape == (3,)
    assert stan_data["N_observed"] == window.observation_count
    assert stan_data["N_prediction"] == window.prediction_count
    assert stan_data["x_observed"] == pytest.approx(
        window.x_meters[window.observed_slice]
    )
    assert stan_data["y_observed"] == pytest.approx(
        window.y_meters[window.observed_slice]
    )
    forbidden_fragments = (
        "gps_speed",
        "speed_observed",
        "shaft",
        "thruster",
        "rudder",
        "command",
    )
    assert not any(
        fragment in key for key in stan_data for fragment in forbidden_fragments
    )
    assert "x_state_prediction" not in stan_data
    assert "y_state_prediction" not in stan_data


def test_build_stan_data_forwards_custom_switching_hyperparameters():
    """Every configured transition and mode hyperparameter should reach Stan."""
    config = SwitchingCTRVConfig(
        initial_mode_probability=(0.2, 0.6, 0.2),
        alpha_transition=((9.0, 2.0, 1.0), (1.0, 8.0, 2.0), (2.0, 1.0, 7.0)),
        position_process_multiplier=(0.4, 1.2, 2.5),
        speed_process_multiplier=(0.2, 1.1, 5.0),
        turn_rate_process_multiplier=(0.1, 1.0, 6.0),
        stop_speed_decay_time=12.0,
        stop_turn_decay_time=8.0,
    )

    stan_data = build_stan_data(create_synthetic_window(), config=config)

    for name in (
        "initial_mode_probability",
        "alpha_transition",
        "position_process_multiplier",
        "speed_process_multiplier",
        "turn_rate_process_multiplier",
        "stop_speed_decay_time",
        "stop_turn_decay_time",
    ):
        np.testing.assert_allclose(stan_data[name], getattr(config, name))
    with pytest.raises(TypeError, match="SwitchingCTRVConfig"):
        build_stan_data(create_synthetic_window(), config={})


def test_stationary_prefix_uses_the_first_clear_position_displacement_for_heading():
    """Initial jitter must not define heading before later visible motion."""
    window = create_synthetic_window()
    observed = window.observed_slice
    x_observed = np.asarray([0.0, 0.1, -0.1, 0.0, 10.0, 20.0, 30.0])
    y_observed = np.asarray([0.0, -0.1, 0.1, 0.0, 5.0, 10.0, 15.0])
    observations = PositionObservations(
        time_seconds=window.time_seconds[observed],
        x_meters=x_observed,
        y_meters=y_observed,
        additional_noise_std_m=0.0,
        noise_seed=0,
    )

    stan_data = build_stan_data(window, position_observations=observations)

    assert stan_data["heading_initial_prior_mean"] == pytest.approx(
        np.arctan2(5.0, 10.0)
    )
    assert stan_data["turn_rate_initial_prior_mean"] == pytest.approx(0.0)


@pytest.mark.parametrize("external_speed", [np.nan, -10.0, 10_000.0])
def test_gps_speed_has_no_effect_on_switching_stan_data(external_speed):
    """External GPS speed must not affect data, priors, or mode inputs."""
    window = create_synthetic_window()
    changed_speed = np.full_like(window.gps_speed_mps, external_speed)
    changed_window = replace(window, gps_speed_mps=changed_speed)

    reference = build_stan_data(window)
    changed = build_stan_data(changed_window)

    assert_stan_data_equal(reference, changed)


def test_held_out_measurements_do_not_affect_switching_stan_data():
    """Prediction x/y and GPS speed must remain evaluation-only values."""
    window = create_synthetic_window()
    x_changed = window.x_meters.copy()
    y_changed = window.y_meters.copy()
    speed_changed = window.gps_speed_mps.copy()
    x_changed[window.prediction_slice] += 50_000.0
    y_changed[window.prediction_slice] -= 25_000.0
    speed_changed[window.prediction_slice] = np.nan
    changed_window = replace(
        window,
        x_meters=x_changed,
        y_meters=y_changed,
        gps_speed_mps=speed_changed,
    )

    assert_stan_data_equal(
        build_stan_data(window),
        build_stan_data(changed_window),
    )


def test_supplied_observations_own_all_position_derived_switching_inputs():
    """Clean observed coordinates must not leak into noise-augmented fitting."""
    window = create_synthetic_window()
    observations = simulate_position_observations(
        window,
        additional_noise_std_m=2.0,
        seed=2026,
    )
    x_changed = window.x_meters.copy()
    y_changed = window.y_meters.copy()
    x_changed[window.observed_slice] += np.linspace(100.0, 700.0, 7)
    y_changed[window.observed_slice] -= np.linspace(50.0, 350.0, 7)
    changed_window = replace(window, x_meters=x_changed, y_meters=y_changed)

    reference = build_stan_data(
        window,
        position_observations=observations,
    )
    changed = build_stan_data(
        changed_window,
        position_observations=observations,
    )

    assert_stan_data_equal(reference, changed)
    assert reference["x_observed"] == pytest.approx(observations.x_meters)
    assert reference["y_observed"] == pytest.approx(observations.y_meters)


def test_forward_log_probabilities_remain_finite_for_extreme_log_densities():
    """Log-domain recursion should avoid probability underflow."""
    transition_count = 500
    log_density = np.full((transition_count, MODE_COUNT), -1_000_000.0)
    transition_probability = np.full((MODE_COUNT, MODE_COUNT), 1 / MODE_COUNT)
    initial_probability = np.full(MODE_COUNT, 1 / MODE_COUNT)

    forward = forward_log_probabilities(
        log_density,
        transition_probability,
        initial_probability,
    )

    assert forward.shape == (transition_count, MODE_COUNT)
    assert np.all(np.isfinite(forward))
    assert forward[-1] == pytest.approx(
        np.full(MODE_COUNT, -transition_count * 1_000_000.0 - np.log(3.0))
    )


@pytest.mark.parametrize(
    ("transition_probability", "message"),
    [
        (np.ones((2, 2)) / 2, "positive simplex rows"),
        (np.full((3, 3), 0.5), "positive simplex rows"),
        (
            np.asarray([[1.0, 0.0, 0.0], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]]),
            "positive simplex rows",
        ),
    ],
)
def test_forward_log_probabilities_reject_invalid_transition_matrices(
    transition_probability,
    message,
):
    """The reference recursion should accept only interior simplex rows."""
    with pytest.raises(ValueError, match=message):
        forward_log_probabilities(
            np.zeros((4, 3)),
            transition_probability,
            (1 / 3, 1 / 3, 1 / 3),
        )


def test_stop_transition_holds_position_and_decays_motion_states():
    """The stop mean should be constant-position dynamics, not CTRV drift."""
    state = CTRVState(
        x=12.0,
        y=-4.0,
        speed=2.0,
        heading=0.4,
        turn_rate=0.01,
    )
    config = SwitchingCTRVConfig(
        stop_speed_decay_time=20.0,
        stop_turn_decay_time=10.0,
    )

    result = expected_mode_transition(state, 10.0, MODE_STOP, config=config)

    assert result.x == state.x
    assert result.y == state.y
    assert result.speed == pytest.approx(2.0 * np.exp(-0.5))
    assert result.heading == pytest.approx(0.5)
    assert result.turn_rate == pytest.approx(0.01 * np.exp(-1.0))


@pytest.mark.parametrize("mode", [MODE_CRUISE, MODE_MANEUVER])
def test_cruise_and_maneuver_share_the_ctrv_transition_mean(mode):
    """The two moving modes should differ only through process scales."""
    state = CTRVState(
        x=2.0,
        y=3.0,
        speed=4.0,
        heading=0.2,
        turn_rate=-0.012,
    )

    result = expected_mode_transition(state, 5.0, mode)
    expected = ctrv_step(state, 5.0)

    assert result == expected
    config = SwitchingCTRVConfig()
    assert (
        config.position_process_multiplier[MODE_MANEUVER - 1]
        > config.position_process_multiplier[MODE_CRUISE - 1]
    )
    assert (
        config.speed_process_multiplier[MODE_MANEUVER - 1]
        > config.speed_process_multiplier[MODE_CRUISE - 1]
    )
    assert (
        config.turn_rate_process_multiplier[MODE_MANEUVER - 1]
        > config.turn_rate_process_multiplier[MODE_CRUISE - 1]
    )


def test_fit_forwards_only_supported_vi_controls(monkeypatch):
    """The wrapper should call variational inference and expose all VI controls."""
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module,
        "compile_switching_bayesian_ctrv_model",
        lambda: fake_model,
    )

    fit = fit_switching_bayesian_ctrv_model(
        create_synthetic_window(),
        algorithm="fullrank",
        iter=500,
        grad_samples=2,
        elbo_samples=20,
        eta=0.5,
        adapt_iter=25,
        tol_rel_obj=0.02,
        eval_elbo=50,
        draws=80,
        seed=19,
        require_converged=False,
        show_console=True,
        variational_options={"refresh": 25},
    )

    assert fit is fake_model.result
    arguments = fake_model.arguments
    assert arguments["algorithm"] == "fullrank"
    assert arguments["iter"] == 500
    assert arguments["grad_samples"] == 2
    assert arguments["elbo_samples"] == 20
    assert arguments["eta"] == pytest.approx(0.5)
    assert arguments["adapt_iter"] == 25
    assert arguments["tol_rel_obj"] == pytest.approx(0.02)
    assert arguments["eval_elbo"] == 50
    assert arguments["draws"] == 80
    assert arguments["seed"] == 19
    assert arguments["require_converged"] is False
    assert arguments["show_console"] is True
    assert arguments["refresh"] == 25
    assert np.asarray(arguments["inits"]["transition_probability"]).shape == (
        3,
        3,
    )
    assert np.allclose(
        np.sum(arguments["inits"]["transition_probability"], axis=1),
        1.0,
    )
    assert np.all(arguments["inits"]["speed_state"] > 0)
    assert np.all(arguments["inits"]["speed_state"] < 100)
    assert np.all(
        np.abs(arguments["inits"]["turn_rate_state"])
        < arguments["data"]["turn_rate_limit"]
    )


def test_gps_speed_does_not_change_seeded_switching_initials(monkeypatch):
    """Equal positions and times should produce equal initials despite GPS speed."""
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module,
        "compile_switching_bayesian_ctrv_model",
        lambda: fake_model,
    )
    window = create_synthetic_window()
    changed_window = replace(
        window,
        gps_speed_mps=np.linspace(100.0, 1_000.0, len(window.gps_speed_mps)),
    )

    fit_switching_bayesian_ctrv_model(window, seed=71)
    reference_initials = fake_model.calls[-1]["inits"]
    fit_switching_bayesian_ctrv_model(changed_window, seed=71)
    changed_initials = fake_model.calls[-1]["inits"]

    assert reference_initials.keys() == changed_initials.keys()
    for name, values in reference_initials.items():
        assert changed_initials[name] == pytest.approx(values)


@pytest.mark.parametrize("algorithm", ["mcmc", "laplace", ""])
def test_fit_rejects_non_vi_algorithms(algorithm):
    """The switching wrapper should expose no MCMC path."""
    with pytest.raises(ValueError, match="algorithm"):
        fit_switching_bayesian_ctrv_model(
            create_synthetic_window(),
            algorithm=algorithm,
        )


def test_fit_rejects_conflicting_variational_options():
    """Generic options must not override explicit wrapper arguments."""
    with pytest.raises(ValueError, match="must not override"):
        fit_switching_bayesian_ctrv_model(
            create_synthetic_window(),
            variational_options={"draws": 10},
        )


def test_prediction_summary_has_states_observations_and_future_modes():
    """Future summaries should retain continuous and categorical uncertainty."""
    window = create_synthetic_window()
    fit = FakeFit(**create_fake_variables(window))

    summary = summarize_switching_predictions(
        fit,
        window,
        credible_interval=0.8,
        include_speed_gps_reference=True,
    )

    assert len(summary) == window.prediction_count
    for prefix in (
        "x_state",
        "y_state",
        "speed_state",
        "heading_state",
        "turn_rate_state",
        "x_observation",
        "y_observation",
    ):
        assert {
            f"{prefix}_median",
            f"{prefix}_lower",
            f"{prefix}_upper",
        }.issubset(summary.columns)
    probability_columns = [
        f"mode_{mode_name}_probability" for mode_name in MODE_NAMES.values()
    ]
    assert np.sum(summary[probability_columns].to_numpy(), axis=1) == pytest.approx(
        np.ones(window.prediction_count)
    )
    assert set(summary["most_likely_mode"]).issubset(MODE_NAMES)
    assert set(summary["most_likely_mode_name"]).issubset(MODE_NAMES.values())
    assert summary["speed_gps_reference"].to_numpy() == pytest.approx(
        window.gps_speed_mps[window.prediction_slice]
    )


def test_mode_summary_has_transition_times_and_valid_probabilities():
    """Observed mode rows should align to destination-state timestamps."""
    window = create_synthetic_window()
    fit = FakeFit(**create_fake_variables(window))

    summary = summarize_mode_probabilities(fit, window, credible_interval=0.8)

    assert len(summary) == window.observation_count - 1
    assert summary["transition_index"].to_numpy() == pytest.approx(
        np.arange(1, window.observation_count)
    )
    assert summary["t"].to_numpy() == pytest.approx(
        window.time_seconds[1 : window.observation_count]
    )
    assert (
        summary["time"].tolist()
        == window.timestamps[1 : window.observation_count].tolist()
    )
    probability_columns = [
        f"{mode_name}_probability" for mode_name in MODE_NAMES.values()
    ]
    assert np.sum(summary[probability_columns].to_numpy(), axis=1) == pytest.approx(
        np.ones(window.observation_count - 1)
    )
    assert set(summary["most_likely_mode_name"]).issubset(MODE_NAMES.values())


def test_transition_summary_has_all_nine_simplex_entries():
    """Transition reporting should retain source and destination identities."""
    window = create_synthetic_window()
    summary = summarize_transition_probabilities(
        FakeFit(**create_fake_variables(window)),
        credible_interval=0.8,
    )

    assert len(summary) == MODE_COUNT**2
    assert set(summary["from_mode"]) == set(MODE_NAMES)
    assert set(summary["to_mode"]) == set(MODE_NAMES)
    assert {
        "probability_mean",
        "probability_median",
        "probability_lower",
        "probability_upper",
    }.issubset(summary.columns)
    assert summary.groupby("from_mode")["probability_mean"].sum().to_numpy() == (
        pytest.approx(np.ones(MODE_COUNT))
    )


def test_mode_summary_rejects_malformed_probability_draws():
    """NaN values and non-simplex rows should not reach saved reports."""
    window = create_synthetic_window()
    variables = create_fake_variables(window)
    variables["mode_probability"][0, 0] = [0.8, 0.8, -0.6]

    with pytest.raises(ValueError, match="probabilities between zero and one"):
        summarize_mode_probabilities(FakeFit(**variables), window)


def test_compare_switching_vi_runs_reports_prediction_and_mode_stability(tmp_path):
    """Cross-seed rows should include ELBO, accuracy, transition, and mode data."""
    window = create_synthetic_window()
    stdout_path = tmp_path / "variational-stdout.txt"
    stdout_path.write_text(
        "  iter             ELBO   delta_ELBO_mean   delta_ELBO_med\n"
        "   100          -12.5             0.01             0.01\n",
        encoding="utf-8",
    )
    fit = FakeFit(stdout_path=stdout_path, **create_fake_variables(window))

    comparison = compare_switching_vi_runs(
        [
            VIRunResult(
                seed=4,
                algorithm="meanfield",
                fit=fit,
                runtime_seconds=1.25,
            )
        ],
        window,
    )

    assert comparison.loc[0, "final_elbo"] == pytest.approx(-12.5)
    assert comparison.loc[0, "runtime_seconds"] == pytest.approx(1.25)
    assert {
        "ade_m",
        "fde_m",
        "radial_coverage",
        "mean_interval_width_m",
        "observed_stop_probability_mean",
        "predicted_maneuver_probability_mean",
        "transition_stop_to_stop_mean",
        "transition_cruise_to_maneuver_std",
        "sigma_position_gps_mean",
    }.issubset(comparison.columns)


def test_synthetic_switching_data_are_reproducible_and_retain_truth():
    """The synthetic scenario should retain evaluation truth with fixed phases."""
    first = simulate_synthetic_switching_ctrv_data(seed=101)
    second = simulate_synthetic_switching_ctrv_data(seed=101)
    different = simulate_synthetic_switching_ctrv_data(seed=102)

    assert first.equals(second)
    assert not first[["gps_longitude", "gps_latitude"]].equals(
        different[["gps_longitude", "gps_latitude"]]
    )
    assert first.loc[0, "mode_true"] == MODE_INITIAL
    assert first.loc[0, "mode_name_true"] == "initial"
    assert set(first.loc[1:, "mode_true"]) == set(MODE_NAMES)
    assert {
        "x_true",
        "y_true",
        "speed_true",
        "heading_true",
        "turn_rate_true",
        "mode_true",
        "mode_name_true",
    }.issubset(first.columns)
    expected_mode_names = [MODE_NAMES[int(mode)] for mode in first.mode_true.iloc[1:]]
    assert first.mode_name_true.iloc[1:].tolist() == expected_mode_names


def test_synthetic_gps_speed_is_external_and_truth_is_excluded_from_stan_data():
    """GPS-speed noise and truth columns must not determine model-ready inputs."""
    without_speed_noise = simulate_synthetic_switching_ctrv_data(
        noise=SyntheticSwitchingCTRVNoise(sigma_speed_gps=0.0),
        seed=33,
    )
    with_speed_noise = simulate_synthetic_switching_ctrv_data(
        noise=SyntheticSwitchingCTRVNoise(sigma_speed_gps=50.0),
        seed=33,
    )
    invariant_columns = [
        "gps_longitude",
        "gps_latitude",
        "x_true",
        "y_true",
        "speed_true",
        "heading_true",
        "turn_rate_true",
        "mode_true",
    ]
    assert without_speed_noise[invariant_columns].equals(
        with_speed_noise[invariant_columns]
    )
    assert not without_speed_noise["gps_speed"].equals(with_speed_noise["gps_speed"])

    window = prepare_trajectory_window(
        without_speed_noise,
        observation_count=27,
        prediction_count=4,
    )
    stan_data = build_stan_data(window)
    assert not any("true" in key for key in stan_data)
    assert "mode_true" not in stan_data
    assert "gps_speed" not in stan_data


def test_switching_module_imports_no_private_baseline_names():
    """The parallel extension should depend only on the public baseline API."""
    source_path = Path(model_module.__file__)
    syntax_tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_names = []
    for node in ast.walk(syntax_tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "ship_trajectory_prediction.models.bayesian_ctrv"
        ):
            imported_names.extend(alias.name for alias in node.names)

    assert imported_names
    assert not any(name.startswith("_") for name in imported_names)


def test_stan_source_marginalizes_modes_and_has_position_only_likelihood():
    """Stan should marginalize discrete modes and expose required predictions."""
    source = STAN_FILE.read_text(encoding="utf-8")
    parameters_match = re.search(
        r"(?ms)^parameters\s*\{(?P<body>.*?)^\}",
        source,
    )

    assert parameters_match is not None
    parameters = parameters_match.group("body")
    assert "int mode" not in parameters
    assert "array[N_observed] int" not in parameters
    assert "array[K] simplex[K] transition_probability" in parameters
    assert "transition_probability[i] ~ dirichlet(alpha_transition[i])" in source
    assert "log_sum_exp" in source
    assert "target += sum(forward_log_scale)" in source
    assert "matrix[N_observed - 1, K] mode_probability" in source
    assert "array[N_prediction] int<lower=1, upper=K> mode_prediction" in source
    assert "x_observed ~ normal(x_state, sigma_position_gps)" in source
    assert "y_observed ~ normal(y_state, sigma_position_gps)" in source
    assert "gps_speed" not in source
    assert "speed_observed" not in source
    assert "real x_previous = x_state[N_observed]" in source
    assert "real y_previous = y_state[N_observed]" in source
    assert re.search(
        r"categorical_rng\s*\(\s*transition_probability\[previous_mode\]\s*\)",
        source,
    )


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_CMDSTAN_INTEGRATION") != "1",
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to run CmdStan inference.",
)
def test_small_synthetic_switching_vi_fit_is_executable():
    """A small marginalized switching fit should produce finite valid draws."""
    data = simulate_synthetic_switching_ctrv_data(
        phase_transition_counts=(2, 2, 2, 2),
        seed=51,
    )
    window = prepare_trajectory_window(
        data,
        observation_count=7,
        prediction_count=2,
    )

    fit = fit_switching_bayesian_ctrv_model(
        window,
        algorithm="meanfield",
        iter=5_000,
        draws=50,
        seed=52,
        require_converged=False,
    )

    assert posterior_variable_samples(fit, "x_state").shape == (50, 7)
    assert posterior_variable_samples(fit, "mode_probability").shape == (50, 6, 3)
    assert posterior_variable_samples(fit, "transition_probability").shape == (
        50,
        3,
        3,
    )
    assert posterior_variable_samples(fit, "mode_prediction").shape == (50, 2)
    assert posterior_variable_samples(fit, "x_state_prediction").shape == (50, 2)
    assert posterior_variable_samples(fit, "log_likelihood").shape == (50, 14)
    mode_probability = posterior_variable_samples(fit, "mode_probability")
    transition_probability = posterior_variable_samples(
        fit,
        "transition_probability",
    )
    assert np.all(np.isfinite(mode_probability))
    assert np.all(np.isfinite(transition_probability))
    assert np.sum(mode_probability, axis=-1) == pytest.approx(np.ones((50, 6)))
    assert np.sum(transition_probability, axis=-1) == pytest.approx(np.ones((50, 3)))


def assert_stan_data_equal(reference, changed):
    """Assert equality for one pair of numeric CmdStan data dictionaries."""
    assert reference.keys() == changed.keys()
    for name in reference:
        assert changed[name] == pytest.approx(reference[name])


def create_fake_variables(window):
    """Create valid posterior arrays for fast summary and comparison tests."""
    draw_count = 3
    prediction_count = window.prediction_count
    transition_count = window.observation_count - 1
    x_actual = window.x_meters[window.prediction_slice]
    y_actual = window.y_meters[window.prediction_slice]
    variables = {
        "x_state_prediction": np.vstack([x_actual - 1.0, x_actual, x_actual + 1.0]),
        "y_state_prediction": np.vstack([y_actual - 1.0, y_actual, y_actual + 1.0]),
        "x_observation_prediction": np.vstack(
            [x_actual - 1.0, x_actual, x_actual + 1.0]
        ),
        "y_observation_prediction": np.vstack(
            [y_actual - 1.0, y_actual, y_actual + 1.0]
        ),
    }
    base = np.arange(prediction_count, dtype=float)
    for name, offset in (
        ("speed_state_prediction", 3.0),
        ("heading_state_prediction", 0.2),
        ("turn_rate_state_prediction", 0.01),
    ):
        variables[name] = np.vstack(
            [base + offset - 0.1, base + offset, base + offset + 0.1]
        )
    variables["mode_prediction"] = np.asarray(
        [
            [MODE_STOP, MODE_CRUISE, MODE_MANEUVER],
            [MODE_STOP, MODE_CRUISE, MODE_MANEUVER],
            [MODE_CRUISE, MODE_CRUISE, MODE_MANEUVER],
        ],
        dtype=int,
    )[:, :prediction_count]
    observed_probability = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    variables["mode_probability"] = np.stack(
        [
            np.vstack(
                [
                    observed_probability[index % MODE_COUNT]
                    for index in range(transition_count)
                ]
            )
            for _ in range(draw_count)
        ]
    )
    transition = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    variables["transition_probability"] = np.stack([transition, transition, transition])
    for parameter_name in (
        "sigma_position_gps",
        "sigma_position_process",
        "sigma_speed_process",
        "sigma_turn_rate_process",
    ):
        variables[parameter_name] = np.asarray([0.1, 0.2, 0.3])
    return variables


class FakeFit:
    """Minimal variational result exposing stored posterior variables."""

    variational_sample = object()

    def __init__(self, stdout_path=None, **variables):
        self.variables = variables
        if stdout_path is not None:
            self.runset = SimpleNamespace(stdout_files=[str(stdout_path)])

    def stan_variable(self, name, mean=False):
        """Return one stored full-draw posterior variable."""
        del mean
        return self.variables[name]


class FakeModel:
    """Minimal CmdStan model recording VI calls and rejecting sampling."""

    def __init__(self):
        self.result = object()
        self.arguments = None
        self.calls = []

    def variational(self, **kwargs):
        """Record one variational call."""
        self.arguments = kwargs
        self.calls.append(kwargs)
        return self.result

    def sample(self, **kwargs):
        """Fail if the VI-only wrapper ever attempts MCMC."""
        del kwargs
        raise AssertionError("The switching wrapper must not call sample().")
