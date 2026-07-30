"""Tests for the Bayesian CTRV state-space model interface."""

from __future__ import annotations

import os
import re
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ship_trajectory_prediction.models import bayesian_ctrv as model_module
from ship_trajectory_prediction.models.bayesian_ctrv import (
    DEFAULT_TURN_RATE_LIMIT,
    NOISE_PARAMETER_NAMES,
    STAN_FILE,
    VIRunResult,
    build_stan_data,
    compare_vi_runs,
    diagnose_observed_turn_rate,
    fit_bayesian_ctrv_model,
    summarize_predictions,
    variational_converged,
)
from ship_trajectory_prediction.models.ctrv import CTRVState
from ship_trajectory_prediction.simulation.synthetic_ctrv import (
    SyntheticCTRVNoise,
    simulate_synthetic_ctrv_data,
)
from ship_trajectory_prediction.trajectory import prepare_trajectory_window


def create_synthetic_window(*, turn_rate=0.012, variable_dt=False):
    """Create one noise-free curved window for fast interface tests."""
    time_seconds = None
    if variable_dt:
        time_seconds = [0.0, 2.0, 5.0, 9.0, 12.0, 17.0, 21.0, 27.0, 32.0, 38.0]
    data = simulate_synthetic_ctrv_data(
        count=10,
        dt_seconds=5.0,
        time_seconds=time_seconds,
        initial_state=CTRVState(
            x=0.0,
            y=0.0,
            speed=3.0,
            heading=0.3,
            turn_rate=turn_rate,
        ),
        noise=SyntheticCTRVNoise(
            sigma_position_gps=0.0,
            sigma_speed_gps=0.0,
            sigma_position_process=0.0,
            sigma_speed_process=0.0,
            sigma_turn_rate_process=0.0,
        ),
        seed=7,
    )
    return prepare_trajectory_window(
        data,
        observation_count=7,
        prediction_count=3,
    )


def test_build_stan_data_contains_observed_states_future_times_and_si_units():
    """Stan data should expose only measured history and future timestamps."""
    window = create_synthetic_window(variable_dt=True)

    stan_data = build_stan_data(window)

    assert stan_data["N_observed"] == 7
    assert stan_data["N_prediction"] == 3
    assert stan_data["time_observed"] == pytest.approx(
        [0.0, 2.0, 5.0, 9.0, 12.0, 17.0, 21.0]
    )
    assert stan_data["time_prediction"] == pytest.approx([27.0, 32.0, 38.0])
    assert stan_data["speed_observed"] == pytest.approx(np.full(7, 3.0))
    assert stan_data["speed_initial_prior_mean"] == pytest.approx(3.0)
    assert stan_data["turn_rate_initial_prior_mean"] == pytest.approx(
        0.012,
        abs=1e-8,
    )
    assert stan_data["turn_rate_state_prior_scale"] == pytest.approx(0.002)
    assert stan_data["turn_rate_limit"] == pytest.approx(DEFAULT_TURN_RATE_LIMIT)
    assert "x_prediction" not in stan_data
    assert "y_prediction" not in stan_data
    assert "speed_prediction" not in stan_data


def test_build_stan_data_does_not_use_held_out_measurements():
    """Changing future GPS positions and speed must leave Stan data unchanged."""
    data = simulate_synthetic_ctrv_data(count=10, seed=9)
    reference_window = prepare_trajectory_window(
        data,
        observation_count=7,
        prediction_count=3,
    )
    changed = data.copy()
    changed.loc[7:, "gps_longitude"] += 2.0
    changed.loc[7:, "gps_latitude"] -= 2.0
    changed.loc[7:, "gps_speed"] = np.nan
    changed_window = prepare_trajectory_window(
        changed,
        observation_count=7,
        prediction_count=3,
    )

    reference_data = build_stan_data(reference_window)
    changed_data = build_stan_data(changed_window)

    assert reference_data.keys() == changed_data.keys()
    for key in reference_data:
        assert changed_data[key] == pytest.approx(reference_data[key])


@pytest.mark.parametrize(
    "scale_name",
    [
        "position_initial_prior_scale",
        "speed_initial_prior_scale",
        "heading_initial_prior_scale",
        "turn_rate_state_prior_scale",
        "sigma_position_gps_prior_scale",
        "sigma_speed_gps_prior_scale",
        "sigma_position_process_prior_scale",
        "sigma_speed_process_prior_scale",
        "sigma_turn_rate_process_prior_scale",
    ],
)
def test_build_stan_data_rejects_non_positive_prior_scales(scale_name):
    """Every configurable prior scale should be positive and finite."""
    with pytest.raises(ValueError, match=scale_name):
        build_stan_data(create_synthetic_window(), **{scale_name: 0.0})


@pytest.mark.parametrize(("turn_rate", "expected_sign"), [(0.012, 1), (-0.012, -1)])
def test_turn_rate_prior_preserves_positive_and_negative_turning(
    turn_rate,
    expected_sign,
):
    """Observed course changes should retain their signed turning direction."""
    stan_data = build_stan_data(create_synthetic_window(turn_rate=turn_rate))

    assert np.sign(stan_data["turn_rate_initial_prior_mean"]) == expected_sign
    assert stan_data["turn_rate_initial_prior_mean"] == pytest.approx(
        turn_rate,
        abs=1e-8,
    )


def test_turn_rate_prior_is_zero_for_straight_motion():
    """A straight trajectory should center the initial turn rate at zero."""
    stan_data = build_stan_data(create_synthetic_window(turn_rate=0.0))

    assert stan_data["turn_rate_initial_prior_mean"] == pytest.approx(0.0, abs=1e-9)


def test_turn_rate_diagnostics_are_robust_and_configurable():
    """Observed course changes should set a bounded robust state prior."""
    window = create_synthetic_window(turn_rate=0.012)

    derived = diagnose_observed_turn_rate(window)
    configured = diagnose_observed_turn_rate(
        window,
        turn_rate_state_prior_scale=0.004,
        turn_rate_limit=0.015,
    )

    assert derived.sample_count == 5
    assert derived.median_rad_s == pytest.approx(0.012, abs=1e-8)
    assert derived.q90_absolute_rad_s == pytest.approx(0.012, abs=1e-8)
    assert derived.prior_scale_rad_s == pytest.approx(0.002)
    assert configured.prior_scale_rad_s == pytest.approx(0.004)
    assert configured.limit_rad_s == pytest.approx(0.015)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"turn_rate_state_prior_scale": 0.0}, "turn_rate_state_prior_scale"),
        ({"turn_rate_limit": 0.0}, "turn_rate_limit"),
    ],
)
def test_turn_rate_diagnostics_reject_invalid_regularization(options, message):
    """Turn-rate regularization controls must remain positive and finite."""
    with pytest.raises(ValueError, match=message):
        diagnose_observed_turn_rate(create_synthetic_window(), **options)


def test_stationary_observations_use_neutral_heading_and_turn_rate_centers():
    """A stationary window should remain fit-ready without a false GPS course."""
    window = create_synthetic_window()
    stationary_window = replace(
        window,
        x_meters=np.zeros_like(window.x_meters),
        y_meters=np.zeros_like(window.y_meters),
    )

    stan_data = build_stan_data(stationary_window)

    assert stan_data["heading_initial_prior_mean"] == 0.0
    assert stan_data["turn_rate_initial_prior_mean"] == 0.0


@pytest.mark.parametrize("invalid_speed", [np.nan, -0.1])
def test_build_stan_data_rejects_invalid_observed_speed(invalid_speed):
    """The GPS speed likelihood requires finite non-negative history."""
    window = create_synthetic_window()
    speed = window.gps_speed_mps.copy()
    speed[2] = invalid_speed
    invalid_window = replace(window, gps_speed_mps=speed)

    with pytest.raises(ValueError, match="gps_speed|speed_observed"):
        build_stan_data(invalid_window)


def test_build_stan_data_rejects_invalid_observation_and_prediction_times():
    """Both parts of the state transition need positive time increments."""
    window = create_synthetic_window()
    invalid_observed_time = window.time_seconds.copy()
    invalid_observed_time[3] = invalid_observed_time[2]
    with pytest.raises(ValueError, match="Observed timestamps"):
        build_stan_data(replace(window, time_seconds=invalid_observed_time))

    invalid_prediction_time = window.time_seconds.copy()
    invalid_prediction_time[7] = invalid_prediction_time[6]
    with pytest.raises(ValueError, match="Prediction timestamps"):
        build_stan_data(replace(window, time_seconds=invalid_prediction_time))


def test_fit_forwards_explicit_meanfield_controls(monkeypatch):
    """The wrapper should expose the complete requested CmdStan VI interface."""
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module,
        "compile_bayesian_ctrv_model",
        lambda: fake_model,
    )

    fit = fit_bayesian_ctrv_model(
        create_synthetic_window(),
        algorithm="meanfield",
        iter=500,
        grad_samples=2,
        elbo_samples=20,
        eta=0.5,
        adapt_iter=25,
        tol_rel_obj=0.01,
        eval_elbo=50,
        draws=80,
        seed=17,
        require_converged=False,
        show_console=True,
    )

    assert fit is fake_model.result
    arguments = fake_model.arguments
    assert arguments["algorithm"] == "meanfield"
    assert arguments["iter"] == 500
    assert arguments["grad_samples"] == 2
    assert arguments["elbo_samples"] == 20
    assert arguments["eta"] == pytest.approx(0.5)
    assert arguments["adapt_iter"] == 25
    assert arguments["tol_rel_obj"] == pytest.approx(0.01)
    assert arguments["eval_elbo"] == 50
    assert arguments["draws"] == 80
    assert arguments["seed"] == 17
    assert arguments["require_converged"] is False
    assert arguments["show_console"] is True


def test_fit_accepts_fullrank_and_reproducible_seeded_initials(monkeypatch):
    """A repeated seed should reproduce initial values for either VI family."""
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module,
        "compile_bayesian_ctrv_model",
        lambda: fake_model,
    )
    window = create_synthetic_window()

    fit_bayesian_ctrv_model(window, algorithm="fullrank", seed=31)
    first_arguments = fake_model.calls[-1]
    fit_bayesian_ctrv_model(window, algorithm="fullrank", seed=31)
    second_arguments = fake_model.calls[-1]

    assert first_arguments["algorithm"] == "fullrank"
    for name, first_value in first_arguments["inits"].items():
        assert second_arguments["inits"][name] == pytest.approx(first_value)


@pytest.mark.parametrize("observed_speed", [0.0, 150.0])
def test_fit_initializes_speed_strictly_inside_stan_bounds(
    monkeypatch,
    observed_speed,
):
    """Zero or extreme GPS speeds must not initialize on a Stan boundary."""
    fake_model = FakeModel()
    monkeypatch.setattr(
        model_module,
        "compile_bayesian_ctrv_model",
        lambda: fake_model,
    )
    window = create_synthetic_window()
    speed = window.gps_speed_mps.copy()
    speed[2] = observed_speed

    fit_bayesian_ctrv_model(replace(window, gps_speed_mps=speed), seed=71)

    initial_speed = fake_model.arguments["inits"]["speed_state"]
    assert np.all(initial_speed > 0.001)
    assert np.all(initial_speed < 100)


@pytest.mark.parametrize(
    "options",
    [
        {"data": {}},
        {"seed": 1},
        {"algorithm": "fullrank"},
        {"tol_rel_obj": 0.2},
    ],
)
def test_fit_rejects_conflicting_variational_options(monkeypatch, options):
    """Generic options must not silently override the typed VI controls."""
    monkeypatch.setattr(
        model_module,
        "compile_bayesian_ctrv_model",
        lambda: FakeModel(),
    )

    with pytest.raises(ValueError, match="must not override"):
        fit_bayesian_ctrv_model(
            create_synthetic_window(),
            variational_options=options,
        )


@pytest.mark.parametrize("algorithm", ["laplace", "mcmc", ""])
def test_fit_rejects_unsupported_inference_algorithms(algorithm):
    """This phase should expose only meanfield and fullrank VI."""
    with pytest.raises(ValueError, match="algorithm"):
        fit_bayesian_ctrv_model(create_synthetic_window(), algorithm=algorithm)


def test_summarize_predictions_includes_all_future_ctrv_states():
    """Prediction summaries should have one interval per requested state."""
    window = create_synthetic_window()
    base = np.arange(window.prediction_count, dtype=float)
    variables = {
        name: np.vstack([base + offset, base + offset + 1, base + offset + 2])
        for name, offset in (
            ("x_prediction", 0),
            ("y_prediction", 3),
            ("speed_prediction", 6),
            ("heading_prediction", 9),
            ("turn_rate_prediction", 12),
        )
    }

    summary = summarize_predictions(FakeFit(**variables), window, 0.8)

    assert len(summary) == window.prediction_count
    for prefix in ("x", "y", "speed", "heading", "turn_rate"):
        assert {
            f"{prefix}_median",
            f"{prefix}_lower",
            f"{prefix}_upper",
        }.issubset(summary.columns)


def test_summarize_predictions_rejects_wrong_prediction_shape():
    """Malformed generated quantities should fail before evaluation."""
    window = create_synthetic_window()
    variables = {
        name: np.ones((5, window.prediction_count))
        for name in (
            "x_prediction",
            "y_prediction",
            "speed_prediction",
            "heading_prediction",
            "turn_rate_prediction",
        )
    }
    variables["heading_prediction"] = np.ones((5, 2))

    with pytest.raises(ValueError, match="unexpected shape"):
        summarize_predictions(FakeFit(**variables), window)


def test_compare_vi_runs_reports_noise_prediction_and_elbo_metrics(tmp_path):
    """Cross-seed comparison should expose all required stability quantities."""
    window = create_synthetic_window()
    prediction_count = window.prediction_count
    x_actual = window.x_meters[window.prediction_slice]
    y_actual = window.y_meters[window.prediction_slice]
    variables = {
        "x_prediction": np.vstack([x_actual - 1, x_actual, x_actual + 1]),
        "y_prediction": np.vstack([y_actual - 1, y_actual, y_actual + 1]),
    }
    for parameter_name in NOISE_PARAMETER_NAMES:
        variables[parameter_name] = np.asarray([0.1, 0.2, 0.3])
    stdout_path = tmp_path / "variational-stdout.txt"
    stdout_path.write_text(
        "  iter             ELBO   delta_ELBO_mean   delta_ELBO_med\n"
        "   100          -12.5             0.01             0.01\n",
        encoding="utf-8",
    )
    fit = FakeFit(stdout_path=stdout_path, **variables)

    comparison = compare_vi_runs(
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
    assert comparison.loc[0, "ade_m"] == pytest.approx(0.0)
    assert comparison.loc[0, "fde_m"] == pytest.approx(0.0)
    assert comparison.loc[0, "radial_coverage"] == pytest.approx(1.0)
    assert comparison.loc[0, "sigma_position_gps_mean"] == pytest.approx(0.2)
    assert prediction_count == 3


@pytest.mark.parametrize(("warning", "expected"), [(False, True), (True, False)])
def test_variational_converged_reads_cmdstan_status(tmp_path, warning, expected):
    """CmdStan's explicit VI warning should determine reported convergence."""
    transcript = "COMPLETED.\n"
    if warning:
        transcript = "The algorithm may not have converged.\n" + transcript
    stdout_path = tmp_path / "variational-status.txt"
    stdout_path.write_text(transcript, encoding="utf-8")

    assert variational_converged(FakeFit(stdout_path=stdout_path)) is expected


def test_synthetic_data_retains_truth_noise_and_is_seed_reproducible():
    """Synthetic validation data should retain every known latent quantity."""
    first = simulate_synthetic_ctrv_data(count=8, seed=101)
    second = simulate_synthetic_ctrv_data(count=8, seed=101)

    assert first.equals(second)
    required_columns = {
        "x_true",
        "y_true",
        "speed_true",
        "heading_true",
        "turn_rate_true",
        *(f"{name}_true" for name in NOISE_PARAMETER_NAMES),
    }
    assert required_columns.issubset(first.columns)


def test_stan_model_contains_ctrv_branches_and_variable_dt_diffusion():
    """The Stan source should retain exact, straight, and scaled transitions."""
    source = STAN_FILE.read_text(encoding="utf-8")

    assert "speed / turn_rate" in source
    assert "x + speed * dt * cos(heading)" in source
    assert "y + speed * dt * sin(heading)" in source
    assert "sigma_speed_process * sqrt(dt)" in source
    assert "sigma_turn_rate_process * sqrt(dt)" in source
    assert "lower=-turn_rate_limit" in source
    assert "upper=turn_rate_limit" in source
    assert "turn_rate_state ~ normal(turn_rate_initial_prior_mean" in source
    assert "real x_previous = x_state[N_observed]" in source
    assert "real y_previous = y_state[N_observed]" in source


def test_stan_ctrv_calls_never_use_observed_positions_as_transition_inputs():
    """Every fitted and future CTRV step should propagate latent positions."""
    source = STAN_FILE.read_text(encoding="utf-8")
    calls = re.findall(
        r"vector\[2\]\s+\w+\s*=\s*ctrv_position\((.*?)\);",
        source,
        flags=re.DOTALL,
    )

    assert len(calls) == 2
    assert all("x_observed" not in call for call in calls)
    assert all("y_observed" not in call for call in calls)
    assert "x_observed ~ normal(x_state, sigma_position_gps)" in source
    assert "y_observed ~ normal(y_state, sigma_position_gps)" in source


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_CMDSTAN_INTEGRATION") != "1",
    reason="Set RUN_CMDSTAN_INTEGRATION=1 to compile and run CmdStan VI.",
)
@pytest.mark.parametrize(
    ("algorithm", "seed"),
    [("meanfield", 12), ("fullrank", 13)],
)
def test_small_synthetic_vi_fit_is_executable(algorithm, seed):
    """Both supported VI approximations should produce finite state draws."""
    fit = fit_bayesian_ctrv_model(
        create_synthetic_window(),
        algorithm=algorithm,
        iter=5_000,
        draws=100,
        seed=seed,
        require_converged=False,
    )

    assert posterior_samples(fit, "x_state").shape == (100, 7)
    assert posterior_samples(fit, "heading_prediction").shape == (100, 3)
    assert np.all(np.isfinite(posterior_samples(fit, "sigma_position_gps")))


def posterior_samples(fit, name):
    """Extract full VI draws for the integration assertions."""
    return np.asarray(fit.stan_variable(name, mean=False))


class FakeFit:
    """Minimal variational CmdStan result for reporting tests."""

    variational_sample = object()

    def __init__(self, stdout_path=None, **variables):
        self.variables = variables
        if stdout_path is not None:
            self.runset = SimpleNamespace(stdout_files=[str(stdout_path)])

    def stan_variable(self, name, mean=False):
        """Return one stored fake posterior variable."""
        del mean
        return self.variables[name]


class FakeModel:
    """Minimal CmdStan model that records variational calls."""

    def __init__(self):
        self.result = object()
        self.arguments = None
        self.calls = []

    def variational(self, **kwargs):
        """Capture variational arguments and return a stable sentinel."""
        self.arguments = kwargs
        self.calls.append(kwargs)
        return self.result
