"""Contracts for causal method-specific raw BR24 inference."""

import numpy as np
import pytest

from bayestraj.inference import ctrv_cmdstan, ctrv_sequential_vi
from bayestraj.inference.br24_radar_smc import BR24BayesianRadarTracker
from bayestraj.inference.ctrv_sequential_vi import SequentialVIConfig
from bayestraj.inference.ctrv_smc import (
    SequentialMonteCarloCTRVConfig,
    SequentialMonteCarloCTRVFilter,
)
from bayestraj.inference.particle_utils import SequentialCTRVFit
from bayestraj.models.bayesian_ctrv import BayesianCTRVPriors
from bayestraj.observations.br24 import BR24Detection, BR24RadarCycle


def _cycle(time, *, ranges=(10.0,), azimuth=0.0):
    return BR24RadarCycle(
        time_seconds=time,
        cycle_counter=int(100 * time) + 1,
        sensor_position=np.array([100.0, -20.0]),
        sensor_yaw_rad=0.0,
        detections=tuple(
            BR24Detection(index + 1, distance, azimuth, 0.5, 0.02)
            for index, distance in enumerate(ranges)
        ),
    )


def _fit(x=130.0):
    spread = np.linspace(-0.01, 0.01, 40)
    return _ConvergedFit(
        {
            "x_at_origin": x + spread,
            "y_at_origin": -20.0 + spread,
            "speed_at_origin": np.full(40, 1e-6),
            "heading_at_origin": spread,
            "turn_rate_at_origin": spread * 1e-4,
            "sigma_position_observation": np.full(40, 0.1),
            "sigma_speed_process": np.full(40, 1e-6),
            "sigma_turn_rate_process": np.full(40, 1e-6),
        }
    )


class _ConvergedFit(SequentialCTRVFit):
    converged = True


def _tracker(method, **options):
    return BR24BayesianRadarTracker.initialize(
        _cycle(0.0),
        initial_detection_id=1,
        priors=BayesianCTRVPriors(),
        config=SequentialMonteCarloCTRVConfig(
            particle_count=128, posterior_draw_count=32
        ),
        inference_method=method,
        observation_count=6,
        sequential_vi_config=SequentialVIConfig(n_bootstrap=3, draws=40),
        seed=42,
        **options,
    )


@pytest.mark.parametrize(
    ("bootstrap_converged", "update_flags", "expected"),
    [(False, [True], False), (True, [False, True], False), (True, [True, True], True)],
)
def test_sequential_convergence_retains_every_carried_fit_failure(
    monkeypatch, bootstrap_converged, update_flags, expected
):
    def batch_fitter(*args, **kwargs):
        fit = _fit()
        fit.converged = bootstrap_converged
        return fit

    flags = iter(update_flags)

    def update_fitter(*args, **kwargs):
        fit = _fit()
        fit.converged = next(flags)
        return fit

    monkeypatch.setattr(ctrv_cmdstan, "fit_bayesian_ctrv_model", batch_fitter)
    monkeypatch.setattr(ctrv_sequential_vi, "fit_sequential_update", update_fitter)
    tracker = _tracker("sequential_vi")
    tracker.process_cycle(_cycle(0.4))
    tracker.process_cycle(_cycle(1.7))
    for time in (2.2, 3.7)[: len(update_flags)]:
        tracker.process_cycle(_cycle(time, ranges=(30.0,)))

    assert tracker.filter.converged is expected
    assert tracker.filter.diagnostics["converged"] is expected


@pytest.mark.parametrize("method", ["vi", "mcmc", "sequential_vi"])
def test_each_backend_associates_using_its_own_fitted_prediction(monkeypatch, method):
    calls = []

    def fitter(window, **kwargs):
        calls.append(kwargs["inference_method"])
        return _fit(150.0 if method == "mcmc" else 130.0)

    monkeypatch.setattr(ctrv_cmdstan, "fit_bayesian_ctrv_model", fitter)
    monkeypatch.setattr(
        ctrv_sequential_vi, "fit_sequential_update", lambda *args, **kwargs: _fit()
    )

    def forbidden_smc(*args, **kwargs):
        raise AssertionError("A non-SMC backend must not run an SMC prepass")

    monkeypatch.setattr(SequentialMonteCarloCTRVFilter, "initialize", forbidden_smc)
    tracker = _tracker(method)
    tracker.process_cycle(_cycle(0.4))
    tracker.process_cycle(_cycle(1.7))
    step = tracker.process_cycle(_cycle(2.2, ranges=(10.0, 30.0, 50.0)))

    assert step.detection_id == (3 if method == "mcmc" else 2)
    assert calls[0] == ("mcmc" if method == "mcmc" else "vi")
    assert tracker.accepted_measurement_count == 4
    assert not tracker.is_prior_predictive
    assert tracker.filter.last_fit is not None
    assert tracker.filter.converged is (None if method == "mcmc" else True)
    assert tracker.filter.diagnostics["posterior_fit_count"] == 2


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_batch_passes_full_cycle_grid_mask_and_correlated_sensor_covariance(
    monkeypatch, method
):
    calls = []

    def fitter(window, **kwargs):
        calls.append((window, kwargs))
        return _fit()

    monkeypatch.setattr(ctrv_cmdstan, "fit_bayesian_ctrv_model", fitter)
    tracker = _tracker(method)
    tracker.process_cycle(_cycle(0.4, ranges=()))
    tracker.process_cycle(_cycle(1.7, azimuth=np.pi / 4))
    tracker.process_cycle(_cycle(2.2, azimuth=np.pi / 4))
    tracker.process_cycle(_cycle(3.7, ranges=()))

    assert len(calls) == 2  # Refit on every cycle, including a miss.
    window, kwargs = calls[-1]
    observations = kwargs["position_observations"]
    np.testing.assert_array_equal(observations.time_seconds, [0, 0.4, 1.7, 2.2, 3.7])
    np.testing.assert_array_equal(observations.observed_mask, [1, 0, 1, 1, 0])
    assert observations.measurement_covariances[2, 0, 1] != 0
    assert window.observation_count == 5
    assert window.prediction_count == 1
    assert window.reference_longitude is None
    assert window.reference_latitude is None
    assert np.isnan(window.gps_speed_mps).all()
    assert tracker.filter.processed_observation_count == 5
    assert tracker.accepted_measurement_count == 3


@pytest.mark.parametrize("method", ["vi", "mcmc", "sequential_vi"])
def test_seed_warmup_keeps_uncertainty_and_does_not_reanchor_on_buffered_fixes(method):
    accepted = _tracker(method)
    missed = _tracker(method)
    draws = accepted.filter.forecast([0.01], seed=100)
    assert np.var(draws.stan_variable("x_at_origin")) > 1.0
    accepted.process_cycle(_cycle(0.4, ranges=(18.0,)))
    missed.process_cycle(_cycle(0.4, ranges=()))

    np.testing.assert_allclose(accepted.estimated_position, missed.estimated_position)
    np.testing.assert_allclose(
        accepted.filter.predict_position_distribution(1.7)[0],
        missed.filter.predict_position_distribution(1.7)[0],
    )
    assert accepted.accepted_measurement_count == 2
    assert missed.accepted_measurement_count == 1
    assert accepted.is_prior_predictive
    assert accepted.fit_status == "seed_conditioned_prior"


@pytest.mark.parametrize("method", ["smc", "vi", "mcmc", "sequential_vi"])
def test_rendering_and_repeated_forecasts_cannot_change_later_association(method):
    rendered = _tracker(method)
    untouched = _tracker(method)
    for seed in (3, 6, 9):
        rendered.filter.forecast([0.4, 1.7], seed=seed)
        rendered.filter.sample_current_posterior(seed=seed)
        rendered.filter.predict_position_distribution(0.4)
    candidates = _cycle(0.4, ranges=(9.0, 12.0, 60.0))
    actual = rendered.process_cycle(candidates)
    expected = untouched.process_cycle(candidates)

    assert actual.detection_id == expected.detection_id
    assert actual.nis == expected.nis
    np.testing.assert_array_equal(
        rendered.estimated_position, untouched.estimated_position
    )


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_batch_retains_last_posterior_when_recent_cycle_window_loses_fixes(
    monkeypatch, method
):
    calls = []

    def fitter(window, **kwargs):
        calls.append(kwargs["position_observations"])
        return _fit()

    monkeypatch.setattr(ctrv_cmdstan, "fit_bayesian_ctrv_model", fitter)
    tracker = _tracker(method)
    tracker.filter.observation_count = 4
    tracker.process_cycle(_cycle(0.4))
    tracker.process_cycle(_cycle(1.7))
    tracker.process_cycle(_cycle(2.2, ranges=()))
    tracker.process_cycle(_cycle(3.7, ranges=()))

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[-1].time_seconds, [0, 0.4, 1.7, 2.2])
    assert tracker.fit_status == "posterior_prediction_insufficient_fixes"
    assert not tracker.is_prior_predictive
    assert tracker.filter.processed_observation_count == 5
    assert tracker.accepted_measurement_count == 3
    assert tracker.filter.diagnostics["window_accepted_measurement_count"] == 2


def test_sequential_vi_bootstraps_on_real_fixes_and_predicts_through_misses(
    monkeypatch,
):
    batches = []
    updates = []

    def batch_fitter(window, **kwargs):
        batches.append(kwargs)
        return _fit()

    def update_fitter(*args, **kwargs):
        updates.append(kwargs)
        return _fit()

    monkeypatch.setattr(ctrv_cmdstan, "fit_bayesian_ctrv_model", batch_fitter)
    monkeypatch.setattr(ctrv_sequential_vi, "fit_sequential_update", update_fitter)
    tracker = _tracker("sequential_vi")
    tracker.process_cycle(_cycle(0.4, ranges=()))
    tracker.process_cycle(_cycle(1.7))
    tracker.process_cycle(_cycle(2.2))
    tracker.process_cycle(_cycle(3.7, ranges=()))
    tracker.process_cycle(_cycle(4.1, ranges=(30.0,)))

    assert len(batches) == 1
    observations = batches[0]["position_observations"]
    np.testing.assert_array_equal(observations.time_seconds, [0, 0.4, 1.7, 2.2])
    np.testing.assert_array_equal(observations.observed_mask, [1, 0, 1, 1])
    assert len(updates) == 1
    assert updates[0]["process_interval_seconds"] == 1.5
    assert updates[0]["observation_interval_seconds"] == pytest.approx(0.4)
    assert "measurement_covariance" in updates[0]
    assert tracker.filter.processed_observation_count == 6
    assert tracker.accepted_measurement_count == 4
    assert tracker.filter.sequential_filter.processed_cycle_count == 6
    assert tracker.filter.sequential_filter.processed_observation_count == 4


@pytest.mark.parametrize("method", ["vi", "mcmc", "sequential_vi"])
def test_fitted_posterior_rendering_cannot_change_next_fit_or_association(
    monkeypatch, method
):
    seeds = []

    def fitter(*args, **kwargs):
        seeds.append(kwargs["seed"])
        return _fit()

    monkeypatch.setattr(ctrv_cmdstan, "fit_bayesian_ctrv_model", fitter)
    monkeypatch.setattr(ctrv_sequential_vi, "fit_sequential_update", fitter)
    rendered = _tracker(method)
    untouched = _tracker(method)
    for tracker in (rendered, untouched):
        tracker.process_cycle(_cycle(0.4))
        tracker.process_cycle(_cycle(1.7))
    before = rendered.estimated_position.copy()
    for seed in (3, 9):
        rendered.filter.forecast([2.2, 3.7], seed=seed)
        rendered.filter.sample_current_posterior(seed=seed)
        rendered.filter.predict_position_distribution(2.2)
    np.testing.assert_array_equal(rendered.estimated_position, before)
    candidates = _cycle(2.2, ranges=(10.0, 30.0, 60.0))
    actual = rendered.process_cycle(candidates)
    expected = untouched.process_cycle(candidates)

    assert actual.detection_id == expected.detection_id == 2
    assert actual.nis == expected.nis
    assert seeds[-1] == seeds[-2]
    np.testing.assert_array_equal(
        rendered.estimated_position, untouched.estimated_position
    )


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_batch_forecast_preserves_transition_informed_first_interval_motion(
    monkeypatch, method
):
    def fitter(*args, **kwargs):
        fit = _fit()
        variables = {
            name: fit.stan_variable(name)
            for name in (
                "x_at_origin",
                "y_at_origin",
                "speed_at_origin",
                "heading_at_origin",
                "turn_rate_at_origin",
                "sigma_position_observation",
                "sigma_speed_process",
                "sigma_turn_rate_process",
            )
        }
        variables.update(
            speed_at_origin=np.ones(40),
            heading_at_origin=np.zeros(40),
            turn_rate_at_origin=np.zeros(40),
            sigma_speed_process=np.full(40, 10.0),
        )
        return _ConvergedFit(variables)

    monkeypatch.setattr(ctrv_cmdstan, "fit_bayesian_ctrv_model", fitter)
    tracker = _tracker(method)
    tracker.process_cycle(_cycle(0.4))
    tracker.process_cycle(_cycle(1.7))
    forecast = tracker.filter.forecast([2.7], seed=10)
    # Existing Stan/SMC/Sequential VI forecasts use transition-informed motion
    # for the first future interval; association uses prepared next motion.
    np.testing.assert_allclose(
        forecast.stan_variable("x_prediction")[:, 0]
        - forecast.stan_variable("x_at_origin"),
        1.0,
    )
    predicted_mean, _ = tracker.filter.predict_position_distribution(2.7)
    assert abs(predicted_mean[0] - forecast.stan_variable("x_prediction").mean()) > 0.5
