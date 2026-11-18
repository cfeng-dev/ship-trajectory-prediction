"""Contracts for causal method-specific raw BR24 inference."""

import numpy as np
import pytest

from bayestraj.inference import ctrv_cmdstan
from bayestraj.inference.br24_radar_tracker import BR24BayesianRadarTracker
from bayestraj.inference.ctrv_smc import (
    SequentialMonteCarloCTRVConfig,
    SequentialMonteCarloCTRVFilter,
)
from bayestraj.inference.particle_utils import SequentialCTRVFit
from bayestraj.models.bayesian_ctrv import BayesianCTRVPriors
from bayestraj.observations.br24 import (
    BR24Detection,
    BR24RadarCycle,
    br24_measurement_covariance,
)


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


def _fit(x=130.0, y=-20.0):
    spread = np.linspace(-0.01, 0.01, 40)
    return _ConvergedFit(
        {
            "x_at_origin": x + spread,
            "y_at_origin": y + spread,
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


def _fit_at_latest_fix(options):
    observations = options["position_observations"]
    mask = observations.observed_mask
    if mask is None:
        mask = np.ones(len(observations.x_meters), dtype=bool)
    return _fit(observations.x_meters[mask][-1], observations.y_meters[mask][-1])


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
        seed=42,
        **options,
    )


@pytest.mark.parametrize(
    ("method", "adapter_name"),
    [
        ("smc", "BR24SMCAdapter"),
        ("mcmc", "BR24BatchAdapter"),
        ("vi", "BR24BatchAdapter"),
    ],
)
def test_tracker_selects_a_method_specific_radar_adapter(method, adapter_name):
    adapter = _tracker(method).filter
    assert type(adapter).__name__ == adapter_name
    assert adapter.estimated_position.shape == (2,)
    assert adapter.processed_observation_count == 1
    assert adapter.last_observation_time_seconds == 0.0


def test_smc_adapter_preserves_native_filter_state_draws_and_diagnostics():
    adapter = _tracker("smc").filter
    assert type(adapter).__name__ == "BR24SMCAdapter"
    covariance = br24_measurement_covariance(
        range_m=10.0, azimuth_rad=0.0, std_range_m=0.5, std_azimuth_rad=0.02
    )
    native = SequentialMonteCarloCTRVFilter.initialize(
        np.array([0.0]),
        np.array([110.0]),
        np.array([-20.0]),
        priors=BayesianCTRVPriors(),
        config=SequentialMonteCarloCTRVConfig(
            particle_count=128, posterior_draw_count=32
        ),
        seed=42,
        initial_measurement_covariance=covariance,
    )
    for radar_filter in (adapter, native):
        radar_filter.update(0.4, 110.2, -20.0, measurement_covariance=covariance)
        radar_filter.advance_without_measurement(1.7)
        radar_filter.update(2.2, 110.8, -20.0, measurement_covariance=covariance)

    expected_position = np.sum(
        native.weights[:, None] * native.forecast_origin_particles[:, :2], axis=0
    )
    np.testing.assert_array_equal(adapter.estimated_position, expected_position)
    for name in (
        "processed_observation_count",
        "last_observation_time_seconds",
        "effective_sample_size",
        "last_effective_sample_size",
        "resample_count",
        "last_log_predictive_density",
        "predictive_log_density_history",
        "expected_position_observation_variance",
    ):
        assert getattr(adapter, name) == getattr(native, name)
    for actual, expected in zip(
        adapter.predict_position_distribution(2.7),
        native.predict_position_distribution(2.7),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    for method_name, args in (
        ("sample_current_posterior", ()),
        ("forecast", ([2.7, 3.7],)),
    ):
        actual = getattr(adapter, method_name)(*args, seed=100)
        expected = getattr(native, method_name)(*args, seed=100)
        variables = (
            "speed_at_origin",
            "heading_at_origin",
            "turn_rate_at_origin",
            "sigma_position_observation",
            "sigma_speed_process",
            "sigma_turn_rate_process",
        )
        if method_name == "forecast":
            variables += ("x_at_origin", "y_at_origin", "x_prediction", "y_prediction")
        for name in variables:
            np.testing.assert_array_equal(
                actual.stan_variable(name), expected.stan_variable(name)
            )


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_log_density_scores_the_new_fix_before_fitting_with_full_covariance(
    monkeypatch, method
):
    def fitter(*args, **kwargs):
        fit = _fit(1_000.0)
        fit.converged = False  # A usable nonconverged fit must still be scored.
        return fit

    monkeypatch.setattr(ctrv_cmdstan, "fit_bayesian_ctrv_model", fitter)
    tracker = _tracker(method)
    tracker.process_cycle(_cycle(0.4))
    radar_filter = tracker.filter
    population = radar_filter._population
    # At t=1.7 every predictive component has mean (1.7, 0), sigma=2.
    population.next_states[:] = [0.4, 0.0, 1.0, 0.0, 0.0]
    population.draws["sigma_position_observation"] = np.full(
        population.draws["sigma_position_observation"].shape, 2.0
    )

    radar_filter.update(
        1.7, 4.7, -1.0, measurement_covariance=np.array([[4.0, 1.0], [1.0, 2.0]])
    )

    # R + sigma^2 I = [[8, 1], [1, 6]], det=47, residual=(3, -1).
    expected = -np.log(2 * np.pi) - 0.5 * np.log(47.0) - 34.0 / 47.0
    assert radar_filter.last_log_predictive_density == pytest.approx(expected)
    assert radar_filter.predictive_log_density_history[-1] == pytest.approx(expected)
    assert len(radar_filter.predictive_log_density_history) == 2
    assert radar_filter.posterior_fit_count == 2
    assert radar_filter.converged is (None if method == "mcmc" else False)


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_log_density_updates_after_first_fit_and_is_cleared_for_a_missed_fix(
    monkeypatch, method
):
    def fitter(*args, **kwargs):
        if kwargs["position_observations"].time_seconds[-1] < 1.7:
            return _fit_at_latest_fix(kwargs)
        return _fit()

    monkeypatch.setattr(ctrv_cmdstan, "fit_bayesian_ctrv_model", fitter)
    tracker = _tracker(method)
    tracker.process_cycle(_cycle(0.4))
    tracker.process_cycle(_cycle(1.7))
    tracker.process_cycle(_cycle(2.2, ranges=(30.0,)))
    radar_filter = tracker.filter

    assert np.isfinite(radar_filter.last_log_predictive_density)
    assert len(radar_filter.predictive_log_density_history) == 3
    history = radar_filter.predictive_log_density_history.copy()
    tracker.process_cycle(_cycle(3.7, ranges=()))
    assert radar_filter.last_log_predictive_density is None
    assert radar_filter.predictive_log_density_history == history
    tracker.process_cycle(_cycle(4.4, ranges=(30.0,)))
    assert np.isfinite(radar_filter.last_log_predictive_density)
    assert len(radar_filter.predictive_log_density_history) == 4


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_failed_fit_does_not_publish_a_new_log_density(monkeypatch, method):
    def failed_fit(*args, **kwargs):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(ctrv_cmdstan, "fit_bayesian_ctrv_model", failed_fit)
    tracker = _tracker(method)
    previous = tracker.filter.last_log_predictive_density
    history = tracker.filter.predictive_log_density_history.copy()
    with pytest.raises(RuntimeError, match="fit failed"):
        tracker.process_cycle(_cycle(0.4))
    assert tracker.filter.last_log_predictive_density == previous
    assert tracker.filter.predictive_log_density_history == history


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_each_backend_associates_using_its_own_fitted_prediction(monkeypatch, method):
    calls = []

    def fitter(window, **kwargs):
        calls.append(kwargs["inference_method"])
        if kwargs["position_observations"].time_seconds[-1] < 1.7:
            return _fit_at_latest_fix(kwargs)
        return _fit(150.0 if method == "mcmc" else 130.0)

    monkeypatch.setattr(ctrv_cmdstan, "fit_bayesian_ctrv_model", fitter)

    def forbidden_smc(*args, **kwargs):
        raise AssertionError("A non-SMC backend must not run an SMC prepass")

    monkeypatch.setattr(SequentialMonteCarloCTRVFilter, "initialize", forbidden_smc)
    tracker = _tracker(method)
    tracker.process_cycle(_cycle(0.4))
    tracker.process_cycle(_cycle(1.7))
    step = tracker.process_cycle(_cycle(2.2, ranges=(10.0, 30.0, 50.0)))

    assert step.detection_id == (3 if method == "mcmc" else 2)
    assert calls[0] == method
    assert tracker.accepted_measurement_count == 4
    assert not tracker.is_prior_predictive
    assert tracker.filter.last_fit is not None
    assert tracker.filter.converged is (None if method == "mcmc" else True)
    assert tracker.filter.diagnostics["posterior_fit_count"] == 3


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_batch_passes_full_cycle_grid_mask_and_correlated_sensor_covariance(
    monkeypatch, method
):
    calls = []

    def fitter(window, **kwargs):
        calls.append((window, kwargs))
        return _fit_at_latest_fix(kwargs)

    monkeypatch.setattr(ctrv_cmdstan, "fit_bayesian_ctrv_model", fitter)
    tracker = _tracker(method)
    tracker.process_cycle(_cycle(0.4, ranges=()))
    tracker.process_cycle(_cycle(1.7, azimuth=np.pi / 4))
    tracker.process_cycle(_cycle(2.2, azimuth=np.pi / 4))
    tracker.process_cycle(_cycle(3.7, ranges=()))

    assert len(calls) == 3  # Fit from the second accepted fix, then every cycle.
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


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_batch_seed_prior_keeps_position_uncertainty_through_missed_cycle(method):
    tracker = _tracker(method)
    before = tracker.filter.forecast([0.01], seed=100)
    assert np.var(before.stan_variable("x_at_origin")) > 1.0

    tracker.process_cycle(_cycle(0.4, ranges=()))
    after = tracker.filter.forecast([1.7], seed=100)

    assert np.var(after.stan_variable("x_at_origin")) > 1.0
    assert tracker.accepted_measurement_count == 1
    assert tracker.filter.processed_observation_count == 2
    assert tracker.is_prior_predictive
    assert tracker.fit_status == "seed_conditioned_prior"


@pytest.mark.parametrize("method", ["smc", "vi", "mcmc"])
def test_rendering_and_repeated_forecasts_cannot_change_later_association(
    monkeypatch, method
):
    monkeypatch.setattr(
        ctrv_cmdstan, "fit_bayesian_ctrv_model", lambda *args, **kwargs: _fit()
    )
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
        return _fit_at_latest_fix(kwargs)

    monkeypatch.setattr(ctrv_cmdstan, "fit_bayesian_ctrv_model", fitter)
    tracker = _tracker(method)
    tracker.filter.observation_count = 4
    tracker.process_cycle(_cycle(0.4))
    tracker.process_cycle(_cycle(1.7))
    tracker.process_cycle(_cycle(2.2, ranges=()))
    tracker.process_cycle(_cycle(3.7, ranges=()))
    tracker.process_cycle(_cycle(4.1, ranges=()))

    assert len(calls) == 4
    np.testing.assert_array_equal(calls[-1].time_seconds, [0.4, 1.7, 2.2, 3.7])
    assert tracker.fit_status == "posterior_prediction_insufficient_fixes"
    assert not tracker.is_prior_predictive
    assert tracker.filter.processed_observation_count == 6
    assert tracker.accepted_measurement_count == 3
    assert tracker.filter.diagnostics["window_accepted_measurement_count"] == 1


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_fitted_posterior_rendering_cannot_change_next_fit_or_association(
    monkeypatch, method
):
    seeds = []

    def fitter(*args, **kwargs):
        seeds.append(kwargs["seed"])
        return _fit()

    monkeypatch.setattr(ctrv_cmdstan, "fit_bayesian_ctrv_model", fitter)
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
    # Stan and SMC forecasts use transition-informed motion
    # for the first future interval; association uses prepared next motion.
    np.testing.assert_allclose(
        forecast.stan_variable("x_prediction")[:, 0]
        - forecast.stan_variable("x_at_origin"),
        1.0,
    )
    predicted_mean, _ = tracker.filter.predict_position_distribution(2.7)
    assert abs(predicted_mean[0] - forecast.stan_variable("x_prediction").mean()) > 0.5


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_batch_reads_final_positions_from_stan_state_draws(monkeypatch, method):
    origin_fit = _fit()

    class StateHistoryFit:
        converged = True

        def stan_variable(self, name, *, mean=False):
            assert not mean
            if name in {"x_at_origin", "y_at_origin"}:
                raise ValueError(name)
            if name in {"x_state", "y_state"}:
                origin = origin_fit.stan_variable(f"{name[0]}_at_origin")
                return np.column_stack((np.zeros_like(origin), origin))
            return origin_fit.stan_variable(name)

    fit = StateHistoryFit()
    if method == "vi":
        fit.variational_sample = True
    monkeypatch.setattr(
        ctrv_cmdstan, "fit_bayesian_ctrv_model", lambda *args, **kwargs: fit
    )
    tracker = _tracker(method)
    tracker.process_cycle(_cycle(0.4))

    np.testing.assert_allclose(tracker.estimated_position, [130.0, -20.0])
