"""Tests for non-Rao-Blackwellized online CTRV particle filtering."""

import copy

import numpy as np
import pytest

import bayestraj.inference.ctrv_smc as smc
import bayestraj.models.bayesian_ctrv as ctrv_model


def test_smc_config_has_comparable_particle_filter_defaults():
    config = smc.SequentialMonteCarloCTRVConfig()

    assert config.particle_count == 4_000
    assert config.posterior_draw_count == 1_000
    assert config.resample_ess_fraction == 0.5
    assert config.rejuvenation_scale == 0.05


def test_smc_initializes_weighted_full_state_and_parameter_particles():
    particle_count = 64

    online_filter = smc.SequentialMonteCarloCTRVFilter.initialize(
        np.array([0.0]),
        np.array([12.0]),
        np.array([-4.0]),
        priors=ctrv_model.BayesianCTRVPriors(),
        config=smc.SequentialMonteCarloCTRVConfig(
            particle_count=particle_count,
            posterior_draw_count=16,
        ),
        seed=42,
    )

    assert online_filter.state_particles.shape == (particle_count, 5)
    assert online_filter.parameter_particles.shape == (particle_count, 3)
    assert online_filter.weights.shape == (particle_count,)
    assert not hasattr(online_filter, "state_covariances")
    assert np.all(np.isfinite(online_filter.state_particles))
    assert np.all(np.isfinite(online_filter.parameter_particles))
    assert np.all(np.exp(online_filter.parameter_particles) > 0.0)
    assert np.sum(online_filter.weights) == np.float64(1.0)
    assert online_filter.effective_sample_size == particle_count
    assert online_filter.processed_observation_count == 1


def test_smc_update_propagates_full_states_and_weights_the_observation():
    online_filter = smc.SequentialMonteCarloCTRVFilter(
        config=smc.SequentialMonteCarloCTRVConfig(
            particle_count=2,
            posterior_draw_count=2,
            resample_ess_fraction=0.01,
        ),
        parameter_particles=np.log(
            np.array(
                [
                    [1.0, 1e-12, 1e-12],
                    [1.0, 1e-12, 1e-12],
                ]
            )
        ),
        state_particles=np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 10.0, 1.0, 0.0, 0.0],
            ]
        ),
        weights=np.array([0.5, 0.5]),
        generator=np.random.default_rng(42),
        last_observation_time_seconds=0.0,
        processed_observation_count=1,
    )

    online_filter.update(1.0, 1.0, 0.0)

    assert online_filter.state_particles[:, 0] == pytest.approx([1.0, 1.0])
    assert online_filter.state_particles[:, 1] == pytest.approx([0.0, 10.0])
    assert online_filter.weights[0] > 0.999
    assert np.sum(online_filter.weights) == pytest.approx(1.0)
    assert online_filter.last_effective_sample_size == pytest.approx(1.0)
    assert online_filter.processed_observation_count == 2
    assert online_filter.last_observation_time_seconds == 1.0


def test_smc_update_propagates_current_motion_before_evolving_next_motion():
    initial_states = np.array(
        [
            [0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0, 0.0],
        ]
    )
    online_filter = smc.SequentialMonteCarloCTRVFilter(
        config=smc.SequentialMonteCarloCTRVConfig(
            particle_count=2,
            posterior_draw_count=2,
            resample_ess_fraction=0.01,
        ),
        parameter_particles=np.log(
            np.broadcast_to(np.array([100.0, 5.0, 1e-12]), (2, 3)).copy()
        ),
        state_particles=initial_states.copy(),
        weights=np.array([0.5, 0.5]),
        generator=np.random.default_rng(42),
        last_observation_time_seconds=0.0,
        processed_observation_count=1,
    )

    online_filter.update(1.0, 3.0, 0.0)

    assert online_filter.forecast_origin_particles[:, 0] == pytest.approx([2.0, 4.0])
    assert online_filter.forecast_origin_particles[:, 2] == pytest.approx([2.0, 4.0])
    assert online_filter.state_particles[:, 0] == pytest.approx([2.0, 4.0])
    assert not np.allclose(online_filter.state_particles[:, 2], [2.0, 4.0])


def test_smc_forecast_evolves_motion_after_the_completed_interval():
    particle_count = 64
    origin_state = np.array([0.0, 0.0, 2.0, 0.0, 0.0])
    origin_particles = np.broadcast_to(origin_state, (particle_count, 5)).copy()
    online_filter = smc.SequentialMonteCarloCTRVFilter(
        config=smc.SequentialMonteCarloCTRVConfig(
            particle_count=particle_count,
            posterior_draw_count=particle_count,
        ),
        parameter_particles=np.log(
            np.broadcast_to(
                np.array([1.0, 2.0, 1e-12]),
                (particle_count, 3),
            ).copy()
        ),
        state_particles=origin_particles.copy(),
        weights=np.full(particle_count, 1.0 / particle_count),
        generator=np.random.default_rng(42),
        last_observation_time_seconds=0.0,
        processed_observation_count=1,
        forecast_origin_particles=origin_particles,
    )

    fit = online_filter.forecast(np.array([1.0, 2.0]), seed=43)
    x_prediction = fit.stan_variable("x_prediction")

    assert fit.stan_variable("speed_at_origin") == pytest.approx(
        np.full(particle_count, 2.0)
    )
    assert x_prediction[:, 0] == pytest.approx(np.full(particle_count, 2.0))
    assert np.std(x_prediction[:, 1]) > 0.0


def test_smc_observation_updates_parameter_particle_weights():
    online_filter = smc.SequentialMonteCarloCTRVFilter(
        config=smc.SequentialMonteCarloCTRVConfig(
            particle_count=2,
            posterior_draw_count=2,
            resample_ess_fraction=0.01,
        ),
        parameter_particles=np.log(
            np.array(
                [
                    [1.0, 1e-12, 1e-12],
                    [10.0, 1e-12, 1e-12],
                ]
            )
        ),
        state_particles=np.zeros((2, 5)),
        weights=np.array([0.5, 0.5]),
        generator=np.random.default_rng(42),
        last_observation_time_seconds=0.0,
        processed_observation_count=1,
    )

    online_filter.update(1.0, 5.0, 0.0)

    assert online_filter.weights[1] > 0.999


def test_smc_initialization_processes_every_supplied_observation_once():
    online_filter = smc.SequentialMonteCarloCTRVFilter.initialize(
        np.arange(5, dtype=float),
        np.arange(5, dtype=float),
        np.zeros(5),
        priors=ctrv_model.BayesianCTRVPriors(),
        config=smc.SequentialMonteCarloCTRVConfig(
            particle_count=128,
            posterior_draw_count=16,
        ),
        seed=42,
    )

    assert online_filter.processed_observation_count == 5
    assert online_filter.last_observation_time_seconds == 4.0
    assert online_filter.last_effective_sample_size is not None
    assert np.sum(online_filter.weights) == pytest.approx(1.0)


def test_smc_resamples_state_and_parameter_ancestry_together():
    expected_parameters = np.log(np.array([0.1, 1e-12, 1e-12]))
    online_filter = smc.SequentialMonteCarloCTRVFilter(
        config=smc.SequentialMonteCarloCTRVConfig(
            particle_count=4,
            posterior_draw_count=2,
            resample_ess_fraction=1.0,
            rejuvenation_scale=0.0,
        ),
        parameter_particles=np.array(
            [
                expected_parameters,
                np.log(np.array([0.1, 2e-12, 1e-12])),
                np.log(np.array([0.1, 3e-12, 1e-12])),
                np.log(np.array([0.1, 4e-12, 1e-12])),
            ]
        ),
        state_particles=np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 10.0, 1.0, 0.0, 0.0],
                [0.0, 20.0, 1.0, 0.0, 0.0],
                [0.0, 30.0, 1.0, 0.0, 0.0],
            ]
        ),
        weights=np.full(4, 0.25),
        generator=np.random.default_rng(42),
        last_observation_time_seconds=0.0,
        processed_observation_count=1,
    )

    online_filter.update(1.0, 1.0, 0.0)

    assert online_filter.resample_count == 1
    assert online_filter.last_effective_sample_size == pytest.approx(1.0)
    assert online_filter.weights == pytest.approx(np.full(4, 0.25))
    assert online_filter.state_particles[:, 1] == pytest.approx(np.zeros(4))
    assert online_filter.parameter_particles == pytest.approx(
        np.broadcast_to(expected_parameters, (4, 3))
    )


def test_smc_samples_reproducible_current_posterior_without_advancing():
    online_filter = smc.SequentialMonteCarloCTRVFilter.initialize(
        np.arange(5, dtype=float),
        np.arange(5, dtype=float),
        np.zeros(5),
        priors=ctrv_model.BayesianCTRVPriors(),
        config=smc.SequentialMonteCarloCTRVConfig(
            particle_count=128,
            posterior_draw_count=16,
        ),
        seed=42,
    )
    states_before = online_filter.state_particles.copy()
    parameters_before = online_filter.parameter_particles.copy()
    weights_before = online_filter.weights.copy()
    generator_state_before = copy.deepcopy(online_filter.generator.bit_generator.state)
    observation_count_before = online_filter.processed_observation_count

    first_fit = online_filter.sample_current_posterior(seed=43)
    second_fit = online_filter.sample_current_posterior(seed=43)

    for variable_name in ctrv_model.PARAMETER_NAMES:
        first_samples = first_fit.stan_variable(variable_name)
        second_samples = second_fit.stan_variable(variable_name)
        assert first_samples.shape == (16,)
        assert np.array_equal(first_samples, second_samples)
    assert online_filter.state_particles == pytest.approx(states_before)
    assert online_filter.parameter_particles == pytest.approx(parameters_before)
    assert online_filter.weights == pytest.approx(weights_before)
    assert online_filter.generator.bit_generator.state == generator_state_before
    assert online_filter.processed_observation_count == observation_count_before


def test_smc_forecast_exposes_shared_latent_and_observation_draws():
    online_filter = smc.SequentialMonteCarloCTRVFilter.initialize(
        np.arange(5, dtype=float),
        np.arange(5, dtype=float),
        np.zeros(5),
        priors=ctrv_model.BayesianCTRVPriors(),
        config=smc.SequentialMonteCarloCTRVConfig(
            particle_count=128,
            posterior_draw_count=16,
        ),
        seed=42,
    )
    states_before = online_filter.state_particles.copy()
    observation_count_before = online_filter.processed_observation_count

    fit = online_filter.forecast(np.arange(5, 8, dtype=float), seed=43)

    for variable_name in ctrv_model.PARAMETER_NAMES:
        samples = fit.stan_variable(variable_name)
        assert samples.shape == (16,)
        assert np.all(np.isfinite(samples))
    for variable_name in (
        "x_prediction",
        "y_prediction",
        "x_observation_prediction",
        "y_observation_prediction",
    ):
        samples = fit.stan_variable(variable_name)
        assert samples.shape == (16, 3)
        assert np.all(np.isfinite(samples))
    assert not np.array_equal(
        fit.stan_variable("x_prediction"),
        fit.stan_variable("x_observation_prediction"),
    )
    assert online_filter.state_particles == pytest.approx(states_before)
    assert online_filter.processed_observation_count == observation_count_before
