"""Tests for shared Bayesian CTRV forecasting configuration."""

from ship_trajectory_prediction.forecasting.bayesian_ctrv import (
    DEFAULT_FULLRANK_GRAD_SAMPLES,
    create_default_mcmc_config,
    create_default_vi_config,
)
from ship_trajectory_prediction.models.bayesian_ctrv import (
    DEFAULT_MEANFIELD_GRAD_SAMPLES,
    DEFAULT_VI_ADAPT_ITER,
)


def test_default_vi_config_returns_independent_complete_options():
    """Each experiment should receive its own VI option dictionary."""
    first = create_default_vi_config()
    second = create_default_vi_config()

    assert first == {
        "algorithm": "meanfield",
        "iter": 20_000,
        "grad_samples": DEFAULT_MEANFIELD_GRAD_SAMPLES,
        "elbo_samples": 100,
        "eta": 1.0,
        "adapt_iter": DEFAULT_VI_ADAPT_ITER,
        "tol_rel_obj": 0.01,
        "eval_elbo": 100,
        "draws": 1_000,
        "require_converged": False,
    }
    assert first is not second
    assert DEFAULT_FULLRANK_GRAD_SAMPLES == 10
    assert DEFAULT_MEANFIELD_GRAD_SAMPLES == 2


def test_default_mcmc_config_returns_independent_complete_options():
    """Each experiment should receive its own MCMC option dictionary."""
    first = create_default_mcmc_config()
    second = create_default_mcmc_config()

    assert first == {
        "chains": 4,
        "parallel_chains": 4,
        "iter_warmup": 1_000,
        "iter_sampling": 1_000,
        "adapt_delta": 0.9,
        "max_treedepth": 10,
    }
    assert first is not second
