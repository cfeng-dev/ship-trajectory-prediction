"""Tests for the Bayesian Position Model single-window workflow."""

import numpy as np

import ship_trajectory_prediction.forecasting.bayesian_position_model as config
import ship_trajectory_prediction.forecasting.bayesian_position_model_workflow as workflow
import ship_trajectory_prediction.forecasting.inference as inference
import ship_trajectory_prediction.models.bayesian_position_model as position_model
from ship_trajectory_prediction.models.deterministic_ctrv import CTRVState
from ship_trajectory_prediction.simulation.synthetic_ctrv import (
    SyntheticCTRVNoise,
    simulate_synthetic_ctrv_data,
)


def test_single_window_workflow_uses_observation_level_predictions(monkeypatch):
    """The new workflow should fit K positions and evaluate recorded positions."""
    trajectory_data = simulate_synthetic_ctrv_data(
        count=23,
        dt_seconds=10.0,
        initial_state=CTRVState(
            x=0.0,
            y=0.0,
            speed=2.0,
            heading=0.1,
            turn_rate=0.005,
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
    experiment = config.ExperimentConfig(
        run_id=0,
        start_index=0,
        observation_count=20,
        prediction_count=3,
        history_position_count=10,
        additional_position_noise_std_m=0.0,
        position_noise_seed=2026,
        inference_method="vi",
        inference_seed=42,
    )
    priors = position_model.BayesianPositionModelPriors(
        log_displacement_scale_prior_scale=0.016354,
        rotation_angle_prior_scale=0.016980,
        sigma_displacement_residual_prior_scale=1.989083,
    )
    fit_calls = []
    plot_calls = []

    def fake_fit(window, **kwargs):
        fit_calls.append(kwargs)
        prediction = window.prediction_slice
        x_values = np.tile(window.x_meters[prediction], (100, 1))
        y_values = np.tile(window.y_meters[prediction], (100, 1))
        return FakeFit(
            x_observation_prediction=x_values,
            y_observation_prediction=y_values,
        )

    monkeypatch.setattr(
        workflow.observations_io,
        "read_ship_data",
        lambda *args, **kwargs: trajectory_data,
    )
    monkeypatch.setattr(
        workflow.position_model,
        "fit_bayesian_position_model",
        fake_fit,
    )
    monkeypatch.setattr(
        workflow.position_model,
        "variational_converged",
        lambda fit: True,
    )
    monkeypatch.setattr(
        workflow.reporting,
        "print_variational_diagnostics",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        workflow.reporting,
        "posterior_parameter_summary",
        lambda *args, **kwargs: "parameter summary",
    )
    monkeypatch.setattr(
        workflow.metrics,
        "print_position_evaluation",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        workflow.prediction_plotting,
        "plot_prediction",
        lambda *args, **kwargs: plot_calls.append(kwargs),
    )

    result = workflow.run_bayesian_position_prediction(
        data_file="unused.csv",
        experiment=experiment,
        priors=priors,
        vi_config=inference.create_default_vi_config(),
        mcmc_config=inference.create_default_mcmc_config(),
        fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
        credible_interval=0.9,
        history_position_count=10,
        inference_method="vi",
        vi_algorithm="meanfield",
        seed=42,
        position_noise_std_m=0.0,
        position_noise_seed=2026,
        require_converged=False,
        plot_coordinate_mode="m",
    )

    assert result["stan_data"]["N_history"] == 10
    assert result["evaluation"].ade_m == 0.0
    assert fit_calls[0]["history_position_count"] == 10
    assert plot_calls[0]["state_prediction_variable_names"] == (
        "x_observation_prediction",
        "y_observation_prediction",
    )


class FakeFit:
    """Minimal variational result exposing posterior prediction arrays."""

    variational_sample = object()

    def __init__(self, **variables):
        self.variables = variables

    def stan_variable(self, name, mean=False):
        """Return one stored posterior variable."""
        del mean
        return self.variables[name]
