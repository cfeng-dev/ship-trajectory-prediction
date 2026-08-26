"""Single-window workflow for the parametric Bayesian CTRV model."""

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

import bayestraj.forecasting.bayesian_ctrv as forecasting
import bayestraj.forecasting.inference as inference
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.observations.io as observations_io
import bayestraj.observations.window as observation_window
import bayestraj.validation.metrics as metrics
import bayestraj.validation.prediction_plotting as prediction_plotting
import bayestraj.validation.reporting as reporting


def run_bayesian_ctrv_prediction(
    *,
    data_file,
    experiment: forecasting.ExperimentConfig,
    priors: bayesian_model.BayesianCTRVPriors,
    vi_config: Mapping[str, Any],
    mcmc_config: Mapping[str, Any],
    fullrank_grad_samples: int,
    credible_interval: float,
    inference_method: str,
    vi_algorithm: str,
    seed: int,
    position_noise_std_m: float,
    position_noise_seed: int,
    require_converged: bool,
    plot_coordinate_mode: str,
    show_time_labels: bool,
):
    """Fit and evaluate one constant-parameter Bayesian CTRV prediction."""
    inference_method, inference_config = inference.select_inference_config(
        inference_method,
        vi_algorithm=vi_algorithm,
        require_converged=require_converged,
        vi_config=vi_config,
        mcmc_config=mcmc_config,
        fullrank_grad_samples=fullrank_grad_samples,
    )
    plot_coordinate_mode = prediction_plotting.normalize_plot_coordinate_mode(
        plot_coordinate_mode
    )
    trajectory_data = observations_io.read_ship_data(
        data_file,
        run_id=experiment.run_id,
    )
    window = observation_window.prepare_trajectory_window(
        trajectory_data,
        observation_count=experiment.observation_count,
        prediction_count=experiment.prediction_count,
        start_index=experiment.start_index,
    )
    position_observations = bayesian_model.simulate_position_observations(
        window,
        additional_noise_std_m=position_noise_std_m,
        seed=position_noise_seed,
    )
    stan_data = bayesian_model.build_stan_data(
        window,
        priors=priors,
        position_observations=position_observations,
    )
    forecast_horizon_seconds = float(
        stan_data["time_prediction"][-1] - stan_data["time_observed"][-1]
    )
    reporting.print_prediction_setup(
        "Parametric Bayesian CTRV Prediction",
        data_file=data_file,
        run_id=experiment.run_id,
        window=window,
        extra_rows=[
            (
                "Model parameters",
                "speed, heading_initial, turn_rate, sigma_position_observation",
            ),
            ("Inference method", inference_method.upper()),
            ("Inference seed", seed),
            (
                "Hidden synthetic noise",
                (
                    f"{position_noise_std_m:g} m (seed={position_noise_seed})"
                    if position_noise_std_m > 0
                    else "disabled"
                ),
            ),
            (
                "Observation-noise prior",
                "Exponential("
                f"rate={priors.sigma_position_observation_prior_rate:.4f} 1/m; "
                "P(sigma > "
                f"{priors.sigma_position_observation_prior_upper_m:g} m)="
                f"{priors.sigma_position_observation_prior_tail_probability:g})",
            ),
            ("Forecast horizon", f"{forecast_horizon_seconds:g} s"),
            ("Plot coordinates", plot_coordinate_mode),
            ("Prior status", "provisional transfer; model-specific validation pending"),
        ],
    )

    computation_started = time.perf_counter()
    fit = bayesian_model.fit_bayesian_ctrv_model(
        window,
        priors=priors,
        position_observations=position_observations,
        inference_method=inference_method,
        seed=seed,
        **inference_config,
    )
    evaluation = metrics.evaluate_position_predictions(
        fit,
        window,
        credible_interval=credible_interval,
        position_variable_names=("x_prediction", "y_prediction"),
    )
    computation_time_seconds = time.perf_counter() - computation_started

    if inference_method == "vi":
        converged = bayesian_model.variational_converged(fit)
        reporting.print_variational_diagnostics(fit, converged=converged)
    else:
        converged = None
        print("\nMCMC diagnostics:")
        print(fit.diagnose())

    print("\nPosterior parameter summary:")
    print(reporting.posterior_parameter_summary(fit, bayesian_model.PARAMETER_NAMES))
    parameter_rows = [
        (
            "Speed [m/s]",
            f"{np.median(reporting.posterior_variable_samples(fit, 'speed')):.3f}",
        ),
        (
            "Initial heading [rad]",
            f"{np.median(reporting.posterior_variable_samples(fit, 'heading_initial')):.4f}",
        ),
        (
            "Turn rate [rad/s]",
            f"{np.median(reporting.posterior_variable_samples(fit, 'turn_rate')):.6f}",
        ),
        (
            "Observation noise [m]",
            f"{np.median(reporting.posterior_variable_samples(fit, 'sigma_position_observation')):.3f}",
        ),
    ]
    print("\nPosterior parameter medians:")
    print(reporting.format_aligned_rows(parameter_rows))
    print("Only position observations were used for fitting.")
    metrics.print_position_evaluation(
        evaluation,
        computation_time_seconds=computation_time_seconds,
    )

    prediction_plotting.plot_prediction(
        window,
        fit,
        state_prediction_variable_names=("x_prediction", "y_prediction"),
        observed_position_values=(
            position_observations.x_meters,
            position_observations.y_meters,
        ),
        observed_trajectory_label=(
            "Für Fit verwendete verrauschte Beobachtungen"
            if position_observations.additional_noise_std_m > 0
            else "Für Fit verwendete Beobachtungen"
        ),
        additional_position_noise_std_m=(position_observations.additional_noise_std_m),
        coordinate_mode=plot_coordinate_mode,
        forecast_label="Median der parametrischen CTRV-Trajektorie",
        sample_label="Trajektorien aus Posterior-Parameterziehungen",
        show_time_labels=show_time_labels,
    )
    return {
        "fit": fit,
        "window": window,
        "position_observations": position_observations,
        "stan_data": stan_data,
        "evaluation": evaluation,
        "converged": converged,
    }
