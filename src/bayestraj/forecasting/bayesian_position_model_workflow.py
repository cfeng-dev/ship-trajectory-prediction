"""Single-window workflow for the latent Bayesian position model."""

from __future__ import annotations

import time

import numpy as np

import bayestraj.forecasting.inference as inference
import bayestraj.models.bayesian_position_model as position_model
import bayestraj.observations.io as observations_io
import bayestraj.observations.window as observation_window
import bayestraj.validation.metrics as metrics
import bayestraj.validation.prediction_plotting as prediction_plotting
import bayestraj.validation.reporting as reporting


def run_bayesian_position_prediction(
    *,
    data_file,
    experiment,
    priors,
    vi_config,
    mcmc_config,
    fullrank_grad_samples,
    credible_interval,
    observation_count,
    inference_mode,
    inference_method,
    vi_algorithm,
    seed,
    position_noise_std_m,
    position_noise_seed,
    require_converged,
    plot_coordinate_mode,
    show_time_labels,
):
    """Fit and evaluate one latent-position measurement-error forecast."""
    inference_mode, inference_method = inference.normalize_inference_configuration(
        inference_mode,
        inference_method,
    )
    inference_method, inference_config = inference.select_inference_config(
        inference_mode,
        inference_method,
        vi_algorithm=vi_algorithm,
        require_converged=require_converged,
        vi_config=vi_config,
        mcmc_config=mcmc_config,
        fullrank_grad_samples=fullrank_grad_samples,
    )
    trajectory_data = observations_io.read_ship_data(
        data_file,
        run_id=experiment.run_id,
    )
    window = observation_window.prepare_trajectory_window(
        trajectory_data,
        observation_count=observation_count,
        prediction_count=experiment.prediction_count,
        start_index=experiment.start_index,
    )
    position_observations = position_model.simulate_position_observations(
        window,
        position_noise_std_m=position_noise_std_m,
        seed=position_noise_seed,
    )
    stan_data = position_model.build_stan_data(
        window,
        priors=priors,
        position_observations=position_observations,
    )
    print(
        reporting.format_aligned_rows(
            [
                (
                    "Model",
                    "Bayesian latent-position autoregressive measurement-error model",
                ),
                ("Run ID", experiment.run_id),
                ("Observed positions", window.observation_count),
                ("Prediction positions", window.prediction_count),
                ("Observation intervals", _time_interval_description(window)),
                (
                    "Motion reference interval",
                    f"{position_model.POSITION_MODEL_REFERENCE_INTERVAL_SECONDS:g} s",
                ),
                ("Inference mode", inference_mode.upper()),
                ("Inference method", inference_method.upper()),
                ("Position-only input", "yes"),
                (
                    "Log-scale-rate prior",
                    _displacement_scale_prior_description(priors),
                ),
                (
                    "Rotation-rate prior",
                    _rotation_rate_prior_description(priors),
                ),
                (
                    "Injected position noise",
                    f"{position_observations.position_noise_std_m:g} m per axis",
                ),
                (
                    "Observation-noise prior",
                    _noise_prior_description(
                        priors.sigma_position_observation_prior_upper_m,
                        priors.sigma_position_observation_prior_tail_probability,
                    ),
                ),
                (
                    "Motion-residual prior",
                    _noise_prior_description(
                        priors.sigma_motion_residual_prior_upper_m,
                        priors.sigma_motion_residual_prior_tail_probability,
                    )
                    + " at the reference interval",
                ),
            ]
        )
    )

    computation_started = time.perf_counter()
    fit = position_model.fit_bayesian_position_model(
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
        position_variable_names=(
            "x_model_prediction",
            "y_model_prediction",
        ),
    )
    computation_time_seconds = time.perf_counter() - computation_started
    if inference_method == "vi":
        converged = position_model.variational_converged(fit)
        reporting.print_variational_diagnostics(fit, converged=converged)
    else:
        print("\nMCMC diagnostics:")
        print(fit.diagnose())
    print("\nPosterior parameter summary:")
    print(
        reporting.posterior_parameter_summary(
            fit,
            position_model.PARAMETER_NAMES,
            credible_interval=credible_interval,
        )
    )
    _print_time_based_motion_posterior(fit)
    metrics.print_position_evaluation(
        evaluation,
        computation_time_seconds=computation_time_seconds,
    )
    prediction_plotting.plot_prediction(
        window,
        fit,
        state_prediction_variable_names=(
            "x_model_prediction",
            "y_model_prediction",
        ),
        observed_position_values=(
            position_observations.x_meters,
            position_observations.y_meters,
        ),
        observed_trajectory_label=(
            "Verrauschte Beobachtungen"
            if position_observations.position_noise_std_m > 0
            else "Beobachtungen"
        ),
        position_noise_std_m=position_noise_std_m,
        coordinate_mode=plot_coordinate_mode,
        forecast_label="Median der latenten Trajektorienprognose",
        sample_label="Latente Trajektorienprognosen",
        show_time_labels=show_time_labels,
    )
    return {
        "fit": fit,
        "window": window,
        "position_observations": position_observations,
        "stan_data": stan_data,
        "evaluation": evaluation,
    }


def _time_interval_description(window) -> str:
    """Describe the possibly irregular observed time intervals."""
    observed_times = window.time_seconds[window.observed_slice]
    intervals = np.diff(observed_times)
    return (
        f"min/median/max={np.min(intervals):g}/"
        f"{np.median(intervals):g}/{np.max(intervals):g} s"
    )


def _noise_prior_description(upper_m: float, tail_probability: float) -> str:
    """Describe one ship-independent exponential noise prior."""
    return f"Exponential; P(sigma > {upper_m:g} m)={tail_probability:g}"


def _print_time_based_motion_posterior(fit) -> None:
    """Print rate parameters and their reference-interval interpretation."""
    reference = position_model.POSITION_MODEL_REFERENCE_INTERVAL_SECONDS
    scale_rate = np.median(
        reporting.posterior_variable_samples(fit, "log_displacement_scale_rate")
    )
    rotation_rate = np.median(
        reporting.posterior_variable_samples(fit, "rotation_rate")
    )
    scale_at_reference = np.median(
        reporting.posterior_variable_samples(fit, "displacement_scale_at_reference")
    )
    rotation_at_reference_deg = np.rad2deg(
        np.median(
            reporting.posterior_variable_samples(fit, "rotation_angle_at_reference")
        )
    )
    print("\nTime-based motion interpretation:")
    print(
        reporting.format_aligned_rows(
            [
                ("Motion scale rate", f"{scale_rate:.6g} 1/s"),
                ("Rotation rate", f"{rotation_rate:.6g} rad/s"),
                (f"Scale over {reference:g} s", f"{scale_at_reference:.6g}"),
                (
                    f"Rotation over {reference:g} s",
                    f"{rotation_at_reference_deg:.6g} deg",
                ),
            ]
        )
    )


def _displacement_scale_prior_description(priors) -> str:
    """Describe the ship-independent log-scale-rate prior."""
    central_probability = 1.0 - priors.displacement_scale_prior_tail_probability
    factor = priors.displacement_scale_prior_factor
    reference = position_model.POSITION_MODEL_REFERENCE_INTERVAL_SECONDS
    return (
        f"{central_probability:.0%} reference factor between "
        f"{1.0 / factor:g}x and {factor:g}x over {reference:g} s"
    )


def _rotation_rate_prior_description(priors) -> str:
    """Describe the ship-independent rotation-rate prior."""
    central_probability = 1.0 - priors.rotation_angle_prior_tail_probability
    upper_deg = priors.rotation_angle_prior_abs_upper_deg
    reference = position_model.POSITION_MODEL_REFERENCE_INTERVAL_SECONDS
    return f"{central_probability:.0%} within +/-{upper_deg:g} deg over {reference:g} s"
