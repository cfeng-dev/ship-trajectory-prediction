"""Single-window workflow for the latent Bayesian position model."""

from __future__ import annotations

import time

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
    inference_method, inference_config = inference.select_inference_config(
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
                ("Sampling interval", f"{_sampling_interval(window):g} s"),
                ("Inference method", inference_method.upper()),
                ("Position-only input", "yes"),
                (
                    "Fixed observation noise",
                    f"{position_observations.observation_noise_std_m:g} m per axis",
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


def _sampling_interval(window) -> float:
    """Return the displacement-model interval for reporting."""
    observed_times = window.time_seconds[window.observed_slice]
    return float(observed_times[-1] - observed_times[-2])
