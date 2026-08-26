"""Single-window prediction workflow for the Bayesian Position Model."""

from __future__ import annotations

import time

import ship_trajectory_prediction.forecasting.inference as inference
import ship_trajectory_prediction.models.bayesian_position_model as position_model
import ship_trajectory_prediction.observations.io as observations_io
import ship_trajectory_prediction.observations.window as observation_window
import ship_trajectory_prediction.validation.metrics as metrics
import ship_trajectory_prediction.validation.prediction_plotting as prediction_plotting
import ship_trajectory_prediction.validation.reporting as reporting


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
):
    """Fit and evaluate one configured local position-model forecast."""
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
        additional_noise_std_m=position_noise_std_m,
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
                ("Model", "Bayesian Position Model"),
                ("Run ID", experiment.run_id),
                ("Observed positions", window.observation_count),
                ("Prediction positions", window.prediction_count),
                ("Sampling interval", f"{_sampling_interval(window):g} s"),
                ("Inference method", inference_method.upper()),
                ("Position-only input", "yes"),
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
            "x_observation_prediction",
            "y_observation_prediction",
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
            "x_observation_prediction",
            "y_observation_prediction",
        ),
        observed_position_values=(
            position_observations.x_meters,
            position_observations.y_meters,
        ),
        observed_trajectory_label=(
            "Verrauschte Beobachtungen"
            if position_observations.additional_noise_std_m > 0
            else "Beobachtungen"
        ),
        additional_position_noise_std_m=position_noise_std_m,
        coordinate_mode=plot_coordinate_mode,
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
