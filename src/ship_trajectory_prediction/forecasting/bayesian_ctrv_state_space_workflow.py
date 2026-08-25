"""Single-window Bayesian CTRV state-space prediction workflow."""

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

import ship_trajectory_prediction.forecasting.bayesian_ctrv_state_space as forecasting
import ship_trajectory_prediction.forecasting.inference as inference
import ship_trajectory_prediction.models.bayesian_ctrv_state_space as bayesian_model
import ship_trajectory_prediction.observations.io as observations_io
import ship_trajectory_prediction.observations.window as observation_window
import ship_trajectory_prediction.validation.metrics as metrics
import ship_trajectory_prediction.validation.prediction_plotting as prediction_plotting
import ship_trajectory_prediction.validation.reporting as reporting


def run_fully_bayesian_ctrv_prediction(
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
) -> None:
    """Fit and evaluate one fully Bayesian CTRV prediction."""
    return run_bayesian_ctrv_prediction(
        data_file=data_file,
        experiment=experiment,
        priors=priors,
        vi_config=vi_config,
        mcmc_config=mcmc_config,
        fullrank_grad_samples=fullrank_grad_samples,
        credible_interval=credible_interval,
        inference_method=inference_method,
        vi_algorithm=vi_algorithm,
        seed=seed,
        position_noise_std_m=position_noise_std_m,
        position_noise_seed=position_noise_seed,
        require_converged=require_converged,
        plot_coordinate_mode=plot_coordinate_mode,
        model_label="Fully Bayesian CTRV",
        stan_data_builder=bayesian_model.build_stan_data,
        fit_model=bayesian_model.fit_bayesian_ctrv_model,
        build_motion_setup_rows=_fully_bayesian_motion_setup_rows,
        build_motion_state_rows=_fully_bayesian_motion_state_rows,
    )


def run_bayesian_ctrv_prediction(
    *,
    data_file,
    experiment,
    priors,
    vi_config,
    mcmc_config,
    fullrank_grad_samples,
    credible_interval,
    inference_method,
    vi_algorithm,
    seed,
    position_noise_std_m,
    position_noise_seed,
    require_converged,
    plot_coordinate_mode,
    model_label,
    stan_data_builder,
    fit_model,
    build_motion_setup_rows,
    build_motion_state_rows,
    noise_parameter_names=bayesian_model.NOISE_PARAMETER_NAMES,
):
    """Run the shared single-window fitting and evaluation workflow."""
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
    stan_data = stan_data_builder(
        window,
        priors=priors,
        position_observations=position_observations,
    )
    forecast_horizon_seconds = float(
        stan_data["time_prediction"][-1] - stan_data["time_observed"][-1]
    )
    inference_rows = [
        ("Model", model_label),
        ("Inference method", inference_method.upper()),
    ]
    inference_rows.extend(build_motion_setup_rows(stan_data))
    if inference_method == "vi":
        inference_rows.append(("VI algorithm", inference_config["algorithm"]))
    else:
        inference_rows.extend(
            [
                ("MCMC chains", inference_config["chains"]),
                ("MCMC parallel chains", inference_config["parallel_chains"]),
                ("MCMC warmup per chain", inference_config["iter_warmup"]),
                ("MCMC samples per chain", inference_config["iter_sampling"]),
            ]
        )
    _print_ctrv_setup(
        model_label=model_label,
        window=window,
        inference_rows=inference_rows,
        inference_seed=seed,
        priors=priors,
        position_observations=position_observations,
        forecast_horizon_seconds=forecast_horizon_seconds,
        plot_coordinate_mode=plot_coordinate_mode,
        data_file=data_file,
        run_id=experiment.run_id,
    )

    computation_started = time.perf_counter()
    fit = fit_model(
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
            "x_state_prediction",
            "y_state_prediction",
        ),
    )
    computation_time_seconds = time.perf_counter() - computation_started
    if inference_method == "vi":
        converged = bayesian_model.variational_converged(fit)
        reporting.print_variational_diagnostics(fit, converged=converged)
        if not converged:
            print(
                "WARNING: Treat this posterior and its plot as preliminary; "
                "the VI convergence criterion was not met."
            )
    else:
        print("\nMCMC diagnostics:")
        print(fit.diagnose())

    print("\nPosterior parameter summary:")
    print(
        reporting.posterior_parameter_summary(
            fit,
            noise_parameter_names,
        )
    )

    speed_state = reporting.posterior_variable_samples(fit, "speed_state")
    heading_state = reporting.posterior_variable_samples(fit, "heading_state")
    state_rows = [
        (
            "Speed [m/s]",
            f"{np.median(speed_state[:, 0]):.3f} -> "
            f"{np.median(speed_state[:, -1]):.3f}",
        ),
        (
            "Heading [rad]",
            f"{np.median(heading_state[:, 0]):.4f} -> "
            f"{np.median(heading_state[:, -1]):.4f}",
        ),
    ]
    state_rows.extend(build_motion_state_rows(fit, stan_data))
    print("\nPosterior state medians:")
    print(reporting.format_aligned_rows(state_rows))
    print(
        "GPS speed was not used for fitting; it is retained only as an "
        "external post-fit plausibility reference."
    )

    metrics.print_position_evaluation(
        evaluation,
        computation_time_seconds=computation_time_seconds,
    )
    observed_trajectory_label = (
        "Verrauschte Beobachtungen"
        if position_observations.additional_noise_std_m > 0
        else "Beobachtungen"
    )
    prediction_plotting.plot_prediction(
        window,
        fit,
        state_prediction_variable_names=(
            "x_state_prediction",
            "y_state_prediction",
        ),
        observed_position_values=(
            position_observations.x_meters,
            position_observations.y_meters,
        ),
        observed_trajectory_label=observed_trajectory_label,
        additional_position_noise_std_m=(position_observations.additional_noise_std_m),
        coordinate_mode=plot_coordinate_mode,
    )


def _fully_bayesian_motion_setup_rows(stan_data):
    """Describe the inferred motion quantities of the fully Bayesian model."""
    del stan_data
    return [("Heading and turn rate", "latent posterior")]


def _fully_bayesian_motion_state_rows(fit, stan_data):
    """Return posterior turn-rate rows for the fully Bayesian model."""
    del stan_data
    turn_rate_state = reporting.posterior_variable_samples(fit, "turn_rate_state")
    return [
        (
            "Turn rate [rad/s]",
            f"{np.median(turn_rate_state[:, 0]):.5f} -> "
            f"{np.median(turn_rate_state[:, -1]):.5f}",
        )
    ]


def _print_ctrv_setup(
    *,
    model_label,
    window,
    inference_rows,
    inference_seed,
    priors,
    position_observations,
    forecast_horizon_seconds,
    plot_coordinate_mode,
    data_file,
    run_id,
):
    """Print the concise, reproducible setup for one Bayesian CTRV run."""
    noise_std_m = position_observations.additional_noise_std_m
    noise_description = (
        f"{noise_std_m:g} m (seed={position_observations.noise_seed})"
        if noise_std_m > 0
        else "disabled"
    )
    reporting.print_prediction_setup(
        f"{model_label} State-Space Prediction",
        data_file=data_file,
        run_id=run_id,
        window=window,
        extra_rows=[
            *inference_rows,
            ("Inference seed", inference_seed),
            ("Hidden synthetic noise", noise_description),
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
        ],
    )
