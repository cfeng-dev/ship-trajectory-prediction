"""Single-window Bayesian CTRV prediction workflow."""

from collections.abc import Mapping
from functools import partial
from time import perf_counter
from typing import Any

import numpy as np

import ship_trajectory_prediction.forecasting.bayesian_ctrv as forecasting
import ship_trajectory_prediction.models.bayesian_ctrv as bayesian_model
import ship_trajectory_prediction.models.hybrid_bayesian_ctrv as hybrid_model
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
    return _run_bayesian_ctrv_prediction(
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
    )


def run_hybrid_bayesian_ctrv_prediction(
    *,
    data_file,
    experiment: forecasting.ExperimentConfig,
    priors: hybrid_model.HybridBayesianCTRVPriors,
    hybrid_config: hybrid_model.HybridBayesianCTRVConfig,
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
    """Fit and evaluate one hybrid Bayesian CTRV prediction."""
    return _run_bayesian_ctrv_prediction(
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
        model_label="Hybrid Bayesian CTRV",
        stan_data_builder=partial(
            hybrid_model.build_stan_data,
            hybrid_config=hybrid_config,
        ),
        fit_model=partial(
            hybrid_model.fit_hybrid_bayesian_ctrv_model,
            hybrid_config=hybrid_config,
        ),
        terminal_motion_history_seconds=(hybrid_config.final_motion_history_seconds),
        noise_parameter_names=hybrid_model.NOISE_PARAMETER_NAMES,
        has_latent_turn_rate=False,
    )


def _run_bayesian_ctrv_prediction(
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
    terminal_motion_history_seconds=None,
    noise_parameter_names=bayesian_model.NOISE_PARAMETER_NAMES,
    has_latent_turn_rate=True,
):
    """Run the shared single-window fitting and evaluation workflow."""
    inference_method, inference_config = (
        forecasting.select_bayesian_ctrv_inference_config(
            inference_method,
            vi_algorithm=vi_algorithm,
            require_converged=require_converged,
            vi_config=vi_config,
            mcmc_config=mcmc_config,
            fullrank_grad_samples=fullrank_grad_samples,
        )
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
    if terminal_motion_history_seconds is not None:
        inference_rows.extend(
            [
                (
                    "Deterministic terminal heading",
                    f"{stan_data['heading_final']:.4f} rad "
                    f"({np.degrees(stan_data['heading_final']):.1f} deg)",
                ),
                (
                    "Deterministic terminal turn rate",
                    f"{stan_data['turn_rate_final']:.5f} rad/s",
                ),
                (
                    "Terminal-motion history",
                    f"{terminal_motion_history_seconds:g} s",
                ),
            ]
        )
    else:
        inference_rows.append(("Terminal heading and turn rate", "latent posterior"))
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
        position_observations=position_observations,
        forecast_horizon_seconds=forecast_horizon_seconds,
        plot_coordinate_mode=plot_coordinate_mode,
        data_file=data_file,
        run_id=experiment.run_id,
    )

    fit_started = perf_counter()
    fit = fit_model(
        window,
        priors=priors,
        position_observations=position_observations,
        inference_method=inference_method,
        seed=seed,
        **inference_config,
    )
    fit_and_forecast_runtime_seconds = perf_counter() - fit_started
    print(f"\nModel fit and forecast runtime: {fit_and_forecast_runtime_seconds:.2f} s")
    if inference_method == "vi":
        converged = bayesian_model.variational_converged(fit)
        reporting.print_variational_diagnostics(fit)
        print(f"CmdStan convergence criterion met: {converged}")
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
    print("\nPosterior state medians:")
    print(
        "Speed [m/s]       : "
        f"{np.median(speed_state[:, 0]):.3f} -> "
        f"{np.median(speed_state[:, -1]):.3f}"
    )
    print(
        "Heading [rad]     : "
        f"{np.median(heading_state[:, 0]):.4f} -> "
        f"{np.median(heading_state[:, -1]):.4f}"
    )
    if has_latent_turn_rate:
        turn_rate_state = reporting.posterior_variable_samples(fit, "turn_rate_state")
        print(
            "Turn rate [rad/s] : "
            f"{np.median(turn_rate_state[:, 0]):.5f} -> "
            f"{np.median(turn_rate_state[:, -1]):.5f}"
        )
    else:
        print(f"Turn rate [rad/s] : fixed at {stan_data['turn_rate_final']:.5f}")
    print(
        "GPS speed was not used for fitting; it is retained only as an "
        "external post-fit plausibility reference."
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
    metrics.print_position_evaluation(evaluation)
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


def _print_ctrv_setup(
    *,
    model_label,
    window,
    inference_rows,
    inference_seed,
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
            ("Additional position noise", noise_description),
            ("Forecast horizon", f"{forecast_horizon_seconds:g} s"),
            ("Plot coordinates", plot_coordinate_mode),
        ],
    )
