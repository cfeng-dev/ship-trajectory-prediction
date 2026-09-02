"""Single-window workflow for the parametric Bayesian CTRV model."""

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

import bayestraj.forecasting.bayesian_ctrv as forecasting
import bayestraj.forecasting.inference as inference
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.models.sequential_monte_carlo_ctrv as smc_model
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
    rbpf_config: bayesian_model.SequentialCTRVFilterConfig,
    smc_config: smc_model.SequentialMonteCarloCTRVConfig,
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
    inference_mode, inference_method = inference.normalize_inference_method(
        inference_method,
        online_inference_methods=inference.CTRV_ONLINE_INFERENCE_METHODS,
    )
    online_mode = inference_mode == "online"
    if online_mode:
        if inference_method == "rbpf":
            if not isinstance(rbpf_config, bayesian_model.SequentialCTRVFilterConfig):
                raise TypeError(
                    "rbpf_config must be a SequentialCTRVFilterConfig instance."
                )
            filter_type = bayesian_model.SequentialBayesianCTRVFilter
            particle_filter_config = rbpf_config
        else:
            if not isinstance(smc_config, smc_model.SequentialMonteCarloCTRVConfig):
                raise TypeError(
                    "smc_config must be a SequentialMonteCarloCTRVConfig instance."
                )
            filter_type = smc_model.SequentialMonteCarloCTRVFilter
            particle_filter_config = smc_config
        inference_config = {}
    else:
        inference_method, inference_config = inference.select_inference_config(
            inference_mode,
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
        position_noise_std_m=position_noise_std_m,
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
                (
                    "dynamic speed/turn-rate states, heading, "
                    "sigma_position_observation, sigma_speed_process, "
                    "sigma_turn_rate_process"
                ),
            ),
            (
                "Position transition",
                "deterministic conditional on latent speed and turn rate",
            ),
            ("Inference mode", inference_mode.upper()),
            ("Inference method", inference_method.upper()),
            ("Inference seed", seed),
            (
                "Position noise",
                (
                    f"{position_noise_std_m:g} m (seed={position_noise_seed})"
                    if position_noise_std_m > 0
                    else "disabled"
                ),
            ),
            (
                "Speed prior",
                (
                    "Half-Normal; P(speed > "
                    f"{priors.speed_prior_upper_mps:g} m/s)="
                    f"{priors.speed_prior_tail_probability:g}"
                ),
            ),
            (
                "Turn-rate prior",
                (
                    "Normal(0, scale); P(|heading change over "
                    f"{priors.turn_rate_prior_reference_interval_seconds:g} s| > "
                    f"{priors.turn_rate_prior_abs_heading_change_deg:g} deg)="
                    f"{priors.turn_rate_prior_tail_probability:g}"
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
            (
                "Speed-process prior",
                "Exponential("
                f"rate={priors.sigma_speed_process_prior_rate:.4f} s/m; "
                "P(sigma > "
                f"{priors.sigma_speed_process_prior_upper_mps:g} m/s)="
                f"{priors.sigma_speed_process_prior_tail_probability:g})",
            ),
            (
                "Turn-process prior",
                "Exponential("
                f"rate={priors.sigma_turn_rate_process_prior_rate:.4f} s/rad; "
                "P(sigma > "
                f"{priors.sigma_turn_rate_process_prior_upper_deg_s:g} deg/s)="
                f"{priors.sigma_turn_rate_process_prior_tail_probability:g})",
            ),
            (
                "Process reference",
                f"{bayesian_model.PROCESS_REFERENCE_INTERVAL_SECONDS:g} s",
            ),
            ("Forecast horizon", f"{forecast_horizon_seconds:g} s"),
            ("Plot coordinates", plot_coordinate_mode),
            ("Prior status", "ship-independent domain assumptions"),
        ],
    )

    computation_started = time.perf_counter()
    online_filter = None
    if online_mode:
        online_filter = filter_type.initialize(
            position_observations.time_seconds,
            position_observations.x_meters,
            position_observations.y_meters,
            priors=priors,
            config=particle_filter_config,
            seed=seed,
        )
        fit = online_filter.forecast(
            stan_data["time_prediction"],
            seed=1_000_000 + seed,
        )
    else:
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
    elif inference_method == "mcmc":
        converged = None
        print("\nMCMC diagnostics:")
        print(fit.diagnose())
    else:
        converged = None
        latest_update_ess = online_filter.last_effective_sample_size
        if latest_update_ess is None:
            latest_update_ess = online_filter.effective_sample_size
        print(f"\n{inference_method.upper()} diagnostics:")
        print(
            reporting.format_aligned_rows(
                [
                    (
                        "Processed observations",
                        online_filter.processed_observation_count,
                    ),
                    (
                        "Latest-update ESS",
                        f"{latest_update_ess:.0f}/"
                        f"{particle_filter_config.particle_count}",
                    ),
                    ("Resamples", online_filter.resample_count),
                ]
            )
        )

    print("\nPosterior parameter summary:")
    print(reporting.posterior_parameter_summary(fit, bayesian_model.PARAMETER_NAMES))
    parameter_rows = [
        (
            "Speed at origin [m/s]",
            f"{np.median(reporting.posterior_variable_samples(fit, 'speed_at_origin')):.3f}",
        ),
        (
            "Heading at origin [rad]",
            f"{np.median(reporting.posterior_variable_samples(fit, 'heading_at_origin')):.4f}",
        ),
        (
            "Turn rate at origin [rad/s]",
            f"{np.median(reporting.posterior_variable_samples(fit, 'turn_rate_at_origin')):.6f}",
        ),
        (
            "Observation noise [m]",
            f"{np.median(reporting.posterior_variable_samples(fit, 'sigma_position_observation')):.3f}",
        ),
        (
            "Speed process [m/s]",
            f"{np.median(reporting.posterior_variable_samples(fit, 'sigma_speed_process')):.3f}",
        ),
        (
            "Turn-rate process [rad/s]",
            f"{np.median(reporting.posterior_variable_samples(fit, 'sigma_turn_rate_process')):.6f}",
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
            if position_observations.position_noise_std_m > 0
            else "Für Fit verwendete Beobachtungen"
        ),
        position_noise_std_m=position_observations.position_noise_std_m,
        coordinate_mode=plot_coordinate_mode,
        forecast_label="Median der latenten CTRV-Trajektorie",
        sample_label="Latente CTRV-Trajektorienprognosen",
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
