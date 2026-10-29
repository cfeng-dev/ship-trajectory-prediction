"""Single-window workflow for the parametric Bayesian CTRV model."""

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import bayestraj.forecasting.bayesian_ctrv as forecasting
import bayestraj.inference.br24_radar_smc as radar_inference
import bayestraj.inference.cmdstan as cmdstan_inference
import bayestraj.inference.configuration as inference
import bayestraj.inference.ctrv_cmdstan as batch_inference
import bayestraj.inference.ctrv_sequential_vi as sequential_vi_model
import bayestraj.inference.ctrv_smc as smc_model
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.observations.br24 as radar_observations
import bayestraj.observations.io as observations_io
import bayestraj.observations.window as observation_window
import bayestraj.validation.metrics as metrics
import bayestraj.validation.prediction_plotting as prediction_plotting
import bayestraj.validation.reporting as reporting


def _resolve_inference_configs(
    vi_config,
    mcmc_config,
    smc_config,
    sequential_vi_config=None,
):
    """Return explicit inference settings or the source-owned defaults."""
    return (
        inference.create_default_vi_config() if vi_config is None else vi_config,
        inference.create_default_mcmc_config() if mcmc_config is None else mcmc_config,
        inference.create_default_ctrv_smc_config()
        if smc_config is None
        else smc_config,
        inference.create_default_sequential_vi_config()
        if sequential_vi_config is None
        else sequential_vi_config,
    )


def run_bayesian_ctrv_prediction(
    *,
    data_file,
    experiment: forecasting.ExperimentConfig,
    priors: bayesian_model.BayesianCTRVPriors,
    vi_config: Mapping[str, Any] | None = None,
    mcmc_config: Mapping[str, Any] | None = None,
    smc_config: smc_model.SequentialMonteCarloCTRVConfig | None = None,
    sequential_vi_config: sequential_vi_model.SequentialVIConfig | None = None,
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
    sample_trajectories_per_forecast=prediction_plotting.DEFAULT_SAMPLE_TRAJECTORIES,
):
    """Fit and evaluate one constant-parameter Bayesian CTRV prediction."""
    vi_config, mcmc_config, smc_config, sequential_vi_config = (
        _resolve_inference_configs(
            vi_config,
            mcmc_config,
            smc_config,
            sequential_vi_config,
        )
    )
    inference_mode, inference_method = inference.normalize_inference_method(
        inference_method,
        online_inference_methods=inference.CTRV_ONLINE_INFERENCE_METHODS,
    )
    online_mode = inference_mode == "online"
    if online_mode:
        if inference_method == "smc":
            if not isinstance(smc_config, smc_model.SequentialMonteCarloCTRVConfig):
                raise TypeError(
                    "smc_config must be a SequentialMonteCarloCTRVConfig instance."
                )
            filter_type = smc_model.SequentialMonteCarloCTRVFilter
            particle_filter_config = smc_config
        else:
            if not isinstance(
                sequential_vi_config,
                sequential_vi_model.SequentialVIConfig,
            ):
                raise TypeError(
                    "sequential_vi_config must be a SequentialVIConfig instance."
                )
            filter_type = sequential_vi_model.SequentialCTRVVI
            particle_filter_config = sequential_vi_config
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
    if Path(data_file).is_file() and radar_observations.is_br24_tracking_input(
        data_file
    ):
        return _run_radar_prediction(
            data_file=data_file,
            experiment=experiment,
            priors=priors,
            inference_method=inference_method,
            vi_config=inference_config if inference_method == "vi" else vi_config,
            mcmc_config=mcmc_config,
            smc_config=smc_config,
            sequential_vi_config=sequential_vi_config,
            seed=seed,
            position_noise_std_m=position_noise_std_m,
            position_noise_seed=position_noise_seed,
            credible_interval=credible_interval,
            plot_coordinate_mode=plot_coordinate_mode,
            show_time_labels=show_time_labels,
            sample_trajectories_per_forecast=sample_trajectories_per_forecast,
        )
    trajectory_data = observations_io.read_processed_trajectory(data_file)
    window = observation_window.prepare_processed_trajectory_window(
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
                    "Normal(0, scale); P(|turn rate| > "
                    f"{priors.turn_rate_prior_abs_rate_deg_s:g} deg/s)="
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
        forecast_seed = (
            seed if inference_method == "sequential_vi" else 1_000_000 + seed
        )
        fit = online_filter.forecast(stan_data["time_prediction"], seed=forecast_seed)
    else:
        fit = batch_inference.fit_bayesian_ctrv_model(
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
        converged = cmdstan_inference.variational_converged(fit)
        reporting.print_variational_diagnostics(fit, converged=converged)
    elif inference_method == "mcmc":
        converged = None
        print("\nMCMC diagnostics:")
        print(fit.diagnose())
    elif inference_method == "smc":
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
    else:
        converged = online_filter.bootstrap_converged and all(
            update.converged for update in online_filter.updates
        )
        update_runtime = sum(update.runtime_seconds for update in online_filter.updates)
        print("\nSequential VI diagnostics:")
        print(
            reporting.format_aligned_rows(
                [
                    (
                        "Processed observations",
                        online_filter.processed_observation_count,
                    ),
                    (
                        "Bootstrap runtime [s]",
                        f"{online_filter.bootstrap_runtime_seconds:.3f}",
                    ),
                    ("Update runtime [s]", f"{update_runtime:.3f}"),
                    ("All VI steps converged", converged),
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
        title=f"Bayesian-CTRV ({inference_method.upper()})",
        forecast_label="Median der latenten CTRV-Trajektorie",
        sample_label="Latente CTRV-Trajektorienprognosen",
        max_sample_trajectories=sample_trajectories_per_forecast,
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


def _run_radar_prediction(
    *,
    data_file,
    experiment,
    priors,
    inference_method,
    vi_config,
    mcmc_config,
    smc_config,
    sequential_vi_config,
    seed,
    position_noise_std_m,
    position_noise_seed,
    credible_interval,
    plot_coordinate_mode,
    show_time_labels,
    sample_trajectories_per_forecast,
):
    """Forecast from past raw cycles, reading optional labels only afterward."""
    if experiment.start_index != 0:
        raise ValueError("BR24 radar tracking starts at the lowest cycle counter.")
    if position_noise_std_m != 0.0:
        raise ValueError(
            "BR24 radar prediction does not support artificial position noise."
        )
    cycles = radar_observations.read_br24_tracking_input(data_file)
    observed_count = experiment.observation_count
    stop = observed_count + experiment.prediction_count
    if stop > len(cycles):
        raise ValueError(
            "The requested observations and predictions exceed BR24 cycles."
        )
    if not cycles[0].detections:
        raise ValueError("The first BR24 radar cycle needs a detection to initialize.")
    started = time.perf_counter()
    tracker = radar_inference.BR24BayesianRadarTracker.initialize(
        cycles[0],
        initial_detection_id=cycles[0].detections[0].detection_id,
        priors=priors,
        inference_method=inference_method,
        observation_count=observed_count,
        config=smc_config,
        vi_config=vi_config,
        mcmc_config=mcmc_config,
        sequential_vi_config=sequential_vi_config,
        seed=seed,
    )
    detection_ids = [cycles[0].detections[0].detection_id]
    tracked = [tracker.estimated_position]
    for cycle in cycles[1:observed_count]:
        step = tracker.process_cycle(cycle)
        detection_ids.append(step.detection_id)
        tracked.append(tracker.estimated_position)
    future_times = np.array(
        [cycle.time_seconds for cycle in cycles[observed_count:stop]]
    )
    fit = tracker.filter.forecast(future_times, seed=1_000_000 + seed)
    runtime = time.perf_counter() - started

    # These arrays contain real fixes only; missing-cycle placeholders are masked.
    positions = np.zeros((observed_count, 2))
    covariances = np.zeros((observed_count, 2, 2))
    mask = np.zeros(observed_count, dtype=bool)
    for index, (cycle, detection_id) in enumerate(
        zip(cycles, detection_ids, strict=False)
    ):
        if detection_id is None:
            continue
        detection = next(
            item for item in cycle.detections if item.detection_id == detection_id
        )
        positions[index] = radar_observations.br24_measurement_position(
            sensor_position=cycle.sensor_position,
            range_m=detection.range_m,
            azimuth_rad=detection.azimuth_rad,
        )
        covariances[index] = radar_observations.br24_measurement_covariance(
            range_m=detection.range_m,
            azimuth_rad=detection.azimuth_rad,
            std_range_m=detection.std_range_m,
            std_azimuth_rad=detection.std_azimuth_rad,
        )
        mask[index] = True
    times = np.array([cycle.time_seconds for cycle in cycles[:stop]])
    observations = bayesian_model.PositionObservations(
        times[:observed_count],
        positions[:, 0],
        positions[:, 1],
        0.0,
        position_noise_seed,
        measurement_covariances=covariances,
        observed_mask=mask,
    )
    reference = radar_observations.read_br24_tracking_reference(data_file)
    display_positions = np.vstack(
        (
            tracked,
            np.repeat(
                np.asarray(tracked[-1])[None, :], experiment.prediction_count, axis=0
            ),
        )
    )
    window = observation_window.TrajectoryWindowData(
        timestamps=pd.DatetimeIndex(pd.to_datetime(times, unit="s", utc=True)),
        time_seconds=times,
        x_meters=reference.x_meters[:stop]
        if reference is not None
        else display_positions[:, 0],
        y_meters=reference.y_meters[:stop]
        if reference is not None
        else display_positions[:, 1],
        reference_longitude=None,
        reference_latitude=None,
        gps_speed_mps=np.full(stop, np.nan),
        observation_count=observed_count,
    )
    evaluation = None
    print(f"Bayesian CTRV BR24 prediction ({inference_method.upper()})")
    print(f"Radar state: {tracker.fit_status}; processed cycles: {observed_count}")
    converged = getattr(tracker.filter, "converged", None)
    mcmc_diagnostics_ok = None
    if tracker.is_prior_predictive:
        print("Prior-predictive startup: no posterior fit is available yet.")
    elif inference_method == "vi":
        print(f"VI convergence: {converged}")
    elif inference_method == "sequential_vi":
        print(f"Sequential VI convergence: {converged}")
    elif inference_method == "mcmc":
        diagnosis = tracker.filter.last_fit.diagnose()
        mcmc_diagnostics_ok = "no problems detected" in diagnosis.lower()
        print("MCMC sampler diagnostics:")
        print(diagnosis)
    print(reporting.posterior_parameter_summary(fit, bayesian_model.PARAMETER_NAMES))
    if reference is not None:
        evaluation = metrics.evaluate_position_predictions(
            fit,
            window,
            credible_interval=credible_interval,
            position_variable_names=("x_prediction", "y_prediction"),
        )
        metrics.print_position_evaluation(evaluation, computation_time_seconds=runtime)
    prediction_plotting.plot_prediction(
        window,
        fit,
        plot_mode="evaluation" if reference is not None else "operational",
        state_prediction_variable_names=("x_prediction", "y_prediction"),
        observed_position_values=(np.asarray(tracked)[:, 0], np.asarray(tracked)[:, 1]),
        observed_trajectory_label="Bayesian CTRV track",
        position_noise_std_m=0.0,
        coordinate_mode=plot_coordinate_mode,
        title=f"Bayesian-CTRV ({inference_method.upper()})",
        forecast_label="Median der latenten CTRV-Trajektorie",
        sample_label="Latente CTRV-Trajektorienprognosen",
        max_sample_trajectories=sample_trajectories_per_forecast,
        show_time_labels=show_time_labels,
    )
    return dict(
        fit=fit,
        window=window,
        position_observations=observations,
        stan_data=None,
        evaluation=evaluation,
        converged=converged,
        mcmc_diagnostics_ok=mcmc_diagnostics_ok,
        tracker=tracker,
        is_prior_predictive=tracker.is_prior_predictive,
    )
