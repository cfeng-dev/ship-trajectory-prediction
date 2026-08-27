"""Rolling evaluation for the latent Bayesian position model."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bayestraj.forecasting.inference as inference
import bayestraj.models.bayesian_position_model as position_model
import bayestraj.observations.coordinates as coordinates
import bayestraj.observations.io as observations_io
import bayestraj.observations.window as observation_window
import bayestraj.validation.metrics as metrics
import bayestraj.validation.prediction_plotting as prediction_plotting
import bayestraj.validation.reporting as reporting
import bayestraj.validation.rolling as rolling


@dataclass(frozen=True, slots=True)
class _RollingPosteriorGroup:
    forecast_origin_x: float
    forecast_origin_y: float
    x_samples: np.ndarray
    y_samples: np.ndarray
    forecast_time_seconds: np.ndarray


def run_bayesian_position_evaluation(
    *,
    data_file,
    experiment,
    priors,
    vi_config,
    mcmc_config,
    rbpf_config,
    fullrank_grad_samples,
    credible_interval,
    sample_trajectories_per_forecast,
    options,
    show_time_labels=False,
):
    """Evaluate the latent-position model over leakage-free rolling windows."""
    configured_experiment = dataclasses.replace(
        experiment,
        window_mode=options.window_mode,
        observation_count=options.observation_count,
        prediction_count=options.prediction_count,
        stride=options.stride,
        inference_mode=options.inference_mode,
        inference_method=options.inference_method,
        inference_seed=options.inference_seed,
        position_noise_std_m=options.position_noise_std_m,
        position_noise_seed=options.position_noise_seed,
    )
    online_mode = configured_experiment.inference_mode == "online"
    if online_mode and not isinstance(
        rbpf_config,
        position_model.SequentialPositionFilterConfig,
    ):
        raise TypeError(
            "rbpf_config must be a SequentialPositionFilterConfig instance."
        )
    if online_mode:
        inference_method = configured_experiment.inference_method
        inference_config = {}
    else:
        inference_method, inference_config = inference.select_inference_config(
            configured_experiment.inference_mode,
            configured_experiment.inference_method,
            vi_algorithm=options.vi_algorithm,
            require_converged=options.require_converged,
            vi_config=vi_config,
            mcmc_config=mcmc_config,
            fullrank_grad_samples=fullrank_grad_samples,
        )
    trajectory_data = (
        observations_io.read_ship_data(
            data_file,
            run_id=configured_experiment.run_id,
        )
        .sort_values("time")
        .reset_index(drop=True)
    )
    if trajectory_data.empty:
        raise ValueError(
            f"No trajectory rows found for run_id={configured_experiment.run_id}."
        )
    if online_mode:
        windows = rolling.build_online_forecast_specs(
            len(trajectory_data),
            initial_observation_count=configured_experiment.observation_count,
            prediction_count=configured_experiment.prediction_count,
            stride=configured_experiment.stride,
        )
    else:
        windows = rolling.build_rolling_window_specs(
            len(trajectory_data),
            initial_observation_count=configured_experiment.observation_count,
            prediction_count=configured_experiment.prediction_count,
            stride=configured_experiment.stride,
            window_mode=configured_experiment.window_mode,
        )
    if options.max_windows is not None:
        if isinstance(options.max_windows, bool) or options.max_windows < 1:
            raise ValueError("max_windows must be a positive integer or None.")
        windows = windows[: options.max_windows]

    route_x, route_y, longitude, latitude = _prepare_route_coordinates(trajectory_data)
    route_time_seconds = _route_time_seconds(trajectory_data)
    route_noise_x, route_noise_y = _simulate_route_position_noise(
        len(trajectory_data),
        position_noise_std_m=configured_experiment.position_noise_std_m,
        seed=configured_experiment.position_noise_seed,
    )
    setup_rows = [
        (
            "Model",
            "Bayesian latent-position autoregressive measurement-error model",
        ),
        ("Run ID", configured_experiment.run_id),
        ("Inference mode", configured_experiment.inference_mode.upper()),
        ("Inference method", inference_method.upper()),
        ("Prediction positions", configured_experiment.prediction_count),
        ("Forecast origins", len(windows)),
        (
            "Log-scale-rate prior",
            _displacement_scale_prior_description(priors),
        ),
        (
            "Rotation-rate prior",
            _rotation_rate_prior_description(priors),
        ),
        ("Route intervals", _time_interval_description(route_time_seconds)),
        (
            "Motion reference interval",
            f"{position_model.POSITION_MODEL_REFERENCE_INTERVAL_SECONDS:g} s",
        ),
        (
            "Injected position noise",
            f"{configured_experiment.position_noise_std_m:g} m per axis",
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
    if online_mode:
        setup_rows.insert(
            4,
            ("Initial observations", configured_experiment.observation_count),
        )
        setup_rows.extend(
            [
                ("Particles", rbpf_config.particle_count),
                ("Posterior draws", rbpf_config.posterior_draw_count),
                (
                    "Resampling ESS",
                    f"{rbpf_config.resample_ess_fraction:.0%} of particles",
                ),
                (
                    "Parameter rejuvenation",
                    f"Liu-West scale {rbpf_config.rejuvenation_scale:g}",
                ),
            ]
        )
    else:
        setup_rows[4:4] = [
            ("Window mode", configured_experiment.window_mode.upper()),
            ("Observations/window", configured_experiment.observation_count),
        ]
    print(reporting.format_aligned_rows(setup_rows))

    prediction_tables = []
    posterior_groups = []
    online_filter = None
    noisy_route_x = route_x + route_noise_x
    noisy_route_y = route_y + route_noise_y
    for number, specification in enumerate(windows, start=1):
        window_seed = configured_experiment.inference_seed + specification.window_index
        print(
            f"{'Forecast' if online_mode else 'Window'} {number}/{len(windows)}: "
            f"observations={specification.observation_count}, "
            f"predictions={specification.prediction_count}"
        )
        window = observation_window.prepare_trajectory_window(
            trajectory_data,
            observation_count=specification.observation_count,
            prediction_count=specification.prediction_count,
            start_index=specification.start_index,
        )
        position_observations = _build_window_position_observations(
            window,
            route_start_index=specification.start_index,
            route_noise_x=route_noise_x,
            route_noise_y=route_noise_y,
            position_noise_std_m=(configured_experiment.position_noise_std_m),
            noise_seed=configured_experiment.position_noise_seed,
        )
        runtime_started = time.perf_counter()
        if online_mode:
            online_filter = _advance_online_filter(
                online_filter,
                specification=specification,
                route_time_seconds=route_time_seconds,
                noisy_route_x=noisy_route_x,
                noisy_route_y=noisy_route_y,
                priors=priors,
                rbpf_config=rbpf_config,
                inference_seed=configured_experiment.inference_seed,
            )
            fit = online_filter.forecast(
                route_time_seconds[
                    specification.forecast_start_index : (
                        specification.forecast_start_index
                        + specification.prediction_count
                    )
                ],
                seed=1_000_000 + window_seed,
            )
            converged = None
            mcmc_diagnostics_ok = None
        else:
            fit = position_model.fit_bayesian_position_model(
                window,
                priors=priors,
                position_observations=position_observations,
                inference_method=inference_method,
                seed=window_seed,
                **inference_config,
            )
        if inference_method == "vi":
            converged = position_model.variational_converged(fit)
            mcmc_diagnostics_ok = None
        elif inference_method == "mcmc":
            converged = None
            mcmc_diagnostics_ok = _mcmc_diagnostics_ok(fit)
        evaluation = metrics.evaluate_position_predictions(
            fit,
            window,
            credible_interval=credible_interval,
            position_variable_names=(
                "x_model_prediction",
                "y_model_prediction",
            ),
        )
        posterior_sigma_position_observation_median_m = float(
            np.median(
                reporting.posterior_variable_samples(
                    fit,
                    "sigma_position_observation",
                )
            )
        )
        posterior_sigma_motion_residual_median_m = float(
            np.median(
                reporting.posterior_variable_samples(
                    fit,
                    "sigma_motion_residual",
                )
            )
        )
        window_runtime_seconds = time.perf_counter() - runtime_started
        table = _build_route_prediction_table(
            evaluation.prediction_table,
            specification=specification,
            window=window,
            route_x=route_x,
            route_y=route_y,
            longitude=longitude,
            latitude=latitude,
            inference_mode=configured_experiment.inference_mode,
            inference_method=inference_method,
            converged=converged,
            mcmc_diagnostics_ok=mcmc_diagnostics_ok,
            posterior_sigma_position_observation_median_m=(
                posterior_sigma_position_observation_median_m
            ),
            posterior_sigma_motion_residual_median_m=(
                posterior_sigma_motion_residual_median_m
            ),
            position_noise_std_m=configured_experiment.position_noise_std_m,
            position_noise_seed=configured_experiment.position_noise_seed,
        )
        table["window_runtime_seconds"] = window_runtime_seconds
        prediction_tables.append(table)
        posterior_groups.append(
            _build_rolling_posterior_group(
                fit,
                specification=specification,
                window=window,
                route_x=route_x,
                route_y=route_y,
                longitude=longitude,
                latitude=latitude,
            )
        )
        print(
            f"ADE={evaluation.ade_m:.2f} m, FDE={evaluation.fde_m:.2f} m, "
            f"runtime={window_runtime_seconds:.3f} s"
        )
        _print_time_based_motion_posterior(fit)
        if online_filter is not None:
            print(
                "  RBPF posterior: "
                f"processed={online_filter.processed_observation_count}, "
                f"ESS={online_filter.effective_sample_size:.0f}/"
                f"{rbpf_config.particle_count}, "
                f"resamples={online_filter.resample_count}"
            )
        if options.plot_each_window:
            figure, _ = prediction_plotting.plot_prediction(
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
                    if configured_experiment.position_noise_std_m > 0
                    else "Beobachtungen"
                ),
                position_noise_std_m=configured_experiment.position_noise_std_m,
                forecast_label="Median der latenten Trajektorienprognose",
                sample_label="Latente Trajektorienprognosen",
                show_time_labels=show_time_labels,
            )
            plt.close(figure)

    predictions = pd.concat(prediction_tables, ignore_index=True)
    summary = rolling.summarize_rolling_predictions(predictions)
    _print_summary(summary, credible_interval=credible_interval)
    _plot_rolling_predictions(
        route_x,
        route_y,
        posterior_groups,
        initial_observation_count=configured_experiment.observation_count,
        sample_trajectories_per_forecast=sample_trajectories_per_forecast,
        sample_seed=configured_experiment.inference_seed,
        observed_route_x=route_x + route_noise_x,
        observed_route_y=route_y + route_noise_y,
        observed_trajectory_label=(
            "Verrauschte Anfangsbeobachtungen"
            if configured_experiment.position_noise_std_m > 0
            else "Anfangsbeobachtungen"
        ),
        show_time_labels=show_time_labels,
    )
    return predictions, summary


def _advance_online_filter(
    online_filter,
    *,
    specification,
    route_time_seconds,
    noisy_route_x,
    noisy_route_y,
    priors,
    rbpf_config,
    inference_seed,
):
    """Initialize once or update one persistent position RBPF with new points."""
    update_stop_index = specification.forecast_start_index
    if online_filter is None:
        return position_model.SequentialBayesianPositionFilter.initialize(
            route_time_seconds[:update_stop_index],
            noisy_route_x[:update_stop_index],
            noisy_route_y[:update_stop_index],
            priors=priors,
            config=rbpf_config,
            seed=inference_seed,
        )

    update_start_index = online_filter.processed_observation_count
    if update_start_index > update_stop_index:
        raise RuntimeError(
            "Online forecast origins must advance monotonically without reset."
        )
    online_filter.update_many(
        route_time_seconds[update_start_index:update_stop_index],
        noisy_route_x[update_start_index:update_stop_index],
        noisy_route_y[update_start_index:update_stop_index],
    )
    return online_filter


def _prepare_route_coordinates(trajectory_data):
    """Return the complete recorded run in one local coordinate frame."""
    longitude = pd.to_numeric(
        trajectory_data["gps_longitude"], errors="coerce"
    ).to_numpy(dtype=float)
    latitude = pd.to_numeric(trajectory_data["gps_latitude"], errors="coerce").to_numpy(
        dtype=float
    )
    route_x, route_y = coordinates.gps_to_local_coordinates(
        longitude,
        latitude,
        unit="m",
    )
    return route_x, route_y, longitude, latitude


def _route_time_seconds(trajectory_data) -> np.ndarray:
    """Return finite increasing route times without imposing regular spacing."""
    timestamps = pd.to_datetime(trajectory_data["time"], utc=True)
    time_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy()
    time_steps = np.diff(time_seconds)
    if (
        time_steps.size == 0
        or not np.all(np.isfinite(time_steps))
        or np.any(time_steps <= 0.0)
    ):
        raise ValueError("Position-model route times must be finite and increasing.")
    return np.asarray(time_seconds, dtype=float)


def _time_interval_description(time_seconds) -> str:
    """Describe the possibly irregular route time intervals."""
    intervals = np.diff(np.asarray(time_seconds, dtype=float))
    return (
        f"min/median/max={np.min(intervals):g}/"
        f"{np.median(intervals):g}/{np.max(intervals):g} s"
    )


def _simulate_route_position_noise(position_count, *, position_noise_std_m, seed):
    """Generate one reproducible perturbation for every route position."""
    if (
        isinstance(position_noise_std_m, bool)
        or not np.isfinite(position_noise_std_m)
        or position_noise_std_m < 0
    ):
        raise ValueError("position_noise_std_m must be non-negative.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("position_noise_seed must be a non-negative integer.")
    if position_count < 1:
        raise ValueError("position_count must be positive.")
    if position_noise_std_m == 0:
        zeros = np.zeros(int(position_count), dtype=float)
        return zeros, zeros.copy()
    noise = np.random.default_rng(int(seed)).normal(
        0.0,
        float(position_noise_std_m),
        size=(int(position_count), 2),
    )
    return noise[:, 0], noise[:, 1]


def _build_window_position_observations(
    window,
    *,
    route_start_index,
    route_noise_x,
    route_noise_y,
    position_noise_std_m,
    noise_seed,
):
    """Apply the fixed route perturbation to one rolling observed window."""
    route_stop_index = route_start_index + window.observation_count
    route_slice = slice(route_start_index, route_stop_index)
    observed = window.observed_slice
    return position_model.PositionObservations(
        time_seconds=window.time_seconds[observed],
        x_meters=window.x_meters[observed] + route_noise_x[route_slice],
        y_meters=window.y_meters[observed] + route_noise_y[route_slice],
        position_noise_std_m=position_noise_std_m,
        noise_seed=noise_seed,
    )


def _build_route_prediction_table(
    prediction_table,
    *,
    specification,
    window,
    route_x,
    route_y,
    longitude,
    latitude,
    inference_mode,
    inference_method,
    converged,
    mcmc_diagnostics_ok,
    posterior_sigma_position_observation_median_m,
    posterior_sigma_motion_residual_median_m,
    position_noise_std_m,
    position_noise_seed,
):
    """Add rolling and common-route metadata to one prediction table."""
    table = prediction_table.copy()
    target_indices = np.arange(
        specification.forecast_start_index,
        specification.forecast_start_index + specification.prediction_count,
    )
    predicted_longitude, predicted_latitude = coordinates.local_to_gps_coordinates(
        table["x_median"],
        table["y_median"],
        reference_longitude=longitude[specification.start_index],
        reference_latitude=latitude[specification.start_index],
        unit="m",
    )
    x_median_route, y_median_route = _gps_to_route_coordinates(
        predicted_longitude,
        predicted_latitude,
        reference_longitude=longitude[0],
        reference_latitude=latitude[0],
    )
    forecast_origin_index = specification.forecast_start_index - 1
    table.insert(0, "window_index", specification.window_index)
    table.insert(1, "window_start_index", specification.start_index)
    table.insert(2, "forecast_start_index", specification.forecast_start_index)
    table.insert(3, "target_index", target_indices)
    table.insert(4, "horizon_step", np.arange(1, len(table) + 1))
    table["observation_count"] = specification.observation_count
    table["prediction_count"] = specification.prediction_count
    table["inference_mode"] = inference_mode
    table["inference_method"] = inference_method
    table["model_variant"] = "bayesian_position_model"
    table["converged"] = converged
    table["mcmc_diagnostics_ok"] = mcmc_diagnostics_ok
    table["posterior_sigma_position_observation_median_m"] = (
        posterior_sigma_position_observation_median_m
    )
    table["posterior_sigma_motion_residual_median_m"] = (
        posterior_sigma_motion_residual_median_m
    )
    table["position_noise_std_m"] = position_noise_std_m
    table["position_noise_seed"] = position_noise_seed
    table["forecast_origin_x_route"] = route_x[forecast_origin_index]
    table["forecast_origin_y_route"] = route_y[forecast_origin_index]
    table["x_actual_route"] = route_x[target_indices]
    table["y_actual_route"] = route_y[target_indices]
    table["x_median_route"] = x_median_route
    table["y_median_route"] = y_median_route
    return table


def _build_rolling_posterior_group(
    fit,
    *,
    specification,
    window,
    route_x,
    route_y,
    longitude,
    latitude,
):
    """Transform posterior predictive positions into the common route frame."""
    x_samples = reporting.posterior_variable_samples(fit, "x_model_prediction")
    y_samples = reporting.posterior_variable_samples(fit, "y_model_prediction")
    expected_shape = (x_samples.shape[0], specification.prediction_count)
    if (
        x_samples.ndim != 2
        or x_samples.shape[0] == 0
        or x_samples.shape != expected_shape
        or y_samples.shape != expected_shape
        or not np.all(np.isfinite(x_samples))
        or not np.all(np.isfinite(y_samples))
    ):
        raise ValueError("Rolling posterior position draws must be finite matrices.")
    predicted_longitude, predicted_latitude = coordinates.local_to_gps_coordinates(
        x_samples.ravel(),
        y_samples.ravel(),
        reference_longitude=longitude[specification.start_index],
        reference_latitude=latitude[specification.start_index],
        unit="m",
    )
    x_route, y_route = _gps_to_route_coordinates(
        predicted_longitude,
        predicted_latitude,
        reference_longitude=longitude[0],
        reference_latitude=latitude[0],
    )
    prediction_start_time = float(
        window.time_seconds[specification.observation_count - 1]
    )
    prediction_times = np.asarray(
        window.time_seconds[window.prediction_slice],
        dtype=float,
    )
    forecast_origin_index = specification.forecast_start_index - 1
    return _RollingPosteriorGroup(
        forecast_origin_x=float(route_x[forecast_origin_index]),
        forecast_origin_y=float(route_y[forecast_origin_index]),
        x_samples=x_route.reshape(expected_shape),
        y_samples=y_route.reshape(expected_shape),
        forecast_time_seconds=np.concatenate(
            ([0.0], prediction_times - prediction_start_time)
        ),
    )


def _gps_to_route_coordinates(
    longitude,
    latitude,
    *,
    reference_longitude,
    reference_latitude,
):
    """Convert GPS arrays to the complete route's local coordinate frame."""
    longitude_with_reference = np.concatenate(([reference_longitude], longitude))
    latitude_with_reference = np.concatenate(([reference_latitude], latitude))
    x_route, y_route = coordinates.gps_to_local_coordinates(
        longitude_with_reference,
        latitude_with_reference,
        unit="m",
    )
    return x_route[1:], y_route[1:]


def _plot_rolling_predictions(
    route_x,
    route_y,
    posterior_groups,
    *,
    initial_observation_count,
    sample_trajectories_per_forecast,
    sample_seed,
    observed_route_x,
    observed_route_y,
    observed_trajectory_label,
    show_time_labels,
):
    """Plot rolling position-model forecasts with shared trajectory styling."""
    posterior_groups = tuple(posterior_groups)
    generator = np.random.default_rng(sample_seed)
    forecast_paths = []
    sample_paths = []
    origin_x = []
    origin_y = []
    for group in posterior_groups:
        forecast_paths.append(
            (
                np.concatenate(
                    ([group.forecast_origin_x], np.median(group.x_samples, axis=0))
                ),
                np.concatenate(
                    ([group.forecast_origin_y], np.median(group.y_samples, axis=0))
                ),
            )
        )
        origin_x.append(group.forecast_origin_x)
        origin_y.append(group.forecast_origin_y)
        sample_count = min(
            sample_trajectories_per_forecast,
            group.x_samples.shape[0],
        )
        indices = generator.choice(
            group.x_samples.shape[0],
            size=sample_count,
            replace=False,
        )
        for sample_index in indices:
            sample_paths.append(
                (
                    np.concatenate(
                        ([group.forecast_origin_x], group.x_samples[sample_index])
                    ),
                    np.concatenate(
                        ([group.forecast_origin_y], group.y_samples[sample_index])
                    ),
                )
            )
    figure, axis = prediction_plotting.plot_trajectory_paths(
        observed_path=(
            observed_route_x[:initial_observation_count],
            observed_route_y[:initial_observation_count],
        ),
        reference_path=(route_x, route_y),
        forecast_paths=forecast_paths,
        sample_paths=sample_paths,
        prediction_origins=(origin_x, origin_y),
        posterior_draw_groups=tuple(
            (group.x_samples, group.y_samples) for group in posterior_groups
        ),
        forecast_time_groups=tuple(
            group.forecast_time_seconds for group in posterior_groups
        ),
        annotate_prediction_regions=show_time_labels,
        title=None,
        observed_label=observed_trajectory_label,
        reference_label="Aufgezeichnete Trajektorie",
        forecast_label="Rollierende Mediane der latenten Prognosen",
        sample_label="Latente Trajektorienprognosen",
        prediction_origin_label="Startpunkte der Prognosen",
        figsize=(11, 8),
        forecast_alpha=0.35,
        forecast_linewidth=1.5,
    )
    plt.show()
    return figure, axis


def _print_summary(summary, *, credible_interval):
    """Print aggregate rolling metrics for the position model."""
    interval_percent = 100 * credible_interval
    evaluation_count_label = (
        "Forecast origins" if summary.inference_mode == "online" else "Rolling windows"
    )
    rows = [
        (evaluation_count_label, summary.window_count),
        ("Inference mode", summary.inference_mode.upper()),
        ("Inference method", summary.inference_method.upper()),
        ("Forecast positions", summary.forecast_count),
        ("ADE", f"{summary.ade_m:.2f} m"),
        ("FDE", f"{summary.fde_m:.2f} m"),
        (f"Joint 2D {interval_percent:g}% coverage", f"{summary.radial_coverage:.1%}"),
        ("Mean prediction radius", f"{summary.mean_prediction_radius_m:.2f} m"),
        ("Mean window runtime", f"{summary.mean_window_runtime_seconds:.3f} s"),
        ("Total computation time", f"{summary.total_computation_time_seconds:.3f} s"),
    ]
    if summary.vi_convergence_rate is not None:
        rows.append(("VI convergence rate", f"{summary.vi_convergence_rate:.1%}"))
    if summary.mcmc_diagnostics_pass_rate is not None:
        rows.append(
            ("MCMC diagnostics pass rate", f"{summary.mcmc_diagnostics_pass_rate:.1%}")
        )
    print("\nRolling Bayesian latent-position model summary")
    print(reporting.format_aligned_rows(rows))
    print("\nPer-horizon evaluation:")
    print(rolling.format_per_horizon_table(summary.per_horizon_table))


def _mcmc_diagnostics_ok(fit) -> bool:
    """Return whether CmdStan reports no MCMC diagnostic problems."""
    diagnostic = fit.diagnose().lower()
    return "no problems detected" in diagnostic


def _noise_prior_description(upper_m: float, tail_probability: float) -> str:
    """Describe one ship-independent exponential noise prior."""
    return f"Exponential; P(sigma > {upper_m:g} m)={tail_probability:g}"


def _print_time_based_motion_posterior(fit) -> None:
    """Print concise rate and reference-interval posterior medians."""
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
    print(
        "  Time-based motion: "
        f"kappa={scale_rate:.6g} 1/s, omega={rotation_rate:.6g} rad/s, "
        f"scale({reference:g} s)={scale_at_reference:.6g}, "
        f"rotation({reference:g} s)={rotation_at_reference_deg:.6g} deg"
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
