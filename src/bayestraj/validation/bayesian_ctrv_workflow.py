"""Rolling evaluation workflow for the parametric Bayesian CTRV model."""

import dataclasses
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bayestraj.forecasting.bayesian_ctrv as forecasting
import bayestraj.forecasting.inference as inference
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.models.bayesian_observations as observation_support
import bayestraj.observations.coordinates as coordinates
import bayestraj.observations.io as observations_io
import bayestraj.observations.window as observation_window
import bayestraj.validation.cli as validation_cli
import bayestraj.validation.metrics as metrics
import bayestraj.validation.plotting as plotting
import bayestraj.validation.prediction_plotting as prediction_plotting
import bayestraj.validation.reporting as reporting
import bayestraj.validation.rolling as rolling_validation

VI_EXECUTION_RETRIES = 2


def run_bayesian_ctrv_evaluation(
    *,
    data_file,
    experiment: forecasting.RollingExperimentConfig,
    priors: bayesian_model.BayesianCTRVPriors,
    vi_config,
    mcmc_config,
    fullrank_grad_samples,
    credible_interval,
    sample_trajectories_per_forecast,
    options: validation_cli.BayesianCTRVEvaluationOptions,
):
    """Evaluate constant-parameter Bayesian CTRV forecasts across one run."""
    configured_experiment = dataclasses.replace(
        experiment,
        window_mode=options.window_mode,
        observation_count=options.observation_count,
        prediction_count=options.prediction_count,
        history_position_count=options.history_position_count,
        stride=options.stride,
        inference_method=options.inference_method,
        inference_seed=options.inference_seed,
        additional_position_noise_std_m=options.additional_position_noise_std_m,
        position_noise_seed=options.position_noise_seed,
    )
    configured_priors = dataclasses.replace(
        priors,
        turn_rate_prior_scale=options.turn_rate_prior_scale,
    )
    return _run_evaluation(
        data_file=data_file,
        experiment=configured_experiment,
        priors=configured_priors,
        vi_config=vi_config,
        mcmc_config=mcmc_config,
        fullrank_grad_samples=fullrank_grad_samples,
        credible_interval=credible_interval,
        sample_trajectories_per_forecast=sample_trajectories_per_forecast,
        vi_algorithm=options.vi_algorithm,
        require_converged=options.require_converged,
        max_windows=options.max_windows,
        plot_each_window=options.plot_each_window,
    )


def _run_evaluation(
    *,
    data_file,
    experiment,
    priors,
    vi_config,
    mcmc_config,
    fullrank_grad_samples,
    credible_interval,
    sample_trajectories_per_forecast,
    vi_algorithm,
    require_converged,
    max_windows,
    plot_each_window,
):
    """Run one configured rolling evaluation."""
    history_position_count = observation_support.validate_history_position_count(
        experiment.history_position_count,
        observation_count=experiment.observation_count,
    )
    inference_method, inference_config = inference.select_inference_config(
        experiment.inference_method,
        vi_algorithm=vi_algorithm,
        require_converged=require_converged,
        vi_config=vi_config,
        mcmc_config=mcmc_config,
        fullrank_grad_samples=fullrank_grad_samples,
    )
    trajectory_data = (
        observations_io.read_ship_data(
            data_file,
            run_id=experiment.run_id,
        )
        .sort_values("time")
        .reset_index(drop=True)
    )
    if trajectory_data.empty:
        raise ValueError(f"No trajectory rows found for run_id={experiment.run_id}.")

    windows = rolling_validation.build_rolling_window_specs(
        len(trajectory_data),
        initial_observation_count=experiment.observation_count,
        prediction_count=experiment.prediction_count,
        stride=experiment.stride,
        window_mode=experiment.window_mode,
    )
    if max_windows is not None:
        if isinstance(max_windows, bool) or max_windows < 1:
            raise ValueError("max_windows must be a positive integer or None.")
        windows = windows[:max_windows]

    route_x, route_y, longitude, latitude = _prepare_route_coordinates(trajectory_data)
    route_noise_x, route_noise_y = _simulate_route_position_noise(
        len(trajectory_data),
        additional_noise_std_m=experiment.additional_position_noise_std_m,
        seed=experiment.position_noise_seed,
    )
    effective_stride = (
        experiment.prediction_count if experiment.stride is None else experiment.stride
    )
    print("=" * 72)
    print("Parametric Bayesian CTRV Rolling-Window Evaluation")
    print("=" * 72)
    print(f"Data file             : {data_file}")
    print(f"Run ID                : {experiment.run_id}")
    print(f"Window mode           : {experiment.window_mode}")
    print(f"Initial observations  : {experiment.observation_count}")
    print(f"History positions K   : {history_position_count}")
    print(f"Prediction horizon    : {experiment.prediction_count}")
    print(f"Stride                : {effective_stride}")
    print(f"Rolling windows       : {len(windows)}")
    print(f"Inference method      : {inference_method.upper()}")
    noise_description = (
        f"{experiment.additional_position_noise_std_m:g} m "
        f"(seed={experiment.position_noise_seed})"
        if experiment.additional_position_noise_std_m > 0
        else "disabled"
    )
    print(f"Hidden synthetic noise: {noise_description}")
    print(
        "Observation-noise prior: Exponential("
        f"rate={priors.sigma_position_observation_prior_rate:.4f} 1/m; "
        "P(sigma > "
        f"{priors.sigma_position_observation_prior_upper_m:g} m)="
        f"{priors.sigma_position_observation_prior_tail_probability:g})"
    )
    print("Prior status          : provisional; model-specific validation pending")
    print(f"Plot each window      : {plot_each_window}")

    prediction_tables = []
    posterior_plot_groups = []
    for number, specification in enumerate(windows, start=1):
        window_seed = experiment.inference_seed + specification.window_index
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
            additional_noise_std_m=experiment.additional_position_noise_std_m,
            noise_seed=experiment.position_noise_seed,
        )
        print(
            f"\nWindow {number}/{len(windows)}: "
            f"observations={specification.observation_count}, "
            f"history={history_position_count}, "
            f"predictions={specification.prediction_count}, seed={window_seed}"
        )
        runtime_started = time.perf_counter()
        fit, window_seed = _fit_window(
            window,
            priors=priors,
            history_position_count=history_position_count,
            position_observations=position_observations,
            inference_method=inference_method,
            inference_config=inference_config,
            initial_seed=window_seed,
        )
        if inference_method == "vi":
            converged = bayesian_model.variational_converged(fit)
            mcmc_diagnostics_ok = None
            inference_status = f"VI converged={converged}"
        else:
            converged = None
            mcmc_diagnostics_ok = _mcmc_diagnostics_ok(fit)
            inference_status = f"MCMC diagnostics passed={mcmc_diagnostics_ok}"

        evaluation = metrics.evaluate_position_predictions(
            fit,
            window,
            credible_interval=credible_interval,
            position_variable_names=("x_prediction", "y_prediction"),
        )
        diagnostics = _posterior_diagnostics(fit, window)
        table = _build_route_prediction_table(
            evaluation.prediction_table,
            specification=specification,
            window=window,
            route_x=route_x,
            route_y=route_y,
            longitude=longitude,
            latitude=latitude,
            inference_method=inference_method,
            converged=converged,
            mcmc_diagnostics_ok=mcmc_diagnostics_ok,
            diagnostics=diagnostics,
            history_position_count=history_position_count,
            additional_position_noise_std_m=(
                experiment.additional_position_noise_std_m
            ),
            position_noise_seed=experiment.position_noise_seed,
        )
        posterior_plot_groups.append(
            _build_rolling_plot_data(
                fit,
                specification=specification,
                window=window,
                route_x=route_x,
                route_y=route_y,
                longitude=longitude,
                latitude=latitude,
            )
        )
        window_runtime_seconds = time.perf_counter() - runtime_started
        table["window_runtime_seconds"] = window_runtime_seconds
        prediction_tables.append(table)
        print(
            f"ADE={evaluation.ade_m:.2f} m, FDE={evaluation.fde_m:.2f} m, "
            f"runtime={window_runtime_seconds:.3f} s, {inference_status}"
        )
        if plot_each_window:
            figure, _ = prediction_plotting.plot_prediction(
                window,
                fit,
                state_prediction_variable_names=("x_prediction", "y_prediction"),
                observed_position_values=(
                    position_observations.x_meters,
                    position_observations.y_meters,
                ),
                observed_trajectory_label=(
                    "Für Fit verwendete verrauschte Beobachtungen"
                    if experiment.additional_position_noise_std_m > 0
                    else "Für Fit verwendete Beobachtungen"
                ),
                fit_history_position_count=history_position_count,
                additional_position_noise_std_m=(
                    experiment.additional_position_noise_std_m
                ),
                forecast_label="Median der parametrischen CTRV-Trajektorie",
                sample_label="Trajektorien aus Posterior-Parameterziehungen",
            )
            plt.close(figure)

    predictions = pd.concat(prediction_tables, ignore_index=True)
    summary = rolling_validation.summarize_rolling_predictions(predictions)
    _print_summary(summary, credible_interval=credible_interval)
    _print_parameter_summary(predictions)
    plotting.plot_bayesian_rolling_predictions(
        route_x,
        route_y,
        posterior_plot_groups,
        initial_observation_count=experiment.observation_count,
        window_mode=experiment.window_mode,
        sample_trajectories_per_forecast=sample_trajectories_per_forecast,
        sample_seed=experiment.inference_seed,
        observed_route_x=route_x + route_noise_x,
        observed_route_y=route_y + route_noise_y,
        observed_trajectory_label=(
            "Verrauschte Anfangsbeobachtungen"
            if experiment.additional_position_noise_std_m > 0
            else "Anfängliche Beobachtungen"
        ),
        history_position_count=history_position_count,
        forecast_label="Rollierende parametrische Posterior-Mediane",
        sample_label="Trajektorien aus Posterior-Parameterziehungen",
    )
    return predictions, summary


def _fit_window(
    window,
    *,
    priors,
    history_position_count,
    position_observations,
    inference_method,
    inference_config,
    initial_seed,
):
    """Fit one window and retry recoverable VI execution failures."""
    seed = initial_seed
    for attempt in range(VI_EXECUTION_RETRIES + 1):
        try:
            fit = bayesian_model.fit_bayesian_ctrv_model(
                window,
                priors=priors,
                history_position_count=history_position_count,
                position_observations=position_observations,
                inference_method=inference_method,
                seed=seed,
                **inference_config,
            )
            return fit, seed
        except RuntimeError:
            if inference_method != "vi" or attempt == VI_EXECUTION_RETRIES:
                raise
            seed += 1
            print(f"Retrying VI with seed={seed} after CmdStan execution failure.")
    raise RuntimeError("Unreachable VI retry state.")


def _posterior_diagnostics(fit, window):
    """Return scalar parameter medians and forecast heading change."""
    medians = {
        name: float(np.median(reporting.posterior_variable_samples(fit, name)))
        for name in bayesian_model.PARAMETER_NAMES
    }
    forecast_duration = float(
        window.time_seconds[window.prediction_slice][-1]
        - window.time_seconds[window.observation_count - 1]
    )
    return {
        "posterior_speed_median_mps": medians["speed"],
        "posterior_heading_initial_median_rad": medians["heading_initial"],
        "posterior_turn_rate_median_rad_s": medians["turn_rate"],
        "posterior_sigma_position_observation_median_m": medians[
            "sigma_position_observation"
        ],
        "forecast_heading_change_median_rad": medians["turn_rate"] * forecast_duration,
    }


def _build_rolling_plot_data(
    fit,
    *,
    specification,
    window,
    route_x,
    route_y,
    longitude,
    latitude,
):
    """Transform one forecast's model-position draws to the route frame."""
    x_samples = reporting.posterior_variable_samples(fit, "x_prediction")
    y_samples = reporting.posterior_variable_samples(fit, "y_prediction")
    expected_shape = (x_samples.shape[0], specification.prediction_count)
    if (
        x_samples.ndim != 2
        or x_samples.shape[0] == 0
        or x_samples.shape != expected_shape
        or y_samples.shape != expected_shape
        or not np.all(np.isfinite(x_samples))
        or not np.all(np.isfinite(y_samples))
    ):
        raise ValueError("Rolling posterior position draws must be aligned matrices.")
    sample_shape = x_samples.shape
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
    forecast_origin_index = specification.forecast_start_index - 1
    prediction_start_time = float(
        window.time_seconds[specification.observation_count - 1]
    )
    prediction_times = np.asarray(
        window.time_seconds[window.prediction_slice],
        dtype=float,
    )
    return plotting.RollingPosteriorPlotData(
        forecast_origin_x=float(route_x[forecast_origin_index]),
        forecast_origin_y=float(route_y[forecast_origin_index]),
        x_samples=x_route.reshape(sample_shape),
        y_samples=y_route.reshape(sample_shape),
        forecast_time_seconds=np.concatenate(
            ([0.0], prediction_times - prediction_start_time)
        ),
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
    inference_method,
    converged,
    mcmc_diagnostics_ok,
    diagnostics,
    history_position_count,
    additional_position_noise_std_m,
    position_noise_seed,
):
    """Add rolling metadata and route-wide coordinates."""
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
    table["history_position_count"] = history_position_count
    table["history_start_index"] = (
        specification.forecast_start_index - history_position_count
    )
    table["prediction_count"] = specification.prediction_count
    table["inference_method"] = inference_method
    table["model_variant"] = "bayesian"
    table["converged"] = converged
    table["mcmc_diagnostics_ok"] = mcmc_diagnostics_ok
    table["additional_position_noise_std_m"] = additional_position_noise_std_m
    table["position_observation_noise_std_m"] = diagnostics[
        "posterior_sigma_position_observation_median_m"
    ]
    table["position_noise_seed"] = position_noise_seed
    for name, value in diagnostics.items():
        table[name] = value
    table["forecast_origin_time"] = window.timestamps[
        specification.observation_count - 1
    ]
    table["forecast_origin_x_route"] = route_x[forecast_origin_index]
    table["forecast_origin_y_route"] = route_y[forecast_origin_index]
    table["x_actual_route"] = route_x[target_indices]
    table["y_actual_route"] = route_y[target_indices]
    table["x_median_route"] = x_median_route
    table["y_median_route"] = y_median_route
    return table


def _prepare_route_coordinates(trajectory_data):
    """Return the complete run in one local east/north frame."""
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


def _simulate_route_position_noise(position_count, *, additional_noise_std_m, seed):
    """Generate one reproducible perturbation for every route position."""
    additional_noise_std_m = observation_support.validate_non_negative_finite(
        "additional_position_noise_std_m",
        additional_noise_std_m,
    )
    seed = observation_support.validate_non_negative_integer(
        "position_noise_seed",
        seed,
    )
    if (
        isinstance(position_count, bool)
        or not isinstance(position_count, (int, np.integer))
        or position_count < 1
    ):
        raise ValueError("position_count must be a positive integer.")
    if additional_noise_std_m == 0:
        zeros = np.zeros(int(position_count), dtype=float)
        return zeros, zeros.copy()
    generator = np.random.default_rng(seed)
    noise = generator.normal(
        0.0,
        additional_noise_std_m,
        size=(int(position_count), 2),
    )
    return noise[:, 0], noise[:, 1]


def _build_window_position_observations(
    window,
    *,
    route_start_index,
    route_noise_x,
    route_noise_y,
    additional_noise_std_m,
    noise_seed,
):
    """Apply the fixed route perturbation to one complete observed window."""
    route_stop_index = route_start_index + window.observation_count
    route_slice = slice(route_start_index, route_stop_index)
    observed = window.observed_slice
    return bayesian_model.PositionObservations(
        time_seconds=window.time_seconds[observed],
        x_meters=window.x_meters[observed] + route_noise_x[route_slice],
        y_meters=window.y_meters[observed] + route_noise_y[route_slice],
        additional_noise_std_m=additional_noise_std_m,
        noise_seed=noise_seed,
        observation_noise_std_m=(
            additional_noise_std_m
            if additional_noise_std_m > 0
            else bayesian_model.DEFAULT_POSITION_OBSERVATION_NOISE_STD_M
        ),
    )


def _gps_to_route_coordinates(
    longitude,
    latitude,
    *,
    reference_longitude,
    reference_latitude,
):
    """Convert GPS points to the local frame of the complete run."""
    longitude_with_reference = np.concatenate(([reference_longitude], longitude))
    latitude_with_reference = np.concatenate(([reference_latitude], latitude))
    x_route, y_route = coordinates.gps_to_local_coordinates(
        longitude_with_reference,
        latitude_with_reference,
        unit="m",
    )
    return x_route[1:], y_route[1:]


def _print_summary(summary, *, credible_interval):
    """Print aggregate and horizon-specific rolling metrics."""
    print("\n" + "=" * 72)
    print("Complete rolling evaluation")
    print("=" * 72)
    rows = [
        ("Evaluated windows", str(summary.window_count)),
        ("Inference method", summary.inference_method.upper()),
        ("Forecasted positions", str(summary.forecast_count)),
        ("Overall ADE", f"{summary.ade_m:.2f} m"),
        ("Mean maximum-horizon FDE", f"{summary.fde_m:.2f} m"),
        ("Mean window runtime", f"{summary.mean_window_runtime_seconds:.3f} s"),
        ("Median window runtime", f"{summary.median_window_runtime_seconds:.3f} s"),
        (
            "Total computation time",
            rolling_validation.format_computation_time(
                summary.total_computation_time_seconds
            ),
        ),
        (
            f"Joint 2D {100 * credible_interval:g}% coverage",
            f"{summary.radial_coverage:.1%}",
        ),
    ]
    if summary.vi_convergence_rate is not None:
        rows.append(("VI convergence rate", f"{summary.vi_convergence_rate:.1%}"))
    if summary.mcmc_diagnostics_pass_rate is not None:
        rows.append(
            (
                "MCMC diagnostics rate",
                f"{summary.mcmc_diagnostics_pass_rate:.1%}",
            )
        )
    print(reporting.format_aligned_rows(rows))
    print("\nPer-horizon evaluation:")
    print(rolling_validation.format_per_horizon_table(summary.per_horizon_table))


def _print_parameter_summary(predictions):
    """Print one-row-per-window posterior parameter diagnostics."""
    windows = predictions.groupby("window_index", sort=True).first()
    rows = [
        (
            "Median fitted speed",
            f"{windows['posterior_speed_median_mps'].median():.3f} m/s",
        ),
        (
            "Median absolute turn rate",
            f"{windows['posterior_turn_rate_median_rad_s'].abs().median():.6f} rad/s",
        ),
    ]
    print("\nParametric CTRV diagnostics:")
    print(reporting.format_aligned_rows(rows))


def _mcmc_diagnostics_ok(fit):
    """Return whether CmdStan reports no MCMC sampler problems."""
    if not hasattr(fit, "diagnose"):
        raise TypeError("MCMC fit must provide diagnose().")
    return "no problems detected" in fit.diagnose().lower()
