"""Shared rolling evaluation workflows for Bayesian CTRV models."""

import dataclasses
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ship_trajectory_prediction.forecasting.bayesian_ctrv as forecasting
import ship_trajectory_prediction.models.bayesian_ctrv as bayesian_model
import ship_trajectory_prediction.models.hybrid_bayesian_ctrv as hybrid_model
import ship_trajectory_prediction.observations.coordinates as coordinates
import ship_trajectory_prediction.observations.io as observations_io
import ship_trajectory_prediction.observations.window as observation_window
import ship_trajectory_prediction.validation.cli as validation_cli
import ship_trajectory_prediction.validation.metrics as metrics
import ship_trajectory_prediction.validation.plotting as plotting
import ship_trajectory_prediction.validation.prediction_plotting as prediction_plotting
import ship_trajectory_prediction.validation.reporting as reporting
import ship_trajectory_prediction.validation.rolling as rolling_validation

VI_NUMERICAL_STABILITY_RETRIES = 2
# Numerical guard only: this does not constrain or trim valid posterior draws.
VI_MAX_NOISE_TO_PRIOR_SCALE_RATIO = 1_000_000.0


def _apply_evaluation_options(
    experiment,
    priors,
    options,
    *,
    configure_turn_rate_prior,
):
    """Return independent experiment and prior configs with CLI overrides."""
    configured_experiment = dataclasses.replace(
        experiment,
        window_mode=options.window_mode,
        observation_count=options.observation_count,
        prediction_count=options.prediction_count,
        stride=options.stride,
        inference_method=options.inference_method,
        inference_seed=options.inference_seed,
        additional_position_noise_std_m=options.additional_position_noise_std_m,
        position_noise_seed=options.position_noise_seed,
    )
    prior_changes = {}
    if configure_turn_rate_prior:
        prior_changes["turn_rate_state_prior_scale"] = (
            options.turn_rate_state_prior_scale
        )
    configured_priors = dataclasses.replace(priors, **prior_changes)
    return configured_experiment, configured_priors


def run_fully_bayesian_ctrv_evaluation(
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
    """Evaluate fully Bayesian CTRV forecasts across one complete run."""
    experiment, priors = _apply_evaluation_options(
        experiment,
        priors,
        options,
        configure_turn_rate_prior=True,
    )
    return _run_bayesian_ctrv_evaluation(
        model_name="bayesian",
        model_label="Fully Bayesian CTRV",
        fit_model=bayesian_model.fit_bayesian_ctrv_model,
        data_file=data_file,
        experiment=experiment,
        priors=priors,
        vi_config=vi_config,
        mcmc_config=mcmc_config,
        fullrank_grad_samples=fullrank_grad_samples,
        credible_interval=credible_interval,
        sample_trajectories_per_forecast=sample_trajectories_per_forecast,
        vi_algorithm=options.vi_algorithm,
        require_converged=options.require_converged,
        max_windows=options.max_windows,
        plot_each_window=options.plot_each_window,
        noise_parameter_names=bayesian_model.NOISE_PARAMETER_NAMES,
        has_latent_turn_rate=True,
    )


def run_hybrid_bayesian_ctrv_evaluation(
    *,
    data_file,
    experiment: forecasting.RollingExperimentConfig,
    priors: hybrid_model.HybridBayesianCTRVPriors,
    hybrid_config: hybrid_model.HybridBayesianCTRVConfig,
    vi_config,
    mcmc_config,
    fullrank_grad_samples,
    credible_interval,
    sample_trajectories_per_forecast,
    options: validation_cli.BayesianCTRVEvaluationOptions,
):
    """Evaluate hybrid Bayesian CTRV forecasts across one complete run."""
    experiment, priors = _apply_evaluation_options(
        experiment,
        priors,
        options,
        configure_turn_rate_prior=False,
    )
    return _run_bayesian_ctrv_evaluation(
        model_name="hybrid",
        model_label="Hybrid Bayesian CTRV",
        fit_model=partial(
            hybrid_model.fit_hybrid_bayesian_ctrv_model,
            hybrid_config=hybrid_config,
        ),
        data_file=data_file,
        experiment=experiment,
        priors=priors,
        vi_config=vi_config,
        mcmc_config=mcmc_config,
        fullrank_grad_samples=fullrank_grad_samples,
        credible_interval=credible_interval,
        sample_trajectories_per_forecast=sample_trajectories_per_forecast,
        vi_algorithm=options.vi_algorithm,
        require_converged=options.require_converged,
        max_windows=options.max_windows,
        plot_each_window=options.plot_each_window,
        noise_parameter_names=hybrid_model.NOISE_PARAMETER_NAMES,
        has_latent_turn_rate=False,
    )


def _run_bayesian_ctrv_evaluation(
    *,
    model_name,
    model_label,
    fit_model,
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
    noise_parameter_names,
    has_latent_turn_rate,
):
    """Fit and evaluate one configured Bayesian CTRV model."""
    window_mode = experiment.window_mode
    observation_count = experiment.observation_count
    prediction_count = experiment.prediction_count
    stride = experiment.stride
    inference_method = experiment.inference_method
    seed = experiment.inference_seed
    position_noise_std_m = experiment.additional_position_noise_std_m
    position_noise_seed = experiment.position_noise_seed
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

    trajectory_data = observations_io.read_ship_data(
        data_file,
        run_id=experiment.run_id,
    )
    trajectory_data = trajectory_data.sort_values("time").reset_index(drop=True)
    if trajectory_data.empty:
        raise ValueError(f"No trajectory rows found for run_id={experiment.run_id}.")

    windows = rolling_validation.build_rolling_window_specs(
        len(trajectory_data),
        initial_observation_count=observation_count,
        prediction_count=prediction_count,
        stride=stride,
        window_mode=window_mode,
    )
    if max_windows is not None:
        if isinstance(max_windows, bool) or max_windows < 1:
            raise ValueError("max_windows must be a positive integer or None.")
        windows = windows[:max_windows]

    route_x, route_y, longitude, latitude = _prepare_route_coordinates(trajectory_data)
    route_noise_x, route_noise_y = _simulate_route_position_noise(
        len(trajectory_data),
        additional_noise_std_m=position_noise_std_m,
        seed=position_noise_seed,
    )
    effective_stride = prediction_count if stride is None else stride
    print("=" * 72)
    print("Bayesian CTRV Rolling-Window Evaluation")
    print("=" * 72)
    print(f"Data file             : {data_file}")
    print(f"Run ID                : {experiment.run_id}")
    print(f"Window mode           : {window_mode}")
    print(f"Initial observations  : {observation_count}")
    print(f"Prediction horizon    : {prediction_count}")
    print(f"Stride                : {effective_stride}")
    print(f"Rolling windows       : {len(windows)}")
    print(f"Model                 : {model_label}")
    print(f"Inference method      : {inference_method.upper()}")
    noise_description = (
        f"{position_noise_std_m:g} m (seed={position_noise_seed})"
        if position_noise_std_m > 0
        else "disabled"
    )
    print(f"Additional pos. noise : {noise_description}")
    if inference_method == "vi":
        print(f"VI algorithm          : {vi_algorithm}")
        print(f"VI adaptation steps   : {inference_config['adapt_iter']}")
    else:
        print(
            "Note: MCMC refits every rolling window and can take much longer than VI."
        )
        print(f"MCMC chains           : {inference_config['chains']}")
        print(f"MCMC warmup/chain     : {inference_config['iter_warmup']}")
        print(f"MCMC samples/chain    : {inference_config['iter_sampling']}")
    if has_latent_turn_rate:
        print(
            "Turn-rate prior scale : "
            + (
                "data-derived"
                if priors.turn_rate_state_prior_scale is None
                else f"{priors.turn_rate_state_prior_scale:.5f} rad/s"
            )
        )
    print(f"Plot each window      : {plot_each_window}")

    prediction_tables = []
    posterior_plot_groups = []
    for number, specification in enumerate(windows, start=1):
        window_seed = seed + specification.window_index
        print(
            f"\nWindow {number}/{len(windows)}: "
            f"observations={specification.observation_count}, "
            f"predictions={specification.prediction_count}, seed={window_seed}"
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
            additional_noise_std_m=position_noise_std_m,
            noise_seed=position_noise_seed,
        )
        observed_turn_rate = bayesian_model.diagnose_observed_turn_rate(
            window,
            turn_rate_state_prior_scale=(
                priors.turn_rate_state_prior_scale if has_latent_turn_rate else None
            ),
            position_observations=position_observations,
        )
        fit, window_seed = _fit_rolling_window(
            window,
            priors=priors,
            position_observations=position_observations,
            inference_method=inference_method,
            inference_config=inference_config,
            initial_seed=window_seed,
            fit_model=fit_model,
            noise_parameter_names=noise_parameter_names,
        )
        if inference_method == "vi":
            converged = bayesian_model.variational_converged(fit)
            mcmc_diagnostics_ok = None
            inference_status = f"VI converged={converged}"
        else:
            converged = None
            mcmc_diagnostics_ok = _mcmc_diagnostics_ok(fit)
            inference_status = f"MCMC diagnostics passed={mcmc_diagnostics_ok}"
        posterior_diagnostics = _posterior_window_diagnostics(
            fit,
            noise_parameter_names=noise_parameter_names,
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
            observed_turn_rate=observed_turn_rate,
            posterior_diagnostics=posterior_diagnostics,
            additional_position_noise_std_m=position_noise_std_m,
            position_noise_seed=position_noise_seed,
            model_name=model_name,
        )
        prediction_tables.append(table)
        posterior_plot_groups.append(
            _build_rolling_posterior_plot_data(
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
            f"ADE={evaluation.ade_m:.2f} m, "
            f"FDE={evaluation.fde_m:.2f} m, "
            f"{inference_status}"
        )
        if plot_each_window:
            figure, _ = prediction_plotting.plot_prediction(
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
                observed_trajectory_label=(
                    "Verrauschte Beobachtungen"
                    if position_noise_std_m > 0
                    else "Beobachtungen"
                ),
                additional_position_noise_std_m=position_noise_std_m,
            )
            plt.close(figure)

    predictions = pd.concat(prediction_tables, ignore_index=True)
    summary = rolling_validation.summarize_rolling_predictions(predictions)
    _print_summary(summary, credible_interval=credible_interval)
    _print_turn_rate_and_noise_summary(
        predictions,
        has_latent_turn_rate=has_latent_turn_rate,
    )
    plotting.plot_bayesian_rolling_predictions(
        route_x,
        route_y,
        posterior_plot_groups,
        initial_observation_count=observation_count,
        window_mode=window_mode,
        sample_trajectories_per_forecast=sample_trajectories_per_forecast,
        sample_seed=seed,
        observed_route_x=route_x + route_noise_x,
        observed_route_y=route_y + route_noise_y,
        observed_trajectory_label=(
            "Verrauschte Anfangsbeobachtungen"
            if position_noise_std_m > 0
            else "Anfängliche Beobachtungen"
        ),
    )
    return predictions, summary


def _build_rolling_posterior_plot_data(
    fit,
    *,
    specification,
    window,
    route_x,
    route_y,
    longitude,
    latitude,
):
    """Transform one window's posterior paths into the common route frame."""
    x_samples = reporting.posterior_variable_samples(fit, "x_state_prediction")
    y_samples = reporting.posterior_variable_samples(fit, "y_state_prediction")
    if x_samples.ndim != 2:
        raise ValueError(
            "Rolling posterior position draws must be finite aligned matrices."
        )
    expected_shape = (x_samples.shape[0], specification.prediction_count)
    if (
        x_samples.shape[0] == 0
        or x_samples.shape != expected_shape
        or y_samples.shape != expected_shape
        or not np.all(np.isfinite(x_samples))
        or not np.all(np.isfinite(y_samples))
    ):
        raise ValueError(
            "Rolling posterior position draws must be finite aligned matrices."
        )

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
    observed_turn_rate,
    posterior_diagnostics,
    additional_position_noise_std_m,
    position_noise_seed,
    model_name,
):
    """Add rolling metadata and one route-wide coordinate frame."""
    table = prediction_table.copy()
    target_indices = np.arange(
        specification.forecast_start_index,
        specification.forecast_start_index + specification.prediction_count,
    )
    reference_index = specification.start_index
    predicted_longitude, predicted_latitude = coordinates.local_to_gps_coordinates(
        table["x_median"],
        table["y_median"],
        reference_longitude=longitude[reference_index],
        reference_latitude=latitude[reference_index],
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
    table["inference_method"] = inference_method
    table["model_variant"] = model_name
    table["converged"] = converged
    table["mcmc_diagnostics_ok"] = mcmc_diagnostics_ok
    table["additional_position_noise_std_m"] = additional_position_noise_std_m
    table["position_noise_seed"] = position_noise_seed
    table["observed_turn_rate_sample_count"] = observed_turn_rate.sample_count
    table["observed_turn_rate_median_rad_s"] = observed_turn_rate.median_rad_s
    table["observed_turn_rate_robust_scale_rad_s"] = (
        observed_turn_rate.robust_scale_rad_s
    )
    table["observed_turn_rate_q90_absolute_rad_s"] = (
        observed_turn_rate.q90_absolute_rad_s
    )
    table["turn_rate_prior_scale_rad_s"] = (
        observed_turn_rate.prior_scale_rad_s if model_name == "bayesian" else np.nan
    )
    for name, value in posterior_diagnostics.items():
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


def _posterior_window_diagnostics(
    fit,
    *,
    noise_parameter_names=bayesian_model.NOISE_PARAMETER_NAMES,
):
    """Return forecast-motion values and posterior noise medians for one window."""
    turn_rate_forecast_origin = reporting.posterior_variable_samples(
        fit,
        "turn_rate_forecast_origin",
    )
    heading_state = reporting.posterior_variable_samples(fit, "heading_state")
    heading_state_prediction = reporting.posterior_variable_samples(
        fit,
        "heading_state_prediction",
    )
    diagnostics = {
        "forecast_origin_turn_rate_rad_s": float(np.median(turn_rate_forecast_origin)),
        "forecast_heading_change_rad": float(
            np.median(heading_state_prediction[:, -1] - heading_state[:, -1])
        ),
    }
    for name in noise_parameter_names:
        diagnostics[f"posterior_{name}_median"] = float(
            np.median(reporting.posterior_variable_samples(fit, name))
        )
    return diagnostics


def _fit_rolling_window(
    window,
    *,
    priors,
    position_observations,
    inference_method,
    inference_config,
    initial_seed,
    fit_model=None,
    noise_parameter_names=bayesian_model.NOISE_PARAMETER_NAMES,
):
    """Fit one window and retry only numerically exploded VI approximations."""
    if fit_model is None:
        fit_model = bayesian_model.fit_bayesian_ctrv_model
    seed = initial_seed
    for attempt in range(VI_NUMERICAL_STABILITY_RETRIES + 1):
        fit = fit_model(
            window,
            priors=priors,
            position_observations=position_observations,
            inference_method=inference_method,
            seed=seed,
            **inference_config,
        )
        if inference_method != "vi":
            return fit, seed

        instability = _vi_numerical_instability_reason(
            fit,
            priors,
            noise_parameter_names=noise_parameter_names,
        )
        if instability is None:
            return fit, seed
        if attempt == VI_NUMERICAL_STABILITY_RETRIES:
            raise RuntimeError(
                "VI remained numerically unstable after "
                f"{VI_NUMERICAL_STABILITY_RETRIES + 1} attempts: {instability}"
            )

        next_seed = seed + 1
        print(
            "WARNING: Discarding numerically unstable VI fit "
            f"(seed={seed}: {instability}); retrying with seed={next_seed}."
        )
        seed = next_seed

    raise AssertionError("Unreachable VI retry state.")


def _vi_numerical_instability_reason(
    fit,
    priors,
    *,
    noise_parameter_names=bayesian_model.NOISE_PARAMETER_NAMES,
):
    """Describe posterior noise draws that indicate a failed VI approximation."""
    for name in noise_parameter_names:
        prior_scale = getattr(priors, f"{name}_prior_scale")
        samples = reporting.posterior_variable_samples(fit, name)
        if samples.size == 0 or not np.all(np.isfinite(samples)):
            return f"{name} contains empty or non-finite posterior draws"
        maximum_ratio = float(np.max(samples) / prior_scale)
        if maximum_ratio > VI_MAX_NOISE_TO_PRIOR_SCALE_RATIO:
            return (
                f"{name} maximum/prior-scale ratio={maximum_ratio:.3e} "
                f"> {VI_MAX_NOISE_TO_PRIOR_SCALE_RATIO:.3e}"
            )
    return None


def _prepare_route_coordinates(trajectory_data):
    """Return the complete run in one common local east/north frame."""
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


def _simulate_route_position_noise(
    position_count,
    *,
    additional_noise_std_m,
    seed,
):
    """Generate one reproducible x/y perturbation for every route position."""
    if (
        isinstance(additional_noise_std_m, bool)
        or not np.isfinite(additional_noise_std_m)
        or additional_noise_std_m < 0
    ):
        raise ValueError(
            "additional_position_noise_std_m must be finite and non-negative."
        )
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("position_noise_seed must be a non-negative integer.")
    if isinstance(position_count, bool) or not isinstance(
        position_count,
        (int, np.integer),
    ):
        raise ValueError("position_count must be a positive integer.")
    if position_count < 1:
        raise ValueError("position_count must be a positive integer.")

    if additional_noise_std_m == 0:
        zeros = np.zeros(int(position_count), dtype=float)
        return zeros, zeros.copy()
    generator = np.random.default_rng(int(seed))
    noise = generator.normal(
        0.0,
        float(additional_noise_std_m),
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
    """Apply the fixed route perturbation to one rolling observed window."""
    route_noise_x = np.asarray(route_noise_x, dtype=float)
    route_noise_y = np.asarray(route_noise_y, dtype=float)
    route_stop_index = route_start_index + window.observation_count
    if (
        route_start_index < 0
        or route_noise_x.ndim != 1
        or route_noise_y.shape != route_noise_x.shape
        or route_stop_index > route_noise_x.size
    ):
        raise ValueError("Route position noise does not cover the rolling window.")
    route_slice = slice(route_start_index, route_stop_index)
    observed = window.observed_slice
    return bayesian_model.PositionObservations(
        time_seconds=window.time_seconds[observed],
        x_meters=window.x_meters[observed] + route_noise_x[route_slice],
        y_meters=window.y_meters[observed] + route_noise_y[route_slice],
        additional_noise_std_m=additional_noise_std_m,
        noise_seed=noise_seed,
    )


def _gps_to_route_coordinates(
    longitude,
    latitude,
    *,
    reference_longitude,
    reference_latitude,
):
    """Convert GPS points to the local coordinate frame of the complete run."""
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
        (
            f"Joint 2D {100 * credible_interval:g}% coverage",
            f"{summary.radial_coverage:.1%}",
        ),
        (
            "Mean equivalent radius",
            f"{summary.mean_prediction_radius_m:.2f} m",
        ),
        (
            "Mean marginal width",
            f"{summary.mean_marginal_interval_width_m:.2f} m",
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
    label_width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{label_width}} : {value}")
    print("\nPer-horizon evaluation:")
    print(rolling_validation.format_per_horizon_table(summary.per_horizon_table))


def _print_turn_rate_and_noise_summary(predictions, *, has_latent_turn_rate):
    """Print one-row-per-window diagnostics for model identifiability."""
    windows = predictions.groupby("window_index", sort=True).first()
    print("\nTurn-rate and process-noise diagnostics:")
    print(
        "Median absolute observed turn rate : "
        f"{windows['observed_turn_rate_median_rad_s'].abs().median():.5f} rad/s"
    )
    print(
        "Maximum forecast-origin turn rate  : "
        f"{windows['forecast_origin_turn_rate_rad_s'].abs().max():.5f} rad/s"
    )
    print(
        "Median position process sigma      : "
        f"{windows['posterior_sigma_position_process_median'].median():.3f} m/sqrt(s)"
    )
    if has_latent_turn_rate:
        print(
            "Median turn-rate process sigma     : "
            f"{windows['posterior_sigma_turn_rate_process_median'].median():.6f} "
            "rad/s/sqrt(s)"
        )


def _mcmc_diagnostics_ok(fit):
    """Return whether CmdStan reports no MCMC sampler problems."""
    if not hasattr(fit, "diagnose"):
        raise TypeError("MCMC fit must provide diagnose().")
    return "no problems detected" in fit.diagnose().lower()
