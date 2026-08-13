"""Evaluate fully Bayesian CTRV forecasts across rolling windows."""

import argparse
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ship_trajectory_prediction.coordinates import (
    gps_to_local_coordinates,
    local_to_gps_coordinates,
)
from ship_trajectory_prediction.evaluation import (
    build_rolling_window_specs,
    evaluate_position_predictions,
    summarize_rolling_predictions,
)
from ship_trajectory_prediction.evaluation import (
    plot_prediction as plot_window_prediction,
)
from ship_trajectory_prediction.evaluation.bayesian_ctrv import (
    RollingExperimentConfig,
    normalize_bayesian_ctrv_model_variant,
    select_bayesian_ctrv_inference_config,
)
from ship_trajectory_prediction.evaluation.plotting import (
    RollingPosteriorPlotData,
    plot_bayesian_rolling_predictions,
)
from ship_trajectory_prediction.evaluation.reporting import (
    posterior_variable_samples,
)
from ship_trajectory_prediction.models.bayesian_ctrv import (
    DEFAULT_VI_ADAPT_ITER,
    NOISE_PARAMETER_NAMES,
    BayesianCTRVPriors,
    PositionObservations,
    diagnose_observed_turn_rate,
    fit_bayesian_ctrv_model,
    variational_converged,
)
from ship_trajectory_prediction.models.hybrid_bayesian_ctrv import (
    fit_hybrid_bayesian_ctrv_model,
)
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import (
    prepare_trajectory_window,
    read_ship_data,
)

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)


EXPERIMENT = RollingExperimentConfig(
    run_id=1,  # Trajectory run to evaluate.
    window_mode="sliding",  # Fixed "sliding" or growing "expanding" history.
    observation_count=20,  # Position points used by the first fit.
    prediction_count=3,  # Held-out future points per rolling forecast.
    additional_position_noise_std_m=2.0,  # Per x/y axis [m]; 0 disables.
    position_noise_seed=2026,  # Reproduces route-wide added position noise.
    stride=None,  # Forecast-origin step; None uses prediction_count.
    inference_method="vi",  # Fast "vi" or reference "mcmc".
    inference_seed=42,  # Reproduces every rolling VI or MCMC fit.
)
PRIORS = BayesianCTRVPriors(
    position_initial_prior_scale=5.0,  # Initial x/y uncertainty [m].
    # Historical calibration from Run IDs 0-99; keep evaluation runs disjoint.
    speed_initial_prior_mean=3.524,  # Robust initial speed center [m/s].
    speed_initial_prior_scale=0.365,  # Robust initial speed scale [m/s].
    turn_rate_initial_prior_mean=0.0,  # Neutral independent center [rad/s].
    turn_rate_state_prior_scale=0.001698,  # Robust turn-rate scale [rad/s].
    sigma_position_gps_prior_scale=5.0,  # Measurement-noise scale [m].
    sigma_position_process_prior_scale=0.5,  # Position drift [m/sqrt(s)].
    sigma_speed_process_prior_scale=0.05,  # Speed drift [(m/s)/sqrt(s)].
    sigma_turn_rate_process_prior_scale=0.001,  # Turn drift [(rad/s)/sqrt(s)].
)
VI_CONFIG = {
    "algorithm": "meanfield",  # "meanfield" or "fullrank".
    "iter": 20_000,  # Maximum optimization iterations.
    "grad_samples": 1,  # Samples per gradient estimate.
    "elbo_samples": 100,  # Samples per ELBO estimate.
    "eta": 1.0,  # Initial step size.
    "adapt_iter": DEFAULT_VI_ADAPT_ITER,  # Step-size adaptation iterations.
    "tol_rel_obj": 0.01,  # Relative ELBO stopping tolerance.
    "eval_elbo": 100,  # ELBO evaluation interval.
    "draws": 1_000,  # Posterior draws to save.
    "require_converged": False,  # Allow preliminary non-converged VI.
}
FULLRANK_GRAD_SAMPLES = 10
MCMC_CONFIG = {
    "chains": 4,  # Independent NUTS chains.
    "parallel_chains": 4,  # Chains run concurrently.
    "iter_warmup": 1_000,  # Warmup iterations per chain.
    "iter_sampling": 1_000,  # Saved draws per chain.
    "adapt_delta": 0.9,  # Target acceptance probability.
    "max_treedepth": 10,  # Maximum NUTS tree depth.
}
CREDIBLE_INTERVAL = 0.9  # Central 90% posterior-predictive region.
MAX_WINDOWS = None  # Optional smoke-test limit; None evaluates every window.
PLOT_EACH_WINDOW = False  # Show the individual fit of every rolling window.
SAMPLE_TRAJECTORIES_PER_FORECAST = 15  # Posterior paths shown per forecast.
MODEL_VARIANT = "bayesian"
VI_NUMERICAL_STABILITY_RETRIES = 2
# Numerical guard only: this does not constrain or trim valid posterior draws.
VI_MAX_NOISE_TO_PRIOR_SCALE_RATIO = 1_000_000.0


def main(
    *,
    model_variant=MODEL_VARIANT,
    window_mode=EXPERIMENT.window_mode,
    observation_count=EXPERIMENT.observation_count,
    prediction_count=EXPERIMENT.prediction_count,
    stride=EXPERIMENT.stride,
    inference_method=EXPERIMENT.inference_method,
    vi_algorithm=VI_CONFIG["algorithm"],
    priors=PRIORS,
    seed=EXPERIMENT.inference_seed,
    position_noise_std_m=EXPERIMENT.additional_position_noise_std_m,
    position_noise_seed=EXPERIMENT.position_noise_seed,
    require_converged=VI_CONFIG["require_converged"],
    max_windows=MAX_WINDOWS,
    plot_each_window=PLOT_EACH_WINDOW,
):
    """Fit and evaluate rolling CTRV forecasts across one complete run."""
    model_variant = normalize_bayesian_ctrv_model_variant(model_variant)
    fit_model = (
        fit_hybrid_bayesian_ctrv_model
        if model_variant == "hybrid"
        else fit_bayesian_ctrv_model
    )
    inference_method, inference_config = select_bayesian_ctrv_inference_config(
        inference_method,
        vi_algorithm=vi_algorithm,
        require_converged=require_converged,
        vi_config=VI_CONFIG,
        mcmc_config=MCMC_CONFIG,
        fullrank_grad_samples=FULLRANK_GRAD_SAMPLES,
    )

    trajectory_data = read_ship_data(DATA_FILE, run_id=EXPERIMENT.run_id)
    trajectory_data = trajectory_data.sort_values("time").reset_index(drop=True)
    if trajectory_data.empty:
        raise ValueError(f"No trajectory rows found for run_id={EXPERIMENT.run_id}.")

    windows = build_rolling_window_specs(
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
    print(f"Data file             : {DATA_FILE}")
    print(f"Run ID                : {EXPERIMENT.run_id}")
    print(f"Window mode           : {window_mode}")
    print(f"Initial observations  : {observation_count}")
    print(f"Prediction horizon    : {prediction_count}")
    print(f"Stride                : {effective_stride}")
    print(f"Rolling windows       : {len(windows)}")
    print(f"Model variant         : {model_variant}")
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
        window = prepare_trajectory_window(
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
        observed_turn_rate = diagnose_observed_turn_rate(
            window,
            turn_rate_state_prior_scale=priors.turn_rate_state_prior_scale,
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
        )
        if inference_method == "vi":
            converged = variational_converged(fit)
            mcmc_diagnostics_ok = None
            inference_status = f"VI converged={converged}"
        else:
            converged = None
            mcmc_diagnostics_ok = _mcmc_diagnostics_ok(fit)
            inference_status = f"MCMC diagnostics passed={mcmc_diagnostics_ok}"
        posterior_diagnostics = _posterior_window_diagnostics(fit)
        evaluation = evaluate_position_predictions(
            fit,
            window,
            credible_interval=CREDIBLE_INTERVAL,
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
            model_variant=model_variant,
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
            f"turn-rate observed/forecast="
            f"{observed_turn_rate.median_rad_s:+.5f}/"
            f"{posterior_diagnostics['forecast_origin_turn_rate_rad_s']:+.5f} "
            f"rad/s, {inference_status}"
        )
        if plot_each_window:
            figure, _ = plot_window_prediction(
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
    summary = summarize_rolling_predictions(predictions)
    _print_summary(summary, credible_interval=CREDIBLE_INTERVAL)
    _print_turn_rate_and_noise_summary(predictions)
    plot_bayesian_rolling_predictions(
        route_x,
        route_y,
        posterior_plot_groups,
        initial_observation_count=observation_count,
        window_mode=window_mode,
        sample_trajectories_per_forecast=SAMPLE_TRAJECTORIES_PER_FORECAST,
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
    x_samples = posterior_variable_samples(fit, "x_state_prediction")
    y_samples = posterior_variable_samples(fit, "y_state_prediction")
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
    predicted_longitude, predicted_latitude = local_to_gps_coordinates(
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
    return RollingPosteriorPlotData(
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
    model_variant="bayesian",
):
    """Add rolling metadata and one route-wide coordinate frame."""
    table = prediction_table.copy()
    target_indices = np.arange(
        specification.forecast_start_index,
        specification.forecast_start_index + specification.prediction_count,
    )
    reference_index = specification.start_index
    predicted_longitude, predicted_latitude = local_to_gps_coordinates(
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
    table["model_variant"] = model_variant
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
    table["turn_rate_prior_scale_rad_s"] = observed_turn_rate.prior_scale_rad_s
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


def _posterior_window_diagnostics(fit):
    """Return forecast-motion values and posterior noise medians for one window."""
    turn_rate_forecast_origin = posterior_variable_samples(
        fit,
        "turn_rate_forecast_origin",
    )
    heading_state = posterior_variable_samples(fit, "heading_state")
    heading_state_prediction = posterior_variable_samples(
        fit,
        "heading_state_prediction",
    )
    diagnostics = {
        "forecast_origin_turn_rate_rad_s": float(np.median(turn_rate_forecast_origin)),
        "forecast_heading_change_rad": float(
            np.median(heading_state_prediction[:, -1] - heading_state[:, -1])
        ),
    }
    for name in NOISE_PARAMETER_NAMES:
        diagnostics[f"posterior_{name}_median"] = float(
            np.median(posterior_variable_samples(fit, name))
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
):
    """Fit one window and retry only numerically exploded VI approximations."""
    if fit_model is None:
        fit_model = fit_bayesian_ctrv_model
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

        instability = _vi_numerical_instability_reason(fit, priors)
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


def _vi_numerical_instability_reason(fit, priors):
    """Describe posterior noise draws that indicate a failed VI approximation."""
    prior_scales = {
        "sigma_position_gps": priors.sigma_position_gps_prior_scale,
        "sigma_position_process": priors.sigma_position_process_prior_scale,
        "sigma_speed_process": priors.sigma_speed_process_prior_scale,
        "sigma_turn_rate_process": priors.sigma_turn_rate_process_prior_scale,
    }
    for name, prior_scale in prior_scales.items():
        samples = posterior_variable_samples(fit, name)
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
    route_x, route_y = gps_to_local_coordinates(longitude, latitude, unit="m")
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
    return PositionObservations(
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
    x_route, y_route = gps_to_local_coordinates(
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
    print(f"Evaluated windows        : {summary.window_count}")
    print(f"Inference method         : {summary.inference_method.upper()}")
    print(f"Forecasted positions     : {summary.forecast_count}")
    print(f"Overall ADE              : {summary.ade_m:.2f} m")
    print(f"Mean maximum-horizon FDE : {summary.fde_m:.2f} m")
    print(
        f"Joint 2D {100 * credible_interval:g}% coverage    : "
        f"{summary.radial_coverage:.1%}"
    )
    print(f"Mean equivalent radius  : {summary.mean_prediction_radius_m:.2f} m")
    print(f"Mean marginal width      : {summary.mean_marginal_interval_width_m:.2f} m")
    if summary.vi_convergence_rate is not None:
        print(f"VI convergence rate      : {summary.vi_convergence_rate:.1%}")
    if summary.mcmc_diagnostics_pass_rate is not None:
        print(f"MCMC diagnostics rate    : {summary.mcmc_diagnostics_pass_rate:.1%}")
    print("\nPer-horizon evaluation:")
    print(summary.per_horizon_table.round(3).to_string(index=False))


def _print_turn_rate_and_noise_summary(predictions):
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


def _parse_arguments(*, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--window-mode",
        choices=("sliding", "expanding"),
        default=EXPERIMENT.window_mode,
        help="Keep a fixed history or expand it from the beginning of the run.",
    )
    parser.add_argument(
        "--observations", type=int, default=EXPERIMENT.observation_count
    )
    parser.add_argument("--predictions", type=int, default=EXPERIMENT.prediction_count)
    parser.add_argument(
        "--stride",
        type=int,
        default=EXPERIMENT.stride,
        help="Forecast-origin step; defaults to the prediction horizon.",
    )
    parser.add_argument(
        "--inference",
        choices=("vi", "mcmc"),
        default=EXPERIMENT.inference_method,
        help="Fast variational inference or reference MCMC for every window.",
    )
    parser.add_argument(
        "--vi-algorithm",
        choices=("meanfield", "fullrank"),
        default=VI_CONFIG["algorithm"],
    )
    parser.add_argument(
        "--turn-rate-prior-scale",
        type=float,
        default=PRIORS.turn_rate_state_prior_scale,
        help="Optional fixed state-prior scale; defaults to observed-history MAD.",
    )
    parser.add_argument("--seed", type=int, default=EXPERIMENT.inference_seed)
    parser.add_argument(
        "--position-noise-std-m",
        type=float,
        default=EXPERIMENT.additional_position_noise_std_m,
        help="Additional Gaussian x/y observation noise in meters; 0 disables.",
    )
    parser.add_argument(
        "--position-noise-seed",
        type=int,
        default=EXPERIMENT.position_noise_seed,
        help="Seed for one route-wide reproducible position perturbation.",
    )
    parser.add_argument(
        "--require-converged",
        action="store_true",
        default=VI_CONFIG["require_converged"],
        help="Abort when any rolling VI fit misses its convergence criterion.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=MAX_WINDOWS,
        help="Optional smoke-test limit; omit it to evaluate the complete run.",
    )
    parser.add_argument(
        "--plot-each-window",
        action="store_true",
        default=PLOT_EACH_WINDOW,
        help="Show each fitted window and continue after its plot is closed.",
    )
    return parser.parse_args()


def run_cli(*, model_variant=MODEL_VARIANT, description=__doc__):
    """Parse shared options and evaluate one fixed Bayesian CTRV variant."""
    arguments = _parse_arguments(description=description)
    main(
        model_variant=model_variant,
        window_mode=arguments.window_mode,
        observation_count=arguments.observations,
        prediction_count=arguments.predictions,
        stride=arguments.stride,
        inference_method=arguments.inference,
        vi_algorithm=arguments.vi_algorithm,
        priors=replace(
            PRIORS,
            turn_rate_state_prior_scale=arguments.turn_rate_prior_scale,
        ),
        seed=arguments.seed,
        position_noise_std_m=arguments.position_noise_std_m,
        position_noise_seed=arguments.position_noise_seed,
        require_converged=arguments.require_converged,
        max_windows=arguments.max_windows,
        plot_each_window=arguments.plot_each_window,
    )


if __name__ == "__main__":
    run_cli()
