"""Plotting utilities for Bayesian trajectory evaluation."""

from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ship_trajectory_prediction.evaluation.metrics import (
    evaluate_position_predictions,
)
from ship_trajectory_prediction.evaluation.reporting import (
    posterior_variable_samples,
)

MAX_POSTERIOR_TRAJECTORIES = 30
PREDICTION_REGION_LEVELS = (0.5, 0.9)
TIME_MARKER_SECONDS = (30.0, 60.0, 90.0)


def plot_trajectory_paths(
    observed_path,
    reference_path,
    forecast_paths,
    *,
    title,
    observed_label="Observed history",
    reference_label="Held-out trajectory",
    forecast_label="Posterior median",
    sample_paths=(),
    prediction_origins=None,
    prediction_origin_label="Prediction start",
    posterior_draws=None,
    forecast_time_seconds=None,
    time_marker_seconds=TIME_MARKER_SECONDS,
    annotation_text=None,
    figsize=(10, 7),
    forecast_alpha=1.0,
    forecast_linewidth=2.0,
):
    """Draw observed, reference, sampled, and central forecast paths."""
    title = _non_empty_text("title", title)
    observed_label = _non_empty_text("observed_label", observed_label)
    forecast_label = _non_empty_text("forecast_label", forecast_label)
    observed_x, observed_y = _path_arrays("observed_path", observed_path)
    reference = None
    if reference_path is not None:
        reference = _path_arrays("reference_path", reference_path)
        reference_label = _non_empty_text("reference_label", reference_label)
    forecasts = _path_collection(
        "forecast_paths",
        forecast_paths,
        require_non_empty=True,
    )
    samples = _path_collection("sample_paths", sample_paths)
    if prediction_origins is not None:
        origin_x, origin_y = _path_arrays(
            "prediction_origins",
            prediction_origins,
        )
        prediction_origin_label = _non_empty_text(
            "prediction_origin_label",
            prediction_origin_label,
        )

    figure, axis = plt.subplots(figsize=figsize)
    observed_line = axis.plot(
        observed_x,
        observed_y,
        color="tab:blue",
        linewidth=2,
        label=observed_label,
    )[0]
    reference_line = None
    if reference is not None:
        reference_line = axis.plot(
            *reference,
            color="black",
            linestyle="--",
            linewidth=2,
            label=reference_label,
        )[0]

    region_handles = []
    if posterior_draws is not None:
        region_handles = _draw_prediction_regions(axis, posterior_draws)

    for x_values, y_values in samples:
        axis.plot(
            x_values,
            y_values,
            color="tab:red",
            alpha=0.06,
            linewidth=0.8,
            zorder=2,
        )

    forecast_lines = []
    for path_index, (x_values, y_values) in enumerate(forecasts):
        forecast_lines.append(
            axis.plot(
                x_values,
                y_values,
                color="tab:red",
                alpha=forecast_alpha,
                linewidth=forecast_linewidth,
                label=forecast_label if path_index == 0 else None,
                zorder=4,
            )[0]
        )

    origin_artist = None
    if prediction_origins is not None:
        origin_artist = axis.scatter(
            origin_x,
            origin_y,
            color="tab:blue",
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
            label=prediction_origin_label,
        )

    if forecast_time_seconds is not None:
        _draw_time_markers(
            axis,
            forecasts[0],
            forecast_time_seconds,
            time_marker_seconds,
        )
    if annotation_text is not None:
        annotation_text = _non_empty_text("annotation_text", annotation_text)
        axis.text(
            0.02,
            0.02,
            annotation_text,
            transform=axis.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.82,
            },
            zorder=6,
        )

    legend_handles = [observed_line]
    if reference_line is not None:
        legend_handles.append(reference_line)
    legend_handles.extend(region_handles)
    legend_handles.append(forecast_lines[0])
    if origin_artist is not None:
        legend_handles.append(origin_artist)

    axis.set_title(title)
    axis.set_xlabel("East position [m]")
    axis.set_ylabel("North position [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        framealpha=0.9,
    )
    figure.tight_layout()
    return figure, axis


def plot_prediction(
    window,
    fit,
    *,
    model_name,
    max_posterior_trajectories=MAX_POSTERIOR_TRAJECTORIES,
    trajectory_sample_seed=42,
    state_prediction_variable_names=("x_prediction_mean", "y_prediction_mean"),
    observed_position_values=None,
    observed_trajectory_label="Observed history",
):
    """Plot posterior trajectories against a held-out future."""
    plot_data = _prepare_prediction_plot_data(
        window,
        fit,
        model_name=model_name,
        max_posterior_trajectories=max_posterior_trajectories,
        trajectory_sample_seed=trajectory_sample_seed,
        state_prediction_variable_names=state_prediction_variable_names,
        observed_position_values=observed_position_values,
        observed_trajectory_label=observed_trajectory_label,
    )
    evaluation = evaluate_position_predictions(
        fit,
        window,
        credible_interval=0.9,
        position_variable_names=state_prediction_variable_names,
    )
    held_out_x = np.concatenate(
        ([plot_data["prediction_start_x"]], window.x_meters[window.prediction_slice])
    )
    held_out_y = np.concatenate(
        ([plot_data["prediction_start_y"]], window.y_meters[window.prediction_slice])
    )

    figure, axis = plot_trajectory_paths(
        observed_path=plot_data["observed_path"],
        reference_path=(held_out_x, held_out_y),
        forecast_paths=(plot_data["median_path"],),
        sample_paths=plot_data["sample_paths"],
        prediction_origins=plot_data["prediction_origin"],
        posterior_draws=plot_data["posterior_draws"],
        forecast_time_seconds=plot_data["forecast_time_seconds"],
        annotation_text=_evaluation_annotation(window, evaluation),
        title=f"Bayesian {plot_data['model_name']} Prediction",
        observed_label=plot_data["observed_label"],
    )
    plt.show()
    return figure, axis


def plot_operational_prediction(
    window,
    fit,
    *,
    model_name,
    max_posterior_trajectories=MAX_POSTERIOR_TRAJECTORIES,
    trajectory_sample_seed=42,
    state_prediction_variable_names=("x_prediction_mean", "y_prediction_mean"),
    observed_position_values=None,
    observed_trajectory_label="Observed history",
):
    """Plot an operational posterior forecast without unknown future positions."""
    plot_data = _prepare_prediction_plot_data(
        window,
        fit,
        model_name=model_name,
        max_posterior_trajectories=max_posterior_trajectories,
        trajectory_sample_seed=trajectory_sample_seed,
        state_prediction_variable_names=state_prediction_variable_names,
        observed_position_values=observed_position_values,
        observed_trajectory_label=observed_trajectory_label,
    )
    figure, axis = plot_trajectory_paths(
        observed_path=plot_data["observed_path"],
        reference_path=None,
        forecast_paths=(plot_data["median_path"],),
        sample_paths=plot_data["sample_paths"],
        prediction_origins=plot_data["prediction_origin"],
        posterior_draws=plot_data["posterior_draws"],
        forecast_time_seconds=plot_data["forecast_time_seconds"],
        annotation_text=_operational_annotation(window),
        title=f"Bayesian {plot_data['model_name']} Operational Prediction",
        observed_label=plot_data["observed_label"],
    )
    plt.show()
    return figure, axis


def _prepare_prediction_plot_data(
    window,
    fit,
    *,
    model_name,
    max_posterior_trajectories,
    trajectory_sample_seed,
    state_prediction_variable_names,
    observed_position_values,
    observed_trajectory_label,
):
    """Build plot-ready paths from existing posterior draws."""
    model_name = _non_empty_text("model_name", model_name)
    observed_label = _non_empty_text(
        "observed_trajectory_label",
        observed_trajectory_label,
    )
    variable_names = _prediction_variable_names(state_prediction_variable_names)
    observed_x, observed_y = _resolve_observed_position_values(
        window,
        observed_position_values,
    )
    x_samples = posterior_variable_samples(fit, variable_names[0])
    y_samples = posterior_variable_samples(fit, variable_names[1])
    _validate_prediction_samples(window, x_samples, y_samples)

    prediction_start_x = observed_x[-1]
    prediction_start_y = observed_y[-1]
    connected_x_samples = np.column_stack(
        (np.full(len(x_samples), prediction_start_x), x_samples)
    )
    connected_y_samples = np.column_stack(
        (np.full(len(y_samples), prediction_start_y), y_samples)
    )
    sample_indices = _sample_trajectory_indices(
        len(x_samples),
        max_posterior_trajectories,
        trajectory_sample_seed,
    )
    sample_paths = tuple(
        (connected_x_samples[index], connected_y_samples[index])
        for index in sample_indices
    )
    prediction_start_time = float(window.time_seconds[window.observation_count - 1])
    prediction_times = np.asarray(
        window.time_seconds[window.prediction_slice],
        dtype=float,
    )

    return {
        "model_name": model_name,
        "observed_label": observed_label,
        "observed_path": (observed_x, observed_y),
        "prediction_start_x": prediction_start_x,
        "prediction_start_y": prediction_start_y,
        "prediction_origin": (
            np.asarray([prediction_start_x]),
            np.asarray([prediction_start_y]),
        ),
        "posterior_draws": (x_samples, y_samples),
        "sample_paths": sample_paths,
        "median_path": (
            np.median(connected_x_samples, axis=0),
            np.median(connected_y_samples, axis=0),
        ),
        "forecast_time_seconds": np.concatenate(
            ([0.0], prediction_times - prediction_start_time)
        ),
    }


def _draw_prediction_regions(axis, posterior_draws):
    """Draw empirical joint spatial highest-density regions."""
    x_centers, y_centers, density, thresholds = _joint_prediction_density(
        posterior_draws
    )
    region_styles = {
        0.9: ("#d62728", 0.10, "90% prediction region"),
        0.5: ("#d62728", 0.20, "50% prediction region"),
    }
    handles = {}
    density_max = np.nextafter(float(np.max(density)), np.inf)
    for probability in (0.9, 0.5):
        color, alpha, label = region_styles[probability]
        axis.contourf(
            x_centers,
            y_centers,
            density.T,
            levels=[thresholds[probability], density_max],
            colors=[color],
            alpha=alpha,
            antialiased=True,
            zorder=1,
        )
        handles[probability] = Patch(
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            label=label,
        )
    return [handles[0.5], handles[0.9]]


def _joint_prediction_density(posterior_draws, *, grid_size=80):
    """Estimate a joint x/y density and empirical mass thresholds."""
    x_samples, y_samples = _posterior_draw_arrays(posterior_draws)
    x_values = x_samples.ravel()
    y_values = y_samples.ravel()
    x_limits = _padded_limits(x_values)
    y_limits = _padded_limits(y_values)
    histogram, x_edges, y_edges = np.histogram2d(
        x_values,
        y_values,
        bins=grid_size,
        range=(x_limits, y_limits),
    )
    density = histogram.astype(float)
    for _ in range(2):
        density = _smooth_density(density)
    density /= np.sum(density)
    thresholds = {
        probability: _highest_density_threshold(density, probability)
        for probability in PREDICTION_REGION_LEVELS
    }
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, density, thresholds


def _posterior_draw_arrays(posterior_draws):
    """Return aligned finite matrices of joint position draws."""
    if not isinstance(posterior_draws, (tuple, list)) or len(posterior_draws) != 2:
        raise ValueError("posterior_draws must contain x and y draw matrices.")
    try:
        x_samples = np.asarray(posterior_draws[0], dtype=float)
        y_samples = np.asarray(posterior_draws[1], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "posterior_draws must contain finite x and y draw matrices."
        ) from error
    if (
        x_samples.ndim != 2
        or x_samples.size == 0
        or x_samples.shape != y_samples.shape
        or not np.all(np.isfinite(x_samples))
        or not np.all(np.isfinite(y_samples))
    ):
        raise ValueError(
            "posterior_draws must contain non-empty, finite, aligned matrices."
        )
    return x_samples, y_samples


def _padded_limits(values):
    """Return non-degenerate plot limits around finite values."""
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    padding = 0.05 * span if span > 0 else 1.0
    return lower - padding, upper + padding


def _smooth_density(density):
    """Apply one small separable Gaussian-like smoothing step."""
    padded = np.pad(density, 1, mode="constant")
    smoothed = np.zeros_like(density, dtype=float)
    weights = (1.0, 2.0, 1.0)
    for row_index, row_weight in enumerate(weights):
        for column_index, column_weight in enumerate(weights):
            smoothed += (
                row_weight
                * column_weight
                * padded[
                    row_index : row_index + density.shape[0],
                    column_index : column_index + density.shape[1],
                ]
            )
    return smoothed / 16.0


def _highest_density_threshold(density, probability):
    """Return the density cutoff containing the requested empirical mass."""
    sorted_density = np.sort(density.ravel())[::-1]
    threshold_index = int(
        np.searchsorted(np.cumsum(sorted_density), probability, side="left")
    )
    return float(sorted_density[min(threshold_index, len(sorted_density) - 1)])


def _draw_time_markers(axis, forecast_path, forecast_time_seconds, marker_seconds):
    """Interpolate and annotate requested elapsed times on one median path."""
    x_values, y_values = forecast_path
    time_values = np.asarray(forecast_time_seconds, dtype=float)
    if (
        time_values.ndim != 1
        or time_values.shape != x_values.shape
        or not np.all(np.isfinite(time_values))
        or np.any(np.diff(time_values) <= 0)
    ):
        raise ValueError(
            "forecast_time_seconds must be finite, increasing, and match the "
            "median forecast path."
        )
    for marker_seconds_value in marker_seconds:
        marker_seconds_value = float(marker_seconds_value)
        if not 0 < marker_seconds_value <= time_values[-1]:
            continue
        x_marker = np.interp(marker_seconds_value, time_values, x_values)
        y_marker = np.interp(marker_seconds_value, time_values, y_values)
        axis.scatter(
            [x_marker],
            [y_marker],
            s=28,
            facecolor="white",
            edgecolor="tab:red",
            linewidth=1,
            zorder=5,
        )
        axis.annotate(
            f"+{marker_seconds_value:g} s",
            (x_marker, y_marker),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="0.25",
            zorder=6,
        )


def _sample_trajectory_indices(draw_count, requested_count, seed):
    """Select at most 30 posterior paths reproducibly without replacement."""
    if isinstance(requested_count, bool) or not isinstance(requested_count, Integral):
        raise ValueError("max_posterior_trajectories must be a non-negative integer.")
    if requested_count < 0:
        raise ValueError("max_posterior_trajectories must be a non-negative integer.")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("trajectory_sample_seed must be an integer.")
    sample_count = min(requested_count, MAX_POSTERIOR_TRAJECTORIES, draw_count)
    if sample_count == 0:
        return np.asarray([], dtype=int)
    random_generator = np.random.default_rng(int(seed))
    return np.sort(random_generator.choice(draw_count, sample_count, replace=False))


def _evaluation_annotation(window, evaluation):
    """Format evaluation timing and accuracy for the plot inset."""
    observation_duration, prediction_horizon = _window_durations(window)
    return "\n".join(
        (
            f"Observation duration: {observation_duration:g} s",
            f"Prediction horizon: {prediction_horizon:g} s",
            f"ADE: {evaluation.ade_m:.2f} m",
            f"FDE: {evaluation.fde_m:.2f} m",
            f"90% coverage: {evaluation.radial_coverage:.1%}",
        )
    )


def _operational_annotation(window):
    """Format timing information that is available operationally."""
    observation_duration, prediction_horizon = _window_durations(window)
    return "\n".join(
        (
            f"Observation duration: {observation_duration:g} s",
            f"Prediction horizon: {prediction_horizon:g} s",
        )
    )


def _window_durations(window):
    """Return observation duration and forecast horizon in seconds."""
    time_seconds = np.asarray(window.time_seconds, dtype=float)
    observation_end = window.observation_count - 1
    observation_duration = time_seconds[observation_end] - time_seconds[0]
    prediction_horizon = time_seconds[-1] - time_seconds[observation_end]
    return float(observation_duration), float(prediction_horizon)


def _validate_prediction_samples(window, x_samples, y_samples):
    """Validate posterior matrices against the selected prediction window."""
    expected_count = len(np.asarray(window.x_meters[window.prediction_slice]))
    if (
        x_samples.ndim != 2
        or x_samples.shape != y_samples.shape
        or x_samples.shape[0] == 0
        or x_samples.shape[1] != expected_count
        or not np.all(np.isfinite(x_samples))
        or not np.all(np.isfinite(y_samples))
    ):
        raise ValueError(
            "Posterior position variables must contain aligned finite draws "
            "matching the prediction window."
        )


def _prediction_variable_names(variable_names):
    """Return two validated posterior prediction variable names."""
    if (
        not isinstance(variable_names, (tuple, list))
        or len(variable_names) != 2
        or not all(isinstance(name, str) and name.strip() for name in variable_names)
    ):
        raise ValueError(
            "state_prediction_variable_names must contain non-empty x and y names."
        )
    return tuple(name.strip() for name in variable_names)


def _resolve_observed_position_values(window, observed_position_values):
    """Return finite observed positions, optionally overriding window values."""
    if observed_position_values is None:
        observed = window.observed_slice
        return (
            np.asarray(window.x_meters[observed], dtype=float),
            np.asarray(window.y_meters[observed], dtype=float),
        )
    if (
        not isinstance(observed_position_values, (tuple, list))
        or len(observed_position_values) != 2
    ):
        raise ValueError(
            "observed_position_values must contain observed x and y values."
        )
    observed_x = np.asarray(observed_position_values[0], dtype=float)
    observed_y = np.asarray(observed_position_values[1], dtype=float)
    expected_shape = (window.observation_count,)
    if (
        observed_x.shape != expected_shape
        or observed_y.shape != expected_shape
        or not np.all(np.isfinite(observed_x))
        or not np.all(np.isfinite(observed_y))
    ):
        raise ValueError(
            "observed_position_values must contain finite x and y vectors "
            "matching window.observation_count."
        )
    return observed_x, observed_y


def _path_collection(name, paths, *, require_non_empty=False):
    """Return a validated tuple of finite x/y path pairs."""
    if isinstance(paths, (str, bytes)):
        raise ValueError(f"{name} must contain x/y path pairs.")
    try:
        paths = tuple(paths)
    except TypeError as error:
        raise ValueError(f"{name} must contain x/y path pairs.") from error
    if require_non_empty and not paths:
        raise ValueError(f"{name} must contain at least one path.")
    return tuple(
        _path_arrays(f"{name}[{index}]", path) for index, path in enumerate(paths)
    )


def _path_arrays(name, path):
    """Return one non-empty finite x/y path with aligned one-dimensional arrays."""
    if not isinstance(path, (tuple, list)) or len(path) != 2:
        raise ValueError(f"{name} must contain x and y values.")
    try:
        x_values = np.asarray(path[0], dtype=float)
        y_values = np.asarray(path[1], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite x and y values.") from error
    if (
        x_values.ndim != 1
        or y_values.ndim != 1
        or x_values.size == 0
        or x_values.shape != y_values.shape
        or not np.all(np.isfinite(x_values))
        or not np.all(np.isfinite(y_values))
    ):
        raise ValueError(
            f"{name} must contain non-empty, finite, aligned x and y vectors."
        )
    return x_values, y_values


def _non_empty_text(name, value):
    """Return stripped non-empty plot text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()
