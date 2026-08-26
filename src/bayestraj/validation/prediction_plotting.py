"""Plot posterior-predictive trajectories for Bayesian evaluation."""

import warnings
from numbers import Integral, Real

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Patch

import bayestraj.observations.coordinates as coordinates
import bayestraj.validation.metrics as metrics
import bayestraj.validation.reporting as reporting

MAX_SAMPLE_TRAJECTORIES = 15
PREDICTION_REGION_LEVELS = (0.5, 0.9)
PLOT_COORDINATE_MODES = ("m", "km", "gps")
# Plot typography
PLOT_TITLE_PAD = 16
PLOT_TITLE_FONT_SIZE = 13
PLOT_TITLE_FONT_WEIGHT = "bold"
AXIS_LABEL_FONT_SIZE = 13
AXIS_TICK_FONT_SIZE = 11

# Plot layout
PLOT_FIGURE_SIZE = (11, 9)
PLOT_MAX_WIDTH_TO_HEIGHT_RATIO = 2.0
PLOT_UPPER_PADDING_FRACTION = 0.85
GRID_ALPHA = 0.2
LEGEND_LOCATION = "upper right"
LEGEND_FRAME_ALPHA = 0.9

# Plot footer
FOOTER_BOTTOM = 0.02
FOOTER_LAYOUT_BOTTOM = 0.14
FOOTER_FONT_SIZE = 9
FOOTER_FACE_COLOR = "white"
FOOTER_EDGE_COLOR = "0.75"
FOOTER_ALPHA = 0.82

# Trajectory appearance
OBSERVED_TRAJECTORY_COLOR = "tab:blue"
OBSERVED_TRAJECTORY_LINE_WIDTH = 2.0
REFERENCE_TRAJECTORY_COLOR = "black"
REFERENCE_TRAJECTORY_LINE_STYLE = "--"
REFERENCE_TRAJECTORY_LINE_WIDTH = 2.0
PREDICTION_ORIGIN_COLOR = OBSERVED_TRAJECTORY_COLOR
PREDICTION_ORIGIN_EDGE_COLOR = "white"
PREDICTION_ORIGIN_EDGE_LINE_WIDTH = 0.6

# Posterior appearance
POSTERIOR_COLOR = "tab:red"
POSTERIOR_SAMPLE_ALPHA = 0.12
POSTERIOR_SAMPLE_LINE_WIDTH = 0.8
POSTERIOR_MEDIAN_ALPHA = 1.0
POSTERIOR_MEDIAN_LINE_WIDTH = 2.0
PREDICTION_REGION_50_ALPHA = 0.22
PREDICTION_REGION_90_ALPHA = 0.10
PREDICTION_REGION_EDGE_LINE_WIDTH = 0.7


def plot_trajectory_paths(
    observed_path,
    reference_path,
    forecast_paths,
    *,
    title,
    observed_label="Beobachtungen",
    context_path=None,
    context_label="Frühere Beobachtungen (nicht für Fit)",
    reference_label="Referenztrajektorie",
    forecast_label="Posterior-prädiktiver Median",
    sample_paths=(),
    sample_label="Posterior-prädiktive Trajektorien",
    prediction_origins=None,
    prediction_origin_label="Prognosebeginn",
    posterior_draws=None,
    forecast_time_seconds=None,
    posterior_draw_groups=(),
    forecast_time_groups=(),
    annotate_prediction_regions=True,
    annotation_text=None,
    figsize=PLOT_FIGURE_SIZE,
    forecast_alpha=POSTERIOR_MEDIAN_ALPHA,
    forecast_linewidth=POSTERIOR_MEDIAN_LINE_WIDTH,
    x_axis_label="Ostposition x [m]",
    y_axis_label="Nordposition y [m]",
    spatial_aspect=1.0,
):
    """Draw observed, reference, sampled, and central forecast paths."""
    if title is not None:
        title = _non_empty_text("title", title)
    observed_label = _non_empty_text("observed_label", observed_label)
    forecast_label = _non_empty_text("forecast_label", forecast_label)
    x_axis_label = _non_empty_text("x_axis_label", x_axis_label)
    y_axis_label = _non_empty_text("y_axis_label", y_axis_label)
    spatial_aspect = _positive_finite_number("spatial_aspect", spatial_aspect)
    observed_x, observed_y = _path_arrays("observed_path", observed_path)
    context = None
    if context_path is not None:
        context = _path_arrays("context_path", context_path)
        context_label = _non_empty_text("context_label", context_label)
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
    if samples:
        sample_label = _non_empty_text("sample_label", sample_label)
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
    context_line = None
    if context is not None:
        context_line = axis.plot(
            *context,
            color=OBSERVED_TRAJECTORY_COLOR,
            alpha=0.35,
            linestyle="--",
            linewidth=OBSERVED_TRAJECTORY_LINE_WIDTH,
            label=context_label,
        )[0]
    observed_line = axis.plot(
        observed_x,
        observed_y,
        color=OBSERVED_TRAJECTORY_COLOR,
        linewidth=OBSERVED_TRAJECTORY_LINE_WIDTH,
        label=observed_label,
    )[0]
    reference_line = None
    if reference is not None:
        reference_line = axis.plot(
            *reference,
            color=REFERENCE_TRAJECTORY_COLOR,
            linestyle=REFERENCE_TRAJECTORY_LINE_STYLE,
            linewidth=REFERENCE_TRAJECTORY_LINE_WIDTH,
            label=reference_label,
        )[0]

    region_inputs = _prediction_region_inputs(
        posterior_draws,
        forecast_time_seconds,
        posterior_draw_groups,
        forecast_time_groups,
    )
    region_handles = []
    for group_index, (group_draws, group_times) in enumerate(region_inputs):
        handles = _draw_prediction_regions(
            axis,
            group_draws,
            group_times,
            annotate_time=annotate_prediction_regions,
            group_index=group_index if len(region_inputs) > 1 else None,
        )
        if not region_handles:
            region_handles = handles

    sample_lines = []
    for sample_index, (x_values, y_values) in enumerate(samples):
        sample_lines.append(
            axis.plot(
                x_values,
                y_values,
                color=POSTERIOR_COLOR,
                alpha=POSTERIOR_SAMPLE_ALPHA,
                linewidth=POSTERIOR_SAMPLE_LINE_WIDTH,
                label=(
                    f"{sample_label} (n = {len(samples)})"
                    if sample_index == 0
                    else None
                ),
                zorder=2,
            )[0]
        )

    forecast_lines = []
    for path_index, (x_values, y_values) in enumerate(forecasts):
        forecast_lines.append(
            axis.plot(
                x_values,
                y_values,
                color=POSTERIOR_COLOR,
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
            color=PREDICTION_ORIGIN_COLOR,
            edgecolor=PREDICTION_ORIGIN_EDGE_COLOR,
            linewidth=PREDICTION_ORIGIN_EDGE_LINE_WIDTH,
            zorder=5,
            label=prediction_origin_label,
        )

    legend_handles = []
    if context_line is not None:
        legend_handles.append(context_line)
    legend_handles.append(observed_line)
    if reference_line is not None:
        legend_handles.append(reference_line)
    if sample_lines:
        legend_handles.append(sample_lines[0])
    legend_handles.extend(region_handles)
    legend_handles.append(forecast_lines[0])
    if origin_artist is not None:
        legend_handles.append(origin_artist)

    if title is not None:
        axis.set_title(
            title,
            pad=PLOT_TITLE_PAD,
            fontsize=PLOT_TITLE_FONT_SIZE,
            fontweight=PLOT_TITLE_FONT_WEIGHT,
        )
    axis.set_xlabel(x_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axis.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONT_SIZE)
    axis.tick_params(axis="both", labelsize=AXIS_TICK_FONT_SIZE)
    _reserve_vertical_layout_space(axis, spatial_aspect=spatial_aspect)
    axis.set_aspect(spatial_aspect, adjustable="box")
    axis.grid(alpha=GRID_ALPHA)
    axis.legend(
        handles=legend_handles,
        loc=LEGEND_LOCATION,
        framealpha=LEGEND_FRAME_ALPHA,
    )
    if annotation_text is not None:
        annotation_text = _non_empty_text("annotation_text", annotation_text)
        figure.text(
            0.5,
            FOOTER_BOTTOM,
            annotation_text,
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=FOOTER_FONT_SIZE,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": FOOTER_FACE_COLOR,
                "edgecolor": FOOTER_EDGE_COLOR,
                "alpha": FOOTER_ALPHA,
            },
        )
        figure.tight_layout(rect=(0, FOOTER_LAYOUT_BOTTOM, 1, 1))
    else:
        figure.tight_layout()
    return figure, axis


def _reserve_vertical_layout_space(axis, *, spatial_aspect):
    """Make a wide trajectory plot taller while preserving spatial scale."""
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    x_span = x_max - x_min
    y_span = y_max - y_min
    minimum_y_span = x_span / (PLOT_MAX_WIDTH_TO_HEIGHT_RATIO * spatial_aspect)
    if y_span >= minimum_y_span:
        return

    additional_y_span = minimum_y_span - y_span
    axis.set_ylim(
        y_min - (1 - PLOT_UPPER_PADDING_FRACTION) * additional_y_span,
        y_max + PLOT_UPPER_PADDING_FRACTION * additional_y_span,
    )


def plot_prediction(
    window,
    fit,
    *,
    plot_mode="evaluation",
    show_sample_trajectories=True,
    max_sample_trajectories=MAX_SAMPLE_TRAJECTORIES,
    sample_seed=42,
    state_prediction_variable_names=("x_prediction_mean", "y_prediction_mean"),
    observed_position_values=None,
    observed_trajectory_label="Beobachtungen",
    fit_history_position_count=None,
    additional_position_noise_std_m=None,
    coordinate_mode="m",
    title=None,
    forecast_label="Posterior-prädiktiver Median",
    sample_label="Posterior-prädiktive Trajektorien",
):
    """Plot an evaluation or operational posterior-predictive trajectory."""
    plot_mode = _validate_plot_mode(plot_mode)
    coordinate_mode = normalize_plot_coordinate_mode(coordinate_mode)
    additional_position_noise_std_m = _validate_additional_position_noise_std_m(
        additional_position_noise_std_m
    )
    plot_data = _prepare_prediction_plot_data(
        window,
        fit,
        show_sample_trajectories=show_sample_trajectories,
        max_sample_trajectories=max_sample_trajectories,
        sample_seed=sample_seed,
        state_prediction_variable_names=state_prediction_variable_names,
        observed_position_values=observed_position_values,
        observed_trajectory_label=observed_trajectory_label,
        fit_history_position_count=fit_history_position_count,
        coordinate_mode=coordinate_mode,
    )
    reference_path = None
    annotation_text = _operational_annotation(
        window,
        additional_position_noise_std_m=additional_position_noise_std_m,
    )
    if plot_mode == "evaluation":
        evaluation = metrics.evaluate_position_predictions(
            fit,
            window,
            credible_interval=0.9,
            position_variable_names=plot_data["variable_names"],
        )
        held_out_x_meters = np.concatenate(
            (
                [plot_data["prediction_start_meters"][0]],
                window.x_meters[window.prediction_slice],
            )
        )
        held_out_y_meters = np.concatenate(
            (
                [plot_data["prediction_start_meters"][1]],
                window.y_meters[window.prediction_slice],
            )
        )
        reference_path = _convert_plot_coordinates(
            window,
            held_out_x_meters,
            held_out_y_meters,
            coordinate_mode=coordinate_mode,
        )
        annotation_text = _evaluation_annotation(
            window,
            evaluation,
            additional_position_noise_std_m=additional_position_noise_std_m,
        )

    figure, axis = plot_trajectory_paths(
        observed_path=plot_data["observed_path"],
        context_path=plot_data["context_path"],
        reference_path=reference_path,
        forecast_paths=(plot_data["median_path"],),
        sample_paths=plot_data["sample_paths"],
        prediction_origins=plot_data["prediction_origin"],
        posterior_draws=plot_data["posterior_draws"],
        forecast_time_seconds=plot_data["forecast_time_seconds"],
        annotation_text=annotation_text,
        title=title,
        observed_label=plot_data["observed_label"],
        forecast_label=forecast_label,
        sample_label=sample_label,
        x_axis_label=plot_data["x_axis_label"],
        y_axis_label=plot_data["y_axis_label"],
        spatial_aspect=plot_data["spatial_aspect"],
    )
    if coordinate_mode == "gps":
        axis.ticklabel_format(style="plain", useOffset=False)
    plt.show()
    return figure, axis


def plot_operational_prediction(
    window,
    fit,
    *,
    show_sample_trajectories=True,
    max_sample_trajectories=MAX_SAMPLE_TRAJECTORIES,
    sample_seed=42,
    state_prediction_variable_names=("x_prediction_mean", "y_prediction_mean"),
    observed_position_values=None,
    observed_trajectory_label="Beobachtungen",
    additional_position_noise_std_m=None,
    coordinate_mode="m",
    title=None,
    forecast_label="Posterior-prädiktiver Median",
    sample_label="Posterior-prädiktive Trajektorien",
):
    """Plot an operational forecast through the shared plot-mode interface."""
    return plot_prediction(
        window,
        fit,
        plot_mode="operational",
        show_sample_trajectories=show_sample_trajectories,
        max_sample_trajectories=max_sample_trajectories,
        sample_seed=sample_seed,
        state_prediction_variable_names=state_prediction_variable_names,
        observed_position_values=observed_position_values,
        observed_trajectory_label=observed_trajectory_label,
        additional_position_noise_std_m=additional_position_noise_std_m,
        coordinate_mode=coordinate_mode,
        title=title,
        forecast_label=forecast_label,
        sample_label=sample_label,
    )


def _prepare_prediction_plot_data(
    window,
    fit,
    *,
    show_sample_trajectories,
    max_sample_trajectories,
    sample_seed,
    state_prediction_variable_names,
    observed_position_values,
    observed_trajectory_label,
    fit_history_position_count,
    coordinate_mode,
):
    """Build plot-ready paths from existing posterior draws."""
    observed_label = _non_empty_text(
        "observed_trajectory_label",
        observed_trajectory_label,
    )
    variable_names = _prediction_variable_names(state_prediction_variable_names)
    observed_x, observed_y = _resolve_observed_position_values(
        window,
        observed_position_values,
    )
    x_samples = reporting.posterior_variable_samples(fit, variable_names[0])
    y_samples = reporting.posterior_variable_samples(fit, variable_names[1])
    _validate_prediction_samples(window, x_samples, y_samples)
    if not isinstance(show_sample_trajectories, bool):
        raise ValueError("show_sample_trajectories must be a boolean.")

    prediction_start_x = observed_x[-1]
    prediction_start_y = observed_y[-1]
    connected_x_samples = np.column_stack(
        (np.full(len(x_samples), prediction_start_x), x_samples)
    )
    connected_y_samples = np.column_stack(
        (np.full(len(y_samples), prediction_start_y), y_samples)
    )
    observed_x, observed_y = _convert_plot_coordinates(
        window,
        observed_x,
        observed_y,
        coordinate_mode=coordinate_mode,
    )
    fit_history_position_count = _validate_fit_history_position_count(
        fit_history_position_count,
        observation_count=window.observation_count,
    )
    context_path = None
    if fit_history_position_count is not None:
        history_start = window.observation_count - fit_history_position_count
        if history_start > 0:
            context_path = (
                observed_x[: history_start + 1],
                observed_y[: history_start + 1],
            )
            observed_x = observed_x[history_start:]
            observed_y = observed_y[history_start:]
    connected_x_samples, connected_y_samples = _convert_plot_coordinates(
        window,
        connected_x_samples,
        connected_y_samples,
        coordinate_mode=coordinate_mode,
    )
    x_samples = connected_x_samples[:, 1:]
    y_samples = connected_y_samples[:, 1:]
    sample_indices = (
        _sample_trajectory_indices(
            len(x_samples),
            max_sample_trajectories,
            sample_seed,
        )
        if show_sample_trajectories
        else np.asarray([], dtype=int)
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
    x_axis_label, y_axis_label, spatial_aspect = _coordinate_plot_spec(
        window,
        coordinate_mode,
    )

    return {
        "variable_names": variable_names,
        "observed_label": observed_label,
        "observed_path": (observed_x, observed_y),
        "context_path": context_path,
        "prediction_start_meters": (prediction_start_x, prediction_start_y),
        "prediction_origin": (
            np.asarray([connected_x_samples[0, 0]]),
            np.asarray([connected_y_samples[0, 0]]),
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
        "x_axis_label": x_axis_label,
        "y_axis_label": y_axis_label,
        "spatial_aspect": spatial_aspect,
    }


def _validate_fit_history_position_count(value, *, observation_count):
    """Validate an optional trailing fit-history count used only for styling."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("fit_history_position_count must be an integer or None.")
    value = int(value)
    if value < 3 or value > observation_count:
        raise ValueError(
            "fit_history_position_count must be between 3 and observation_count."
        )
    return value


def _prediction_region_inputs(
    posterior_draws,
    forecast_time_seconds,
    posterior_draw_groups,
    forecast_time_groups,
):
    """Return aligned single- or multiple-forecast uncertainty inputs."""
    draw_groups = tuple(posterior_draw_groups)
    time_groups = tuple(forecast_time_groups)
    if posterior_draws is not None:
        if draw_groups or time_groups:
            raise ValueError(
                "posterior_draws cannot be combined with posterior_draw_groups."
            )
        return ((posterior_draws, forecast_time_seconds),)
    if len(draw_groups) != len(time_groups):
        raise ValueError(
            "posterior_draw_groups and forecast_time_groups must have equal length."
        )
    return tuple(zip(draw_groups, time_groups, strict=True))


def _draw_prediction_regions(
    axis,
    posterior_draws,
    forecast_time_seconds,
    *,
    annotate_time=True,
    group_index=None,
):
    """Draw one empirical covariance region pair per future time point."""
    x_samples, y_samples = _posterior_draw_arrays(posterior_draws)
    horizon_seconds = _prediction_horizon_seconds(
        forecast_time_seconds,
        x_samples.shape[1],
    )
    region_styles = {
        0.9: (
            POSTERIOR_COLOR,
            PREDICTION_REGION_90_ALPHA,
            "Posterior-prädiktiver Bereich (90 %)",
        ),
        0.5: (
            POSTERIOR_COLOR,
            PREDICTION_REGION_50_ALPHA,
            "Posterior-prädiktiver Bereich (50 %)",
        ),
    }
    centers = []
    for time_index in range(x_samples.shape[1]):
        regions = metrics.empirical_covariance_regions(
            x_samples[:, time_index],
            y_samples[:, time_index],
            probabilities=PREDICTION_REGION_LEVELS,
        )
        center = regions[0.9].center
        centers.append(center)
        for probability in (0.9, 0.5):
            color, alpha, _ = region_styles[probability]
            region = regions[probability]
            ellipse = Ellipse(
                xy=center,
                width=region.width,
                height=region.height,
                angle=region.angle_degrees,
                facecolor=color,
                edgecolor=color,
                linewidth=PREDICTION_REGION_EDGE_LINE_WIDTH,
                alpha=alpha,
                zorder=1,
            )
            group_suffix = "" if group_index is None else f"-g{group_index}"
            ellipse.set_gid(
                "posterior-predictive-region-"
                f"{probability:g}{group_suffix}-t{time_index}"
            )
            axis.add_patch(ellipse)

    if annotate_time:
        _label_prediction_regions(axis, centers, horizon_seconds)
    return [
        Patch(
            facecolor=region_styles[0.5][0],
            edgecolor="none",
            alpha=region_styles[0.5][1],
            label=region_styles[0.5][2],
        ),
        Patch(
            facecolor=region_styles[0.9][0],
            edgecolor="none",
            alpha=region_styles[0.9][1],
            label=region_styles[0.9][2],
        ),
    ]


def _prediction_horizon_seconds(forecast_time_seconds, prediction_count):
    """Return positive future horizons aligned with posterior sample columns."""
    time_values = np.asarray(forecast_time_seconds, dtype=float)
    if (
        time_values.ndim != 1
        or time_values.shape != (prediction_count + 1,)
        or time_values[0] != 0
        or not np.all(np.isfinite(time_values))
        or np.any(np.diff(time_values) <= 0)
    ):
        raise ValueError(
            "forecast_time_seconds must start at zero and contain one increasing "
            "time for every posterior prediction column."
        )
    return time_values[1:]


def _label_prediction_regions(axis, centers, horizon_seconds):
    """Label a small non-repeating selection of future-time regions."""
    for time_index in _selected_time_label_indices(len(horizon_seconds)):
        axis.annotate(
            f"+{horizon_seconds[time_index]:g} s",
            centers[time_index],
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=8,
            color="0.25",
            zorder=6,
        )


def _selected_time_label_indices(prediction_count):
    """Select at most three useful, unique horizon labels."""
    if prediction_count <= 3:
        return tuple(range(prediction_count))
    candidates = (0, prediction_count // 2, prediction_count - 1)
    return tuple(sorted(set(candidates)))


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


def _sample_trajectory_indices(draw_count, requested_count, seed):
    """Select at most 15 posterior paths reproducibly without replacement."""
    if isinstance(requested_count, bool) or not isinstance(requested_count, Integral):
        raise ValueError("max_sample_trajectories must be a non-negative integer.")
    if requested_count < 0:
        raise ValueError("max_sample_trajectories must be a non-negative integer.")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("sample_seed must be an integer.")
    sample_count = min(requested_count, MAX_SAMPLE_TRAJECTORIES, draw_count)
    if sample_count == 0:
        return np.asarray([], dtype=int)
    random_generator = np.random.default_rng(int(seed))
    return np.sort(random_generator.choice(draw_count, sample_count, replace=False))


def _evaluation_annotation(
    window,
    evaluation,
    *,
    additional_position_noise_std_m,
):
    """Format evaluation timing and accuracy for the figure footer."""
    observation_duration, prediction_horizon = _window_durations(window)
    covered_count = int(evaluation.prediction_table["radial_covered"].sum())
    prediction_count = len(evaluation.prediction_table)
    ade_m = _format_decimal_comma(evaluation.ade_m, decimal_places=2)
    fde_m = _format_decimal_comma(evaluation.fde_m, decimal_places=2)
    coverage_percent = _format_decimal_comma(
        100 * evaluation.radial_coverage,
        decimal_places=1,
    )
    interval_percent = _format_general_decimal_comma(100 * evaluation.credible_interval)
    return "\n".join(
        (
            _timing_annotation(
                observation_duration,
                prediction_horizon,
                additional_position_noise_std_m=additional_position_noise_std_m,
            ),
            " | ".join(
                (
                    f"ADE: {ade_m} m",
                    f"FDE: {fde_m} m",
                    f"Empirische 2D-Abdeckung ({interval_percent} %): "
                    f"{coverage_percent} % "
                    f"({covered_count}/{prediction_count} Punkte)",
                )
            ),
        )
    )


def _operational_annotation(window, *, additional_position_noise_std_m):
    """Format operationally available timing for the figure footer."""
    observation_duration, prediction_horizon = _window_durations(window)
    return _timing_annotation(
        observation_duration,
        prediction_horizon,
        additional_position_noise_std_m=additional_position_noise_std_m,
    )


def _timing_annotation(
    observation_duration,
    prediction_horizon,
    *,
    additional_position_noise_std_m,
):
    """Format timing and optional noise settings as one compact line."""
    parts = [
        f"Beobachtungsdauer: {observation_duration:g} s",
        f"Prognosehorizont: {prediction_horizon:g} s",
    ]
    if additional_position_noise_std_m is not None:
        noise_std_m = _format_general_decimal_comma(additional_position_noise_std_m)
        parts.append(f"Zusatzrauschen: σ_add = {noise_std_m} m je Achse")
    return " | ".join(parts)


def _validate_additional_position_noise_std_m(value):
    """Return one optional finite non-negative plotting noise value."""
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            "additional_position_noise_std_m must be a finite non-negative "
            "number or None."
        )
    return float(value) if value > 0 else None


def _format_decimal_comma(value, *, decimal_places):
    """Format a decimal number using the German decimal separator."""
    return f"{value:.{decimal_places}f}".replace(".", ",")


def _format_general_decimal_comma(value):
    """Format a compact number using the German decimal separator."""
    return f"{value:g}".replace(".", ",")


def _window_durations(window):
    """Return observation duration and forecast horizon in seconds."""
    time_seconds = np.asarray(window.time_seconds, dtype=float)
    observation_end = window.observation_count - 1
    observation_duration = time_seconds[observation_end] - time_seconds[0]
    prediction_horizon = time_seconds[-1] - time_seconds[observation_end]
    return float(observation_duration), float(prediction_horizon)


def _validate_prediction_samples(window, x_samples, y_samples):
    """Validate posterior matrices against the selected prediction window."""
    expected_count = len(np.asarray(window.time_seconds[window.prediction_slice]))
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


def _validate_plot_mode(plot_mode):
    """Return one supported posterior-predictive plot mode."""
    if not isinstance(plot_mode, str):
        raise ValueError("plot_mode must be 'evaluation' or 'operational'.")
    plot_mode = plot_mode.strip().lower()
    if plot_mode not in {"evaluation", "operational"}:
        raise ValueError("plot_mode must be 'evaluation' or 'operational'.")
    return plot_mode


def normalize_plot_coordinate_mode(coordinate_mode):
    """Return a supported coordinate mode or warn and fall back to meters."""
    if isinstance(coordinate_mode, str):
        normalized_mode = coordinate_mode.strip().lower()
        if normalized_mode in PLOT_COORDINATE_MODES:
            return normalized_mode
    warnings.warn(
        f"Invalid plot coordinate mode {coordinate_mode!r}; falling back to 'm'. "
        "Use 'm', 'km', or 'gps'.",
        UserWarning,
        stacklevel=2,
    )
    return "m"


def _coordinate_plot_spec(window, coordinate_mode):
    """Return axis labels and a physically meaningful display aspect."""
    if coordinate_mode == "m":
        return "Ostposition x [m]", "Nordposition y [m]", 1.0
    if coordinate_mode == "km":
        return "Ostposition x [km]", "Nordposition y [km]", 1.0

    _, reference_latitude = _reference_gps_coordinates(window)
    spatial_aspect = 1.0 / np.cos(np.radians(reference_latitude))
    return "Längengrad [°]", "Breitengrad [°]", float(spatial_aspect)


def _convert_plot_coordinates(
    window,
    x_meters,
    y_meters,
    *,
    coordinate_mode,
):
    """Convert aligned local-meter values solely for plot presentation."""
    x_meters = np.asarray(x_meters, dtype=float)
    y_meters = np.asarray(y_meters, dtype=float)
    if (
        x_meters.size == 0
        or x_meters.shape != y_meters.shape
        or not np.all(np.isfinite(x_meters))
        or not np.all(np.isfinite(y_meters))
    ):
        raise ValueError(
            "Plot coordinates must contain non-empty, finite, aligned values."
        )
    if coordinate_mode == "m":
        return x_meters, y_meters
    if coordinate_mode == "km":
        return (
            x_meters / coordinates.METERS_PER_KILOMETER,
            y_meters / coordinates.METERS_PER_KILOMETER,
        )

    reference_longitude, reference_latitude = _reference_gps_coordinates(window)
    longitude, latitude = coordinates.local_to_gps_coordinates(
        x_meters.ravel(),
        y_meters.ravel(),
        reference_longitude,
        reference_latitude,
        unit="m",
    )
    return longitude.reshape(x_meters.shape), latitude.reshape(y_meters.shape)


def _reference_gps_coordinates(window):
    """Return the GPS origin stored with a trajectory window."""
    try:
        reference_longitude = float(window.reference_longitude)
        reference_latitude = float(window.reference_latitude)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "coordinate_mode='gps' requires finite reference_longitude and "
            "reference_latitude values on the trajectory window."
        ) from error
    if (
        not np.isfinite(reference_longitude)
        or not np.isfinite(reference_latitude)
        or not -90 < reference_latitude < 90
    ):
        raise ValueError(
            "coordinate_mode='gps' requires finite reference_longitude and "
            "reference_latitude values, with latitude between -90 and 90 degrees."
        )
    return reference_longitude, reference_latitude


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


def _positive_finite_number(name, value):
    """Return a plotting value that is finite and strictly positive."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number.")
    return float(value)
