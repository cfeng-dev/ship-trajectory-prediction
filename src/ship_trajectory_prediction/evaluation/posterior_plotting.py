"""Plot distributions and latent states from Bayesian CTRV posteriors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ship_trajectory_prediction.evaluation.reporting import (
    posterior_variable_samples,
)

if TYPE_CHECKING:
    from ship_trajectory_prediction.models.bayesian_ctrv import VIRunResult
    from ship_trajectory_prediction.trajectory import TrajectoryWindowData

DEFAULT_CREDIBLE_INTERVAL = 0.9
DEFAULT_HISTOGRAM_BINS = 40
DEFAULT_MAX_COMPARISON_RUNS = 6
PRIOR_DISPLAY_SCALE_MULTIPLIER = 3.0

NOISE_PARAMETER_METADATA = {
    "sigma_position_gps": (
        "Approximate posterior of position measurement noise",
        "Position measurement noise",
        "m",
    ),
    "sigma_position_process": (
        "Approximate posterior of position process noise",
        "Position process noise",
        "m/√s",
    ),
    "sigma_speed_process": (
        "Approximate posterior of speed process noise",
        "Speed process noise",
        "(m/s)/√s",
    ),
    "sigma_turn_rate_process": (
        "Approximate posterior of turn-rate process noise",
        "Turn-rate process noise",
        "(rad/s)/√s",
    ),
}

STATE_METADATA = {
    "x_state": ("Latent x position", "m"),
    "y_state": ("Latent y position", "m"),
    "speed_state": ("Latent speed", "m/s"),
    "heading_state": ("Latent heading", "rad"),
    "turn_rate_state": ("Latent turn rate", "rad/s"),
}


def plot_scalar_posterior(
    fit: Any,
    variable_name: str,
    *,
    credible_interval: float = DEFAULT_CREDIBLE_INTERVAL,
    reference_value: float | None = None,
    prior_scale: float | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    unit: str | None = None,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot actual scalar draws from a variational posterior approximation.

    No parametric distribution is fitted to the posterior. The histogram and
    all summary markers are computed directly from ``posterior_variable_samples``.
    An optional half-normal prior curve uses its configured scale exactly.
    """
    credible_interval = _validate_credible_interval(credible_interval)
    bins = _validate_bins(bins)
    samples = _scalar_samples(fit, variable_name)
    reference_value = _optional_finite_scalar("reference_value", reference_value)
    prior_scale = _optional_positive_finite_scalar("prior_scale", prior_scale)
    default_title, default_label, default_unit = _scalar_metadata(variable_name)
    if prior_scale is not None:
        default_title = default_title.replace(
            "Approximate posterior",
            "Prior and approximate posterior",
            1,
        )
    display_unit = default_unit if unit is None else unit
    figure, axis = _figure_and_axis(ax)
    _plot_distribution(
        axis,
        samples,
        credible_interval=credible_interval,
        reference_value=reference_value,
        prior_scale=prior_scale,
        title=default_title if title is None else title,
        xlabel=_label_with_unit(
            default_label if xlabel is None else xlabel, display_unit
        ),
        bins=bins,
    )
    figure.tight_layout()
    return figure, axis


def plot_scalar_prior_to_posterior_update(
    posterior_fits: Mapping[int, Any],
    variable_name: str,
    *,
    prior_scale: float,
    credible_interval: float = DEFAULT_CREDIBLE_INTERVAL,
    title: str | None = None,
    xlabel: str | None = None,
    unit: str | None = None,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot one fixed half-normal prior and posteriors from growing prefixes."""
    if not isinstance(posterior_fits, Mapping) or not posterior_fits:
        raise ValueError("posterior_fits must be a non-empty mapping.")
    variable_name = _validate_variable_name(variable_name)
    prior_scale = _optional_positive_finite_scalar("prior_scale", prior_scale)
    if prior_scale is None:
        raise ValueError("prior_scale must be a positive finite scalar.")
    credible_interval = _validate_credible_interval(credible_interval)
    bins = _validate_bins(bins)

    posterior_updates = []
    for observation_count, fit in posterior_fits.items():
        validated_count = _validate_positive_integer(
            "posterior_fits observation count",
            observation_count,
        )
        samples = _scalar_samples(fit, variable_name)
        if np.any(samples < 0):
            raise ValueError(
                "Half-normal prior comparisons require non-negative posterior draws."
            )
        posterior_updates.append((validated_count, samples))
    posterior_updates.sort(key=lambda update: update[0])

    default_title, default_label, default_unit = _scalar_metadata(variable_name)
    display_unit = default_unit if unit is None else unit
    if title is None:
        title = default_title.replace(
            "Approximate posterior of",
            "Prior-to-posterior update of",
            1,
        )
    combined_samples = np.concatenate([samples for _, samples in posterior_updates])
    display_max = max(
        float(np.max(combined_samples)),
        PRIOR_DISPLAY_SCALE_MULTIPLIER * prior_scale,
    )
    bin_edges = np.linspace(0.0, display_max, bins + 1)
    interval_percent = 100 * credible_interval

    figure, axis = _figure_and_axis(ax)
    _plot_half_normal_prior(axis, prior_scale, display_max=display_max)
    colors = plt.get_cmap("tab10").colors
    for update_index, (observation_count, samples) in enumerate(posterior_updates):
        lower, median, upper = _central_summary(samples, credible_interval)
        axis.hist(
            samples,
            bins=bin_edges,
            density=True,
            histtype="step",
            linewidth=1.7,
            color=colors[update_index % len(colors)],
            label=(
                f"Posterior after n={observation_count}: "
                f"median={_format_number(median)}, "
                f"{interval_percent:g}% CI="
                f"[{_format_number(lower)}, {_format_number(upper)}]"
            ),
        )
    axis.set_title(title)
    axis.set_xlabel(
        _label_with_unit(default_label if xlabel is None else xlabel, display_unit)
    )
    axis.set_ylabel("Density")
    axis.set_xlim(0.0, display_max)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_state_posterior_at_time(
    fit: Any,
    variable_name: str,
    time_index: int,
    *,
    time_values: Sequence[Any] | np.ndarray | None = None,
    credible_interval: float = DEFAULT_CREDIBLE_INTERVAL,
    reference_value: float | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    unit: str | None = None,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot actual approximate-posterior draws for one latent-state time index."""
    credible_interval = _validate_credible_interval(credible_interval)
    bins = _validate_bins(bins)
    samples = _state_samples(fit, variable_name)
    time_index = _validate_time_index(time_index, samples.shape[1])
    validated_time_values = _optional_time_values(
        time_values,
        expected_length=samples.shape[1],
    )
    reference_value = _optional_finite_scalar("reference_value", reference_value)

    default_label, default_unit = _state_metadata(variable_name)
    display_unit, convert_to_degrees = _state_display_unit(
        variable_name,
        unit,
        default_unit,
    )
    selected_samples = samples[:, time_index]
    if convert_to_degrees:
        selected_samples = np.degrees(selected_samples)
        if reference_value is not None:
            reference_value = float(np.degrees(reference_value))

    if title is None:
        time_description = f"index {time_index}"
        if validated_time_values is not None:
            time_description += ", " + _format_time_value(
                validated_time_values[time_index]
            )
        title = f"Approximate posterior of {variable_name} at {time_description}"

    figure, axis = _figure_and_axis(ax)
    _plot_distribution(
        axis,
        selected_samples,
        credible_interval=credible_interval,
        reference_value=reference_value,
        title=title,
        xlabel=_label_with_unit(
            default_label if xlabel is None else xlabel, display_unit
        ),
        bins=bins,
    )
    figure.tight_layout()
    return figure, axis


def plot_state_credible_band(
    fit: Any,
    variable_name: str,
    time_values: Sequence[Any] | np.ndarray,
    *,
    credible_interval: float = DEFAULT_CREDIBLE_INTERVAL,
    observed_values: Sequence[float] | np.ndarray | None = None,
    reference_values: Sequence[float] | np.ndarray | None = None,
    observed_label: str = "Observed values",
    reference_label: str = "Known reference",
    title: str | None = None,
    ylabel: str | None = None,
    unit: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a latent-state median and central credible band over observations.

    State draws must cover the observed inference window only. In particular,
    ``heading_state`` is summarized in its model-provided unwrapped form; draws
    are never wrapped to ``[-pi, pi]`` before quantiles are computed.
    """
    credible_interval = _validate_credible_interval(credible_interval)
    samples = _state_samples(fit, variable_name)
    time_values = _time_values(time_values, expected_length=samples.shape[1])
    observed_values = _optional_finite_vector(
        "observed_values",
        observed_values,
        expected_length=samples.shape[1],
    )
    reference_values = _optional_finite_vector(
        "reference_values",
        reference_values,
        expected_length=samples.shape[1],
    )
    observed_label = _validate_non_empty_label("observed_label", observed_label)
    reference_label = _validate_non_empty_label("reference_label", reference_label)
    default_label, default_unit = _state_metadata(variable_name)
    display_unit, convert_to_degrees = _state_display_unit(
        variable_name,
        unit,
        default_unit,
    )

    lower, median, upper = _central_summary(samples, credible_interval)
    if convert_to_degrees:
        lower = np.degrees(lower)
        median = np.degrees(median)
        upper = np.degrees(upper)
        if observed_values is not None:
            observed_values = np.degrees(observed_values)
        if reference_values is not None:
            reference_values = np.degrees(reference_values)

    interval_percent = 100 * credible_interval
    figure, axis = _figure_and_axis(ax)
    axis.plot(
        time_values,
        median,
        color="tab:blue",
        linewidth=2,
        label="Posterior median",
    )
    axis.fill_between(
        time_values,
        lower,
        upper,
        color="tab:blue",
        alpha=0.25,
        label=f"Central {interval_percent:g}% credible interval",
    )
    if observed_values is not None:
        axis.plot(
            time_values,
            observed_values,
            color="black",
            marker="o",
            markersize=3,
            linewidth=1,
            label=observed_label,
        )
    if reference_values is not None:
        axis.plot(
            time_values,
            reference_values,
            color="tab:green",
            linestyle="--",
            linewidth=1.5,
            label=reference_label,
        )
    axis.set_title(
        f"Approximate posterior of {variable_name}" if title is None else title
    )
    axis.set_xlabel(_time_axis_label(time_values))
    axis.set_ylabel(
        _label_with_unit(default_label if ylabel is None else ylabel, display_unit)
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_scalar_posterior_comparison(
    runs: Sequence[VIRunResult],
    variable_name: str,
    *,
    credible_interval: float = DEFAULT_CREDIBLE_INTERVAL,
    reference_value: float | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    unit: str | None = None,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    max_runs: int = DEFAULT_MAX_COMPARISON_RUNS,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Compare actual scalar variational posterior draws across VI runs.

    At most ``max_runs`` outlined density histograms are shown to keep the figure
    interpretable. Each legend entry reports algorithm, seed, posterior median,
    and the requested central credible interval.
    """
    credible_interval = _validate_credible_interval(credible_interval)
    bins = _validate_bins(bins)
    max_runs = _validate_positive_integer("max_runs", max_runs)
    runs = tuple(runs)
    if not runs:
        raise ValueError("runs must contain at least one VI run.")
    if len(runs) > max_runs:
        raise ValueError(f"At most {max_runs} VI runs can be plotted at once.")

    run_samples = []
    for run in runs:
        try:
            algorithm = run.algorithm
            seed = run.seed
            fit = run.fit
        except AttributeError as error:
            raise TypeError(
                "Each run must provide algorithm, seed, and fit attributes."
            ) from error
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise ValueError("Each run algorithm must be a non-empty string.")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("Each run seed must be an integer.")
        run_samples.append(
            (algorithm.strip(), int(seed), _scalar_samples(fit, variable_name))
        )

    reference_value = _optional_finite_scalar("reference_value", reference_value)
    default_title, default_label, default_unit = _scalar_metadata(variable_name)
    display_unit = default_unit if unit is None else unit
    combined_samples = np.concatenate([samples for _, _, samples in run_samples])
    bin_edges = np.histogram_bin_edges(combined_samples, bins=bins)
    interval_percent = 100 * credible_interval

    figure, axis = _figure_and_axis(ax)
    colors = plt.get_cmap("tab10").colors
    for run_index, (algorithm, seed, samples) in enumerate(run_samples):
        lower, median, upper = _central_summary(samples, credible_interval)
        label = (
            f"{algorithm}, seed {seed}: median={_format_number(median)}, "
            f"{interval_percent:g}% CI=[{_format_number(lower)}, "
            f"{_format_number(upper)}]"
        )
        axis.hist(
            samples,
            bins=bin_edges,
            density=True,
            histtype="step",
            linewidth=1.7,
            color=colors[run_index % len(colors)],
            label=label,
        )
    if reference_value is not None:
        axis.axvline(
            reference_value,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Known reference: {_format_number(reference_value)}",
        )
    axis.set_title(
        f"VI comparison: {default_title.lower()}" if title is None else title
    )
    axis.set_xlabel(
        _label_with_unit(default_label if xlabel is None else xlabel, display_unit)
    )
    axis.set_ylabel("Density")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def save_bayesian_ctrv_posterior_plots(
    fit: Any,
    window: TrajectoryWindowData,
    output_directory: str | Path,
    *,
    credible_interval: float = DEFAULT_CREDIBLE_INTERVAL,
    selected_time_indices: Sequence[int] | None = None,
    reference_parameters: Mapping[str, float] | None = None,
    reference_states: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    prior_scales: Mapping[str, float] | None = None,
    include_speed_gps_reference: bool = False,
    dpi: int = 200,
) -> dict[str, Path]:
    """Save standard Bayesian CTRV posterior-draw diagnostics.

    Only latent states over the observed inference window are plotted. If
    requested, GPS speed is labelled as an external post-fit reference; it was
    not used to fit the position-only model. The function reuses the public plot
    functions, closes every saved figure, and returns stable output names.
    """
    credible_interval = _validate_credible_interval(credible_interval)
    dpi = _validate_positive_integer("dpi", dpi)
    observation_count = _window_observation_count(window)
    time_values = _finite_window_vector(
        "window.time_seconds",
        window.time_seconds,
        window.observed_slice,
        observation_count,
    )
    if not isinstance(include_speed_gps_reference, bool):
        raise TypeError("include_speed_gps_reference must be a boolean.")
    speed_gps_reference = None
    if include_speed_gps_reference:
        speed_gps_reference = _finite_window_vector(
            "window.gps_speed_mps",
            window.gps_speed_mps,
            window.observed_slice,
            observation_count,
        )
    selected_time_indices = _selected_time_indices(
        selected_time_indices,
        observation_count,
    )
    reference_parameters = _reference_parameters(reference_parameters)
    reference_states = _reference_states(reference_states, observation_count)
    prior_scales = _validated_noise_prior_scales(prior_scales)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    generated_paths: dict[str, Path] = {}

    for variable_name in NOISE_PARAMETER_METADATA:
        figure, _ = plot_scalar_posterior(
            fit,
            variable_name,
            credible_interval=credible_interval,
            reference_value=reference_parameters.get(variable_name),
            prior_scale=prior_scales.get(variable_name),
        )
        _save_and_close(
            figure,
            output_directory / f"posterior_{variable_name}.png",
            dpi=dpi,
            generated_paths=generated_paths,
        )

    state_plot_options = {
        "speed_state": {
            "observed_values": speed_gps_reference,
            "observed_label": "External GPS-speed reference",
            "reference_values": reference_states.get("speed_state"),
        },
        "turn_rate_state": {
            "observed_values": None,
            "reference_values": reference_states.get("turn_rate_state"),
        },
    }
    for variable_name, options in state_plot_options.items():
        figure, _ = plot_state_credible_band(
            fit,
            variable_name,
            time_values,
            credible_interval=credible_interval,
            **options,
        )
        _save_and_close(
            figure,
            output_directory / f"posterior_{variable_name}_over_time.png",
            dpi=dpi,
            generated_paths=generated_paths,
        )

    for time_index in selected_time_indices:
        for variable_name in state_plot_options:
            reference_values = reference_states.get(variable_name)
            reference_value = (
                None if reference_values is None else reference_values[time_index]
            )
            figure, _ = plot_state_posterior_at_time(
                fit,
                variable_name,
                time_index,
                time_values=time_values,
                credible_interval=credible_interval,
                reference_value=reference_value,
            )
            _save_and_close(
                figure,
                output_directory / f"posterior_{variable_name}_t_{time_index:03d}.png",
                dpi=dpi,
                generated_paths=generated_paths,
            )
    return generated_paths


def show_bayesian_ctrv_posterior_plots(
    fit: Any,
    window: TrajectoryWindowData,
    *,
    selected_time_indices: Sequence[int] | None = None,
    credible_interval: float = DEFAULT_CREDIBLE_INTERVAL,
    prior_scales: Mapping[str, float] | None = None,
    include_speed_gps_reference: bool = False,
) -> None:
    """Display position-only CTRV posterior figures one at a time.

    GPS speed can optionally be overlaid as an explicitly external post-fit
    reference. It is not part of the fitted observation model.
    """
    credible_interval = _validate_credible_interval(credible_interval)
    observation_count = _window_observation_count(window)
    time_values = _finite_window_vector(
        "window.time_seconds",
        window.time_seconds,
        window.observed_slice,
        observation_count,
    )
    if not isinstance(include_speed_gps_reference, bool):
        raise TypeError("include_speed_gps_reference must be a boolean.")
    speed_gps_reference = None
    if include_speed_gps_reference:
        speed_gps_reference = _finite_window_vector(
            "window.gps_speed_mps",
            window.gps_speed_mps,
            window.observed_slice,
            observation_count,
        )
    selected_time_indices = _selected_time_indices(
        selected_time_indices,
        observation_count,
    )
    prior_scales = _validated_noise_prior_scales(prior_scales)

    for variable_name in NOISE_PARAMETER_METADATA:
        figure, _ = plot_scalar_posterior(
            fit,
            variable_name,
            credible_interval=credible_interval,
            prior_scale=prior_scales.get(variable_name),
        )
        _show_and_close(figure)

    state_plot_options = {
        "speed_state": {
            "observed_values": speed_gps_reference,
            "observed_label": "External GPS-speed reference",
        },
        "turn_rate_state": {"observed_values": None},
    }
    for variable_name, options in state_plot_options.items():
        figure, _ = plot_state_credible_band(
            fit,
            variable_name,
            time_values,
            credible_interval=credible_interval,
            **options,
        )
        _show_and_close(figure)

    for time_index in selected_time_indices:
        for variable_name in state_plot_options:
            figure, _ = plot_state_posterior_at_time(
                fit,
                variable_name,
                time_index,
                time_values=time_values,
                credible_interval=credible_interval,
            )
            _show_and_close(figure)


def show_bayesian_ctrv_prior_update_plots(
    posterior_fits: Mapping[int, Any],
    *,
    prior_scales: Mapping[str, float],
    credible_interval: float = DEFAULT_CREDIBLE_INTERVAL,
) -> None:
    """Display fixed-prior updates for all Bayesian CTRV noise parameters."""
    prior_scales = _validated_noise_prior_scales(
        prior_scales,
        require_all=True,
    )
    for variable_name in NOISE_PARAMETER_METADATA:
        figure, _ = plot_scalar_prior_to_posterior_update(
            posterior_fits,
            variable_name,
            prior_scale=prior_scales[variable_name],
            credible_interval=credible_interval,
        )
        _show_and_close(figure)


def _plot_distribution(
    axis: Axes,
    samples: np.ndarray,
    *,
    credible_interval: float,
    reference_value: float | None,
    prior_scale: float | None = None,
    title: str,
    xlabel: str,
    bins: int,
) -> None:
    """Plot one empirical histogram and summaries from posterior draws."""
    lower, median, upper = _central_summary(samples, credible_interval)
    mean = float(np.mean(samples))
    interval_percent = 100 * credible_interval
    if prior_scale is not None:
        display_max = max(
            float(np.max(samples)),
            PRIOR_DISPLAY_SCALE_MULTIPLIER * prior_scale,
        )
        _plot_half_normal_prior(axis, prior_scale, display_max=display_max)
        axis.set_xlim(0.0, display_max)
    axis.hist(
        samples,
        bins=bins,
        density=True,
        color="tab:blue",
        alpha=0.65,
        edgecolor="white",
        linewidth=0.5,
        label="Variational posterior draws",
    )
    axis.axvline(
        mean,
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Posterior mean: {_format_number(mean)}",
    )
    axis.axvline(
        median,
        color="tab:red",
        linewidth=1.8,
        label=f"Posterior median: {_format_number(median)}",
    )
    axis.axvline(
        lower,
        color="tab:purple",
        linestyle=":",
        linewidth=1.5,
        label=(
            f"Central {interval_percent:g}% credible interval: "
            f"[{_format_number(lower)}, {_format_number(upper)}]"
        ),
    )
    axis.axvline(
        upper,
        color="tab:purple",
        linestyle=":",
        linewidth=1.5,
        label="_nolegend_",
    )
    if reference_value is not None:
        axis.axvline(
            reference_value,
            color="black",
            linestyle="-.",
            linewidth=1.5,
            label=f"Known reference: {_format_number(reference_value)}",
        )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")
    axis.grid(alpha=0.25)
    axis.legend()


def _plot_half_normal_prior(axis: Axes, scale: float, *, display_max: float) -> None:
    """Draw an exact half-normal prior density over its positive support."""
    x_values = np.linspace(0.0, display_max, 500)
    density = np.sqrt(2.0 / np.pi) / scale * np.exp(-(x_values**2) / (2.0 * scale**2))
    axis.plot(
        x_values,
        density,
        color="black",
        linewidth=2.0,
        label=f"Half-normal prior: scale={_format_number(scale)}",
    )


def _scalar_samples(fit: Any, variable_name: str) -> np.ndarray:
    """Extract one non-empty finite scalar draw sequence."""
    variable_name = _validate_variable_name(variable_name)
    samples = posterior_variable_samples(fit, variable_name)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError(
            f"Posterior variable {variable_name!r} must contain "
            "non-empty one-dimensional draws."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError(
            f"Posterior variable {variable_name!r} must contain only finite draws."
        )
    return samples


def _state_samples(fit: Any, variable_name: str) -> np.ndarray:
    """Extract one non-empty finite draw-by-time latent-state matrix."""
    variable_name = _validate_variable_name(variable_name)
    samples = posterior_variable_samples(fit, variable_name)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError(
            f"Posterior variable {variable_name!r} must have shape "
            "(number_of_draws, number_of_time_steps)."
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError(
            f"Posterior variable {variable_name!r} must contain only finite draws."
        )
    return samples


def _central_summary(
    samples: np.ndarray,
    credible_interval: float,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """Return lower quantile, median, and upper quantile over draws."""
    lower_probability = (1 - credible_interval) / 2
    upper_probability = 1 - lower_probability
    return (
        np.quantile(samples, lower_probability, axis=0),
        np.median(samples, axis=0),
        np.quantile(samples, upper_probability, axis=0),
    )


def _validate_credible_interval(value: float) -> float:
    """Return a finite probability strictly between zero and one."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("credible_interval must be between 0 and 1.") from error
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("credible_interval must be between 0 and 1.")
    return value


def _validate_bins(bins: int) -> int:
    """Return a positive integer histogram bin count."""
    return _validate_positive_integer("bins", bins)


def _validate_positive_integer(name: str, value: int) -> int:
    """Return one positive integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_variable_name(variable_name: str) -> str:
    """Return a non-empty posterior variable name."""
    if not isinstance(variable_name, str) or not variable_name.strip():
        raise ValueError("variable_name must be a non-empty string.")
    return variable_name.strip()


def _validate_non_empty_label(name: str, value: str) -> str:
    """Return a non-empty plot label."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _validate_time_index(time_index: int, time_count: int) -> int:
    """Return a valid zero-based latent-state time index."""
    if isinstance(time_index, bool) or not isinstance(time_index, (int, np.integer)):
        raise ValueError("time_index must be a zero-based integer.")
    if not 0 <= time_index < time_count:
        raise IndexError(
            f"time_index must be between 0 and {time_count - 1} (zero-based)."
        )
    return int(time_index)


def _time_values(values: Sequence[Any] | np.ndarray, *, expected_length: int):
    """Return one finite numeric or datetime time axis of the expected length."""
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != expected_length:
        raise ValueError(f"time_values must contain exactly {expected_length} values.")
    if np.issubdtype(result.dtype, np.number):
        if not np.all(np.isfinite(result.astype(float))):
            raise ValueError("time_values must contain only finite values.")
    elif np.issubdtype(result.dtype, np.datetime64):
        if np.any(np.isnat(result)):
            raise ValueError("time_values must not contain missing timestamps.")
    elif any(value is None for value in result):
        raise ValueError("time_values must not contain missing values.")
    return result


def _optional_time_values(values, *, expected_length: int):
    """Validate an optional time axis."""
    if values is None:
        return None
    return _time_values(values, expected_length=expected_length)


def _optional_finite_vector(
    name: str,
    values: Sequence[float] | np.ndarray | None,
    *,
    expected_length: int,
) -> np.ndarray | None:
    """Validate one optional finite numeric vector."""
    if values is None:
        return None
    result = np.asarray(values, dtype=float)
    if result.shape != (expected_length,):
        raise ValueError(f"{name} must contain exactly {expected_length} values.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _optional_finite_scalar(name: str, value: float | None) -> float | None:
    """Validate one optional finite numeric scalar."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite scalar.") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar.")
    return result


def _optional_positive_finite_scalar(
    name: str,
    value: float | None,
) -> float | None:
    """Validate one optional strictly positive finite scalar."""
    result = _optional_finite_scalar(name, value)
    if result is not None and result <= 0:
        raise ValueError(f"{name} must be a positive finite scalar.")
    return result


def _figure_and_axis(ax: Axes | None) -> tuple[Figure, Axes]:
    """Create an axis or reuse the caller-provided axis."""
    if ax is None:
        return plt.subplots(figsize=(9, 6))
    if not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib Axes instance or None.")
    return ax.figure, ax


def _scalar_metadata(variable_name: str) -> tuple[str, str, str | None]:
    """Return default title, quantity label, and unit for a scalar."""
    return NOISE_PARAMETER_METADATA.get(
        variable_name,
        (f"Approximate posterior of {variable_name}", variable_name, None),
    )


def _state_metadata(variable_name: str) -> tuple[str, str | None]:
    """Return default quantity label and unit for a latent state."""
    return STATE_METADATA.get(variable_name, (variable_name, None))


def _state_display_unit(
    variable_name: str,
    requested_unit: str | None,
    default_unit: str | None,
) -> tuple[str | None, bool]:
    """Return display unit and whether an unwrapped heading needs conversion."""
    display_unit = default_unit if requested_unit is None else requested_unit
    convert_to_degrees = variable_name == "heading_state" and display_unit in {
        "deg",
        "°",
    }
    if convert_to_degrees:
        display_unit = "°"
    return display_unit, convert_to_degrees


def _label_with_unit(label: str, unit: str | None) -> str:
    """Append a unit to an axis label when one is available."""
    return label if not unit else f"{label} [{unit}]"


def _format_number(value: float | np.ndarray) -> str:
    """Format one posterior summary value compactly."""
    return f"{float(value):.4g}"


def _format_time_value(value: Any) -> str:
    """Format one numeric second value or timestamp for a plot title."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"t = {float(value):g} s"
    return f"time = {value}"


def _time_axis_label(time_values: np.ndarray) -> str:
    """Return the appropriate label for numeric seconds or timestamps."""
    return "Time [s]" if np.issubdtype(time_values.dtype, np.number) else "Time"


def _window_observation_count(window: TrajectoryWindowData) -> int:
    """Validate and return a positive observation count."""
    try:
        observation_count = window.observation_count
        _ = (window.observed_slice, window.time_seconds, window.gps_speed_mps)
    except AttributeError as error:
        raise TypeError(
            "window must provide observed trajectory-window attributes."
        ) from error
    return _validate_positive_integer("window.observation_count", observation_count)


def _finite_window_vector(
    name: str,
    values: Sequence[float] | np.ndarray,
    observed_slice: slice,
    observation_count: int,
) -> np.ndarray:
    """Return one finite observed-window vector."""
    result = np.asarray(values, dtype=float)[observed_slice]
    if result.shape != (observation_count,) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} must provide {observation_count} finite observed values."
        )
    return result


def _selected_time_indices(
    selected_time_indices: Sequence[int] | None,
    observation_count: int,
) -> tuple[int, ...]:
    """Validate, deduplicate, and preserve selected zero-based time indices."""
    if selected_time_indices is None:
        return ()
    try:
        values = tuple(selected_time_indices)
    except TypeError as error:
        raise ValueError(
            "selected_time_indices must be a sequence of integers."
        ) from error
    validated = tuple(
        _validate_time_index(value, observation_count) for value in values
    )
    return tuple(dict.fromkeys(validated))


def _validated_noise_prior_scales(
    prior_scales: Mapping[str, float] | None,
    *,
    require_all: bool = False,
) -> dict[str, float]:
    """Validate optional half-normal scales for known noise parameters."""
    if prior_scales is None:
        if require_all:
            raise ValueError("prior_scales must contain every noise parameter.")
        return {}
    if not isinstance(prior_scales, Mapping):
        raise TypeError("prior_scales must be a mapping or None.")
    unknown_names = set(prior_scales).difference(NOISE_PARAMETER_METADATA)
    if unknown_names:
        raise ValueError(f"Unknown noise prior scales: {sorted(unknown_names)}")
    validated = {}
    for name, scale in prior_scales.items():
        validated_scale = _optional_positive_finite_scalar(
            f"prior_scales[{name!r}]",
            scale,
        )
        if validated_scale is None:
            raise ValueError(f"prior_scales[{name!r}] must not be None.")
        validated[name] = validated_scale
    if require_all:
        missing_names = set(NOISE_PARAMETER_METADATA).difference(validated)
        if missing_names:
            raise ValueError(
                "prior_scales must contain every noise parameter; missing "
                f"{sorted(missing_names)}"
            )
    return validated


def _reference_parameters(
    values: Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate optional known scalar parameter values."""
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("reference_parameters must be a mapping or None.")
    return {
        name: _optional_finite_scalar(f"reference_parameters[{name!r}]", value)
        for name, value in values.items()
    }


def _reference_states(
    values: Mapping[str, Sequence[float] | np.ndarray] | None,
    observation_count: int,
) -> dict[str, np.ndarray]:
    """Validate optional known latent-state trajectories."""
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("reference_states must be a mapping or None.")
    return {
        name: _optional_finite_vector(
            f"reference_states[{name!r}]",
            state_values,
            expected_length=observation_count,
        )
        for name, state_values in values.items()
    }


def _save_and_close(
    figure: Figure,
    path: Path,
    *,
    dpi: int,
    generated_paths: dict[str, Path],
) -> None:
    """Save one owned posterior figure and close it even if saving fails."""
    try:
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
    generated_paths[path.stem] = path


def _show_and_close(figure: Figure) -> None:
    """Show one figure with blocking display and always release it afterward."""
    try:
        plt.show(block=True)
    finally:
        plt.close(figure)


__all__ = [
    "plot_scalar_posterior",
    "plot_scalar_posterior_comparison",
    "plot_scalar_prior_to_posterior_update",
    "plot_state_credible_band",
    "plot_state_posterior_at_time",
    "save_bayesian_ctrv_posterior_plots",
    "show_bayesian_ctrv_posterior_plots",
    "show_bayesian_ctrv_prior_update_plots",
]
