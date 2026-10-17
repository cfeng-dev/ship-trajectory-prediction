"""Plot rolling-evaluation diagnostics for trajectory forecasts."""

import matplotlib.pyplot as plt
import numpy as np

import bayestraj.observations.plotting as observation_plotting

RUNTIME_PLOT_STYLE = observation_plotting.ShipDataPlotStyle()
ROLLING_DIAGNOSTIC_PLOT_STYLES = (
    ("RMSE [m]", "RMSE", "#0072B2", "rmse_m", "Mean RMSE: {mean:.2f} m"),
    (
        "Log predictive density",
        "Log predictive density",
        "#CC79A7",
        "mean_log_predictive_density",
        "Mean log predictive density: {mean:.3f}",
    ),
    ("ADE [m]", "ADE", "#E69F00", "ade_m", "Mean ADE: {mean:.2f} m"),
    ("FDE [m]", "FDE", "#D55E00", "fde_m", "Mean FDE: {mean:.2f} m"),
    (
        "Energy Score [m]",
        "Energy Score",
        "#009E73",
        "energy_score_m",
        "Mean Energy Score: {mean:.2f} m",
    ),
    (
        "Inference time [s]",
        "Inference time",
        "#56B4E9",
        "fit_runtime_seconds",
        "Mean inference time: {mean:.3f} s",
    ),
)


def plot_bayesian_ctrv_inference_runtime(predictions, *, inference_selection=None):
    """Plot Bayesian CTRV inference time per rolling forecast window."""
    required_columns = {
        "window_index",
        "observation_count",
        "fit_runtime_seconds",
        "inference_method",
    }
    missing_columns = sorted(required_columns.difference(predictions.columns))
    if missing_columns:
        raise ValueError(f"Missing fit-runtime columns: {missing_columns}")
    windows = predictions.groupby("window_index", sort=True).first()
    inference_methods = windows["inference_method"].dropna().unique()
    if len(inference_methods) != 1:
        raise ValueError(
            "Runtime plot requires exactly one inference_method per configuration."
        )
    inference_title = _format_inference_title(
        inference_selection or str(inference_methods[0])
    )
    return _plot_window_runtime(
        predictions,
        runtime_column="fit_runtime_seconds",
        title=f"Bayesian-CTRV ({inference_title})",
        label="Inferenzzeit pro Vorhersagefenster",
        y_axis_label="Inferenzzeit pro Vorhersagefenster [s]",
    )


def plot_bayesian_ctrv_rolling_diagnostics(predictions, *, inference_selection=None):
    """Plot each rolling Bayesian CTRV metric in its own sequential figure."""
    required_columns = {
        "window_index",
        "forecast_origin_seconds",
        "horizon_step",
        "position_error_m",
        "log_predictive_density",
        "energy_score_m",
        "fit_runtime_seconds",
        "inference_method",
    }
    missing_columns = sorted(required_columns.difference(predictions.columns))
    if missing_columns:
        raise ValueError(f"Missing rolling-diagnostic columns: {missing_columns}")

    inference_methods = predictions["inference_method"].dropna().unique()
    if len(inference_methods) != 1:
        raise ValueError(
            "Rolling diagnostics require exactly one inference_method per "
            "configuration."
        )
    time_seconds, metric_values = _rolling_window_metric_series(predictions)
    elapsed_time_seconds = time_seconds - time_seconds[0]
    inference_title = _format_inference_title(
        inference_selection or str(inference_methods[0])
    )
    figures = []
    for (
        y_axis_label,
        label,
        color,
        value_name,
        mean_label,
    ) in ROLLING_DIAGNOSTIC_PLOT_STYLES:
        figure, axis = plt.subplots(figsize=RUNTIME_PLOT_STYLE.speed_figure_size)
        axis.plot(
            elapsed_time_seconds,
            metric_values[value_name],
            marker="o",
            markersize=3,
            color=color,
            label=label,
        )
        mean = float(np.mean(metric_values[value_name]))
        axis.axhline(
            mean,
            color=color,
            linestyle="--",
            label=mean_label.format(mean=mean),
        )
        axis.set_xlabel(
            "Time [s]",
            fontsize=RUNTIME_PLOT_STYLE.axis_label_font_size,
        )
        axis.set_ylabel(
            y_axis_label,
            fontsize=RUNTIME_PLOT_STYLE.axis_label_font_size,
        )
        axis.tick_params(
            axis="both",
            labelsize=RUNTIME_PLOT_STYLE.axis_tick_font_size,
        )
        axis.legend(loc=RUNTIME_PLOT_STYLE.legend_location)
        figure.suptitle(f"Bayesian CTRV ({inference_title})")
        figure.tight_layout()
        plt.show()
        figures.append((figure, axis))
    return tuple(figures)


def _rolling_window_metric_series(predictions):
    """Aggregate each forecast window into one time-indexed diagnostic row."""
    windows = []
    for _, window in predictions.groupby("window_index", sort=False):
        time_seconds = window["forecast_origin_seconds"].to_numpy(dtype=float)
        if not np.all(np.isfinite(time_seconds)) or not np.allclose(
            time_seconds,
            time_seconds[0],
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "forecast_origin_seconds must be one finite time per forecast window."
            )
        errors_m = window["position_error_m"].to_numpy(dtype=float)
        log_predictive_density = window["log_predictive_density"].to_numpy(dtype=float)
        energy_score_m = window["energy_score_m"].to_numpy(dtype=float)
        fit_runtime_seconds = window["fit_runtime_seconds"].to_numpy(dtype=float)
        if not (
            np.all(np.isfinite(errors_m))
            and np.all(np.isfinite(log_predictive_density))
            and np.all(np.isfinite(energy_score_m))
            and np.all(np.isfinite(fit_runtime_seconds))
        ):
            raise ValueError("Rolling diagnostic values must be finite.")
        final_row = window.loc[window["horizon_step"].idxmax()]
        windows.append(
            (
                time_seconds[0],
                float(np.sqrt(np.mean(errors_m**2))),
                float(np.mean(log_predictive_density)),
                float(np.mean(errors_m)),
                float(final_row["position_error_m"]),
                float(np.mean(energy_score_m)),
                float(fit_runtime_seconds[0]),
            )
        )
    if not windows:
        raise ValueError("Rolling diagnostics require at least one forecast window.")
    windows.sort(key=lambda values: values[0])
    values = np.asarray(windows, dtype=float)
    return values[:, 0], {
        "rmse_m": values[:, 1],
        "mean_log_predictive_density": values[:, 2],
        "ade_m": values[:, 3],
        "fde_m": values[:, 4],
        "energy_score_m": values[:, 5],
        "fit_runtime_seconds": values[:, 6],
    }


def _plot_window_runtime(
    predictions,
    *,
    runtime_column,
    title,
    label,
    y_axis_label,
):
    """Draw one consistently styled per-window runtime series."""
    windows = predictions.groupby("window_index", sort=True).first()
    figure, axis = plt.subplots(figsize=RUNTIME_PLOT_STYLE.speed_figure_size)
    axis.plot(
        windows["observation_count"].tolist(),
        windows[runtime_column].tolist(),
        marker="o",
        markersize=3,
        color=RUNTIME_PLOT_STYLE.derived_data_color,
        label=label,
    )
    axis.set_xlabel(
        "Anzahl bisher beobachteter Positionen",
        fontsize=RUNTIME_PLOT_STYLE.axis_label_font_size,
    )
    axis.set_ylabel(
        y_axis_label,
        fontsize=RUNTIME_PLOT_STYLE.axis_label_font_size,
    )
    axis.set_title(
        title,
        pad=RUNTIME_PLOT_STYLE.title_pad,
        fontsize=RUNTIME_PLOT_STYLE.title_font_size,
        fontweight=RUNTIME_PLOT_STYLE.title_font_weight,
    )
    axis.tick_params(axis="both", labelsize=RUNTIME_PLOT_STYLE.axis_tick_font_size)
    axis.legend(loc=RUNTIME_PLOT_STYLE.legend_location)
    figure.tight_layout()
    plt.show()
    return figure, axis


def _format_inference_title(inference_selection):
    """Return a concise title label for the selected inference configuration."""
    inference_method, separator, window_mode = inference_selection.partition("_")
    title = inference_method.upper()
    if separator and window_mode in {"sliding", "expanding"}:
        return f"{title}, {window_mode.title()} Window"
    return title
