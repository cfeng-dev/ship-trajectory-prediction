"""Plot computation-time diagnostics for trajectory evaluations."""

import matplotlib.pyplot as plt

import bayestraj.observations.plotting as observation_plotting

RUNTIME_PLOT_STYLE = observation_plotting.ShipDataPlotStyle()


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
