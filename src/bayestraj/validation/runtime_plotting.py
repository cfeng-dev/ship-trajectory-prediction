"""Plot rolling-evaluation diagnostics for trajectory forecasts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bayestraj.observations.plotting as observation_plotting

RUNTIME_PLOT_STYLE = observation_plotting.ShipDataPlotStyle()
COMPARISON_MODEL_COLORS = ("#0072B2", "#E69F00")
COMPARISON_LEGEND_PADDING_POINTS = 8.0
COMPARISON_INFERENCE_COLORS = {
    "imm": "#E69F00",
    "smc": "#0072B2",
    "vi": "#009E73",
    "mcmc": "#CC79A7",
    "sequential_vi": "#D55E00",
}
COMPARISON_METRIC_Y_LABELS = {
    "rmse_m": "Root mean squared error [m]",
    "mean_negative_log_predictive_density": "Negative log predictive density",
    "ade_m": "Average displacement error [m]",
    "fde_m": "Final displacement error [m]",
    "fit_runtime_seconds": "Laufzeit [s]",
}
ROLLING_DIAGNOSTIC_PLOT_STYLES = (
    ("RMSE [m]", "RMSE", "#0072B2", "rmse_m", "Mittlerer RMSE: {mean:.2f} m"),
    (
        "NLPD",
        "NLPD",
        "#CC79A7",
        "mean_negative_log_predictive_density",
        "Mittlere NLPD: {mean:.3f}",
    ),
    ("ADE [m]", "ADE", "#E69F00", "ade_m", "Mittlerer ADE: {mean:.2f} m"),
    ("FDE [m]", "FDE", "#D55E00", "fde_m", "Mittlerer FDE: {mean:.2f} m"),
    (
        "Energy Score [m]",
        "Energy Score",
        "#009E73",
        "energy_score_m",
        "Mittlerer Energy Score: {mean:.2f} m",
    ),
    (
        "Inferenzzeit [s]",
        "Inferenzzeit",
        "#56B4E9",
        "fit_runtime_seconds",
        "Mittlere Inferenzzeit: {mean:.3f} s",
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
    return _plot_rolling_diagnostics(
        predictions,
        title_template="Bayesian CTRV ({inference_title})",
        inference_selection=inference_selection,
    )


def plot_imm_rolling_diagnostics(predictions):
    """Plot rolling IMM metrics in the same style as the Bayesian evaluator."""
    return _plot_rolling_diagnostics(
        predictions,
        title_template="IMM",
        runtime_label="Update-/Forecast-Laufzeit",
    )


def plot_bayesian_imm_comparison(bayesian, imm, *, show=True, output_dir=None):
    """Plot per-origin metrics and final coverage over all matched predictions."""
    return plot_model_comparison(
        {"bayesian_ctrv": bayesian, "imm": imm},
        labels={
            "bayesian_ctrv": f"Bayesian CTRV ({bayesian['inference_method'].iloc[0].upper()})",
            "imm": "IMM",
        },
        title="Bayesian CTRV vs. IMM",
        show=show,
        output_dir=output_dir,
    )


def plot_model_comparison(
    models,
    *,
    labels=None,
    title="Model comparison",
    show=True,
    output_dir=None,
    show_points=True,
    show_mean_lines=True,
    metric_y_scale="linear",
):
    """Plot all selected models using the same metric and coverage styles."""
    if len(models) < 2:
        raise ValueError("Select at least two models for comparison.")
    if metric_y_scale not in {"linear", "symlog"}:
        raise ValueError("metric_y_scale must be 'linear' or 'symlog'.")
    names = list(models)
    tables = list(models.values())
    labels = [name if labels is None else labels[name] for name in names]
    series = [_rolling_window_metric_series(table) for table in tables]
    elapsed = series[0][0] - series[0][0][0]
    style = RUNTIME_PLOT_STYLE
    colors = [
        COMPARISON_INFERENCE_COLORS.get(
            table["inference_method"].iloc[0],
            COMPARISON_MODEL_COLORS[index % len(COMPARISON_MODEL_COLORS)],
        )
        for index, table in enumerate(tables)
    ]
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    figures, rows = [], []
    plot_specs = [
        (COMPARISON_METRIC_Y_LABELS.get(name, ylabel), name, None)
        for ylabel, _, _, name, _ in ROLLING_DIAGNOSTIC_PLOT_STYLES
    ] + [
        (f"Joint {percent}% coverage [%]", f"joint_coverage_{percent}", percent / 100)
        for percent in (50, 90)
    ]
    for ylabel, name, nominal in plot_specs:
        figure, axis = plt.subplots(figsize=style.speed_figure_size)
        if nominal is None and metric_y_scale == "symlog":
            # Adapt the linear range to each metric's units and typical low values.
            magnitudes = np.abs(np.concatenate([values[name] for _, values in series]))
            positive = magnitudes[magnitudes > 0]
            threshold = float(np.percentile(positive, 10)) if positive.size else 1.0
            axis.set_yscale("symlog", linthresh=threshold)
        if nominal is None:
            averages = []
            for (_, values), label, color in zip(series, labels, colors, strict=True):
                metric = values[name]
                average = float(np.mean(metric))
                averages.append(average)
                axis.plot(
                    elapsed,
                    metric,
                    marker="o" if show_points else "None",
                    markersize=3,
                    color=color,
                    label=label,
                )
                if show_mean_lines:
                    axis.axhline(
                        average,
                        color=color,
                        linestyle="--",
                        label=f"{label}: Mittel = {average:.3f}",
                    )
            lower, upper = axis.get_ylim()
            axis.set_ylim(min(lower, 0.0), max(upper, 0.0))
            axis.set_xlabel("Zeit [s]", fontsize=style.axis_label_font_size)
        else:
            column = f"joint_covered_{100 * nominal:.0f}"
            averages = [float(table[column].mean()) for table in tables]
            bars = axis.bar(
                np.arange(len(labels)),
                100 * np.asarray(averages),
                color=colors,
                width=0.55,
            )
            axis.set_xticks(
                np.arange(len(labels)),
                [
                    label.replace(" (", "\n(") if len(labels) > 2 else label
                    for label in labels
                ],
            )
            for bar in bars:
                percent = bar.get_height()
                inside = percent >= 95
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    percent - 2 if inside else percent + 2,
                    f"{percent:.1f}%",
                    ha="center",
                    va="top" if inside else "bottom",
                    fontsize=style.axis_tick_font_size,
                )
            axis.axhline(
                100 * nominal,
                color="0.35",
                linestyle=":",
                label=f"Nominal {100 * nominal:.0f}%",
            )
            axis.set_ylim(0, 100)
        axis.set_ylabel(ylabel, fontsize=style.axis_label_font_size)
        axis.tick_params(axis="both", labelsize=style.axis_tick_font_size)
        axis.set_axisbelow(True)
        axis.grid(
            True,
            which="major",
            axis="both" if nominal is None else "y",
            color="0.7",
            alpha=0.2,
        )
        legend = axis.legend(loc="upper right", ncol=1)
        figure.suptitle(
            title,
            fontsize=style.title_font_size,
            fontweight=style.title_font_weight,
        )
        figure.tight_layout()
        if nominal is None:
            _reserve_legend_headroom(figure, axis, legend)
        if output_dir is not None:
            figure.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        figures.append((figure, axis))
        rows.append(
            {
                "metric": name,
                **dict(zip(names, averages, strict=True)),
                "preferred": "lower"
                if nominal is None
                else f"near {100 * nominal:.0f}%",
            }
        )
    return pd.DataFrame(rows), tuple(figures)


def _reserve_legend_headroom(figure, axis, legend):
    """Keep the legend inside the axes and above the metric curves."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    padding = COMPARISON_LEGEND_PADDING_POINTS * figure.dpi / 72.0
    legend_height = legend.get_window_extent(renderer).height
    width, height = figure.get_size_inches()
    figure.set_size_inches(width, height + (legend_height + 2 * padding) / figure.dpi)
    figure.tight_layout()

    renderer = figure.canvas.get_renderer()
    plot_box = axis.get_window_extent(renderer)
    legend_bottom = legend.get_window_extent(renderer).y0
    data_fraction = (legend_bottom - padding - plot_box.y0) / plot_box.height
    lower, upper = axis.get_ylim()
    transform = axis.yaxis.get_transform()
    scaled_lower, scaled_upper = transform.transform([lower, upper])
    expanded_upper = transform.inverted().transform(
        [scaled_lower + (scaled_upper - scaled_lower) / data_fraction]
    )[0]
    axis.set_ylim(lower, expanded_upper)


def _plot_rolling_diagnostics(
    predictions, *, title_template, inference_selection=None, runtime_label=None
):
    """Validate, aggregate and render one shared set of rolling metric figures."""
    required_columns = {
        "window_index",
        "forecast_origin_seconds",
        "horizon_step",
        "position_error_m",
        "negative_log_predictive_density",
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
        if value_name == "fit_runtime_seconds" and runtime_label is not None:
            y_axis_label = f"{runtime_label} [s]"
            label = runtime_label
            mean_label = f"Mittlere {runtime_label}: {{mean:.3f}} s"
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
            "Zeit [s]",
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
        axis.set_axisbelow(True)
        axis.grid(True, which="major", axis="both", color="0.7", alpha=0.2)
        axis.legend(loc=RUNTIME_PLOT_STYLE.legend_location)
        figure.suptitle(
            title_template.format(inference_title=inference_title),
            fontsize=RUNTIME_PLOT_STYLE.title_font_size,
            fontweight=RUNTIME_PLOT_STYLE.title_font_weight,
        )
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
        negative_log_predictive_density = window[
            "negative_log_predictive_density"
        ].to_numpy(dtype=float)
        energy_score_m = window["energy_score_m"].to_numpy(dtype=float)
        fit_runtime_seconds = window["fit_runtime_seconds"].to_numpy(dtype=float)
        if not (
            np.all(np.isfinite(errors_m))
            and np.all(np.isfinite(negative_log_predictive_density))
            and np.all(np.isfinite(energy_score_m))
            and np.all(np.isfinite(fit_runtime_seconds))
        ):
            raise ValueError("Rolling diagnostic values must be finite.")
        final_row = window.loc[window["horizon_step"].idxmax()]
        windows.append(
            (
                time_seconds[0],
                float(np.sqrt(np.mean(errors_m**2))),
                float(np.mean(negative_log_predictive_density)),
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
        "mean_negative_log_predictive_density": values[:, 2],
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
        window_labels = {
            "sliding": "gleitendes Fenster",
            "expanding": "wachsendes Fenster",
        }
        return f"{title}, {window_labels[window_mode]}"
    return title
