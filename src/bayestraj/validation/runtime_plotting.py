"""Plot rolling-evaluation diagnostics for trajectory forecasts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bayestraj.observations.plotting as observation_plotting

RUNTIME_PLOT_STYLE = observation_plotting.ShipDataPlotStyle()
COMPARISON_MODEL_COLORS = ("#0072B2", "#E69F00")
ROLLING_DIAGNOSTIC_PLOT_STYLES = (
    ("RMSE [m]", "RMSE", "#0072B2", "rmse_m", "Mittlerer RMSE: {mean:.2f} m"),
    (
        "Log predictive density",
        "Log predictive density",
        "#CC79A7",
        "mean_log_predictive_density",
        "Mittlere log predictive density: {mean:.3f}",
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
    """Plot already aligned tables with the evaluator's aggregation and style."""
    series = [_rolling_window_metric_series(table) for table in (bayesian, imm)]
    elapsed = series[0][0] - series[0][0][0]
    style = RUNTIME_PLOT_STYLE
    labels = (f"Bayesian CTRV ({bayesian['inference_method'].iloc[0].upper()})", "IMM")
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    figures, rows = [], []
    for ylabel, _, _, name, _ in ROLLING_DIAGNOSTIC_PLOT_STYLES:
        if name == "fit_runtime_seconds":
            ylabel = "Laufzeit [s]"
        figure, axis = plt.subplots(figsize=style.speed_figure_size)
        averages = []
        for (_, values), label, color in zip(
            series, labels, COMPARISON_MODEL_COLORS, strict=True
        ):
            metric = values[name]
            average = float(np.mean(metric))
            averages.append(average)
            axis.plot(
                elapsed, metric, marker="o", markersize=3, color=color, label=label
            )
            axis.axhline(
                average,
                color=color,
                linestyle="--",
                label=f"{label}: Mittel = {average:.3f}",
            )
        axis.set_xlabel("Zeit [s]", fontsize=style.axis_label_font_size)
        axis.set_ylabel(ylabel, fontsize=style.axis_label_font_size)
        axis.tick_params(axis="both", labelsize=style.axis_tick_font_size)
        axis.set_axisbelow(True)
        axis.grid(True, which="major", axis="both", color="0.7", alpha=0.2)
        axis.legend(loc=style.legend_location)
        figure.suptitle(
            "Bayesian CTRV vs. IMM",
            fontsize=style.title_font_size,
            fontweight=style.title_font_weight,
        )
        figure.tight_layout()
        if output_dir is not None:
            figure.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        figures.append((figure, axis))
        rows.append(
            {
                "metric": name,
                "bayesian_ctrv": averages[0],
                "imm": averages[1],
                "preferred": "higher"
                if name == "mean_log_predictive_density"
                else "lower",
            }
        )
    return pd.DataFrame(rows), tuple(figures)


def _plot_rolling_diagnostics(
    predictions, *, title_template, inference_selection=None, runtime_label=None
):
    """Validate, aggregate and render one shared set of rolling metric figures."""
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
        window_labels = {
            "sliding": "gleitendes Fenster",
            "expanding": "wachsendes Fenster",
        }
        return f"{title}, {window_labels[window_mode]}"
    return title
