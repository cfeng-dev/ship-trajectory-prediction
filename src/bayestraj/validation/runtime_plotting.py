"""Plot rolling-evaluation diagnostics for trajectory forecasts."""

from collections import Counter
from pathlib import Path
from textwrap import fill
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from matplotlib.ticker import MaxNLocator

import bayestraj.observations.plotting as observation_plotting

DEFAULT_METRIC_WINDOW_WIDTH = 60.0


METRIC_DISPLAY_GROUPS = {
    "Point prediction": ("show_metric_ade", "show_metric_rmse", "show_metric_fde"),
    "Probabilistic prediction": ("show_metric_energy_score", "show_metric_nlpd"),
}


METRIC_DISPLAY_FIELDS = tuple(
    field for fields in METRIC_DISPLAY_GROUPS.values() for field in fields
)


METRIC_CURVES = {
    "show_metric_ade": ("ade_m", "ADE", "#0072b2"),
    "show_metric_rmse": ("rmse_m", "RMSE", "#009e73"),
    "show_metric_fde": ("fde_m", "FDE", "#d55e00"),
    "show_metric_energy_score": ("energy_score_m", "Energy Score", "#0072b2"),
    "show_metric_nlpd": ("negative_log_predictive_density", "NLPD", "#d55e00"),
}


def _metric_empty_message(axis, message):
    axis.text(
        0.5,
        0.5,
        message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )


def draw_forecast_metric_history(
    axes,
    density_axis,
    origins,
    histories,
    current_time,
    *,
    time_label,
    display_options,
    has_reference,
    show_legend,
    follow_time,
    window_width,
):
    """Draw cached forecast scores with the same controls in both Predictors."""
    point_axis, probability_axis = axes
    for axis in (*axes, density_axis):
        axis.clear()
        axis.relim()  # Reset data limits before adding the time cursor.
        axis.tick_params(labelsize=9)
    density_axis.yaxis.set_label_position("right")
    density_axis.yaxis.tick_right()
    left, right = float(origins[0]), float(origins[-1])
    if follow_time:
        left = max(left, current_time - 0.9 * window_width)
        right = left + window_width
    elif left == right:
        right = left + 1
    visible = (origins >= left) & (origins <= right)
    options = display_options
    dual_scale = options["show_metric_energy_score"] and options["show_metric_nlpd"]
    density_axis.set_visible(dual_scale)
    for axis, title in (
        (point_axis, "Point prediction"),
        (probability_axis, "Probabilistic prediction"),
    ):
        fields = [field for field in METRIC_DISPLAY_GROUPS[title] if options[field]]
        axis.set_title(title, fontsize=12)
        axis.set_xlabel(time_label, fontsize=10)
        axis.set_xlim(left, right)
        axis.grid(alpha=0.2)
        axis.axvline(
            current_time,
            color="0.4",
            linestyle="--",
            linewidth=1,
            label="_current_time",
        )
        if not fields:
            _metric_empty_message(axis, "No metrics selected")
            continue
        handles = []
        for metric_field in fields:
            _attribute, label, color = METRIC_CURVES[metric_field]
            target = (
                density_axis
                if dual_scale and metric_field == "show_metric_nlpd"
                else axis
            )
            handles.extend(
                target.plot(
                    origins[visible],
                    histories[metric_field][visible],
                    label=label,
                    color=color,
                    linewidth=1.3,
                    marker=".",
                    markersize=3,
                    linestyle="--"
                    if dual_scale and metric_field == "show_metric_nlpd"
                    else "-",
                )
            )
        if title == "Point prediction":
            axis.set_ylabel("Position error [m]", fontsize=10)
            axis.set_ylim(bottom=0)
        else:
            axis.set_ylabel(
                "Energy Score [m]" if options["show_metric_energy_score"] else "NLPD",
                fontsize=10,
                color="black",
            )
            if dual_scale:
                density_axis.set_ylabel("NLPD", color="black", fontsize=10)
                density_axis.tick_params(
                    axis="y",
                    colors="black",
                    labelsize=9,
                )
        if not any(np.isfinite(histories[field][visible]).any() for field in fields):
            message = (
                "No reference trajectory"
                if not has_reference
                else "No complete evaluated forecasts"
            )
            _metric_empty_message(axis, message)
        if show_legend:
            axis.legend(
                handles=handles, fontsize=8, loc="upper right", ncols=len(handles)
            )
    return {axis: axis.get_ylim() for axis in (*axes, density_axis)}


def reserve_metric_legend_space(
    figure, axes, density_axis, value_limits, display_limits
):
    """Keep metric curves below the legends, including after resize."""
    get_renderer = getattr(figure.canvas, "get_renderer", None)
    if get_renderer is None:
        return
    renderer = get_renderer()
    padding = renderer.points_to_pixels(5)
    for index, axis in enumerate(axes):
        legend = axis.get_legend()
        if legend is None or axis not in value_limits:
            continue
        available = legend.get_window_extent(renderer).y0 - axis.bbox.y0 - padding
        data_fraction = np.clip(available / max(axis.bbox.height, 1), 0.03, 1)
        data_axes = [axis]
        if index == 1 and density_axis.get_visible():
            data_axes.append(density_axis)
        for data_axis in data_axes:
            previous = display_limits.get(data_axis)
            if previous is not None and not np.allclose(data_axis.get_ylim(), previous):
                continue  # Preserve the user's manual zoom or pan.
            lower, upper = value_limits[data_axis]
            data_axis.set_ylim(lower, lower + (upper - lower) / data_fraction)
            display_limits[data_axis] = data_axis.get_ylim()


RUNTIME_PLOT_STYLE = observation_plotting.ShipDataPlotStyle()
COMPARISON_MODEL_COLORS = ("#0072B2", "#E69F00")
COMPARISON_LEGEND_PADDING_POINTS = 8.0
COMPARISON_INFERENCE_COLORS = {
    "imm": "#E69F00",
    "smc": "#0072B2",
    "vi": "#009E73",
    "mcmc": "#CC79A7",
}
COMPARISON_METRIC_LABELS = {
    "ade_m": ("ADE [m]", "Average Displacement Error"),
    "fde_m": ("FDE [m]", "Final Displacement Error"),
    "rmse_m": ("RMSE [m]", "Root Mean Squared Error"),
    "energy_score_m": ("ES [m]", "Energy Score"),
    "mean_negative_log_predictive_density": ("NLPD", "Negative Log Predictive Density"),
    "joint_coverage_50": ("Coverage [%]", "Joint 50% Coverage"),
    "joint_coverage_90": ("Coverage [%]", "Joint 90% Coverage"),
    "fit_runtime_seconds": ("Runtime [s]", "Runtime"),
}
OBSERVATION_COUNT_METRICS = {
    "ade_m": COMPARISON_METRIC_LABELS["ade_m"],
    "fde_m": COMPARISON_METRIC_LABELS["fde_m"],
    "rmse_m": COMPARISON_METRIC_LABELS["rmse_m"],
    "mean_energy_score_m": COMPARISON_METRIC_LABELS["energy_score_m"],
    "mean_negative_log_predictive_density": COMPARISON_METRIC_LABELS[
        "mean_negative_log_predictive_density"
    ],
    "joint_coverage_50": COMPARISON_METRIC_LABELS["joint_coverage_50"],
    "joint_coverage_90": COMPARISON_METRIC_LABELS["joint_coverage_90"],
    "mean_fit_runtime_seconds": ("Runtime [s]", "Runtime"),
}
PREDICTION_HORIZON_METRICS = {
    name: labels
    for name, labels in OBSERVATION_COUNT_METRICS.items()
    if name != "mean_fit_runtime_seconds"
}
OBSERVATION_COUNT_DIAGNOSTICS = {
    "mcmc_rhat_max_mean": (
        "mcmc",
        "R_hat",
        "R_hat",
        "near 1",
    ),
    "mcmc_divergences": ("mcmc", "Divergenzen gesamt", "Total divergences", "lower"),
    "mcmc_ess_bulk_min": ("mcmc", "Min. Bulk-ESS", "Min. Bulk ESS", "higher"),
    "mcmc_ess_tail_min": ("mcmc", "Min. Tail-ESS", "Min. Tail ESS", "higher"),
    "mcmc_treedepth_hits": (
        "mcmc",
        "Baumtiefentreffer gesamt",
        "Total maximum-treedepth hits",
        "lower",
    ),
    "mcmc_diagnostics_pass_rate": (
        "mcmc",
        "Diagnose-Erfolgsquote [%]",
        "Diagnostics pass rate [%]",
        "higher",
    ),
    "vi_convergence_rate": (
        "vi",
        "VI-Konvergenzquote [%]",
        "VI convergence rate [%]",
        "higher",
    ),
    "vi_iteration_limit_rate": (
        "vi",
        "Fits am Iterationslimit [%]",
        "Fits at iteration limit [%]",
        "lower",
    ),
    "vi_elbo_delta_mean_max": (
        "vi",
        "Max. letzte relative ELBO-Änderung (Mittel)",
        "Max. final relative ELBO change (mean)",
        "lower",
    ),
    "vi_elbo_delta_median_max": (
        "vi",
        "Max. letzte relative ELBO-Änderung (Median)",
        "Max. final relative ELBO change (median)",
        "lower",
    ),
}
COMPARISON_PLOT_TEXT = {
    "de": {
        "title": "Modellvergleich",
        "time": "Zeit [s]",
        "runtime": "Laufzeit",
        "mean": "Mittelwert",
        "bayesian_model": "Bayessches CTRV",
        "forecast_horizon": "Vorhersagehorizont: H = {count}",
        "nominal": "Sollwert: {percent}%",
        "metric": "Metrik",
        "summary": "Metrikübersicht",
        "horizon_note": (
            "H: Anzahl der Positionen, die ab jedem Prognosestart "
            "in die Zukunft vorhergesagt werden."
        ),
        "window_note": (
            "T: maximale Anzahl der Zeitschritte im Beobachtungsfenster "
            "von MCMC und VI."
        ),
        "summary_note": (
            "Mittelwerte über gemeinsame Vorhersageursprünge.\n"
            "Coverage über alle zugeordneten Vorhersagepunkte."
        ),
        "forecast_summary_note": (
            "Gemeinsame Prognosestartpunkte.\n"
            "Coverage über alle vorhergesagten Positionen."
        ),
    },
    "en": {
        "title": "Model comparison",
        "time": "Time [s]",
        "runtime": "Runtime",
        "mean": "Mean",
        "bayesian_model": "Bayesian CTRV",
        "forecast_horizon": "Forecast horizon: H = {count}",
        "nominal": "Nominal {percent}%",
        "metric": "Metric",
        "summary": "Metric summary",
        "horizon_note": (
            "H: number of future positions predicted from each forecast origin."
        ),
        "window_note": (
            "T: maximum number of time steps in the observation window of MCMC and VI."
        ),
        "summary_note": (
            "Mean metrics over common forecast origins.\n"
            "Coverage over all matched forecast positions."
        ),
        "forecast_summary_note": (
            "Identical forecast origins.\nCoverage over all predicted positions."
        ),
    },
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


def _comparison_subplots(*, figsize):
    """Create a comparison window without interactive auto-display."""
    with plt.ioff():
        return plt.subplots(figsize=figsize)


def _show_comparison_figure(figure):
    """Render the final layout before making comparison windows visible."""
    if any(axis.tables for axis in figure.axes):
        _prepare_scrollable_comparison_table(figure)
    figure.canvas.draw()
    plt.show()


def _prepare_scrollable_comparison_table(figure):
    """Keep Tk tables at their full height inside a scrollable viewport."""
    canvas = figure.canvas
    get_widget = getattr(canvas, "get_tk_widget", None)
    if get_widget is None or hasattr(canvas, "_comparison_table_scrollbar"):
        return
    widget = get_widget()
    minimum_height = figure.get_figheight()
    content_height = round(minimum_height * figure.dpi)
    viewport_height = min(
        content_height, max(1, int(widget.winfo_screenheight() * 0.8))
    )

    from tkinter import ttk

    scrollbar = ttk.Scrollbar(widget.master, orient="vertical", command=widget.yview)
    scrollbar_visible = False

    def show_scrollbar(visible):
        nonlocal scrollbar_visible
        if visible == scrollbar_visible:
            return
        if visible:
            scrollbar.pack(side="right", fill="y", before=widget)
        else:
            scrollbar.pack_forget()
        scrollbar_visible = visible

    show_scrollbar(content_height > viewport_height)
    widget.configure(
        height=viewport_height,
        yscrollcommand=scrollbar.set,
        yscrollincrement=24,
        scrollregion=(0, 0, round(figure.bbox.width), content_height),
    )

    def resize(event):
        height = max(event.height, round(minimum_height * figure.dpi))
        show_scrollbar(event.height < height)
        canvas.resize(SimpleNamespace(width=event.width, height=height))
        widget.configure(
            height=min(
                round(minimum_height * figure.dpi),
                max(1, int(widget.winfo_screenheight() * 0.8)),
            ),
            scrollregion=(0, 0, event.width, height),
        )

    def scroll(event):
        button = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        if button in (4, 5):
            steps = -1 if button == 4 else 1
        elif delta:
            steps = (-1 if delta > 0 else 1) * max(1, abs(delta) // 120)
        else:
            return "break"
        widget.yview_scroll(steps, "units")
        return "break"

    widget.bind("<Configure>", resize)
    for name in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        widget.bind(name, scroll)
    canvas._comparison_table_scrollbar = scrollbar


def comparison_model_label(label, table):
    """Label batch window limits from explicit CSV configuration only."""
    if (
        table["inference_method"].iloc[0] not in {"vi", "mcmc"}
        or "configured_observation_count" not in table
        or table["configured_observation_count"].isna().all()
    ):
        return label
    counts = pd.to_numeric(
        table["configured_observation_count"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(counts).all()
        or (counts <= 0).any()
        or (counts != np.floor(counts)).any()
        or np.unique(counts).size != 1
    ):
        raise ValueError(
            "configured_observation_count must contain one consistent positive "
            "integer for every row of a batch evaluation."
        )
    count = int(counts[0])
    if f", T={count})" in label or f"(T={count})" in label:
        return label
    if label.endswith(")"):
        return f"{label[:-1]}, T={count})"
    return f"{label} (T={count})"


def _comparison_configuration_note(text, labels):
    """Use the same horizon and observation-window definitions in both views."""
    notes = [text["horizon_note"]]
    if any("T=" in label for label in labels):
        notes.append(text["window_note"])
    return "\n".join(notes)


def _comparison_series_colors(tables):
    """Keep method colors and distinguish multiple runs with lighter shades."""
    methods = [table["inference_method"].iloc[0] for table in tables]
    totals, seen = Counter(methods), Counter()
    colors = []
    for index, method in enumerate(methods):
        color = COMPARISON_INFERENCE_COLORS.get(
            method, COMPARISON_MODEL_COLORS[index % len(COMPARISON_MODEL_COLORS)]
        )
        if seen[method]:
            fraction = 0.55 * seen[method] / (totals[method] - 1)
            rgb = np.asarray(to_rgb(color))
            color = to_hex(rgb * (1 - fraction) + fraction)
        colors.append(color)
        seen[method] += 1
    return colors


def plot_model_comparison(
    models,
    *,
    labels=None,
    title=None,
    show=True,
    output_dir=None,
    show_points=True,
    show_mean_lines=True,
    show_summary_table=False,
    metric_y_scale="linear",
    plot_language="en",
):
    """Plot all selected models using the same metric and coverage styles."""
    if len(models) < 2:
        raise ValueError("Select at least two models for comparison.")
    if metric_y_scale not in {"linear", "symlog"}:
        raise ValueError("metric_y_scale must be 'linear' or 'symlog'.")
    if plot_language not in COMPARISON_PLOT_TEXT:
        raise ValueError("plot_language must be 'de' or 'en'.")
    text = COMPARISON_PLOT_TEXT[plot_language]
    names = list(models)
    tables = list(models.values())
    prediction_counts = sorted(
        {int(count) for table in tables for count in table["prediction_count"].unique()}
    )
    legend_title = text["forecast_horizon"].format(
        count=", ".join(str(count) for count in prediction_counts)
    )
    labels = [
        comparison_model_label(
            str(name).replace("Bayesian CTRV", text["bayesian_model"], 1)
            if labels is None
            else labels[name],
            models[name],
        )
        for name in names
    ]
    configuration_note = _comparison_configuration_note(text, labels)
    layout_rect = (0, 0.08 if "\n" in configuration_note else 0.06, 1, 1)
    series = [_rolling_window_metric_series(table) for table in tables]
    elapsed = series[0][0] - series[0][0][0]
    style = RUNTIME_PLOT_STYLE
    colors = _comparison_series_colors(tables)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    figures, rows = [], []
    for name, (ylabel, metric_name) in COMPARISON_METRIC_LABELS.items():
        nominal = (
            int(name.rsplit("_", 1)[1]) / 100
            if name.startswith("joint_coverage_")
            else None
        )
        if name == "fit_runtime_seconds":
            metric_name = text["runtime"]
            ylabel = f"{metric_name} [s]"
        figure, axis = _comparison_subplots(figsize=style.speed_figure_size)
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
                        label=f"{label}: {text['mean']} = {average:.3f}",
                    )
            lower, upper = axis.get_ylim()
            axis.set_ylim(min(lower, 0.0), max(upper, 0.0))
            axis.set_xlabel(text["time"], fontsize=style.axis_label_font_size)
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
                label=text["nominal"].format(percent=f"{100 * nominal:.0f}"),
            )
            axis.set_ylim(0, 100)
            axis.set_yticks(np.arange(0, 101, 20))
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
        legend = axis.legend(
            loc="upper right", ncol=1, title=legend_title, alignment="left"
        )
        figure.suptitle(
            title if title is not None else f"{text['title']} ({metric_name})",
            fontsize=style.title_font_size,
            fontweight=style.title_font_weight,
        )
        figure.text(0.5, 0.02, configuration_note, ha="center", va="bottom", fontsize=9)
        figure.tight_layout(rect=layout_rect)
        _reserve_legend_headroom(figure, axis, legend, layout_rect=layout_rect)
        if output_dir is not None:
            figure.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        if show:
            _show_comparison_figure(figure)
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
    summary = pd.DataFrame(rows)
    if show_summary_table:
        summary_text = {
            **text,
            "summary_note": f"{text['summary_note']}\n{configuration_note}",
        }
        figure, axis = _plot_comparison_summary_table(
            summary,
            [label.replace(" (", "\n(") for label in labels],
            text={
                **summary_text,
                "metric": "Modell" if plot_language == "de" else "Model",
            },
            title=title,
            transpose=True,
            metric_column_width=1.1,
        )
        if output_dir is not None:
            figure.savefig(
                output_dir / "metrics_summary.png", dpi=150, bbox_inches="tight"
            )
        if show:
            _show_comparison_figure(figure)
        figures.append((figure, axis))
    return summary, tuple(figures)


def _smc_diagnostic_series(predictions):
    """Validate particle diagnostics and retain one row per forecast update."""
    fields = (
        "particle_effective_sample_size",
        "particle_count",
        "particle_effective_sample_size_fraction",
        "particle_resample_count",
    )
    values = predictions[["window_index", "forecast_origin_seconds"]].copy()
    for field in fields:
        values[field] = pd.to_numeric(
            predictions.get(field, pd.Series(np.nan, index=predictions.index)),
            errors="raise",
        )
        defined = values[field].dropna()
        if not np.isfinite(defined).all() or (defined < 0).any():
            raise ValueError(f"{field} must be finite and non-negative or missing.")
        if (values.groupby("window_index")[field].nunique(dropna=False) > 1).any():
            raise ValueError(f"{field} must be constant within a forecast window.")
    windows = values.drop_duplicates("window_index").sort_values(
        "forecast_origin_seconds"
    )
    counts = windows["particle_count"].dropna()
    if (counts < 1).any() or not counts.eq(np.floor(counts)).all():
        raise ValueError("particle_count must contain positive integers.")
    ess = windows["particle_effective_sample_size"]
    if (ess.dropna() < 1 - 1e-9).any() or (
        ess > windows["particle_count"] + 1e-9
    ).any():
        raise ValueError(
            "particle_effective_sample_size must be between 1 and particle_count."
        )
    fraction = windows["particle_effective_sample_size_fraction"]
    derived = ess / windows["particle_count"]
    defined = fraction.notna() & derived.notna()
    if (fraction.dropna() > 1 + 1e-9).any() or not np.allclose(
        fraction[defined], derived[defined], rtol=1e-9, atol=1e-12
    ):
        raise ValueError(
            "particle_effective_sample_size_fraction must equal ESS / particle_count."
        )
    windows["particle_effective_sample_size_fraction"] = fraction.combine_first(derived)
    resamples = windows["particle_resample_count"].dropna()
    if (
        not resamples.eq(np.floor(resamples)).all()
        or (resamples.diff().dropna() < 0).any()
    ):
        raise ValueError(
            "particle_resample_count must contain nondecreasing integer counts."
        )
    return windows


def _plot_smc_diagnostics(
    series, *, colors, show, output_dir=None, show_points=True, plot_language="de"
):
    """Plot SMC health separately from forecast scores and convergence claims."""
    if not series:
        return ()
    origin = min(
        float(table["forecast_origin_seconds"].iloc[0]) for table in series.values()
    )
    styles = (
        (
            "particle_effective_sample_size_fraction",
            "smc_effective_sample_size_fraction",
            "Relative ESS [%]",
            "Relative ESS [%]",
            "Gewichts-ESS; hohe Werte allein garantieren keine ausreichende Partikelvielfalt.",
            "Weight ESS; high values alone do not guarantee particle diversity.",
        ),
        (
            "particle_resample_count",
            "smc_resample_count",
            "Kumulative Resampling-Ereignisse",
            "Cumulative resampling events",
            "Einschließlich Hilfs-Resampling; die Häufigkeit ist kein Konvergenztest.",
            "Includes auxiliary resampling; frequency is not a convergence test.",
        ),
    )
    figures = []
    for field, filename, ylabel_de, ylabel_en, note_de, note_en in styles:
        selected = {
            label: table
            for label, table in series.items()
            if table[field].notna().any()
        }
        if not selected:
            continue
        percentage = field.endswith("_fraction")
        figure, axis = _comparison_subplots(
            figsize=RUNTIME_PLOT_STYLE.speed_figure_size
        )
        for label, table in selected.items():
            axis.plot(
                table["forecast_origin_seconds"] - origin,
                table[field] * (100 if percentage else 1),
                label=label,
                color=colors[label],
                marker="o" if show_points else "None",
                markersize=3,
                drawstyle="default" if percentage else "steps-post",
            )
        if percentage:
            axis.set_ylim(0, 100)
            axis.set_yticks(np.arange(0, 101, 20))
        else:
            axis.set_ylim(bottom=0)
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlabel(
            "Zeit [s]" if plot_language == "de" else "Time [s]",
            fontsize=RUNTIME_PLOT_STYLE.axis_label_font_size,
        )
        axis.set_ylabel(
            ylabel_de if plot_language == "de" else ylabel_en,
            fontsize=RUNTIME_PLOT_STYLE.axis_label_font_size,
        )
        axis.tick_params(labelsize=RUNTIME_PLOT_STYLE.axis_tick_font_size)
        axis.grid(True, color="0.7", alpha=0.2)
        axis.set_axisbelow(True)
        legend = axis.legend(loc="upper right")
        figure.suptitle(
            "SMC-Diagnose" if plot_language == "de" else "SMC diagnostics",
            fontsize=RUNTIME_PLOT_STYLE.title_font_size,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.02,
            note_de if plot_language == "de" else note_en,
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.07, 1, 0.95))
        _reserve_legend_headroom(figure, axis, legend, layout_rect=(0, 0.07, 1, 0.95))
        if output_dir is not None:
            figure.savefig(
                Path(output_dir) / f"{filename}.png", dpi=150, bbox_inches="tight"
            )
        if show:
            _show_comparison_figure(figure)
        figures.append((figure, axis))
    return tuple(figures)


def plot_observation_count_comparison(
    models,
    *,
    show=True,
    output_dir=None,
    show_points=True,
    show_summary_table=False,
    metric_y_scale="linear",
    plot_language="de",
):
    """Plot saved sweep metrics against window size using shared plot styles."""
    if metric_y_scale not in {"linear", "symlog"}:
        raise ValueError("metric_y_scale must be 'linear' or 'symlog'.")
    if plot_language not in COMPARISON_PLOT_TEXT:
        raise ValueError("plot_language must be 'de' or 'en'.")
    text = COMPARISON_PLOT_TEXT[plot_language]
    title = "Fenstervergleich" if plot_language == "de" else "Window-size comparison"
    xlabel = (
        "Beobachtungsfenstergröße T [Zeitschritte]"
        if plot_language == "de"
        else "Observation window size T [time steps]"
    )
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    style = RUNTIME_PLOT_STYLE
    figures = []
    metrics = dict(OBSERVATION_COUNT_METRICS)
    for metric, (
        method,
        label_de,
        label_en,
        _,
    ) in OBSERVATION_COUNT_DIAGNOSTICS.items():
        if any(
            table["inference_method"].iloc[0] == method
            and metric in table
            and table[metric].notna().any()
            for table in models.values()
        ):
            label = label_de if plot_language == "de" else label_en
            metrics[metric] = (label, label)
    for metric, (ylabel, metric_name) in metrics.items():
        figure, axis = _comparison_subplots(figsize=style.speed_figure_size)
        coverage = metric.startswith("joint_coverage_")
        percentage = coverage or metric.endswith("_rate")
        metric_models = {
            label: table
            for label, table in models.items()
            if metric not in OBSERVATION_COUNT_DIAGNOSTICS
            or (
                table["inference_method"].iloc[0]
                == OBSERVATION_COUNT_DIAGNOSTICS[metric][0]
                and metric in table
            )
        }
        if not percentage and metric_y_scale == "symlog":
            # Use the same adaptive linear range as the forecast metric curves.
            magnitudes = np.abs(
                np.concatenate(
                    [
                        table[metric].to_numpy(dtype=float)
                        for table in metric_models.values()
                    ]
                )
            )
            positive = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0)]
            threshold = float(np.percentile(positive, 10)) if positive.size else 1.0
            axis.set_yscale("symlog", linthresh=threshold)
        for label, table in metric_models.items():
            method = table["inference_method"].iloc[0]
            values = table[metric].to_numpy(dtype=float)
            axis.plot(
                table["observation_count"],
                values * 100 if percentage else values,
                label=label,
                color=COMPARISON_INFERENCE_COLORS[method],
                marker="o" if show_points else "None",
                markersize=3,
            )
        if coverage:
            nominal = int(metric.rsplit("_", 1)[1])
            axis.axhline(
                nominal,
                color="0.35",
                linestyle=":",
                label=text["nominal"].format(percent=nominal),
            )
            axis.set_ylim(0, 100)
        elif percentage:
            axis.set_ylim(0, 100)
        elif metric_y_scale == "symlog":
            lower, upper = axis.get_ylim()
            axis.set_ylim(min(lower, 0.0), max(upper, 0.0))
        if metric == "mean_fit_runtime_seconds":
            ylabel = f"{text['runtime']} [s]"
            metric_name = text["runtime"]
        axis.set_xlabel(xlabel, fontsize=style.axis_label_font_size)
        axis.set_ylabel(ylabel, fontsize=style.axis_label_font_size)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        axis.tick_params(axis="both", labelsize=style.axis_tick_font_size)
        axis.set_axisbelow(True)
        axis.grid(True, color="0.7", alpha=0.2)
        legend = axis.legend(loc=style.legend_location)
        figure.suptitle(
            f"{title} ({metric_name})",
            fontsize=style.title_font_size,
            fontweight=style.title_font_weight,
        )
        figure.tight_layout()
        if not percentage:
            _reserve_legend_headroom(figure, axis, legend)
        if output_dir is not None:
            figure.savefig(output_dir / f"{metric}.png", dpi=150, bbox_inches="tight")
        if show:
            _show_comparison_figure(figure)
        figures.append((figure, axis))
    if show_summary_table:
        configurations = {
            f"{label} (T={int(row['observation_count'])})": [
                row.get(metric, np.nan) for metric in metrics
            ]
            for label, table in models.items()
            for _, row in table.iterrows()
        }
        summary = pd.DataFrame(
            {
                "metric": list(metrics),
                **configurations,
                "preferred": [
                    f"near {metric.rsplit('_', 1)[1]}%"
                    if metric.startswith("joint_coverage_")
                    else OBSERVATION_COUNT_DIAGNOSTICS[metric][3]
                    if metric in OBSERVATION_COUNT_DIAGNOSTICS
                    else "lower"
                    for metric in metrics
                ],
            }
        )
        summary_text = {
            **text,
            "metric": "Konfiguration" if plot_language == "de" else "Configuration",
            "summary_note": (
                "T: maximale Anzahl der Zeitschritte im Beobachtungsfenster. "
                "Mittelwerte je Fenster.\n"
                "Coverage über alle Vorhersagepunkte; fett: nächster Sollwert."
                + (" R_hat nahe 1." if "mcmc_rhat_max_mean" in metrics else "")
                + (
                    " VI-Konvergenzquote: höher ist besser."
                    if "vi_convergence_rate" in metrics
                    else ""
                )
                if plot_language == "de"
                else "T: maximum number of time steps in the observation window. "
                "Mean metrics per window.\n"
                "Coverage over all forecast positions; bold: closest to nominal."
                + (" R_hat near 1." if "mcmc_rhat_max_mean" in metrics else "")
                + (
                    " VI convergence rate: higher is better."
                    if "vi_convergence_rate" in metrics
                    else ""
                )
            ),
        }
        performance_metrics = [
            metric
            for metric in metrics
            if metric in OBSERVATION_COUNT_METRICS
            or metric in {"mcmc_rhat_max_mean", "vi_convergence_rate"}
        ]
        groups = [
            (
                performance_metrics,
                "Leistungsvergleich"
                if plot_language == "de"
                else "Performance comparison",
                "metrics_summary.png",
                summary_text["summary_note"],
            )
        ]
        diagnostics = [
            metric for metric in metrics if metric not in performance_metrics
        ]
        if diagnostics:
            diagnostic_note = (
                "T: maximale Fenstergröße. Diagnosewerte der ausgewerteten Fits.\n"
                "Ereignisse summiert; ESS minimal; ELBO-Änderungen maximal.\n"
                "Quoten je Fit; fett: günstigerer Wert (ESS/Erfolgsquoten höher)."
                if plot_language == "de"
                else "T: maximum window size. Diagnostics of the evaluated fits.\n"
                "Event totals; minimum ESS; maximum ELBO changes.\n"
                "Rates per fit; bold: preferred value (higher ESS/pass rates)."
            )
            groups.append(
                (
                    diagnostics,
                    "Inferenzdiagnosen"
                    if plot_language == "de"
                    else "Inference diagnostics",
                    "diagnostics_summary.png",
                    diagnostic_note,
                )
            )
        metric_labels = {metric: labels[0] for metric, labels in metrics.items()}
        metric_labels["mean_fit_runtime_seconds"] = f"{text['runtime']} [s]"
        for suffix, label in (
            ("mean", "Mittel" if plot_language == "de" else "mean"),
            ("median", "Median" if plot_language == "de" else "median"),
        ):
            metric_labels[f"vi_elbo_delta_{suffix}_max"] = f"Max. rel. ΔELBO\n({label})"
        for group_metrics, group_title, filename, note in groups:
            figure, axis = _plot_comparison_summary_table(
                summary.loc[summary["metric"].isin(group_metrics)],
                list(configurations),
                text={**summary_text, "summary_note": note},
                title=f"{title} ({group_title})",
                metric_labels=metric_labels,
                transpose=True,
            )
            if output_dir is not None:
                figure.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
            if show:
                _show_comparison_figure(figure)
            figures.append((figure, axis))
    return tuple(figures)


def _forecast_configuration_label(label, prediction_count):
    """Keep the model and its method/window/horizon parameters on two lines."""
    if label.endswith(")"):
        label = f"{label[:-1]}, H={prediction_count})"
    else:
        label = f"{label} (H={prediction_count})"
    return label.replace(" (", "\n(")


def plot_prediction_horizon_comparison(
    models,
    *,
    show=True,
    output_dir=None,
    show_points=True,
    show_summary_table=True,
    metric_y_scale="linear",
    plot_language="de",
):
    """Plot saved prefix metrics against the number of future positions."""
    if metric_y_scale not in {"linear", "symlog"}:
        raise ValueError("metric_y_scale must be 'linear' or 'symlog'.")
    if plot_language not in COMPARISON_PLOT_TEXT:
        raise ValueError("plot_language must be 'de' or 'en'.")
    if not models:
        raise ValueError("Select at least one model for horizon comparison.")
    text = COMPARISON_PLOT_TEXT[plot_language]
    title = (
        "Vergleich der Vorhersagehorizonte"
        if plot_language == "de"
        else "Prediction-horizon comparison"
    )
    xlabel = (
        "Vorhersagehorizont [zukünftige Positionen]"
        if plot_language == "de"
        else "Prediction horizon [future positions]"
    )
    labels = [
        label.replace("Bayesian CTRV", text["bayesian_model"], 1) for label in models
    ]
    configuration_note = _comparison_configuration_note(text, labels)
    layout_rect = (0, 0.08 if "\n" in configuration_note else 0.06, 1, 1)
    colors = _comparison_series_colors(list(models.values()))
    style = RUNTIME_PLOT_STYLE
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for metric, (ylabel, metric_name) in PREDICTION_HORIZON_METRICS.items():
        figure, axis = _comparison_subplots(figsize=style.speed_figure_size)
        coverage = metric.startswith("joint_coverage_")
        if not coverage and metric_y_scale == "symlog":
            values = np.abs(
                np.concatenate(
                    [table[metric].to_numpy(dtype=float) for table in models.values()]
                )
            )
            positive = values[np.isfinite(values) & (values > 0)]
            threshold = float(np.percentile(positive, 10)) if positive.size else 1.0
            axis.set_yscale("symlog", linthresh=threshold)
        for table, label, color in zip(models.values(), labels, colors, strict=True):
            axis.plot(
                table["prediction_count"],
                table[metric] * (100 if coverage else 1),
                label=label,
                color=color,
                marker="o" if show_points else "None",
                markersize=3,
            )
        if coverage:
            nominal = int(metric.rsplit("_", 1)[1])
            axis.axhline(
                nominal,
                color="0.35",
                linestyle=":",
                label=text["nominal"].format(percent=nominal),
            )
            axis.set_ylim(0, 100)
            axis.set_yticks(np.arange(0, 101, 20))
        else:
            lower, upper = axis.get_ylim()
            axis.set_ylim(min(0.0, lower), max(0.0, upper))
        axis.set_xticks(
            sorted(
                {
                    int(count)
                    for table in models.values()
                    for count in table["prediction_count"]
                }
            )
        )
        axis.set_xlabel(xlabel, fontsize=style.axis_label_font_size)
        axis.set_ylabel(ylabel, fontsize=style.axis_label_font_size)
        axis.tick_params(axis="both", labelsize=style.axis_tick_font_size)
        axis.set_axisbelow(True)
        axis.grid(True, color="0.7", alpha=0.2)
        legend = axis.legend(loc="upper right")
        figure.suptitle(
            f"{title} ({metric_name})",
            fontsize=style.title_font_size,
            fontweight=style.title_font_weight,
        )
        figure.text(0.5, 0.02, configuration_note, ha="center", va="bottom", fontsize=9)
        figure.tight_layout(rect=layout_rect)
        _reserve_legend_headroom(figure, axis, legend, layout_rect=layout_rect)
        if output_dir is not None:
            figure.savefig(output_dir / f"{metric}.png", dpi=150, bbox_inches="tight")
        if show:
            _show_comparison_figure(figure)
        figures.append((figure, axis))
    if show_summary_table:
        configurations = {
            _forecast_configuration_label(label, int(row["prediction_count"])): [
                row[metric] for metric in PREDICTION_HORIZON_METRICS
            ]
            for label, table in zip(labels, models.values(), strict=True)
            for _, row in table.iterrows()
        }
        summary = pd.DataFrame(
            {
                "metric": list(PREDICTION_HORIZON_METRICS),
                **configurations,
                "preferred": [
                    f"near {metric.rsplit('_', 1)[1]}%"
                    if metric.startswith("joint_coverage_")
                    else "lower"
                    for metric in PREDICTION_HORIZON_METRICS
                ],
            }
        )
        note = f"{text['forecast_summary_note']}\n{configuration_note}"
        figure, axis = _plot_comparison_summary_table(
            summary,
            list(configurations),
            text={
                **text,
                "metric": "Konfiguration" if plot_language == "de" else "Configuration",
                "summary_note": note,
            },
            title=title,
            metric_labels={
                metric: label[0] for metric, label in PREDICTION_HORIZON_METRICS.items()
            },
            transpose=True,
            metric_column_width=1.1,
            label_column_width=2.8,
            row_height=0.5,
        )
        if output_dir is not None:
            figure.savefig(
                output_dir / "metrics_summary.png", dpi=150, bbox_inches="tight"
            )
        if show:
            _show_comparison_figure(figure)
        figures.append((figure, axis))
    return tuple(figures)


def _plot_comparison_summary_table(
    summary,
    labels,
    *,
    text,
    title,
    metric_labels=None,
    transpose=False,
    metric_column_width=1.5,
    label_column_width=2.0,
    row_height=0.35,
):
    """Render the computed comparison summary without recalculating metrics."""
    style = RUNTIME_PLOT_STYLE
    width = (
        max(8.0, label_column_width + metric_column_width * len(summary))
        if transpose
        else max(8.0, label_column_width + 2.0 * len(labels))
    )
    height = max(5.5, 2.0 + row_height * len(labels)) if transpose else 5.5
    figure, axis = _comparison_subplots(figsize=(width, height))
    axis.set_axis_off()
    rows = []
    best_cells = set()
    for row_number, (_, row) in enumerate(summary.iterrows(), start=1):
        name = row["metric"]
        coverage = name.startswith("joint_coverage_")
        percentage = coverage or name.endswith("_rate")
        values = row.iloc[1:-1].to_numpy(dtype=float)
        scores = values
        if coverage:
            percent = int(name.rsplit("_", 1)[1])
            label = f"Joint {percent}% Coverage"
            scores = np.abs(values - percent / 100)
        elif name == "fit_runtime_seconds":
            label = f"{text['runtime']} [s]"
        else:
            label = (
                metric_labels[name]
                if metric_labels is not None
                else COMPARISON_METRIC_LABELS[name][0]
            )
        if row.get("preferred") == "higher":
            scores = -values
        finite = np.isfinite(scores)
        if name == "mcmc_rhat_max_mean":
            scores = np.abs(values - 1.0)
        if finite.any():
            # Allow floating-point roundoff in otherwise equal scores.
            best = finite & np.isclose(
                scores, np.min(scores[finite]), rtol=1e-12, atol=1e-12
            )
            best_cells.update(
                (row_number, column + 1) for column in np.flatnonzero(best)
            )
        rows.append(
            [label]
            + [
                "—"
                if not np.isfinite(value)
                else f"{value:.1%}"
                if percentage
                else f"{value:.3f}"
                for value in values
            ]
        )
    column_labels = [text["metric"], *[label.replace(" (", "\n(") for label in labels]]
    if transpose:
        transposed = list(zip([text["metric"], *labels], *rows, strict=True))
        column_labels = [
            label
            if "\n" in label
            else label.replace(" Coverage", "\nCoverage").replace(" [", "\n[")
            for label in transposed[0]
        ]
        column_labels = [
            "\n".join(fill(line, width=16) for line in label.splitlines())
            for label in column_labels
        ]
        rows = transposed[1:]
        best_cells = {(column, row) for row, column in best_cells}
    metric_width = label_column_width / width
    table = axis.table(
        cellText=rows,
        colLabels=column_labels,
        colWidths=[metric_width]
        + [(1 - metric_width) / (len(column_labels) - 1)] * (len(column_labels) - 1),
        cellLoc="right",
        bbox=(0, 0.12, 1, 0.84),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(style.axis_tick_font_size)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("0.8")
        cell.set_linewidth(0.6)
        cell.set_facecolor("0.92" if row == 0 else ("0.97" if row % 2 else "white"))
        if row == 0 or (row, column) in best_cells:
            cell.get_text().set_weight("bold")
        if column == 0:
            cell.get_text().set_ha("left")
    figure.suptitle(
        title if title is not None else f"{text['title']} ({text['summary']})",
        fontsize=style.title_font_size,
        fontweight=style.title_font_weight,
    )
    figure.text(0.5, 0.04, text["summary_note"], ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.1, 1, 0.93))
    _fit_comparison_table_rows(figure, table)
    return figure, axis


def _fit_comparison_table_rows(figure, table):
    """Allocate each row from its rendered text height and vertical padding."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    padding = renderer.points_to_pixels(3)
    row_heights = {}
    for (row, _column), cell in table.get_celld().items():
        text_height = cell.get_text().get_window_extent(renderer).height
        row_heights[row] = max(row_heights.get(row, 0), text_height + 2 * padding)
    required_height = sum(row_heights.values())
    available_height = table.get_window_extent(renderer).height
    for (row, _column), cell in table.get_celld().items():
        cell.set_height(row_heights[row] / required_height)
    if required_height > available_height:
        width, height = figure.get_size_inches()
        figure.set_size_inches(width, height * required_height / available_height)
        figure.tight_layout(rect=(0, 0.1, 1, 0.93))
    figure.canvas.draw()


def _reserve_legend_headroom(figure, axis, legend, *, layout_rect=None):
    """Keep the legend inside the axes and above the plotted data."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    padding = COMPARISON_LEGEND_PADDING_POINTS * figure.dpi / 72.0
    legend_height = legend.get_window_extent(renderer).height
    width, height = figure.get_size_inches()
    figure.set_size_inches(width, height + (legend_height + 2 * padding) / figure.dpi)
    figure.tight_layout(rect=layout_rect)

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
    smc_diagnostics = (
        {"SMC": _smc_diagnostic_series(predictions)}
        if inference_methods[0] == "smc"
        else {}
    )
    time_seconds, metric_values = _rolling_window_metric_series(predictions)
    elapsed_time_seconds = time_seconds - time_seconds[0]
    inference_title = _format_inference_title(
        inference_selection or str(inference_methods[0])
    )
    figures = []
    metric_order = {name: index for index, name in enumerate(COMPARISON_METRIC_LABELS)}
    plot_styles = sorted(
        ROLLING_DIAGNOSTIC_PLOT_STYLES,
        key=lambda style: metric_order[style[3]],
    )
    for (
        y_axis_label,
        label,
        color,
        value_name,
        mean_label,
    ) in plot_styles:
        if value_name == "fit_runtime_seconds" and runtime_label is not None:
            y_axis_label = f"{runtime_label} [s]"
            label = runtime_label
            mean_label = f"Mittlere {runtime_label}: {{mean:.3f}} s"
        figure, axis = _comparison_subplots(
            figsize=RUNTIME_PLOT_STYLE.speed_figure_size
        )
        axis.plot(
            elapsed_time_seconds,
            metric_values[value_name],
            marker="o",
            markersize=3,
            color=color,
            label=label,
        )
        mean = float(np.mean(metric_values[value_name]))
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
        legend = axis.legend(loc=RUNTIME_PLOT_STYLE.legend_location)
        figure.suptitle(
            title_template.format(inference_title=inference_title),
            fontsize=RUNTIME_PLOT_STYLE.title_font_size,
            fontweight=RUNTIME_PLOT_STYLE.title_font_weight,
        )
        figure.text(
            0.5,
            0.02,
            mean_label.format(mean=mean),
            ha="center",
            va="bottom",
            fontsize=RUNTIME_PLOT_STYLE.axis_tick_font_size,
        )
        figure.tight_layout(rect=(0, 0.07, 1, 1))
        _reserve_legend_headroom(figure, axis, legend, layout_rect=(0, 0.07, 1, 1))
        _show_comparison_figure(figure)
        figures.append((figure, axis))
    figures.extend(
        _plot_smc_diagnostics(
            smc_diagnostics,
            colors={"SMC": COMPARISON_INFERENCE_COLORS["smc"]},
            show=True,
        )
    )
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
