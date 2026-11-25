"""Headless-testable spatial forecasts and two synchronized metric diagrams."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from bayestraj.validation import runtime_plotting

from .localization import text
from .settings import METRICS


class EvaluationPlot:
    def __init__(self, *, background="#f4f9fc"):
        self.figure = Figure(
            figsize=(11.5, 7), facecolor=background, layout="constrained"
        )
        FigureCanvasAgg(self.figure)
        self.figure.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
        grid = self.figure.add_gridspec(2, 2, width_ratios=(1, 1.15))
        self.trajectory_axis = self.figure.add_subplot(grid[:, 0])
        self.metric_axes = (
            self.figure.add_subplot(grid[0, 1]),
            self.figure.add_subplot(grid[1, 1]),
        )
        self.title = self.figure.suptitle("", fontsize=16, fontweight="bold")
        self.note = self.figure.text(
            0.5, 0.015, "", ha="center", va="bottom", fontsize=8
        )
        self.clear()

    def clear(self, language="en"):
        for axis in (self.trajectory_axis, *self.metric_axes):
            axis.clear()
            axis.grid(alpha=0.2)
            axis.tick_params(labelsize=9)
        self.title.set_text(text("Model evaluation", language))
        self.trajectory_axis.set_aspect("equal", adjustable="box")

    def draw(self, data, index, options, selected_models):
        """Plot saved runs at the same origin, preserving each recorded estimate."""
        if not selected_models:
            raise ValueError("Select at least one model.")
        self.clear(options.language)
        language = options.language
        origin = data.origins[index]
        current = data.elapsed[index]
        horizon = data.horizon_seconds[index][-1]
        self.title.set_text(
            f"{text('Model evaluation', language)} — {text('Forecast origin', language)} +{current:.1f} s · H = {data.prediction_counts[index]}"
        )
        axis = self.trajectory_axis
        reference = data.reference_positions
        axis.plot(
            *reference.T,
            color="#98a2b3",
            linestyle="--",
            label=text("Reference trajectory", language),
        )
        segment = (data.reference_times >= origin - 1e-6) & (
            data.reference_times <= origin + horizon + 1e-6
        )
        if segment.any():
            axis.plot(
                *reference[segment].T,
                color="0.35",
                linewidth=2,
                marker=".",
                label=text("Reference forecast segment", language),
            )
        at_origin = np.flatnonzero(
            np.isclose(data.reference_times, origin, rtol=0, atol=1e-6)
        )
        if at_origin.size:
            axis.scatter(
                *reference[at_origin[0]],
                color="0.2",
                marker="x",
                s=55,
                linewidths=1.5,
                zorder=4,
                label=text("Reference at forecast origin", language),
            )
        for label in selected_models:
            model = data.models[label]
            forecast = model.forecasts[index]
            axis.plot(
                *forecast.T,
                color=model.color,
                marker=".",
                linewidth=1.8,
                label=self._model_label(label, language),
            )
        axis.set_title(
            f"{text('Saved point forecasts', language)} · +{horizon:.1f} s", fontsize=12
        )
        axis.set_xlabel(text("Easting x [m]", language))
        axis.set_ylabel(text("Northing y [m]", language))
        if options.follow_ship:
            center = reference[np.argmin(np.abs(data.reference_times - origin))]
            half = options.ship_view_span / 2
            axis.set_xlim(center[0] - half, center[0] + half)
            axis.set_ylim(center[1] - half, center[1] + half)
        if options.show_legend:
            axis.legend(loc="upper right", fontsize=8)
        left, right = 0.0, float(data.elapsed[-1]) or 1.0
        if options.follow_metric_time:
            left = max(0, current - 0.9 * options.metric_window_width)
            right = left + options.metric_window_width
        visible = (data.elapsed >= left) & (data.elapsed <= right)
        for axis, metric in zip(
            self.metric_axes, (options.top_metric, options.bottom_metric), strict=True
        ):
            if metric not in data.available_metrics:
                raise ValueError("Select an available metric for each diagram.")
            values = []
            for label in selected_models:
                model = data.models[label]
                series = model.metrics[metric]
                values.extend(series[visible])
                axis.plot(
                    data.elapsed,
                    series,
                    label=self._model_label(label, language),
                    color=model.color,
                    linewidth=1.3,
                    marker=".",
                    markersize=3,
                )
            axis.axvline(
                current, color="0.4", linestyle="--", linewidth=1, label="_current_time"
            )
            axis.set_xlim(left, right)
            name = text(METRICS[metric], language)
            unit = (
                " [s]"
                if metric == "fit_runtime_seconds"
                else ""
                if "density" in metric
                else " [m]"
            )
            axis.set_title(name, fontsize=12)
            axis.set_ylabel(name + unit)
            axis.set_xlabel(text("Forecast origin time [s]", language))
            if values:
                lower, upper = min(values), max(values)
                span = max(upper - lower, abs(upper) * 0.1, 0.1)
                lower = (
                    lower - 0.06 * span
                    if "density" in metric
                    else max(0, lower - 0.06 * span)
                )
                # Reserve space below the model legend, independently for each metric.
                fraction = (
                    min(0.7, 0.12 + 0.08 * len(selected_models))
                    if options.show_legend
                    else 0.04
                )
                axis.set_ylim(
                    lower, lower + (upper - lower + 0.05 * span) / (1 - fraction)
                )
            if options.show_legend:
                axis.legend(loc="upper right", fontsize=8)
        strings = runtime_plotting.COMPARISON_PLOT_TEXT[language]
        self.note.set_text(
            runtime_plotting._comparison_configuration_note(strings, selected_models)
        )

    @staticmethod
    def _model_label(label, language):
        return (
            label.replace("Bayesian CTRV", "Bayessches CTRV", 1)
            if language == "de"
            else label
        )
