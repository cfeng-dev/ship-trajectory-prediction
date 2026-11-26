"""Headless-testable spatial forecasts and two synchronized metric diagrams."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Button, CheckButtons, Slider

from .localization import text
from .settings import METRICS


class EvaluationPlot:
    def __init__(self, *, background="#f4f9fc"):
        self.figure = Figure(
            figsize=(11.5, 7), facecolor=background, layout="constrained"
        )
        FigureCanvasAgg(self.figure)
        grid = self.figure.add_gridspec(2, 2, width_ratios=(1, 1.15))
        self.trajectory_axis = self.figure.add_subplot(grid[:, 0])
        self.metric_axes = (
            self.figure.add_subplot(grid[0, 1]),
            self.figure.add_subplot(grid[1, 1]),
        )
        self.clear()

    def clear(self, language="en"):
        for axis in (self.trajectory_axis, *self.metric_axes):
            axis.clear()
            axis.grid(alpha=0.2)
            axis.tick_params(labelsize=9)
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
        current_position = None
        observation_origin = None
        if data.observations is not None:
            observed = data.observations.time_seconds <= origin + 1e-6
            history = data.observations.positions[observed]
            if len(history):
                axis.plot(
                    *history.T,
                    color="#24557A",
                    linewidth=2.2,
                    label=text("Observations through N", language),
                    zorder=2,
                )
                current_position = history[-1]
                at_origin = np.isclose(
                    data.observations.time_seconds[observed][-1],
                    origin,
                    rtol=0,
                    atol=1e-6,
                )
                if at_origin:
                    observation_origin = current_position
                axis.plot(
                    *current_position,
                    marker="o",
                    markersize=7,
                    linestyle="none",
                    color="#D97706",
                    label=text(
                        "Current position" if at_origin else "Latest observation",
                        language,
                    ),
                    zorder=4,
                )
        track_positions = [
            data.models[label].origin_positions[index]
            for label in selected_models
            if data.models[label].inference_method == "imm"
            and data.models[label].origin_positions is not None
        ]
        shared_origin = (
            track_positions[0]
            if track_positions
            and np.allclose(track_positions, track_positions[0], rtol=0, atol=1e-6)
            else None
        )
        if data.observations is not None:
            shared_origin = observation_origin
        for label in selected_models:
            model = data.models[label]
            forecast = model.forecasts[index]
            if model.inference_method == "imm" and model.origin_positions is not None:
                tracked_origin = model.origin_positions[index]
                if data.observations is None:
                    axis.plot(
                        *model.origin_positions[: index + 1].T,
                        color="#24557A",
                        linewidth=2.2,
                        label=text(
                            "{model} track through forecast origin", language
                        ).format(model=self._model_label(label, language)),
                        zorder=2,
                    )
                    axis.scatter(
                        *tracked_origin,
                        color="#D97706",
                        s=55,
                        zorder=4,
                        label=text("{model} track at forecast origin", language).format(
                            model=self._model_label(label, language)
                        ),
                    )
            if shared_origin is not None:
                # This display link does not change the model's saved forecast.
                axis.plot(
                    *np.vstack((shared_origin, forecast[0])).T,
                    color=model.color,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                    label="_imm_track_connection",
                )
            axis.plot(
                *forecast.T,
                color=model.color,
                marker=".",
                linewidth=1.8,
                label=self._model_label(label, language),
            )
        axis.set_title(
            text("Ship movement · Forecast +{seconds} s", language).format(
                seconds=f"{horizon:.1f}"
            ),
            fontsize=12,
        )
        axis.set_xlabel(text("Easting x [m]", language))
        axis.set_ylabel(text("Northing y [m]", language))
        if options.follow_ship:
            center = (
                current_position
                if current_position is not None
                else np.mean(track_positions, axis=0)
                if track_positions
                else reference[np.argmin(np.abs(data.reference_times - origin))]
            )
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

    @staticmethod
    def _model_label(label, language):
        return (
            label.replace("Bayesian CTRV", "Bayessches CTRV", 1)
            if language == "de"
            else label
        )


class PlaybackBar:
    """Use the Predictor's Matplotlib controls without coupling to inference."""

    def __init__(
        self, *, on_seek, on_play, on_follow, on_settings, background="#f4f9fc"
    ):
        self.figure = Figure(figsize=(11.5, 0.9), facecolor=background)
        FigureCanvasAgg(self.figure)
        self._count = 0
        self._playing = False
        self.language = "en"
        self.playback_button = Button(
            self.figure.add_axes((0.025, 0.08, 0.085, 0.42)), "Start"
        )
        self.playback_button.on_clicked(lambda _: on_play())
        self.slider = Slider(
            self.figure.add_axes((0.18, 0.18, 0.62, 0.16)),
            "Index",
            0,
            1,
            valinit=0,
            valstep=1,
            valfmt="%0.0f",
        )
        self.slider.on_changed(
            lambda value: on_seek(int(round(value))) if self._count > 1 else None
        )

        def checkbox(label, position, callback):
            axis = self.figure.add_axes(position, facecolor=background)
            axis.set_frame_on(False)
            control = CheckButtons(
                axis,
                (label,),
                (True,),
                useblit=False,
                label_props={"fontsize": [10]},
                frame_props={"s": [196], "linewidth": [1.2]},
                check_props={"s": [196], "linewidth": [1.5]},
            )
            control.on_clicked(lambda _: callback(control.get_status()[0]))
            return control

        self.follow_checkbox = checkbox(
            "Follow ship", (0.025, 0.55, 0.14, 0.4), on_follow
        )
        self.settings_checkbox = checkbox(
            "Show settings", (0.18, 0.55, 0.2, 0.4), on_settings
        )
        self.figure.canvas.mpl_connect("resize_event", self._resize)
        self._resize()
        self.set_count(0)

    def _resize(self, _event=None):
        width = self.figure.get_figwidth() * 72
        button_width = 64 / width
        slider_left = 0.025 + button_width + 50 / width
        self.playback_button.ax.set_position((0.025, 0.08, button_width, 0.42))
        self.slider.ax.set_position(
            (slider_left, 0.18, max(0.05, 0.975 - slider_left - 140 / width), 0.16)
        )
        self.follow_checkbox.ax.set_position((0.025, 0.55, 110 / width, 0.4))
        self.settings_checkbox.ax.set_position(
            (0.025 + 120 / width, 0.55, 160 / width, 0.4)
        )

    def set_count(self, count):
        self._count = count
        self.slider.valmax = max(1, count - 1)
        self.slider.ax.set_xlim(0, self.slider.valmax)
        self.slider.active = count > 1
        self.playback_button.active = count > 0
        self.show_index(0, elapsed_seconds=0)

    def show_index(self, index, *, elapsed_seconds):
        # Programmatic synchronization must not pause or seek active playback.
        eventson, drawon = self.slider.eventson, self.slider.drawon
        self.slider.eventson = self.slider.drawon = False
        try:
            self.slider.set_val(index)
        finally:
            self.slider.eventson, self.slider.drawon = eventson, drawon
        self.slider.valtext.set_text(
            f"{index + 1}/{self._count} · {elapsed_seconds:.1f} s"
            if self._count
            else "—"
        )
        self.figure.canvas.draw_idle()

    def set_playing(self, playing):
        self._playing = playing
        self.playback_button.label.set_text(
            text("Pause" if playing else "Start", self.language)
        )
        self.figure.canvas.draw_idle()

    def set_language(self, language):
        self.language = language
        self.follow_checkbox.labels[0].set_text(text("Follow ship", language))
        self.settings_checkbox.labels[0].set_text(text("Show settings", language))
        self.set_playing(self._playing)

    @staticmethod
    def _sync_checkbox(control, value):
        eventson = control.eventson
        control.eventson = False
        try:
            if control.get_status()[0] != bool(value):
                control.set_active(0)
        finally:
            control.eventson = eventson

    def set_settings_visible(self, visible):
        self._sync_checkbox(self.settings_checkbox, visible)

    def set_follow_ship(self, follow):
        self._sync_checkbox(self.follow_checkbox, follow)
