"""Headless-testable spatial forecasts and two synchronized metric diagrams."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Button, CheckButtons, Slider

from .localization import text
from .settings import METRICS


class _TrajectoryAxes(Axes):
    """Fit the current viewport without undoing manual zoom or pan."""

    _follow_ship = False
    _preserve_view = False
    _view_size = None

    def apply_aspect(self, position=None):
        if self.get_adjustable() == "datalim":
            box = position if position is not None else self.get_position(original=True)
            pixels = box.transformed(self.get_figure().transFigure)
            x_limits, y_limits = (
                np.asarray(self.get_xlim()),
                np.asarray(self.get_ylim()),
            )
            x_span, y_span = abs(np.diff(x_limits)[0]), abs(np.diff(y_limits)[0])
            if min(pixels.width, pixels.height, x_span, y_span) > 0:
                box_aspect = self.get_box_aspect()
                ratio = (
                    pixels.height / pixels.width if box_aspect is None else box_aspect
                ) / float(self.get_aspect())
                if not np.isclose(y_span / x_span, ratio, rtol=1e-9, atol=0):
                    preserve_view = self._preserve_view and self._view_size == tuple(
                        self.get_figure().get_size_inches()
                    )
                    if self._follow_ship or preserve_view or y_span > ratio * x_span:
                        half = y_span / ratio / 2
                        center = x_limits.mean()
                        limits = (center - half, center + half)
                        self.set_xlim(
                            limits if x_limits[0] < x_limits[1] else limits[::-1],
                            emit=False,
                        )
                    else:
                        half = ratio * x_span / 2
                        center = y_limits.mean()
                        limits = (center - half, center + half)
                        self.set_ylim(
                            limits if y_limits[0] < y_limits[1] else limits[::-1],
                            emit=False,
                        )
        super().apply_aspect(position)


class EvaluationPlot:
    def __init__(self, *, background="#f4f9fc"):
        self.figure = Figure(
            figsize=(11.5, 7), facecolor=background, layout="constrained"
        )
        FigureCanvasAgg(self.figure)
        grid = self.figure.add_gridspec(2, 2, width_ratios=(1, 1.15))
        self._grid = grid
        self.trajectory_axis = self.figure.add_subplot(
            grid[:, 0], axes_class=_TrajectoryAxes
        )
        self.metric_axes = (
            self.figure.add_subplot(grid[0, 1]),
            self.figure.add_subplot(grid[1, 1]),
        )
        self._view_data = None
        self._view_config = None
        self._view_size = None
        self._single_metric_layout = False
        self.clear()

    def clear(self, language="en"):
        for axis in (self.trajectory_axis, *self.metric_axes):
            axis.clear()
            axis.grid(alpha=0.2)
            axis.tick_params(labelsize=9)
        self.trajectory_axis.set_aspect("equal", adjustable="box")

    def _set_layout(self, single_metric_layout):
        layout_changed = single_metric_layout != self._single_metric_layout
        self.trajectory_axis.set_box_aspect(None if single_metric_layout else 1)
        self.trajectory_axis.set_adjustable("datalim")
        # Preserve solved layout bounds between frames to avoid cumulative zoom-out.
        if layout_changed:
            self.trajectory_axis.set_subplotspec(
                self._grid[1, :] if single_metric_layout else self._grid[:, 0]
            )
        for row, axis in enumerate(self.metric_axes):
            if layout_changed:
                axis.set_subplotspec(
                    self._grid[row, :] if single_metric_layout else self._grid[row, 1]
                )
            visible = not single_metric_layout or row == 0
            axis.set_visible(visible)
            axis.set_in_layout(visible)
            axis.set_navigate(visible)
        self._single_metric_layout = single_metric_layout

    def draw(self, data, index, options, selected_models):
        """Plot saved runs at the same origin, preserving each recorded estimate."""
        if not selected_models:
            raise ValueError("Select at least one model.")
        view_config = (
            options.single_metric_layout,
            options.ship_view_span,
            options.follow_ship,
        )
        figure_size = tuple(self.figure.get_size_inches())
        view_limits = None
        if self._view_data is data and self._view_config == view_config:
            view_limits = (
                self.trajectory_axis.get_xlim(),
                self.trajectory_axis.get_ylim(),
            )
        self.clear(options.language)
        self._set_layout(options.single_metric_layout)
        self._view_data, self._view_config = data, view_config
        self.trajectory_axis._follow_ship = options.follow_ship
        self.trajectory_axis._preserve_view = (
            view_limits is not None and self._view_size == figure_size
        )
        self.trajectory_axis._view_size = figure_size
        self._view_size = figure_size
        language = options.language
        origin = data.origins[index]
        current = data.elapsed[index]
        horizon = data.horizon_seconds[index][-1]
        axis = self.trajectory_axis
        reference = data.reference_positions
        if options.show_reference_trajectory:
            axis.plot(
                *reference.T,
                color="#98a2b3",
                linestyle="--",
                marker="o" if options.show_reference_points else "None",
                markersize=3,
                label=text("Reference trajectory", language),
            )
        if options.show_reference_forecast_segment:
            segment = (data.reference_times >= origin - 1e-6) & (
                data.reference_times <= origin + horizon + 1e-6
            )
            if segment.any():
                axis.plot(
                    *reference[segment].T,
                    color="0.35",
                    linewidth=2,
                    marker="o" if options.show_reference_points else "None",
                    markersize=3,
                    label=text("Reference forecast segment", language),
                )
        current_position = None
        if data.observations is not None:
            observed = data.observations.time_seconds <= origin + 1e-6
            history = data.observations.positions[observed]
            if len(history):
                if options.show_observed_trajectory:
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
                if options.show_current_position:
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
        for label in selected_models:
            model = data.models[label]
            forecast = model.forecasts[index]
            if model.inference_method == "imm" and model.origin_positions is not None:
                tracked_origin = model.origin_positions[index]
                if data.observations is None:
                    if options.show_observed_trajectory:
                        axis.plot(
                            *model.origin_positions[: index + 1].T,
                            color="#24557A",
                            linewidth=2.2,
                            label=text(
                                "{model} track through forecast origin", language
                            ).format(model=self._model_label(label, language)),
                            zorder=2,
                        )
                    if options.show_current_position:
                        axis.scatter(
                            *tracked_origin,
                            color="#D97706",
                            s=55,
                            zorder=4,
                            label=text(
                                "{model} track at forecast origin", language
                            ).format(model=self._model_label(label, language)),
                        )
            axis.plot(
                *forecast.T,
                color=model.color,
                marker=".",
                linewidth=1.8,
                label=self._model_label(label, language),
            )
        axis.set_xlabel(text("Easting x [m]", language))
        axis.set_ylabel(text("Northing y [m]", language))
        if view_limits is not None:
            axis.set_xlim(view_limits[0])
            axis.set_ylim(view_limits[1])
        if options.follow_ship:
            center = (
                current_position
                if current_position is not None
                else np.mean(track_positions, axis=0)
                if track_positions
                else reference[np.argmin(np.abs(data.reference_times - origin))]
            )
            half_x, half_y = (
                (
                    (view_limits[0][1] - view_limits[0][0]) / 2,
                    (view_limits[1][1] - view_limits[1][0]) / 2,
                )
                if view_limits is not None
                else (options.ship_view_span / 2, options.ship_view_span / 2)
            )
            axis.set_xlim(center[0] - half_x, center[0] + half_x)
            axis.set_ylim(center[1] - half_y, center[1] + half_y)
        if options.show_trajectory_legend:
            axis.legend(loc="upper right", fontsize=8)
        left, right = 0.0, float(data.elapsed[-1]) or 1.0
        if options.follow_metric_time:
            left = max(0, current - 0.9 * options.metric_window_width)
            right = left + options.metric_window_width
        visible = (data.elapsed >= left) & (data.elapsed <= right)
        for axis, metric in zip(
            self.metric_axes, (options.top_metric, options.bottom_metric), strict=True
        ):
            if not axis.get_visible():
                continue
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
                    marker="None",
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
                    if options.show_metric_legend
                    else 0.04
                )
                axis.set_ylim(
                    lower, lower + (upper - lower + 0.05 * span) / (1 - fraction)
                )
            if options.show_metric_legend:
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
        self.figure = Figure(figsize=(11.5, 1.0), facecolor=background)
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
            "N",
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
        height = self.figure.get_figheight() * 72
        # Match the Predictor's point-sized controls and gaps at every canvas size.
        button_width = max(0.085, 64 / width)
        slider_left = 0.07 + button_width
        self.playback_button.ax.set_position(
            (0.025, 6 / height, button_width, 28 / height)
        )
        self.slider.ax.set_position(
            (
                slider_left,
                17 / height,
                max(0.05, 0.975 - slider_left - 50 / width),
                10 / height,
            )
        )
        self.follow_checkbox.ax.set_position(
            (0.025, 42 / height, 95 / width, 24 / height)
        )
        self.settings_checkbox.ax.set_position(
            (0.025 + 100 / width, 42 / height, 110 / width, 24 / height)
        )

    def set_count(self, count):
        self._count = count
        self.slider.valmax = max(1, count - 1)
        self.slider.ax.set_xlim(0, self.slider.valmax)
        self.slider.active = count > 1
        self.playback_button.active = count > 0
        self.show_index(0)

    def show_index(
        self, index, *, observation_count=None, maximum_observation_count=None
    ):
        # Programmatic synchronization must not pause or seek active playback.
        eventson, drawon = self.slider.eventson, self.slider.drawon
        self.slider.eventson = self.slider.drawon = False
        try:
            self.slider.set_val(index)
        finally:
            self.slider.eventson, self.slider.drawon = eventson, drawon
        current = observation_count if observation_count is not None else index + 1
        maximum = (
            maximum_observation_count
            if maximum_observation_count is not None
            else self._count
        )
        self.slider.valtext.set_text(f"{current}/{maximum}" if self._count else "0/0")
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
