"""Embedded Matplotlib route and IMM forecast panels for the Tk player."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.layout_engine import LayoutEngine
from matplotlib.patches import Ellipse, Patch
from matplotlib.widgets import Button, CheckButtons, Slider
from scipy.stats import chi2

from bayestraj.validation import prediction_plotting

from .player import IMMDisplayState, IMMPlaybackStore
from .view import PLOT_BACKGROUND, TEXT_COLOR, configure_plot_toolbar

MODE_NAMES = ("CV", "CTRV", "CA")
MODE_COLORS = ("#3C78A8", "#D27D2D", "#5A9C65")


class _IMMPlayerLayout(LayoutEngine):
    """Use the Predictor's fixed-point plot margins and footer spacing."""

    _adjust_compatible = False
    _colorbar_gridspec = False

    def __init__(self, view):
        super().__init__()
        self.view = view

    def execute(self, figure):
        view = self.view
        width = max(400.0, figure.get_figwidth() * 72)
        height = max(320.0, figure.get_figheight() * 72)
        grid = view.route_axis.get_subplotspec().get_gridspec()
        panel_height = max(1.0, (height - 116 - 52 - 58) / 2)
        grid.update(
            left=max(0.06, 76 / width),
            right=0.98,
            bottom=116 / height,
            top=1 - 52 / height,
            hspace=58 / panel_height,
            wspace=0.28,
        )
        for axis in (view.route_axis, view.mode_axis, view.uncertainty_axis):
            axis.set_position(axis.get_subplotspec().get_position(figure))
        playback_width = max(0.085, 64 / width)
        view.playback_button.ax.set_position(
            (0.025, 18 / height, playback_width, 28 / height)
        )
        slider_left = 0.07 + playback_width
        view.slider.ax.set_position(
            (slider_left, 29 / height, 0.70 - slider_left, 10 / height)
        )
        for checkbox, offset, checkbox_width in (
            (view.follow_checkbox, 0, 95),
            (view.settings_checkbox, 100, 110),
            (view.reference_checkbox, 215, 125),
        ):
            checkbox.ax.set_position(
                (
                    0.025 + offset / width,
                    54 / height,
                    checkbox_width / width,
                    24 / height,
                )
            )


class IMMPlotView(tk.Frame):
    """Render and play the cached online IMM stages inside a Tk container."""

    def __init__(
        self,
        parent,
        *,
        store: IMMPlaybackStore,
        initial_state: IMMDisplayState | None,
        on_stage_changed,
        on_settings_visibility_change,
        settings_visible: bool,
    ) -> None:
        super().__init__(parent, bg=PLOT_BACKGROUND)
        self.store = store
        self.current_state = initial_state
        self._on_stage_changed = on_stage_changed
        self._on_settings_visibility_change = on_settings_visibility_change
        self._playing = False
        self._after_id = None
        self._updating_slider = False
        self._route_has_been_drawn = False
        self._follow_ship_view_needs_focus = True
        self._coordinate_display_mode = "m"
        self.follow_ship_var = tk.BooleanVar(self, value=True)
        self.show_reference_var = tk.BooleanVar(
            self,
            value=store.reference is not None and store.config.show_recorded_trajectory,
        )
        self.settings_visible_var = tk.BooleanVar(self, value=settings_visible)

        self.figure = Figure(figsize=(11, 8), facecolor=PLOT_BACKGROUND)
        grid = self.figure.add_gridspec(
            2,
            2,
            width_ratios=(1.25, 1.0),
            height_ratios=(1.0, 1.0),
            left=0.075,
            right=0.975,
            bottom=0.10,
            top=0.91,
            wspace=0.26,
            hspace=0.36,
        )
        self.route_axis = self.figure.add_subplot(grid[:, 0])
        self.mode_axis = self.figure.add_subplot(grid[0, 1])
        self.uncertainty_axis = self.figure.add_subplot(grid[1, 1])
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self._key_press_connection = self.canvas.mpl_connect(
            "key_press_event", self.handle_key_press
        )
        self._focus_connection = self.canvas.mpl_connect(
            "button_press_event", lambda _: self.canvas.get_tk_widget().focus_set()
        )
        self._build_controls()
        self.figure.set_layout_engine(_IMMPlayerLayout(self))
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        configure_plot_toolbar(self.toolbar)
        self.toolbar.update()
        self.toolbar.pack(fill="x", side="bottom")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw()

    def _build_controls(self) -> None:
        self.playback_button = Button(
            self.figure.add_axes((0.025, 0.05, 0.085, 0.055)), "Start"
        )
        self.playback_button.on_clicked(lambda _: self.toggle_playback())
        self.slider = Slider(
            self.figure.add_axes((0.15, 0.065, 0.55, 0.035)),
            "N",
            0,
            self.store.maximum_observation_count,
            valinit=self._observation_count(),
            valstep=1,
            valfmt="%0.0f",
        )
        self.slider.on_changed(self._on_slider_change)

        def checkbox(label, variable, callback):
            axis = self.figure.add_axes((0.025, 0.12, 0.2, 0.04))
            axis.set_frame_on(False)
            control = CheckButtons(
                axis,
                (label,),
                (variable.get(),),
                useblit=False,
                label_props={"fontsize": [10]},
                frame_props={"s": [196], "linewidth": [1.2]},
                check_props={"s": [196], "linewidth": [1.5]},
            )

            def changed(_label):
                variable.set(control.get_status()[0])
                callback()

            control.on_clicked(changed)
            return control

        self.follow_checkbox = checkbox(
            "Follow ship", self.follow_ship_var, self._follow_ship_changed
        )
        self.settings_checkbox = checkbox(
            "Show settings", self.settings_visible_var, self._toggle_settings
        )
        self.reference_checkbox = checkbox(
            "Show reference", self.show_reference_var, self._draw
        )
        self.reference_checkbox.active = self.store.reference is not None

    def _on_slider_change(self, value) -> None:
        if self._updating_slider:
            return
        observation_count = int(round(float(value)))
        if observation_count == self._observation_count():
            return
        if observation_count == 0:
            self.set_stage(None)
            return
        self.set_stage(self.store.display_state_for(observation_count))

    def set_stage(self, state: IMMDisplayState | None) -> None:
        """Set the online display stage and synchronize every visible component."""
        self.current_state = state
        self._updating_slider = True
        self.slider.set_val(self._observation_count())
        self._updating_slider = False
        self._on_stage_changed(state)
        self._draw()

    def toggle_playback(self) -> None:
        if (
            not self._playing
            and self._observation_count() >= self.store.maximum_observation_count
        ):
            self.set_stage(None)
        self._playing = not self._playing
        self.playback_button.label.set_text("Pause" if self._playing else "Start")
        self.canvas.draw_idle()
        if self._playing:
            self._schedule_next_step()
        else:
            self._cancel_timer()

    def _schedule_next_step(self) -> None:
        self._cancel_timer()
        self._after_id = self.after(
            self.store.config.playback_interval_ms,
            self._advance_playback,
        )

    def _advance_playback(self) -> None:
        self._after_id = None
        if not self._playing:
            return
        current_count = self._observation_count()
        if current_count >= self.store.maximum_observation_count:
            self.toggle_playback()
            return
        self.set_stage(self.store.display_state_for(current_count + 1))
        self._schedule_next_step()

    def _observation_count(self) -> int:
        if self.current_state is None:
            return 0
        return self.current_state.stage.observation_count

    def _cancel_timer(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None

    def handle_key_press(self, event) -> None:
        """Apply the Predictor's playback shortcuts on the Matplotlib canvas."""
        self.handle_key(event.key)

    def handle_key(self, key: str) -> bool:
        """Toggle playback or select the preceding/following online stage."""
        if key in {" ", "space"}:
            self.toggle_playback()
            return True
        if key not in {"left", "right"}:
            return False
        if self._playing:
            self.toggle_playback()
        step = -1 if key == "left" else 1
        observation_count = min(
            max(self._observation_count() + step, 0),
            self.store.maximum_observation_count,
        )
        if observation_count == self._observation_count():
            return True
        if observation_count == 0:
            self.set_stage(None)
        else:
            self.set_stage(self.store.display_state_for(observation_count))
        return True

    def _toggle_settings(self) -> None:
        self._on_settings_visibility_change(self.settings_visible_var.get())

    def set_settings_visible(self, visible: bool) -> None:
        self.settings_visible_var.set(visible)
        self.settings_checkbox.eventson = False
        self.settings_checkbox.set_active(0, state=bool(visible))
        self.settings_checkbox.eventson = True

    def reset_view(self) -> None:
        """Return to the full trajectory extent without touching IMM state."""
        self.follow_ship_var.set(False)
        self.follow_checkbox.set_active(0, state=False)
        self._follow_ship_view_needs_focus = False
        self._route_has_been_drawn = False
        self.toolbar.home()
        self._draw()

    def _coordinate_scale(self) -> float:
        return 0.001 if self._coordinate_display_mode == "km" else 1.0

    def set_coordinate_display_mode(self, mode: str) -> None:
        """Rescale coordinates and the viewport, retaining the physical zoom."""
        if mode not in {"m", "km"}:
            raise ValueError("Coordinate display must be 'm' or 'km'.")
        if mode == self._coordinate_display_mode:
            return
        old_scale = self._coordinate_scale()
        self._coordinate_display_mode = mode
        ratio = self._coordinate_scale() / old_scale
        self.route_axis.set_xlim(np.asarray(self.route_axis.get_xlim()) * ratio)
        self.route_axis.set_ylim(np.asarray(self.route_axis.get_ylim()) * ratio)
        self.toolbar.update()
        self._draw()

    def _draw(self) -> None:
        self._draw_route()
        self._draw_mode_probabilities()
        self._draw_uncertainty()
        self.figure.suptitle(
            f"BR24 IMM online update - N = {self._observation_count()}",
            fontsize=16,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        self.canvas.draw_idle()

    def _draw_route(self) -> None:
        axis = self.route_axis
        view_limits = None
        if self._route_has_been_drawn:
            view_limits = (axis.get_xlim(), axis.get_ylim())
        axis.clear()
        axis.set_facecolor("#FFFFFF")
        axis.grid(True, alpha=0.25, linewidth=0.8)
        axis.set_title("Ship movement", fontsize=13, pad=10)
        scale = self._coordinate_scale()
        axis.set_xlabel(f"Easting x [{self._coordinate_display_mode}]", fontsize=11)
        axis.set_ylabel(f"Northing y [{self._coordinate_display_mode}]", fontsize=11)
        axis.set_aspect("equal", adjustable="box")
        if self.show_reference_var.get() and self.store.reference is not None:
            axis.plot(
                self.store.reference.x_meters * scale,
                self.store.reference.y_meters * scale,
                color="#9CA3AF",
                linewidth=1.5,
                label="Recorded trajectory",
                zorder=1,
            )
        if self.current_state is None:
            self._set_route_limits(
                axis,
                tracked=np.empty((0, 2)),
                means=np.empty((0, 2)),
            )
            self._restore_route_limits(axis, view_limits)
            self._route_has_been_drawn = True
            if self.show_reference_var.get() and self.store.reference is not None:
                if self.store.config.show_legend:
                    axis.legend(loc="upper right", fontsize=9, framealpha=0.9)
            elif self.store.reference is None:
                axis.text(
                    0.5,
                    0.5,
                    "Reference preview is unavailable.\nFirst IMM update: N = 1.",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="#52616B",
                )
            return

        stage = self.current_state.stage
        tracked = np.asarray(stage.tracked_positions, dtype=float) * scale
        if self.store.config.show_imm_track:
            axis.plot(
                tracked[:, 0],
                tracked[:, 1],
                color="#24557A",
                linewidth=2.2,
                label="IMM estimates through N",
                zorder=3,
            )
        if self.store.config.show_current_position:
            axis.scatter(
                tracked[-1, 0],
                tracked[-1, 1],
                s=45,
                color="#D97706",
                label="Current position",
                zorder=5,
            )
        means = (
            np.empty((0, 2))
            if stage.forecast is None
            else np.asarray(stage.forecast.position_means, dtype=float) * scale
        )
        forecast_route = np.vstack((tracked[-1], means))
        if len(means) and self.store.config.show_forecast_mean:
            axis.plot(
                forecast_route[:, 0],
                forecast_route[:, 1],
                color="#DC2626",
                marker=".",
                markersize=4,
                linewidth=2.2,
                label="Forecast (mean)",
                zorder=4,
            )
        region_handles = self._draw_prediction_uncertainty(axis, stage)
        if len(means):
            horizon = stage.forecast.time_seconds[-1] - stage.observation_time_seconds
            axis.set_title(
                f"Ship movement · Forecast +{horizon:g} s", fontsize=13, pad=10
            )
        self._set_route_limits(axis, tracked, means)
        self._restore_route_limits(axis, view_limits)
        self._center_route_on_ship(axis, tracked)
        self._route_has_been_drawn = True
        handles, labels = axis.get_legend_handles_labels()
        if self.store.config.show_legend and (handles or region_handles):
            axis.legend(
                handles=handles + region_handles,
                labels=labels + [handle.get_label() for handle in region_handles],
                loc="upper right",
                fontsize=9,
                framealpha=0.9,
            )

    def _draw_prediction_uncertainty(self, axis, stage) -> list:
        """Render IMM Gaussian marginals using the Predictor's region styles."""
        if stage.forecast is None:
            return []
        config = self.store.config
        probabilities = tuple(
            probability
            for probability, enabled in (
                (0.5, config.show_prediction_region_50),
                (0.9, config.show_prediction_region_90),
            )
            if enabled
        )
        style = config.prediction_region_style
        scale = self._coordinate_scale()
        means = np.asarray(stage.forecast.position_means) * scale
        covariances = np.asarray(stage.forecast.position_covariances) * scale**2
        if style == "none" or not probabilities or not len(means):
            return []
        if style == "ellipses":
            for mean, covariance in zip(means, covariances, strict=True):
                self._draw_covariance_ellipses(axis, mean, covariance, probabilities)
            return [
                Patch(
                    facecolor=prediction_plotting.POSTERIOR_COLOR,
                    edgecolor="none",
                    alpha=_region_alpha(probability),
                    label=f"Covariance ellipse ({100 * probability:g} %)",
                )
                for probability in probabilities
            ]
        # Fixed visualization samples of each Gaussian marginal. Their row
        # pairing has no temporal meaning and is never drawn as sample paths.
        normal_draws = np.random.default_rng(42).standard_normal((512, 2))
        draws = np.stack(
            [
                mean
                + normal_draws
                @ np.linalg.cholesky(_regularize_covariance(covariance)).T
                for mean, covariance in zip(means, covariances, strict=True)
            ],
            axis=1,
        )
        posterior_draws = (draws[:, :, 0], draws[:, :, 1])
        if style in {"funnel", "horizon_density_contours"}:
            draw = (
                prediction_plotting._draw_horizon_prediction_density_regions
                if style == "funnel"
                else prediction_plotting._draw_horizon_prediction_density_contours
            )
            return draw(axis, posterior_draws, region_probabilities=probabilities)
        draw = (
            prediction_plotting._draw_prediction_density_regions
            if style == "density_regions"
            else prediction_plotting._draw_prediction_density_contours
        )
        times = np.concatenate(
            (
                [0.0],
                np.asarray(stage.forecast.time_seconds)
                - stage.observation_time_seconds,
            )
        )
        return draw(
            axis,
            posterior_draws,
            times,
            annotate_time=False,
            region_probabilities=probabilities,
        )

    def _draw_covariance_ellipses(self, axis, mean, covariance, probabilities) -> None:
        covariance = _regularize_covariance(covariance)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, -1]
        angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
        for probability in sorted(probabilities, reverse=True):
            radius = float(np.sqrt(chi2.ppf(probability, df=2)))
            axis.add_patch(
                Ellipse(
                    xy=mean,
                    width=2.0 * radius * np.sqrt(eigenvalues[-1]),
                    height=2.0 * radius * np.sqrt(eigenvalues[0]),
                    angle=angle,
                    facecolor=prediction_plotting.POSTERIOR_COLOR,
                    edgecolor=prediction_plotting.POSTERIOR_COLOR,
                    linewidth=prediction_plotting.PREDICTION_REGION_EDGE_LINE_WIDTH,
                    alpha=_region_alpha(probability),
                    zorder=1,
                )
            )

    def _set_route_limits(self, axis, tracked: np.ndarray, means: np.ndarray) -> None:
        points = [array for array in (tracked, means) if len(array)]
        if self.show_reference_var.get() and self.store.reference is not None:
            points.append(
                np.column_stack(
                    (self.store.reference.x_meters, self.store.reference.y_meters)
                )
                * self._coordinate_scale()
            )
        if not points:
            axis.set_xlim(-1.0, 1.0)
            axis.set_ylim(-1.0, 1.0)
            return
        all_points = np.vstack(points)
        lower = np.min(all_points, axis=0)
        upper = np.max(all_points, axis=0)
        margin = np.maximum(10.0 * self._coordinate_scale(), 0.05 * (upper - lower))
        axis.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
        axis.set_ylim(lower[1] - margin[1], upper[1] + margin[1])

    @staticmethod
    def _restore_route_limits(axis, view_limits) -> None:
        if view_limits is not None:
            axis.set_xlim(view_limits[0])
            axis.set_ylim(view_limits[1])

    def _follow_ship_changed(self) -> None:
        """Focus once on enable; retain manual zoom during later redraws."""
        if self.follow_ship_var.get():
            self._follow_ship_view_needs_focus = True
        else:
            self._follow_ship_view_needs_focus = False
            self._route_has_been_drawn = False
        self._draw()

    def _center_route_on_ship(self, axis, tracked: np.ndarray) -> None:
        """Follow the ship while preserving the current manual x/y spans."""
        if not self.follow_ship_var.get() or not len(tracked):
            return
        center = tracked[-1]
        if self._follow_ship_view_needs_focus:
            half_span = (
                self.store.config.follow_ship_view_span_meters
                * self._coordinate_scale()
                / 2.0
            )
            axis.set_xlim(center[0] - half_span, center[0] + half_span)
            axis.set_ylim(center[1] - half_span, center[1] + half_span)
            self._follow_ship_view_needs_focus = False
            return
        for get_limits, set_limits, position in (
            (axis.get_xlim, axis.set_xlim, center[0]),
            (axis.get_ylim, axis.set_ylim, center[1]),
        ):
            lower, upper = get_limits()
            half_span = (upper - lower) / 2.0
            set_limits(position - half_span, position + half_span)

    def _draw_mode_probabilities(self) -> None:
        axis = self.mode_axis
        axis.clear()
        axis.set_facecolor("#FFFFFF")
        axis.grid(True, axis="y", color="#DCE5EC", linewidth=0.7)
        axis.set_title("Forecast mode probabilities", fontsize=11)
        if self.store.config.prediction_count == 0:
            axis.text(
                0.5,
                0.5,
                "Forecast disabled (0 steps)",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#52616B",
            )
            return
        if self.current_state is None:
            axis.text(
                0.5,
                0.5,
                "Available after\nN = 1",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#52616B",
            )
            return
        stage = self.current_state.stage
        offsets = (
            np.asarray(stage.forecast.time_seconds) - stage.observation_time_seconds
        )
        modes = np.asarray(stage.forecast.mode_probabilities, dtype=float)
        for index, (name, color) in enumerate(
            zip(MODE_NAMES, MODE_COLORS, strict=True)
        ):
            axis.plot(offsets, modes[:, index], marker="o", color=color, label=name)
        axis.set_xlabel("Horizon [s]")
        axis.set_ylabel("Probability")
        axis.set_ylim(-0.02, 1.02)
        axis.legend(loc="best", fontsize=8, framealpha=0.95)

    def _draw_uncertainty(self) -> None:
        axis = self.uncertainty_axis
        axis.clear()
        axis.set_facecolor("#FFFFFF")
        axis.grid(True, axis="y", color="#DCE5EC", linewidth=0.7)
        axis.set_title("Forecast uncertainty", fontsize=11)
        if self.store.config.prediction_count == 0:
            axis.text(
                0.5,
                0.5,
                "Forecast disabled (0 steps)",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#52616B",
            )
            return
        if self.current_state is None:
            axis.text(
                0.5,
                0.5,
                "Available after\nN = 1",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#52616B",
            )
            return
        stage = self.current_state.stage
        offsets = (
            np.asarray(stage.forecast.time_seconds) - stage.observation_time_seconds
        )
        uncertainty = self.current_state.uncertainty
        axis.plot(
            offsets,
            uncertainty.radius_50_meters,
            marker="o",
            color="#5A9C65",
            label="50% radius",
        )
        axis.plot(
            offsets,
            uncertainty.radius_90_meters,
            marker="o",
            color="#D95F02",
            label="90% radius",
        )
        axis.set_xlabel("Horizon [s]")
        axis.set_ylabel("Equivalent radius [m]")
        axis.legend(loc="best", fontsize=8, framealpha=0.95)

    def destroy(self) -> None:
        """Cancel scheduled playback before the enclosing Tk view is discarded."""
        self._playing = False
        self._cancel_timer()
        self.canvas.mpl_disconnect(self._key_press_connection)
        self.canvas.mpl_disconnect(self._focus_connection)
        for widget in (
            self.playback_button,
            self.slider,
            self.follow_checkbox,
            self.settings_checkbox,
            self.reference_checkbox,
        ):
            widget.disconnect_events()
        super().destroy()


def _regularize_covariance(covariance) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    return (eigenvectors * np.maximum(eigenvalues, 1e-12)) @ eigenvectors.T


def _region_alpha(probability: float) -> float:
    return (
        prediction_plotting.PREDICTION_REGION_50_ALPHA
        if probability == 0.5
        else prediction_plotting.PREDICTION_REGION_90_ALPHA
    )
