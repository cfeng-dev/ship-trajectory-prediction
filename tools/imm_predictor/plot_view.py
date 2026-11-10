"""Embedded Matplotlib route and forecast panels for the IMM Predictor."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.layout_engine import LayoutEngine
from matplotlib.patches import Ellipse, Patch
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from scipy.stats import chi2

from bayestraj.validation import prediction_plotting

from .player import IMMDisplayState, IMMPlaybackStore, IMMStateMixture
from .view import PLOT_BACKGROUND, TEXT_COLOR, configure_plot_toolbar

MODE_NAMES = ("CV", "CTRV", "CA")
MODE_COLORS = ("#3C78A8", "#D27D2D", "#5A9C65")
PANEL_GROUP_LABELS = {"motion": "Motion state", "forecast": "Forecast uncertainty"}


def _draw_mixture_marginal(axis, mixture: IMMStateMixture, index, scale, point_span):
    """Draw the continuous density and label any discrete mass without smoothing."""
    weights, means, sigmas = mixture.marginal_parameters(index)
    means *= scale
    sigmas *= scale
    continuous = sigmas > 0
    density_line = None
    point_masses = {}
    for mean, weight in zip(means[~continuous], weights[~continuous], strict=True):
        point_masses[float(mean)] = point_masses.get(float(mean), 0.0) + float(weight)
    if np.any(continuous):
        # A grid per component retains narrow peaks even beside broad components.
        left = float(np.min(means - 5 * sigmas))
        right = float(np.max(means + 5 * sigmas))
        span = max(right - left, point_span)
        grids = []
        for mean, sigma, weight in zip(
            means[continuous], sigmas[continuous], weights[continuous], strict=True
        ):
            # Joining fixed +/-5-sigma grids can bridge distant, very narrow peaks.
            # Extend their tails until endpoint density * span is below 1e-9.
            # This only chooses drawing points, not a variance floor or truncation.
            log_tail_bound = (
                np.log(weight)
                + np.log(span)
                - np.log(sigma)
                - np.log(np.sqrt(2 * np.pi))
                - np.log(1e-9)
            )
            radius = int(np.ceil(np.sqrt(2 * max(12.5, log_tail_bound))))
            grids.append(mean + sigma * np.linspace(-radius, radius, 100 * radius + 1))
        x = np.unique(np.concatenate([*grids, np.array(list(point_masses))]))
        density = np.zeros_like(x)
        for mean, sigma, weight in zip(
            means[continuous], sigmas[continuous], weights[continuous], strict=True
        ):
            density += (
                weight
                * np.exp(-0.5 * ((x - mean) / sigma) ** 2)
                / (sigma * np.sqrt(2 * np.pi))
            )
        (density_line,) = axis.plot(
            x, density, color="#24557A", linewidth=2, label="Posterior density"
        )
        axis.fill_between(x, density, color="#24557A", alpha=0.12)
        axis.set_xlim(left, right)
        axis.set_ylim(bottom=0)
    else:
        left, right = float(np.min(means)), float(np.max(means))
        padding = max(point_span, 0.05 * (right - left))
        axis.set_xlim(left - padding, right + padding)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Probability mass")
    for mean, weight in point_masses.items():
        label = f"Point mass at {mean:g}: {weight:.1%}"
        if np.any(continuous):
            # This guide has no density height: its discrete probability is in the label.
            axis.axvline(
                mean, linestyle=":", color="#52616B", linewidth=1.5, label=label
            )
        else:
            axis.plot(
                [mean, mean], [0, weight], color="#24557A", linewidth=2, label=label
            )
    if point_masses:
        axis.text(
            0.02,
            0.96,
            "\n".join(
                f"Point mass at {mean:g}: {weight:.1%}"
                for mean, weight in point_masses.items()
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    return density_line


class _IMMPredictorLayout(LayoutEngine):
    """Use the Predictor's fixed-point plot margins and footer spacing."""

    _adjust_compatible = False
    _colorbar_gridspec = False

    def __init__(self, view):
        super().__init__()
        self.view = view

    def execute(self, figure):
        view = self.view
        expanded = not view.settings_visible_var.get()
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
        motion_grid = view.motion_axes[0].get_subplotspec().get_gridspec()
        motion_grid.update(
            left=grid.left,
            right=grid.right,
            bottom=grid.bottom,
            top=grid.top,
            wspace=grid.wspace,
            hspace=58 / max(1.0, (height - 116 - 52 - 2 * 58) / 3),
        )
        for axis in (
            view.route_axis,
            view.mode_axis,
            view.uncertainty_axis,
            *view.motion_axes,
        ):
            spec = axis.get_subplotspec()
            if axis is not view.route_axis and not expanded:
                # Compact groups each occupy the combined right-hand columns.
                spec = spec.get_gridspec()[spec.rowspan.start : spec.rowspan.stop, 1:]
            axis.set_position(spec.get_position(figure))
        for axis in view.motion_axes:
            axis.set_visible(expanded or view._panel_group == "motion")
        for axis in (view.mode_axis, view.uncertainty_axis):
            axis.set_visible(expanded or view._panel_group == "forecast")
        view._fit_route_limits_to_axes()
        playback_width = max(0.085, 64 / width)
        view.playback_button.ax.set_position(
            (0.025, 18 / height, playback_width, 28 / height)
        )
        slider_left = 0.07 + playback_width
        selector_width = max(0.21, 160 / width)
        selector_left = 0.98 - selector_width
        slider_right = 0.94 if expanded else min(0.70, selector_left - 24 / width)
        view.slider.ax.set_position(
            (slider_left, 29 / height, slider_right - slider_left, 10 / height)
        )
        view.group_selector.ax.set_position(
            (selector_left, 18 / height, selector_width, 44 / height)
        )
        view.group_selector.ax.set_visible(not expanded)
        view.group_selector.ax.title.set_visible(not expanded)
        view.group_selector._buttons.set_visible(not expanded)
        for label in view.group_selector.labels:
            label.set_visible(not expanded)
        view.group_selector.active = not (expanded or view._failed or view._destroyed)
        for checkbox, offset, checkbox_width in (
            (view.follow_checkbox, 0, 95),
            (view.settings_checkbox, 100, 110),
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
        on_error=None,
        on_playback_change=None,
    ) -> None:
        super().__init__(parent, bg=PLOT_BACKGROUND)
        self._destroyed = False
        self._failed = False
        self._on_error = on_error
        self.store = store
        self.current_state = initial_state
        self._on_stage_changed = on_stage_changed
        self._on_settings_visibility_change = on_settings_visibility_change
        self._playing = False
        self._suspended = False
        self._on_playback_change = on_playback_change
        self._after_id = None
        self._updating_slider = False
        self._route_has_been_drawn = False
        self._follow_ship_view_needs_focus = True
        self._coordinate_display_mode = "m"
        self._panel_group = "motion"
        self.follow_ship_var = tk.BooleanVar(self, value=True)
        self.show_reference_var = tk.BooleanVar(
            self,
            value=store.reference is not None and store.config.show_recorded_trajectory,
        )
        self.settings_visible_var = tk.BooleanVar(self, value=settings_visible)

        self.figure = Figure(figsize=(11, 8), facecolor=PLOT_BACKGROUND)
        self._build_axes()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self._key_press_connection = self.canvas.mpl_connect(
            "key_press_event", self.handle_key_press
        )
        self._focus_connection = self.canvas.mpl_connect(
            "button_press_event", lambda _: self.canvas.get_tk_widget().focus_set()
        )
        self._build_controls()
        self.canvas.callbacks.exception_handler = self._handle_plot_error
        for widget in (
            self.playback_button,
            self.slider,
            self.follow_checkbox,
            self.settings_checkbox,
            self.group_selector,
        ):
            widget._observers.exception_handler = self._handle_plot_error
        self.figure.set_layout_engine(_IMMPredictorLayout(self))
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        configure_plot_toolbar(self.toolbar)
        self.toolbar.update()
        self.toolbar.pack(fill="x", side="bottom")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw()

    def _build_axes(self) -> None:
        """Keep the route fixed while compact groups share the right-hand area."""
        options = dict(
            width_ratios=(1.25, 0.5, 0.5),
            left=0.075,
            right=0.975,
            bottom=0.10,
            top=0.91,
            wspace=0.26,
            hspace=0.36,
        )
        grid = self.figure.add_gridspec(2, 3, **options)
        motion_grid = self.figure.add_gridspec(3, 3, **options)
        self.route_axis = self.figure.add_subplot(grid[:, 0])
        self.route_axis.set_axes_locator(self._locate_route_axes)
        self.mode_axis = self.figure.add_subplot(grid[0, 2])
        self.uncertainty_axis = self.figure.add_subplot(grid[1, 2])
        self.motion_axes = tuple(
            self.figure.add_subplot(motion_grid[index, 1]) for index in range(3)
        )

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
        selector_axis = self.figure.add_axes((0.77, 0.035, 0.21, 0.10))
        selector_axis.set_title("Display group", fontsize=10)
        self.group_selector = RadioButtons(
            selector_axis,
            tuple(PANEL_GROUP_LABELS.values()),
            active=0,
        )
        self.group_selector.on_clicked(self._select_panel_group)

    def _select_panel_group(self, label) -> None:
        if self._failed or self._destroyed:
            return
        self._panel_group = next(
            key for key, text in PANEL_GROUP_LABELS.items() if text == label
        )
        self._draw()

    def _on_slider_change(self, value) -> None:
        if self._updating_slider or self._failed or self._destroyed:
            return
        observation_count = int(round(float(value)))
        if observation_count == self._observation_count():
            return
        self.show_observation_count(observation_count)

    def show_observation_count(self, observation_count) -> bool:
        """Stop a failed session; never retry a possibly partially updated filter."""
        if self._failed or self._destroyed or self._suspended:
            return False
        try:
            state = (
                None
                if observation_count == 0
                else self.store.display_state_for(observation_count)
            )
            return self.set_stage(state)
        except Exception as error:
            self._handle_plot_error(error)
            return False

    def set_stage(self, state: IMMDisplayState | None) -> bool:
        """Set the online display stage and synchronize every visible component."""
        if self._failed or self._destroyed:
            return False
        previous = self.current_state
        try:
            self.current_state = state
            self._updating_slider = True
            try:
                self.slider.set_val(self._observation_count())
            finally:
                self._updating_slider = False
            if self._failed:
                self.current_state = previous
                return False
            self._on_stage_changed(state)
            self._draw()
            return True
        except Exception as error:
            self.current_state = previous
            self._handle_plot_error(error)
            return False

    def disable_navigation(self) -> None:
        self._failed = True
        self._set_playback_state(False)
        self._cancel_timer()
        self.playback_button.label.set_text("Start")
        for widget in (
            self.playback_button,
            self.slider,
            self.follow_checkbox,
            self.settings_checkbox,
            self.group_selector,
        ):
            widget.active = False

    def _handle_plot_error(self, error) -> None:
        if self._destroyed:
            return
        self.disable_navigation()
        if self._on_error is not None:
            self._on_error(error)
        else:
            self.winfo_toplevel().report_callback_exception(
                type(error), error, error.__traceback__
            )

    def toggle_playback(self) -> None:
        if self._failed or self._destroyed:
            return
        if (
            not self._playing
            and self._observation_count() >= self.store.maximum_observation_count
        ):
            if not self.set_stage(None):
                return
        self._set_playback_state(not self._playing)
        self.canvas.draw_idle()
        if self._playing:
            self._schedule_next_step()
        else:
            self._cancel_timer()

    @property
    def is_playing(self) -> bool:
        return self._playing

    def _set_playback_state(self, playing: bool) -> None:
        if self._playing == playing:
            return
        self._playing = playing
        self.playback_button.label.set_text("Pause" if playing else "Start")
        if self._on_playback_change is not None:
            self._on_playback_change(playing)

    def pause_playback(self) -> None:
        """Stop until a manual restart, including across interaction suspension."""
        self._set_playback_state(False)
        self._cancel_timer()
        self.canvas.draw_idle()

    def set_suspended(self, suspended: bool) -> None:
        """Defer IMM steps while scrolling or resizing without losing play intent."""
        if self._destroyed or self._failed or self._suspended == suspended:
            return
        self._suspended = suspended
        if suspended:
            self._cancel_timer()
        elif self._playing:
            self._schedule_next_step()

    def _schedule_next_step(self) -> None:
        if self._failed or self._destroyed or self._suspended or not self._playing:
            return
        self._cancel_timer()
        self._after_id = self.after(
            self.store.config.playback_interval_ms,
            self._advance_playback,
        )

    def _advance_playback(self) -> None:
        self._after_id = None
        if not self._playing or self._failed or self._destroyed or self._suspended:
            return
        current_count = self._observation_count()
        if current_count >= self.store.maximum_observation_count:
            self.pause_playback()
            return
        if self.show_observation_count(current_count + 1):
            if self._observation_count() >= self.store.maximum_observation_count:
                self.pause_playback()
            else:
                self._schedule_next_step()

    def _observation_count(self) -> int:
        if self.current_state is None:
            return 0
        return self.current_state.stage.observation_count

    def _cancel_timer(self) -> None:
        if self._after_id is not None:
            after_id = self._after_id
            self._after_id = None
            try:
                self.after_cancel(after_id)
            except tk.TclError:
                pass  # The enclosing Tk window has already removed its timers.

    def handle_key_press(self, event) -> None:
        """Apply the Predictor's playback shortcuts on the Matplotlib canvas."""
        self.handle_key(event.key)

    def handle_key(self, key: str) -> bool:
        """Toggle playback or select the preceding/following online stage."""
        if self._failed or self._destroyed:
            return False
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
        self.show_observation_count(observation_count)
        return True

    def _toggle_settings(self) -> None:
        self._on_settings_visibility_change(self.settings_visible_var.get())

    def set_settings_visible(self, visible: bool) -> None:
        """Synchronize the checkbox and compact/expanded diagnostic panels."""
        self.settings_visible_var.set(visible)
        eventson = self.settings_checkbox.eventson
        self.settings_checkbox.eventson = False
        try:
            self.settings_checkbox.set_active(0, state=bool(visible))
        finally:
            self.settings_checkbox.eventson = eventson
        self._draw()

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

    def _locate_route_axes(self, axis, _renderer):
        # Figure.draw checks aspect before executing the layout engine. This
        # locator also handles that early pass after resizing or toolbar zoom.
        self._fit_route_limits_to_axes()
        return axis.get_position(original=True)

    def _fit_route_limits_to_axes(self) -> None:
        """Expand the viewport to equal spatial scale before Matplotlib draws.

        Keep the center and contain the selected zoom rectangle. Matching the
        final layout here prevents datalim aspect correction from overriding
        fixed limits, including during window resizing and toolbar navigation.
        """
        axis = self.route_axis
        position = axis.get_position(original=True)
        box_width = position.width * self.figure.get_figwidth()
        box_height = position.height * self.figure.get_figheight()
        x_limits, y_limits = np.asarray(axis.get_xlim()), np.asarray(axis.get_ylim())
        x_span, y_span = abs(np.diff(x_limits)[0]), abs(np.diff(y_limits)[0])
        if min(box_width, box_height, x_span, y_span) <= 0:
            return
        ratio = box_height / box_width
        if np.isclose(y_span / x_span, ratio, rtol=1e-9, atol=0):
            return
        if y_span < ratio * x_span:
            half_span = ratio * x_span / 2
            center = y_limits.mean()
            limits = (center - half_span, center + half_span)
            axis.set_ylim(limits if y_limits[0] < y_limits[1] else limits[::-1])
        else:
            half_span = y_span / ratio / 2
            center = x_limits.mean()
            limits = (center - half_span, center + half_span)
            axis.set_xlim(limits if x_limits[0] < x_limits[1] else limits[::-1])

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
        expanded = not self.settings_visible_var.get()
        if expanded or self._panel_group == "motion":
            self._draw_motion_posteriors()
        if expanded or self._panel_group == "forecast":
            self._draw_mode_probabilities()
            self._draw_uncertainty()
        self.figure.suptitle(
            f"IMM online update - N = {self._observation_count()}",
            fontsize=16,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        self.canvas.draw_idle()

    def _draw_motion_posteriors(self) -> None:
        """Plot cached model mixtures; retain a labelled fused-only backend fallback."""
        stage = None if self.current_state is None else self.current_state.stage
        posterior = None if stage is None else stage.posterior
        mixture = None if stage is None else stage.posterior_mixture
        specifications = (
            ("Speed posterior", 4, 1.0, r"$v_N$ [m/s]", 1.0),
            ("Heading posterior", 2, 180.0 / np.pi, r"$\theta_N$ [$^\circ$]", 10.0),
            ("Turn-rate posterior", 3, 180.0 / np.pi, r"$\omega_N$ [$^\circ$/s]", 1.0),
        )
        for axis, (title, index, scale, xlabel, point_span) in zip(
            self.motion_axes, specifications, strict=True
        ):
            show_legend = self.store.config.show_legend and axis is self.motion_axes[0]
            axis.clear()
            axis.set_facecolor("#FFFFFF")
            axis.grid(True, alpha=0.25, linewidth=0.8)
            axis.set_title(title, fontsize=11)
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Density")
            if posterior is None and mixture is None:
                axis.text(
                    0.5,
                    0.5,
                    "Available after\nN = 1"
                    if self.current_state is None
                    else "Fused state posterior\nunavailable",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="#52616B",
                )
                continue
            if mixture is not None:
                density_line = _draw_mixture_marginal(
                    axis,
                    mixture,
                    index,
                    scale,
                    point_span,
                )
                if show_legend and density_line is not None:
                    axis.legend(
                        handles=[density_line],
                        loc="upper right",
                        fontsize=8,
                        framealpha=0.9,
                    )
                continue
            mean = float(posterior.mean[index] * scale)
            sigma = float(np.sqrt(max(0.0, posterior.covariance[index, index])) * scale)
            if sigma == 0.0:
                axis.axvline(mean, color="#24557A", linewidth=2.0, label="Point mass")
                axis.set_xlim(mean - point_span, mean + point_span)
                axis.set_yticks([])
                axis.set_ylabel("Probability mass")
                axis.text(
                    0.02,
                    0.96,
                    "Fused-state fallback\nPoint mass (100%)",
                    transform=axis.transAxes,
                    va="top",
                    fontsize=8,
                )
            else:
                x = mean + sigma * np.linspace(-4.0, 4.0, 401)
                density = np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (
                    sigma * np.sqrt(2.0 * np.pi)
                )
                axis.plot(
                    x,
                    density,
                    color="#24557A",
                    linewidth=2.0,
                    label="Gaussian approximation",
                )
                axis.set_xlim(x[0], x[-1])
                axis.set_ylim(bottom=0.0)
                if not show_legend:
                    axis.text(
                        0.98,
                        0.96,
                        "Gaussian approximation",
                        transform=axis.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8,
                    )
            if show_legend and sigma > 0:
                axis.legend(loc="upper right", fontsize=8, framealpha=0.9)

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
        axis.set_aspect("equal", adjustable="datalim")
        if self.show_reference_var.get() and self.store.reference is not None:
            axis.plot(
                self.store.reference.x_meters * scale,
                self.store.reference.y_meters * scale,
                color="#9CA3AF",
                linestyle="--",
                linewidth=1.5,
                label="Reference trajectory",
                marker="o" if self.store.config.show_reference_points else "None",
                markersize=3,
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
                label="IMM track through N",
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
        medians = (
            means
            if stage.forecast_median_positions is None
            else np.asarray(stage.forecast_median_positions, dtype=float) * scale
        )
        if (
            self.store.config.show_sample_trajectories
            and stage.sample_positions is not None
        ):
            for index, positions in enumerate(stage.sample_positions):
                positions = positions * scale
                axis.plot(
                    positions[:, 0],
                    positions[:, 1],
                    color="#DC2626",
                    alpha=0.14,
                    linewidth=0.9,
                    zorder=2,
                    label=f"Possible future trajectories (n = {len(stage.sample_positions)})"
                    if index == 0
                    else "_nolegend_",
                )
        for positions, enabled, label, color, linestyle in (
            (
                medians,
                self.store.config.show_forecast_median,
                "Forecast (median)",
                "#DC2626",
                "-",
            ),
            (
                means,
                self.store.config.show_forecast_mean,
                "Forecast (mean)",
                "#B91C1C",
                "--",
            ),
        ):
            if len(positions) and enabled:
                forecast_route = np.vstack((tracked[-1], positions))
                axis.plot(
                    forecast_route[:, 0],
                    forecast_route[:, 1],
                    color=color,
                    linestyle=linestyle,
                    marker=".",
                    markersize=4,
                    linewidth=2.2,
                    label=label,
                    zorder=4,
                )
        region_handles = self._draw_prediction_uncertainty(axis, stage)
        if len(means):
            horizon = stage.forecast.time_seconds[-1] - stage.observation_time_seconds
            axis.set_title(
                f"Ship movement · Forecast +{horizon:g} s", fontsize=13, pad=10
            )
        self._set_route_limits(axis, tracked, np.vstack((means, medians)))
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
            position = axis.get_position(original=True)
            box_ratio = (position.height * self.figure.get_figheight()) / (
                position.width * self.figure.get_figwidth()
            )
            half_width = half_span / max(1.0, box_ratio)
            half_height = half_span * min(1.0, box_ratio)
            axis.set_xlim(center[0] - half_width, center[0] + half_width)
            axis.set_ylim(center[1] - half_height, center[1] + half_height)
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
        if self._destroyed:
            return
        self._destroyed = True
        self._set_playback_state(False)
        self._cancel_timer()
        for name in ("_idle_draw_id", "_event_loop_id"):
            callback = getattr(self.canvas, name, None)
            if callback is not None:
                try:
                    self.canvas.get_tk_widget().after_cancel(callback)
                except tk.TclError:
                    pass
                setattr(self.canvas, name, None)
        self.canvas.mpl_disconnect(self._key_press_connection)
        self.canvas.mpl_disconnect(self._focus_connection)
        for widget in (
            self.playback_button,
            self.slider,
            self.follow_checkbox,
            self.settings_checkbox,
            self.group_selector,
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
