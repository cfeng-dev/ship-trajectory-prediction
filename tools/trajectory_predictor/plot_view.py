"""Embed the existing Matplotlib dashboard in one disposable Tk view."""

import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .dashboard import (
    create_sequential_posterior_dashboard_figure,
)
from .localization import widget_language
from .view import PLOT_BACKGROUND, configure_plot_toolbar


class PosteriorPlotView(tk.Frame):
    """Own the embedded canvas, navigation toolbar and dashboard callbacks."""

    def __init__(
        self,
        parent,
        settings,
        trajectory,
        maximum,
        minimum,
        request_update,
        on_state_change=None,
        on_metrics_change=None,
        on_medians_change=None,
        on_radar_association_change=None,
        settings_visible=True,
        on_settings_visibility_change=None,
        on_playback_change=None,
    ):
        super().__init__(parent, bg=PLOT_BACKGROUND)
        self._destroyed = False
        self.figure = Figure(figsize=(11, 8), facecolor=PLOT_BACKGROUND)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        configure_plot_toolbar(self.toolbar)
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        _, self.navigator = create_sequential_posterior_dashboard_figure(
            trajectory,
            settings.analysis.priors,
            maximum_observation_count=maximum,
            minimum_posterior_observation_count=minimum,
            show_legend=settings.show_legend,
            playback_interval_ms=settings.playback_interval_ms,
            figure=self.figure,
            request_update=request_update,
            prediction_count=settings.analysis.experiment.prediction_count,
            coordinate_display_mode=settings.coordinate_display_mode,
            show_reference_trajectory=settings.show_reference_trajectory,
            show_observed_trajectory=settings.show_observed_trajectory,
            show_current_position=settings.show_current_position,
            show_sample_trajectories=settings.show_sample_trajectories,
            connect_sample_trajectories=settings.connect_sample_trajectories,
            show_median_forecast=settings.show_median_forecast,
            show_mean_forecast=settings.show_mean_forecast,
            show_prediction_region_50=settings.show_prediction_region_50,
            show_prediction_region_90=settings.show_prediction_region_90,
            prediction_region_style=settings.prediction_region_style,
            follow_ship_view_span_m=settings.follow_ship_view_span_m,
            on_state_change=on_state_change,
            on_metrics_change=on_metrics_change,
            on_medians_change=on_medians_change,
            on_radar_association_change=on_radar_association_change,
            settings_visible=settings_visible,
            on_settings_visibility_change=on_settings_visibility_change,
            on_playback_change=on_playback_change,
        )
        self._focus_connection = self.canvas.mpl_connect(
            "button_press_event", lambda _: self.canvas.get_tk_widget().focus_set()
        )
        self.navigator.set_language(widget_language(self))
        self.canvas.callbacks.exception_handler = self._handle_plot_error
        for widget in (
            self.navigator.playback_button,
            self.navigator.slider,
            self.navigator.group_selector,
            self.navigator.follow_checkbox,
            self.navigator.settings_checkbox,
        ):
            if widget is not None:
                widget._observers.exception_handler = self._handle_plot_error
        self.canvas.draw_idle()

    def _handle_plot_error(self, error):
        if not self._destroyed:
            self.winfo_toplevel().report_callback_exception(
                type(error), error, error.__traceback__
            )

    def set_coordinate_display_mode(self, coordinate_display_mode):
        """Redraw the current dashboard in another unit without replacing analysis."""
        self.navigator.set_coordinate_display_mode(coordinate_display_mode)

    def set_language(self, language):
        """Translate the current artists without requesting another model update."""
        self.navigator.set_language(language)

    def set_display_options(self, display_options):
        """Redraw presentation choices without replacing the current analysis."""
        self.navigator.set_display_options(**display_options)

    def set_settings_visible(self, visible):
        """Switch between compact and six-panel posterior presentation."""
        self.navigator.set_settings_visible(visible)

    def disable_navigation(self):
        """Keep the last valid plot visible after an inference error."""
        self.navigator.pause_playback()
        self.navigator.disconnect()
        widgets = [
            self.navigator.playback_button,
            self.navigator.slider,
            self.navigator.group_selector,
        ]
        if self.navigator.settings_checkbox is not None:
            widgets.append(self.navigator.settings_checkbox)
        for widget in widgets:
            # RadioButtons/CheckButtons override set_active to change selection.
            widget.active = False

    def destroy(self):
        """Release timers and pending Tk draws before destroying the canvas."""
        if self._destroyed:
            return
        self._destroyed = True
        self.navigator.disconnect()
        self.canvas.mpl_disconnect(self._focus_connection)
        # TkAgg can have an idle draw scheduled when a new analysis replaces it.
        for name in ("_idle_draw_id", "_event_loop_id"):
            callback = getattr(self.canvas, name, None)
            if callback is not None:
                try:
                    self.canvas.get_tk_widget().after_cancel(callback)
                except tk.TclError:
                    pass  # The enclosing Tk window may already have been destroyed.
                setattr(self.canvas, name, None)
                setattr(self.canvas, name, None)
        self.figure.clear()
        super().destroy()
