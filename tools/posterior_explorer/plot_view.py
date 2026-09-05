"""Embed the existing Matplotlib dashboard in one disposable Tk view."""

import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from bayestraj.validation.bayesian_ctrv_posterior_dashboard import (
    create_sequential_posterior_dashboard_figure,
)

from .view import PLOT_BACKGROUND


class PosteriorPlotView(tk.Frame):
    """Own the embedded canvas, navigation toolbar and dashboard callbacks."""

    def __init__(self, parent, settings, trajectory, maximum, minimum, request_update):
        super().__init__(parent, bg=PLOT_BACKGROUND)
        self.figure = Figure(figsize=(11, 8), facecolor=PLOT_BACKGROUND)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
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
        )
        self._focus_connection = self.canvas.mpl_connect(
            "button_press_event", lambda _: self.canvas.get_tk_widget().focus_set()
        )
        self.canvas.draw_idle()

    def disable_navigation(self):
        """Keep the last valid plot visible after an inference error."""
        self.navigator.pause_playback()
        self.navigator.disconnect()
        for widget in (
            self.navigator.playback_button,
            self.navigator.slider,
            self.navigator.group_selector,
        ):
            widget.set_active(False)

    def destroy(self):
        """Release timers and pending Tk draws before destroying the canvas."""
        self.navigator.disconnect()
        self.canvas.mpl_disconnect(self._focus_connection)
        # TkAgg can have an idle draw scheduled when a new analysis replaces it.
        for name in ("_idle_draw_id", "_event_loop_id"):
            callback = getattr(self.canvas, name, None)
            if callback is not None:
                self.canvas.get_tk_widget().after_cancel(callback)
                setattr(self.canvas, name, None)
        self.figure.clear()
        super().destroy()
