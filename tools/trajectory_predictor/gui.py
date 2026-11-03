"""Coordinate editable settings, background inference and the embedded plot."""

import logging
import sys
import tkinter as tk
from tkinter import messagebox

from . import view
from .controls import SettingsPanel
from .dialogs import PlotDisplayWindow, SettingsDialog
from .help import PosteriorHelpWindow
from .localization import localize_widgets, normalize_language, translate
from .plot_view import PosteriorPlotView
from .session import PosteriorAnalysisWorker
from .settings import (
    DISPLAY_OPTION_FIELDS,
    PLOT_DISPLAY_FIELDS,
    normalize_inference_method,
    parse_settings,
)

LOGGER = logging.getLogger(__name__)
WINDOW_RESIZE_DELAY_MS = 200
SIDEBAR_SCROLL_DELAY_MS = 150


class TrajectoryPredictor:
    """One Tk window, with all widget and Matplotlib access on its main thread."""

    def __init__(self, root, *, worker=None):
        self.root = root
        self.language = "en"
        self.language_var = tk.StringVar(root, value=self.language)
        root._trajectory_language = self.language
        self.plot_view = None
        self.settings_visible = True
        self._settings = None
        self._closing = False
        self._closed = False
        self._resizing = False
        self._scrolling = False
        self._scroll_after_id = None
        self._resize_after_id = None
        self._last_window_size = None
        self._handling_error = False
        root.report_callback_exception = self._report_callback_exception
        self._error = None
        self._computing = None
        self._analysis_active = False
        self._settings_dialog = None
        self._plot_display_window = None
        self._help_window = None
        self.settings_visible_var = tk.BooleanVar(root, value=True)
        view.configure_window(root)
        self.controls = SettingsPanel(root, self.start_analysis, self.cancel_analysis)
        self.controls.variables["data"]["coordinate_display_mode"].trace_add(
            "write", self._update_coordinate_display_mode
        )
        for key in PLOT_DISPLAY_FIELDS:
            self.controls.variables["data"][key].trace_add(
                "write", self._update_display_options
            )
        view.create_menu_bar(self)
        self.controls.grid(row=0, column=0, sticky="ns", padx=(10, 0), pady=10)
        self.plot_host = tk.Frame(root, bg=view.PLOT_BACKGROUND)
        self.plot_host.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.placeholder = tk.Label(
            self.plot_host,
            text="Trajectory and posterior evolution\n\n"
            "Select CSV, run, and method on the left.\n"
            "Then click “Start analysis”.",
            bg=view.PLOT_BACKGROUND,
            fg=view.TEXT_COLOR,
            font=("Arial", 12),
            anchor="center",
        )
        self.placeholder.pack(fill="both", expand=True)
        self.status = tk.StringVar(root, value="Ready. No analysis started yet.")
        root.bind("<space>", self.handle_space, add="+")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.worker = worker if worker is not None else PosteriorAnalysisWorker()
        self.controls.settings_form.on_scroll = self._on_sidebar_scroll
        root.bind("<Configure>", self._on_window_resize, add="+")
        self._update_menu_state()
        self._poll_id = root.after(75, self._poll)

    def _translate(self, text):
        return translate(text, getattr(self, "language", "en"))

    def _menus_locked(self):
        navigator = getattr(self.plot_view, "navigator", None)
        return bool(
            getattr(self, "_analysis_active", False)
            and (
                self.plot_view is None
                or getattr(self, "_resizing", False)
                or getattr(navigator, "is_playing", False)
                or getattr(navigator, "is_waiting", False)
            )
        )

    def _update_menu_state(self, _playing=None):
        """Synchronize menu access immediately when playback or requests change."""
        if self._closing:
            return
        state = (not getattr(self, "_analysis_active", False), self._menus_locked())
        if state == getattr(self, "_menu_state", None):
            return
        view.set_analysis_menu_enabled(self, state[0], lock_header=state[1])
        self._menu_state = state

    def _request_update(self, observation_count):
        self.worker.request(observation_count)
        self._update_menu_state()

    def start_analysis(self):
        """Apply validated settings and replace the analysis, never mixing caches."""
        if self._closing or self._analysis_active:
            return
        try:
            settings = parse_settings(self.controls.values())
        except (ValueError, TypeError) as error:
            messagebox.showerror(
                self._translate("Check settings"),
                self._translate(str(error)),
                parent=self.root,
            )
            return
        if self.plot_view is not None:
            self.plot_view.destroy()
            self.plot_view = None
        self.controls.show_reference_state(None)
        self.controls.show_analysis_metrics(None)
        self.controls.show_posterior_medians(None)
        self.controls.show_radar_association(None)
        self._settings = settings
        LOGGER.info("Starting analysis: %r", settings.analysis)
        self._error = None
        self._computing = None
        self._analysis_active = True
        self.controls.set_analysis_active(True)
        self._update_menu_state()
        self.placeholder.configure(text=self._translate("Preparing analysis …"))
        self.placeholder.pack(fill="both", expand=True)
        self.status.set(
            self._translate(
                "Loading data / initializing inference. Any previous fit is stopped first.",
            )
        )
        self.worker.start(settings.analysis)

    def cancel_analysis(self):
        """Discard the current result and unlock settings for the next run."""
        if self._closing or not self._analysis_active:
            return
        self.worker.request(0)
        self._analysis_active = False
        self.controls.set_analysis_active(False)
        self._update_menu_state()
        self._settings = None
        self._error = None
        self._computing = None
        self.controls.show_reference_state(None)
        self.controls.show_analysis_metrics(None)
        self.controls.show_posterior_medians(None)
        self.controls.show_radar_association(None)
        if self.plot_view is not None:
            self.plot_view.destroy()
            self.plot_view = None
        self.placeholder.configure(
            text=self._translate(
                "Trajectory and posterior evolution\n\n"
                "Select CSV, run, and method on the left.\n"
                "Then click “Start analysis”.",
            )
        )
        self.placeholder.pack(fill="both", expand=True)
        self.status.set(self._translate("Analysis cancelled. Settings unlocked."))

    def toggle_settings(self):
        """Give the trajectory and posterior panels more space without rebuilding."""
        self.set_settings_visible(not self.settings_visible)

    def set_language(self, language):
        """Update presentation without replacing the analysis, drafts or playback."""
        language = normalize_language(language)
        if self._closing:
            return
        if self._menus_locked():
            self.language_var.set(self.language)
            return
        self.language = language
        self.language_var.set(language)
        self.root._trajectory_language = language
        view.create_menu_bar(self)
        self._update_menu_state()
        localize_widgets(self.root, language)
        self.status.set(translate(self.status.get(), language))
        if self.plot_view is not None:
            self.plot_view.set_language(language)

    def set_settings_visible(self, visible):
        """Synchronize settings visibility across the sidebar, menu and dashboard."""
        self.settings_visible = bool(visible)
        if self.settings_visible:
            self.controls.grid()
        else:
            self.controls.grid_remove()
        self.settings_visible_var.set(self.settings_visible)
        if self.plot_view is not None:
            self.plot_view.set_settings_visible(self.settings_visible)

    def open_csv(self):
        """Expose the existing file chooser from the File menu."""
        if not self._closing and not self._analysis_active:
            if not self.settings_visible:
                self.toggle_settings()
            self.controls.choose_file()

    def reset_plot_view(self):
        """Reset only the plot view, without restarting inference or playback."""
        if self.plot_view is not None:
            self.plot_view.toolbar.home()

    def _update_coordinate_display_mode(self, *_):
        """Apply a display-only coordinate selection to the running dashboard."""
        if self._closing or self.plot_view is None:
            return
        coordinate_display_mode = self.controls.variables["data"][
            "coordinate_display_mode"
        ].get()
        try:
            self.plot_view.set_coordinate_display_mode(coordinate_display_mode)
        except ValueError as error:
            messagebox.showerror(
                self._translate("Coordinate display"),
                self._translate(str(error)),
                parent=self.root,
            )

    def set_coordinate_display_mode(self, coordinate_display_mode):
        """Select a presentation unit from the View menu."""
        self.controls.variables["data"]["coordinate_display_mode"].set(
            coordinate_display_mode
        )

    def _update_display_options(self, *_):
        """Apply plot-only checkboxes without starting another analysis."""
        if self._closing or self.plot_view is None:
            return
        display_options = {
            key: self.controls.variables["data"][key].get()
            for key in DISPLAY_OPTION_FIELDS
        }
        display_options["prediction_region_style"] = self.controls.variables["data"][
            "prediction_region_style"
        ].get()
        try:
            display_options["follow_ship_view_span_m"] = float(
                self.controls.variables["data"]["follow_ship_view_span_m"].get()
            )
        except ValueError:
            return
        try:
            self.plot_view.set_display_options(display_options)
        except ValueError as error:
            messagebox.showerror(
                self._translate("Plot display"),
                self._translate(str(error)),
                parent=self.root,
            )

    def show_inference_settings(self):
        """Edit options for the method currently selected in the sidebar."""
        self.show_settings_dialog(
            normalize_inference_method(
                self.controls.variables["data"]["inference_method"].get()
            )
        )

    def show_settings_dialog(self, group):
        """Open one editor at a time, retaining edits until explicitly applied."""
        if self._closing or self._analysis_active:
            return
        if self._settings_dialog is not None and self._settings_dialog.winfo_exists():
            self._settings_dialog.lift()
            self._settings_dialog.focus_set()
            return
        self._settings_dialog = SettingsDialog(self.root, self.controls, group)

    def show_plot_display(self):
        """Open or raise the non-modal live plot-display panel."""
        if self._closing:
            return
        if (
            self._plot_display_window is not None
            and self._plot_display_window.winfo_exists()
        ):
            self._plot_display_window.lift()
            self._plot_display_window.focus_set()
            return
        self._plot_display_window = PlotDisplayWindow(
            self.root,
            self.controls,
            on_close=self._plot_display_closed,
        )

    def _plot_display_closed(self):
        self._plot_display_window = None

    def show_help(self):
        """Open the persistent non-modal help window."""
        if self._closing:
            return
        if self._help_window is not None and self._help_window.winfo_exists():
            self._help_window.lift()
            self._help_window.focus_set()
            return
        self._help_window = PosteriorHelpWindow(
            self.root,
            on_close=self._help_closed,
        )

    def _help_closed(self):
        self._help_window = None

    def handle_space(self, event):
        """Avoid double toggles on the canvas and preserve normal text input."""
        if self._closing or self._error or self.plot_view is None:
            return None
        if event.widget is self.plot_view.canvas.get_tk_widget():
            return None  # Matplotlib already handles space on its canvas.
        if event.widget.winfo_class() in {
            "Entry",
            "TEntry",
            "Text",
            "Spinbox",
            "TSpinbox",
            "TCombobox",
            "Button",
            "TButton",
            "Checkbutton",
            "TCheckbutton",
            "TRadiobutton",
        }:
            return None
        self.plot_view.navigator.toggle_playback(None)
        return "break"

    def _sync_interaction_suspension(self):
        suspended = self._resizing or getattr(self, "_scrolling", False)
        self.worker.set_suspended(suspended)
        if self.plot_view is not None and self._error is None:
            self.plot_view.navigator.set_suspended(suspended)

    def _on_sidebar_scroll(self):
        """Give sidebar input priority over the next expensive plot update."""
        if self._closing or not self._analysis_active or self.plot_view is None:
            return
        if not self._scrolling:
            self._scrolling = True
            self._sync_interaction_suspension()
        if self._scroll_after_id is not None:
            self.root.after_cancel(self._scroll_after_id)
        self._scroll_after_id = self.root.after(
            SIDEBAR_SCROLL_DELAY_MS, self._finish_sidebar_scroll
        )

    def _finish_sidebar_scroll(self):
        self._scroll_after_id = None
        if self._closing:
            return
        self._scrolling = False
        self._sync_interaction_suspension()
        self._update_menu_state()

    def _on_window_resize(self, event):
        """Pause playback until a manual restart when the window size changes."""
        if self._closing or event.widget is not self.root:
            return
        size = (event.width, event.height)
        if size == self._last_window_size:
            return
        self._last_window_size = size
        if not self._analysis_active:
            return
        if not self._resizing:
            self._resizing = True
            self._sync_interaction_suspension()
            if self.plot_view is not None:
                self.plot_view.navigator.pause_playback()
            else:
                self.worker.request(0)
            self._update_menu_state()
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(
            WINDOW_RESIZE_DELAY_MS, self._finish_window_resize
        )

    def _finish_window_resize(self):
        """Release the resize guard while retaining the user's paused playback."""
        self._resize_after_id = None
        if self._closing:
            return
        self._resizing = False
        self._sync_interaction_suspension()
        self._update_menu_state()

    def _poll(self):
        self._poll_id = None
        if self._closed:
            return
        try:
            if self._closing:
                if not self.worker.is_alive:
                    self._closed = True
                    self.root.destroy()
                    return
            elif not (
                getattr(self, "_resizing", False) or getattr(self, "_scrolling", False)
            ):
                for event in self.worker.drain():
                    self._handle_event(event)
                self._update_menu_state()
                if self.plot_view is not None and self._error is None:
                    navigator = self.plot_view.navigator
                    state = self._translate(
                        "Playing" if navigator.is_playing else "Paused"
                    )
                    progress = (
                        f" · {self._translate('Computing')} N={self._computing}"
                        if self._computing is not None
                        else ""
                    )
                    target = (
                        f" · {self._translate('Target')} N={navigator.requested_observation_count}"
                        if navigator.is_waiting
                        else ""
                    )
                    self.status.set(
                        f"{self._settings.analysis.experiment.inference_method.upper()} · "
                        f"{state} · {self._translate('Display')} N={navigator.observation_count}/"
                        f"{navigator.maximum_observation_count}{target}{progress}"
                    )
        except Exception:
            self._report_callback_exception(*sys.exc_info())
        finally:
            if not self._closed:
                try:
                    self._poll_id = self.root.after(75, self._poll)
                except tk.TclError:
                    self._closed = True

    def _handle_event(self, event):
        if self._settings is None or self._error is not None:
            return
        if event.kind == "ready":
            trajectory, maximum, minimum = event.payload
            self.placeholder.pack_forget()
            self.plot_view = PosteriorPlotView(
                self.plot_host,
                self._settings,
                trajectory,
                maximum,
                minimum,
                self._request_update,
                lambda state: self.controls.show_reference_state(
                    state,
                    turn_rate_is_derived=trajectory.reference_turn_rate_is_derived,
                    speed_is_derived=trajectory.reference_speed_is_derived,
                    heading_is_derived=trajectory.reference_heading_is_derived,
                ),
                self.controls.show_analysis_metrics,
                self.controls.show_posterior_medians,
                self.controls.show_radar_association,
                settings_visible=self.settings_visible,
                on_settings_visibility_change=self.set_settings_visible,
                on_playback_change=self._update_menu_state,
            )
            self.plot_view.pack(fill="both", expand=True)
        elif event.kind == "computing":
            self._computing = event.payload
        elif event.kind == "update" and self._error is None:
            self._computing = None
            self.plot_view.navigator.accept_update(event.payload)
        elif event.kind == "error":
            self._show_error(event.payload)

    def _show_error(self, detail):
        if (
            self._closing
            or getattr(self, "_handling_error", False)
            or self._error is not None
        ):
            return
        self._handling_error = True
        try:
            self._error = detail
            self._computing = None
            self._analysis_active = False
            LOGGER.error("Analysis stopped: %s settings=%r", detail, self._settings)
            self.worker.request(0)
            self.controls.set_analysis_active(False)
            self._update_menu_state()
            if self.plot_view is not None:
                self._cleanup(self.plot_view.disable_navigation, "disable navigation")
            else:
                self.placeholder.configure(
                    text=self._translate("Analysis failed. Check settings.")
                )
            self.status.set(
                self._translate(f"Error: {detail} — Start a new analysis to continue.")
            )
            messagebox.showerror(
                self._translate("Analysis failed"),
                self._translate(detail),
                parent=self.root,
            )
        finally:
            self._handling_error = False

    def _report_callback_exception(self, exception_type, error, traceback):
        navigator = getattr(self.plot_view, "navigator", None)
        LOGGER.error(
            "GUI callback failed: N=%s settings=%r",
            getattr(navigator, "observation_count", None),
            self._settings,
            exc_info=(exception_type, error, traceback),
        )
        try:
            self._show_error(f"{exception_type.__name__}: {error}")
        except Exception:
            LOGGER.exception("Error recovery failed")

    def _cleanup(self, action, description):
        try:
            action()
        except Exception:
            LOGGER.exception("Cleanup failed: %s", description)

    def close(self):
        """Keep Tk responsive while a current fit finishes, then close cleanly."""
        if self._closing:
            return
        self._closing = True
        self.worker.close()
        resize_after_id = getattr(self, "_resize_after_id", None)
        if resize_after_id is not None:
            self._cleanup(
                lambda: self.root.after_cancel(resize_after_id), "cancel resize timer"
            )
            self._resize_after_id = None
        scroll_after_id = getattr(self, "_scroll_after_id", None)
        if scroll_after_id is not None:
            self._cleanup(
                lambda: self.root.after_cancel(scroll_after_id), "cancel scroll timer"
            )
            self._scroll_after_id = None
        self._cleanup(
            lambda: self.controls.apply_button.configure(state="disabled"),
            "disable start",
        )
        if self._settings_dialog is not None:
            self._cleanup(
                lambda: (
                    self._settings_dialog.cancel()
                    if self._settings_dialog.winfo_exists()
                    else None
                ),
                "close settings",
            )
        plot_display_window = getattr(self, "_plot_display_window", None)
        if plot_display_window is not None:
            self._cleanup(
                lambda: (
                    plot_display_window.close()
                    if plot_display_window.winfo_exists()
                    else None
                ),
                "close plot settings",
            )
        help_window = getattr(self, "_help_window", None)
        if help_window is not None:
            self._cleanup(
                lambda: help_window.close() if help_window.winfo_exists() else None,
                "close help",
            )
        if self.plot_view is not None:
            plot_view = self.plot_view
            self.plot_view = None
            self._cleanup(plot_view.destroy, "destroy plot")
        self._cleanup(
            lambda: self.status.set(
                self._translate("Closing … a running calculation is finishing.")
            ),
            "update status",
        )
