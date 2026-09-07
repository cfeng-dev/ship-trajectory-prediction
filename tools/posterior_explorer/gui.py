"""Coordinate editable settings, background inference and the embedded plot."""

import tkinter as tk
from tkinter import messagebox

from bayestraj.validation.posterior_session import PosteriorAnalysisWorker

from . import view
from .controls import SettingsPanel
from .dialogs import SettingsDialog
from .plot_view import PosteriorPlotView
from .settings import (
    DISPLAY_OPTION_FIELDS,
    normalize_inference_method,
    parse_settings,
)


class PosteriorExplorer:
    """One Tk window, with all widget and Matplotlib access on its main thread."""

    def __init__(self, root, *, worker=None):
        self.root = root
        self.plot_view = None
        self.settings_visible = True
        self._settings = None
        self._closing = False
        self._error = None
        self._computing = None
        self._settings_dialog = None
        self.settings_visible_var = tk.BooleanVar(root, value=True)
        view.configure_window(root)
        self.controls = SettingsPanel(root, self.start_analysis, self.reset_analysis)
        self.controls.variables["data"]["coordinate_display_mode"].trace_add(
            "write", self._update_coordinate_display_mode
        )
        for key in DISPLAY_OPTION_FIELDS:
            self.controls.variables["data"][key].trace_add(
                "write", self._update_display_options
            )
        view.create_menu_bar(self)
        self.controls.grid(row=0, column=0, sticky="ns", padx=(10, 0), pady=10)
        self.plot_host = tk.Frame(root, bg=view.PLOT_BACKGROUND)
        self.plot_host.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.placeholder = tk.Label(
            self.plot_host,
            text="Trajektorie und Posterior-Entwicklung\n\n"
            "Links CSV, Run und Methode auswählen.\n"
            "Mit „Analyse starten“ beginnen.",
            bg=view.PLOT_BACKGROUND,
            fg=view.TEXT_COLOR,
            font=("Arial", 12),
            anchor="center",
        )
        self.placeholder.pack(fill="both", expand=True)
        self.status = tk.StringVar(root, value="Bereit. Noch keine Analyse gestartet.")
        root.bind("<space>", self.handle_space, add="+")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.worker = worker if worker is not None else PosteriorAnalysisWorker()
        self._poll_id = root.after(75, self._poll)

    def start_analysis(self):
        """Apply validated settings and replace the analysis, never mixing caches."""
        if self._closing:
            return
        try:
            settings = parse_settings(self.controls.values())
        except (ValueError, TypeError) as error:
            messagebox.showerror("Einstellungen prüfen", str(error), parent=self.root)
            return
        if self.plot_view is not None:
            self.plot_view.destroy()
            self.plot_view = None
        self._settings = settings
        self._error = None
        self._computing = None
        self.placeholder.configure(text="Analyse wird vorbereitet …")
        self.placeholder.pack(fill="both", expand=True)
        self.status.set(
            "Lade Daten / initialisiere Inferenz. "
            "Ein eventuell laufender alter Fit wird zuerst beendet."
        )
        self.worker.start(settings.analysis)

    def reset_analysis(self):
        """Clear the current result and leave the selected settings untouched."""
        if self._closing:
            return
        self.worker.request(0)
        self._settings = None
        self._error = None
        self._computing = None
        if self.plot_view is not None:
            self.plot_view.destroy()
            self.plot_view = None
        self.placeholder.configure(
            text="Trajektorie und Posterior-Entwicklung\n\n"
            "Links CSV, Run und Methode auswählen.\n"
            "Mit „Analyse starten“ beginnen."
        )
        self.placeholder.pack(fill="both", expand=True)
        self.status.set("Bereit. Analyse zurückgesetzt.")

    def toggle_settings(self):
        """Give the trajectory and posterior panels more space without rebuilding."""
        self.settings_visible = not self.settings_visible
        if self.settings_visible:
            self.controls.grid()
        else:
            self.controls.grid_remove()
        self.settings_visible_var.set(self.settings_visible)

    def open_csv(self):
        """Expose the existing file chooser from the File menu."""
        if not self._closing:
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
            messagebox.showerror("Koordinatenanzeige", str(error), parent=self.root)

    def _update_display_options(self, *_):
        """Apply plot-only checkboxes without starting another analysis."""
        if self._closing or self.plot_view is None:
            return
        display_options = {
            key: self.controls.variables["data"][key].get()
            for key in DISPLAY_OPTION_FIELDS
        }
        self.plot_view.set_display_options(display_options)

    def show_inference_settings(self):
        """Edit options for the method currently selected in the sidebar."""
        self.show_settings_dialog(
            normalize_inference_method(
                self.controls.variables["data"]["inference_method"].get()
            )
        )

    def show_settings_dialog(self, group):
        """Open one editor at a time, retaining edits until explicitly applied."""
        if self._closing:
            return
        if self._settings_dialog is not None and self._settings_dialog.winfo_exists():
            self._settings_dialog.lift()
            self._settings_dialog.focus_set()
            return
        self._settings_dialog = SettingsDialog(self.root, self.controls, group)

    def show_help(self):
        """Keep keyboard instructions available without occupying the plot header."""
        messagebox.showinfo(
            "Posterior Explorer — Hilfe",
            "1. CSV, Run und Methode auswählen; „Analyse starten“ klicken.\n"
            "2. Im Plot Start/Pause oder die Leertaste verwenden.\n"
            "3. Mit dem Slider N auswählen; Pfeiltasten: Einzelschritt.\n\n"
            "N = 0 zeigt den Prior. Die Posterior-Gruppe wechselt zwischen\n"
            "Bewegungs- und Rauschparametern. Zoom bleibt bei Updates erhalten.\n\n"
            "Settings enthält Priors, Inferenzparameter sowie Daten- und\n"
            "Wiedergabeoptionen. Übernommene Werte gelten erst bei „Analyse starten“.\n\n"
            "RBPF/SMC: Online-Updates. VI/MCMC: neuer Batch-Fit je N ab N = 3.\n"
            "Pause hält den angezeigten Stand fest; laufende Fits dürfen fertig werden.",
            parent=self.root,
        )

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

    def _poll(self):
        if self._closing:
            if not self.worker.is_alive:
                self.root.destroy()
                return
        else:
            for event in self.worker.drain():
                try:
                    self._handle_event(event)
                except Exception as error:
                    self._show_error(f"{type(error).__name__}: {error}")
            if self.plot_view is not None and self._error is None:
                navigator = self.plot_view.navigator
                state = "Wiedergabe" if navigator.is_playing else "Pausiert"
                progress = (
                    f" · Berechne N={self._computing}"
                    if self._computing is not None
                    else ""
                )
                target = (
                    f" · Ziel N={navigator.requested_observation_count}"
                    if navigator.is_waiting
                    else ""
                )
                self.status.set(
                    f"{self._settings.analysis.experiment.inference_method.upper()} · "
                    f"{state} · Anzeige N={navigator.observation_count}/"
                    f"{navigator.maximum_observation_count}{target}{progress}"
                )
        self._poll_id = self.root.after(75, self._poll)

    def _handle_event(self, event):
        if self._settings is None:
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
                self.worker.request,
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
        self._error = detail
        self._computing = None
        self.worker.request(0)
        if self.plot_view is not None:
            self.plot_view.disable_navigation()
        else:
            self.placeholder.configure(
                text="Analyse fehlgeschlagen. Einstellungen prüfen."
            )
        self.status.set(f"Fehler: {detail} — Zum Fortsetzen eine neue Analyse starten.")

    def close(self):
        """Keep Tk responsive while a current fit finishes, then close cleanly."""
        if self._closing:
            return
        self._closing = True
        self.controls.apply_button.configure(state="disabled")
        self.controls.reset_button.configure(state="disabled")
        if self._settings_dialog is not None and self._settings_dialog.winfo_exists():
            self._settings_dialog.cancel()
        if self.plot_view is not None:
            self.plot_view.destroy()
            self.plot_view = None
        self.worker.close()
        self.status.set("Schließe … eine laufende Berechnung wird noch beendet.")
