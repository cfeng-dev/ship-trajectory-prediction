"""Coordinate editable settings, background inference and the embedded plot."""

import tkinter as tk
from tkinter import messagebox, ttk

from bayestraj.validation.posterior_session import PosteriorAnalysisWorker

from .controls import SettingsPanel
from .plot_view import PosteriorPlotView
from .settings import parse_settings


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
        root.title("Bayesian CTRV — Posterior Explorer")
        root.geometry("1500x900")
        root.minsize(1100, 700)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)
        header = ttk.Frame(root, padding=8)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.toggle_button = ttk.Button(
            header, text="Einstellungen ausblenden", command=self.toggle_settings
        )
        self.toggle_button.pack(side="left")
        ttk.Label(
            header,
            text="  Leertaste: Start/Pause  ·  Pfeiltasten im Plot: Einzelschritt  ·  "
            "Werkzeugleiste: Zoom / Ansicht zurücksetzen",
        ).pack(side="left", padx=10)
        self.controls = SettingsPanel(root, self.start_analysis)
        self.controls.grid(row=1, column=0, sticky="ns")
        self.plot_host = ttk.Frame(root)
        self.plot_host.grid(row=1, column=1, sticky="nsew")
        self.placeholder = ttk.Label(
            self.plot_host,
            text="CSV, Run und Methode auswählen, dann „Neue Analyse“ starten.",
            anchor="center",
        )
        self.placeholder.pack(fill="both", expand=True)
        self.status = tk.StringVar(root, value="Bereit. Noch keine Analyse gestartet.")
        ttk.Label(root, textvariable=self.status, padding=8, wraplength=1400).grid(
            row=2, column=0, columnspan=2, sticky="ew"
        )
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

    def toggle_settings(self):
        """Give the trajectory and posterior panels more space without rebuilding."""
        self.settings_visible = not self.settings_visible
        if self.settings_visible:
            self.controls.grid()
        else:
            self.controls.grid_remove()
        self.toggle_button.configure(
            text="Einstellungen ausblenden"
            if self.settings_visible
            else "Einstellungen anzeigen"
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
        if self.plot_view is not None:
            self.plot_view.destroy()
            self.plot_view = None
        self.worker.close()
        self.status.set("Schließe … eine laufende Berechnung wird noch beendet.")
