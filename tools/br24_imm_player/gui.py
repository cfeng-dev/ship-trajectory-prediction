"""Tk application orchestration for the local BR24 IMM player."""

from __future__ import annotations

import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox

from . import view
from .controls import IMMControls
from .dialogs import PlaybackSettingsDialog, show_help
from .player import PLOT_DISPLAY_OPTIONS, IMMPlaybackStore, IMMPlayerConfig
from .plot_view import IMMPlotView

INITIAL_PLACEHOLDER_TEXT = (
    "Trajectory and IMM evolution\n\n"
    "Select a BR24 CSV and configure the player on the left.\n"
    "Then click 'Start analysis'."
)


class BR24IMMPlayerApp:
    """Keep all Tk and embedded Matplotlib access on the main thread."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.config = IMMPlayerConfig()
        self.store: IMMPlaybackStore | None = None
        self.plot_view: IMMPlotView | None = None
        self.settings_visible = True
        self.settings_visible_var = tk.BooleanVar(root, value=True)
        self.coordinate_display_mode_var = tk.StringVar(root, value="m")
        self._settings_dialog: PlaybackSettingsDialog | None = None
        self._plot_settings_dialog: PlaybackSettingsDialog | None = None
        self._closing = False
        self._analysis_active = False

        view.configure_window(root)
        self.controls = IMMControls(
            root,
            on_open_file=self.open_csv,
            on_cancel=self.reload_session,
            on_start=self.start_analysis,
        )
        self.controls.grid(row=0, column=0, sticky="ns", padx=(10, 0), pady=10)
        self.plot_host = tk.Frame(root, bg=view.PLOT_BACKGROUND)
        self.plot_host.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.placeholder = tk.Label(
            self.plot_host,
            text=INITIAL_PLACEHOLDER_TEXT,
            bg=view.PLOT_BACKGROUND,
            fg=view.TEXT_COLOR,
            font=("Arial", 12),
            justify="center",
        )
        self.placeholder.pack(fill="both", expand=True)
        view.create_menu_bar(self)
        root.bind("<space>", self.handle_key, add="+")
        root.bind("<Left>", self.handle_key, add="+")
        root.bind("<Right>", self.handle_key, add="+")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._prepare_analysis(self.config)

    def _prepare_analysis(self, config: IMMPlayerConfig) -> None:
        """Return to the unstarted state without initializing the tracker."""
        if self.plot_view is not None:
            self.plot_view.destroy()
            self.plot_view = None
        self.store = None
        self._analysis_active = False
        view.set_analysis_menu_enabled(self, True)
        self.config = config
        self.controls.set_data_file(config.data_file)
        self.controls.clear_display_state()
        self.controls.set_analysis_active(False)
        self.placeholder.configure(text=INITIAL_PLACEHOLDER_TEXT)
        self.placeholder.pack(fill="both", expand=True)

    def _load_session(
        self, config: IMMPlayerConfig, *, show_error: bool = True
    ) -> bool:
        try:
            store = IMMPlaybackStore(config)
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            if show_error:
                messagebox.showerror("BR24 IMM player", detail, parent=self.root)
            return False

        previous_view = self.plot_view
        if previous_view is not None:
            previous_view.destroy()
        self.store = store
        self.config = config
        self.controls.set_data_file(config.data_file)
        self.controls.clear_display_state()
        self.controls.set_analysis_active(True)
        self._analysis_active = True
        view.set_analysis_menu_enabled(self, False)
        self.placeholder.pack_forget()
        self.plot_view = IMMPlotView(
            self.plot_host,
            store=store,
            initial_state=None,
            on_stage_changed=self._show_stage,
            on_settings_visibility_change=self.set_settings_visible,
            settings_visible=self.settings_visible,
        )
        self.plot_view.set_coordinate_display_mode(
            self.coordinate_display_mode_var.get()
        )
        self.plot_view.pack(fill="both", expand=True)
        return True

    def _show_stage(self, state) -> None:
        if state is None:
            self.controls.clear_display_state()
            return
        self.controls.show_display_state(
            state, coordinate_display_mode=self.coordinate_display_mode_var.get()
        )

    def start_analysis(self) -> None:
        """Show preparation feedback before loading the initial N = 0 view."""
        if not self._closing and not self._analysis_active:
            self._prepare_analysis(self.config)
            self._analysis_active = True
            view.set_analysis_menu_enabled(self, False)
            self.controls.set_analysis_active(True)
            self.placeholder.configure(text="Preparing analysis …")
            self.root.update_idletasks()
            if not self._load_session(self.config):
                self._prepare_analysis(self.config)

    def open_csv(self, data_file: Path | None = None) -> None:
        """Select a CSV and return to the unstarted analysis state."""
        if self._closing or self._analysis_active:
            return
        if data_file is None:
            self.controls.choose_file()
            return
        self._prepare_analysis(replace(self.config, data_file=data_file))

    def reload_session(self) -> None:
        """Discard any displayed session and return to the unstarted state."""
        if not self._closing:
            self._prepare_analysis(self.config)

    def toggle_settings(self) -> None:
        """Toggle the sidebar from the View menu."""
        self.set_settings_visible(not self.settings_visible)

    def set_settings_visible(self, visible: bool) -> None:
        """Synchronize sidebar visibility with menu and embedded controls."""
        self.settings_visible = bool(visible)
        self.settings_visible_var.set(self.settings_visible)
        if self.settings_visible:
            self.controls.grid()
        else:
            self.controls.grid_remove()
        if self.plot_view is not None:
            self.plot_view.set_settings_visible(self.settings_visible)

    def reset_plot_view(self) -> None:
        """Reset only the route viewport, without restarting the tracker."""
        if self.plot_view is not None:
            self.plot_view.reset_view()

    def set_coordinate_display_mode(self, mode: str) -> None:
        """Change plot units without restarting the IMM session."""
        if mode not in {"m", "km"}:
            raise ValueError("Coordinate display must be 'm' or 'km'.")
        self.coordinate_display_mode_var.set(mode)
        if self.plot_view is not None:
            self.plot_view.set_coordinate_display_mode(mode)
            if self.plot_view.current_state is not None:
                self._show_stage(self.plot_view.current_state)

    def handle_key(self, event):
        """Route Predictor-style playback shortcuts outside editable controls."""
        if self._closing or self.plot_view is None:
            return None
        if event.widget is self.plot_view.canvas.get_tk_widget():
            return None  # The Matplotlib canvas handles its own keyboard events.
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
        if self.plot_view.handle_key(event.keysym.lower()):
            return "break"
        return None

    def show_settings(self) -> None:
        """Open analysis setup only while the session is inactive."""
        if self._closing or self._analysis_active:
            return
        self._open_settings_dialog(plot_only=False)

    def show_plot_settings(self) -> None:
        """Keep visual options available without changing analysis settings."""
        if not self._closing:
            self._open_settings_dialog(plot_only=True)

    def _open_settings_dialog(self, *, plot_only: bool) -> None:
        attribute = "_plot_settings_dialog" if plot_only else "_settings_dialog"
        dialog = getattr(self, attribute)
        if dialog is not None and dialog.window.winfo_exists():
            dialog.window.lift()
            dialog.window.focus_set()
            return
        config = self.config
        if plot_only and self.plot_view is not None:
            config = replace(
                config, show_recorded_trajectory=self.plot_view.show_reference_var.get()
            )
        dialog = PlaybackSettingsDialog(
            self.root,
            config,
            self._apply_plot_settings if plot_only else self._apply_settings,
            plot_only=plot_only,
        )
        setattr(self, attribute, dialog)

    def _apply_plot_settings(self, config: IMMPlayerConfig) -> bool:
        """Commit visual fields only, even if the analysis changed since opening."""
        return self._apply_settings(
            replace(
                self.config,
                **{name: getattr(config, name) for name in PLOT_DISPLAY_OPTIONS},
                prediction_region_style=config.prediction_region_style,
                follow_ship_view_span_meters=config.follow_ship_view_span_meters,
            )
        )

    def _apply_settings(self, config: IMMPlayerConfig) -> bool:
        previous_session_config = replace(
            config,
            **{name: getattr(self.config, name) for name in PLOT_DISPLAY_OPTIONS},
            prediction_region_style=self.config.prediction_region_style,
            follow_ship_view_span_meters=self.config.follow_ship_view_span_meters,
        )
        if previous_session_config == self.config:
            span_changed = (
                config.follow_ship_view_span_meters
                != self.config.follow_ship_view_span_meters
            )
            self.config = config
            if self.store is not None:
                self.store.config = config
            if self.plot_view is not None:
                visible = (
                    config.show_recorded_trajectory and self.store.reference is not None
                )
                self.plot_view.show_reference_var.set(visible)
                self.plot_view.reference_checkbox.eventson = False
                self.plot_view.reference_checkbox.set_active(0, state=visible)
                self.plot_view.reference_checkbox.eventson = True
                if span_changed:
                    self.plot_view._follow_ship_view_needs_focus = True
                self.plot_view._draw()
        else:
            if self._analysis_active:
                return False
            self._prepare_analysis(config)
        return True

    def show_help(self) -> None:
        if not self._closing:
            show_help(self.root)

    def close(self) -> None:
        """Stop playback and close all local Tk resources."""
        if self._closing:
            return
        self._closing = True
        for dialog in (self._settings_dialog, self._plot_settings_dialog):
            if dialog is not None and dialog.window.winfo_exists():
                dialog.close()
        if self.plot_view is not None:
            self.plot_view.destroy()
            self.plot_view = None
        self.root.destroy()
