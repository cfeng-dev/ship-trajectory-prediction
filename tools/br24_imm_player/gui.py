"""Tk application orchestration for the local BR24 IMM player."""

from __future__ import annotations

import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox

from . import view
from .controls import IMMControls
from .dialogs import PlaybackSettingsDialog, show_help
from .player import IMMPlaybackStore, IMMPlayerConfig
from .plot_view import IMMPlotView


class BR24IMMPlayerApp:
    """Keep all Tk and embedded Matplotlib access on the main thread."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.config = IMMPlayerConfig()
        self.store: IMMPlaybackStore | None = None
        self.plot_view: IMMPlotView | None = None
        self.settings_visible = True
        self.settings_visible_var = tk.BooleanVar(root, value=True)
        self._settings_dialog: PlaybackSettingsDialog | None = None
        self._closing = False

        view.configure_window(root)
        self.controls = IMMControls(
            root,
            on_open_file=self.open_csv,
            on_reload=self.reload_session,
        )
        self.controls.grid(row=0, column=0, sticky="ns", padx=(10, 0), pady=10)
        self.plot_host = tk.Frame(root, bg=view.PLOT_BACKGROUND)
        self.plot_host.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.placeholder = tk.Label(
            self.plot_host,
            text="BR24 IMM online playback\n\nLoading local radar input...",
            bg=view.PLOT_BACKGROUND,
            fg=view.TEXT_COLOR,
            font=("Arial", 12),
            justify="center",
        )
        self.placeholder.pack(fill="both", expand=True)
        self.status = tk.StringVar(root, value="Ready.")
        tk.Label(
            root,
            textvariable=self.status,
            bg="#E7EEF4",
            fg=view.TEXT_COLOR,
            anchor="w",
            font=("Arial", 9),
            padx=8,
            pady=3,
        ).grid(row=1, column=0, columnspan=2, sticky="ew")
        view.create_menu_bar(self)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._load_session(self.config)

    def _load_session(
        self, config: IMMPlayerConfig, *, show_error: bool = True
    ) -> bool:
        try:
            store = IMMPlaybackStore(config)
            initial_state = store.display_state_for(config.initial_observation_count)
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            self.status.set(f"Load failed: {detail}")
            if show_error:
                messagebox.showerror("BR24 IMM player", detail, parent=self.root)
            return False

        previous_view = self.plot_view
        if previous_view is not None:
            previous_view.destroy()
        self.store = store
        self.config = config
        self.controls.set_data_file(config.data_file)
        self.controls.show_display_state(initial_state)
        self.placeholder.pack_forget()
        self.plot_view = IMMPlotView(
            self.plot_host,
            store=store,
            initial_state=initial_state,
            on_stage_changed=self._show_stage,
            on_settings_visibility_change=self.set_settings_visible,
            settings_visible=self.settings_visible,
        )
        self.plot_view.pack(fill="both", expand=True)
        self.status.set(
            f"Loaded {config.data_file.name}. "
            f"Display N={initial_state.stage.observation_count}/"
            f"{store.maximum_observation_count}."
        )
        return True

    def _show_stage(self, state) -> None:
        self.controls.show_display_state(state)
        if self.store is not None:
            self.status.set(
                f"Online IMM update: N={state.stage.observation_count}/"
                f"{self.store.maximum_observation_count}, "
                f"cycle={state.stage.cycle_counter}."
            )

    def open_csv(self, data_file: Path | None = None) -> None:
        """Open a selected CSV, preserving the previous player if it is invalid."""
        if self._closing:
            return
        if data_file is None:
            self.controls.choose_file()
            return
        self._load_session(replace(self.config, data_file=data_file))

    def reload_session(self) -> None:
        """Discard cache and restart the current file with the current settings."""
        if not self._closing:
            self._load_session(self.config)

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

    def show_settings(self) -> None:
        """Open a single transactional playback settings dialog."""
        if (
            self._settings_dialog is not None
            and self._settings_dialog.window.winfo_exists()
        ):
            self._settings_dialog.window.lift()
            self._settings_dialog.window.focus_set()
            return
        self._settings_dialog = PlaybackSettingsDialog(
            self.root,
            self.config,
            self._apply_settings,
        )

    def _apply_settings(self, config: IMMPlayerConfig) -> bool:
        return self._load_session(config)

    def show_help(self) -> None:
        if not self._closing:
            show_help(self.root)

    def close(self) -> None:
        """Stop playback and close all local Tk resources."""
        if self._closing:
            return
        self._closing = True
        if (
            self._settings_dialog is not None
            and self._settings_dialog.window.winfo_exists()
        ):
            self._settings_dialog.close()
        if self.plot_view is not None:
            self.plot_view.destroy()
            self.plot_view = None
        self.root.destroy()
