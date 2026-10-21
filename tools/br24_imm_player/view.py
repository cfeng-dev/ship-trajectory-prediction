"""Shared Tk styling and menus for the local BR24 IMM desktop player."""

from __future__ import annotations

import tkinter as tk

WINDOW_TITLE = "BR24 IMM - Ship Trajectory Player"
WINDOW_BACKGROUND = "#F3F7FA"
CONTROL_BACKGROUND = "#E3EFF6"
PLOT_BACKGROUND = "#F9FBFD"
TEXT_COLOR = "#172B3A"
ACCENT_COLOR = "#2878B5"
BUTTON_BACKGROUND = "#EAF1F6"
FONT = ("Arial", 10)
HEADING_FONT = ("Arial", 12, "bold")


def configure_window(root: tk.Tk) -> None:
    """Apply the shared Predictor-like desktop layout to ``root``."""
    root.title(WINDOW_TITLE)
    root.configure(bg=WINDOW_BACKGROUND)
    root.minsize(1_180, 720)
    root.geometry("1600x920")
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)


def create_menu_bar(app) -> None:
    """Create the File, View, Settings, and Help menus for the player."""
    menu_bar = tk.Menu(app.root)

    file_menu = tk.Menu(menu_bar, tearoff=False)
    file_menu.add_command(label="Open BR24 CSV...", command=app.open_csv)
    file_menu.add_command(label="Reload", command=app.reload_session)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=app.close)
    menu_bar.add_cascade(label="File", menu=file_menu)

    view_menu = tk.Menu(menu_bar, tearoff=False)
    view_menu.add_checkbutton(
        label="Show settings",
        variable=app.settings_visible_var,
        command=app.toggle_settings,
    )
    view_menu.add_command(label="Reset plot view", command=app.reset_plot_view)
    menu_bar.add_cascade(label="View", menu=view_menu)

    settings_menu = tk.Menu(menu_bar, tearoff=False)
    settings_menu.add_command(
        label="Playback and forecast...", command=app.show_settings
    )
    menu_bar.add_cascade(label="Settings", menu=settings_menu)

    help_menu = tk.Menu(menu_bar, tearoff=False)
    help_menu.add_command(label="BR24 IMM Player", command=app.show_help)
    menu_bar.add_cascade(label="Help", menu=help_menu)

    app.root.configure(menu=menu_bar)


def create_button(parent, text: str, command, *, foreground: str | None = None):
    """Return a consistently styled Tk button."""
    return tk.Button(
        parent,
        text=text,
        command=command,
        bg=BUTTON_BACKGROUND,
        fg=foreground or TEXT_COLOR,
        activebackground="#D8E8F2",
        activeforeground=foreground or TEXT_COLOR,
        font=FONT,
        relief="solid",
        borderwidth=1,
        padx=6,
        pady=3,
    )
