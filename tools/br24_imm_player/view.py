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
MAIN_WINDOW_WIDTH = 1_400
MAIN_WINDOW_HEIGHT = 700
MAIN_WINDOW_VERTICAL_OFFSET = 40


def configure_window(root: tk.Tk) -> None:
    """Apply the shared Predictor-like desktop layout to ``root``."""
    root.title(WINDOW_TITLE)
    root.configure(bg=WINDOW_BACKGROUND)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    left = (screen_width - MAIN_WINDOW_WIDTH) // 2
    top = (screen_height - MAIN_WINDOW_HEIGHT) // 2 - MAIN_WINDOW_VERTICAL_OFFSET
    root.geometry(f"{MAIN_WINDOW_WIDTH}x{MAIN_WINDOW_HEIGHT}+{left}+{top}")
    root.minsize(1_000, 620)
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)


def create_menu_bar(app) -> None:
    """Create the File, View, Settings, and Help menus for the player."""
    menu_bar = tk.Menu(app.root)

    file_menu = tk.Menu(menu_bar, tearoff=False)
    file_menu.add_command(label="Open BR24 CSV...", command=app.open_csv)
    file_menu.add_command(label="Reset analysis", command=app.reload_session)
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


def create_button(
    parent,
    text: str,
    command,
    *,
    foreground: str | None = None,
    width: int = 18,
    padx: int = 10,
    pady: int = 3,
):
    """Match the Predictor's button sizing and blue hover feedback."""
    button = tk.Button(
        parent,
        text=text,
        command=command,
        width=width,
        bg="#ffffff",
        fg=foreground or TEXT_COLOR,
        activebackground="#b8d8e8",
        activeforeground=foreground or TEXT_COLOR,
        font=FONT,
        disabledforeground="#7a8790",
        relief="solid",
        borderwidth=1,
        highlightthickness=0,
        padx=padx,
        pady=pady,
    )
    button.bind(
        "<Enter>",
        lambda _: (
            button.configure(bg="#b8d8e8")
            if button.cget("state") != "disabled"
            else None
        ),
    )
    button.bind("<Leave>", lambda _: button.configure(bg="#ffffff"))
    return button
