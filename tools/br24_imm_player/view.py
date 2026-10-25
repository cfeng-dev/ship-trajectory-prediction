"""Shared Tk styling and menus for the local BR24 IMM desktop player."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

WINDOW_TITLE = "BR24 IMM - Ship Trajectory Player"
WINDOW_BACKGROUND = "#F3F7FA"
CONTROL_BACKGROUND = "#E3EFF6"
PLOT_BACKGROUND = "#f4f9fc"
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
    configure_input_styles(root)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    left = (screen_width - MAIN_WINDOW_WIDTH) // 2
    top = (screen_height - MAIN_WINDOW_HEIGHT) // 2 - MAIN_WINDOW_VERTICAL_OFFSET
    root.geometry(f"{MAIN_WINDOW_WIDTH}x{MAIN_WINDOW_HEIGHT}+{left}+{top}")
    root.minsize(1_000, 620)
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)


def configure_input_styles(root) -> None:
    """Match the Predictor's themed CSV field and compact file button."""
    style = ttk.Style(root)
    if root.tk.call("tk", "windowingsystem") == "aqua":
        style.theme_use("clam")
    style.configure(
        "IMMPlayer.TEntry",
        fieldbackground="#ffffff",
        foreground="#1f2933",
        insertcolor="#1f2933",
        insertwidth=2,
    )
    style.map(
        "IMMPlayer.TEntry",
        fieldbackground=[("disabled", "#e7edf1"), ("readonly", "#ffffff")],
        foreground=[("disabled", "#687783"), ("!disabled", "#1f2933")],
    )
    style.configure(
        "IMMPlayer.TButton",
        background="#ffffff",
        foreground="#1f2933",
        padding=(4, 0),
    )
    style.map(
        "IMMPlayer.TButton",
        background=[("disabled", "#e7edf1"), ("active", "#b8d8e8")],
        foreground=[("disabled", "#687783"), ("!disabled", "#1f2933")],
    )


def create_menu_bar(app) -> None:
    """Create the File, View, Settings, and Help menus for the player."""
    menu_bar = tk.Menu(app.root)

    file_menu = tk.Menu(menu_bar, tearoff=False)
    file_menu.add_command(label="Open CSV…", command=app.open_csv)
    app._open_csv_menu_index = file_menu.index("end")
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=app.close)
    menu_bar.add_cascade(label="File", menu=file_menu)

    view_menu = tk.Menu(menu_bar, tearoff=False)
    view_menu.add_checkbutton(
        label="Show settings",
        variable=app.settings_visible_var,
        command=app.toggle_settings,
    )
    coordinate_menu = tk.Menu(view_menu, tearoff=False)
    for label, mode in (("Local [m]", "m"), ("Local [km]", "km")):
        coordinate_menu.add_radiobutton(
            label=label,
            value=mode,
            variable=app.coordinate_display_mode_var,
            command=lambda mode=mode: app.set_coordinate_display_mode(mode),
        )
    view_menu.add_cascade(label="Coordinate display", menu=coordinate_menu)
    view_menu.add_command(label="Plot display…", command=app.show_plot_settings)
    menu_bar.add_cascade(label="View", menu=view_menu)

    settings_menu = tk.Menu(menu_bar, tearoff=False)
    settings_menu.add_command(label="Analysis setup...", command=app.show_settings)
    menu_bar.add_cascade(label="Settings", menu=settings_menu)
    app._settings_menu_index = menu_bar.index("end")

    help_menu = tk.Menu(menu_bar, tearoff=False)
    help_menu.add_command(label="Show Help", command=app.show_help)
    menu_bar.add_cascade(label="Help", menu=help_menu)

    app.root.configure(menu=menu_bar)
    app._menu_bar = menu_bar
    app._file_menu = file_menu


def set_analysis_menu_enabled(app, enabled: bool) -> None:
    """Lock session settings and file selection while analysis is active."""
    state = "normal" if enabled else "disabled"
    app._menu_bar.entryconfigure(app._settings_menu_index, state=state)
    app._file_menu.entryconfigure(app._open_csv_menu_index, state=state)


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
    set_button_palette(button, background="#ffffff", hover_background="#b8d8e8")
    return button


def set_button_palette(button, *, background: str, hover_background: str) -> None:
    """Keep a button's resting, pressed, and hover colors consistent."""
    button.configure(bg=background, activebackground=hover_background)
    button.bind(
        "<Enter>",
        lambda _: (
            button.configure(bg=hover_background)
            if button.cget("state") != "disabled"
            else None
        ),
    )
    button.bind("<Leave>", lambda _: button.configure(bg=background))


def configure_plot_toolbar(toolbar) -> None:
    """Match the Predictor's light Matplotlib toolbar and blue active buttons."""
    toolbar.configure(bg=PLOT_BACKGROUND)
    buttons = tuple(toolbar._buttons.values())
    for child in toolbar.winfo_children():
        if child in buttons:
            child.configure(
                bg="#ffffff",
                fg=TEXT_COLOR,
                activebackground="#b8d8e8",
                activeforeground=TEXT_COLOR,
                highlightthickness=0,
            )
            if child.winfo_class() == "Checkbutton":
                child.configure(selectcolor="#b8d8e8")
            toolbar._set_image_for_button(child)
        else:
            child.configure(bg=PLOT_BACKGROUND)
