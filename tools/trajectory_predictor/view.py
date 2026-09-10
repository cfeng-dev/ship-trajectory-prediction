"""Desktop presentation matching the ship simulator, without coupling the tools."""

import tkinter as tk
from tkinter import ttk

APP_BACKGROUND = "#eef7fb"
CONTROL_BACKGROUND = "#e3eef4"
PLOT_BACKGROUND = "#f4f9fc"
TEXT_COLOR = "#1f2933"
INPUT_BACKGROUND = "#ffffff"
DISABLED_INPUT_BACKGROUND = "#e7edf1"
DISABLED_INPUT_TEXT_COLOR = "#687783"
FONT = ("Arial", 10)

MAIN_WINDOW_WIDTH = 1400
MAIN_WINDOW_HEIGHT = 700
MAIN_WINDOW_VERTICAL_OFFSET = 40


def input_style_settings():
    """Return the light input palette, including macOS disabled states."""
    return {
        style_name: {
            "configure": {
                "fieldbackground": INPUT_BACKGROUND,
                "foreground": TEXT_COLOR,
            },
            "map": {
                "fieldbackground": [
                    ("disabled", DISABLED_INPUT_BACKGROUND),
                    ("readonly", INPUT_BACKGROUND),
                ],
                "foreground": [
                    ("disabled", DISABLED_INPUT_TEXT_COLOR),
                    ("!disabled", TEXT_COLOR),
                ],
            },
        }
        for style_name in ("Predictor.TEntry", "Predictor.TCombobox")
    }


def configure_input_styles(root):
    """Keep themed inputs aligned with the application's light palette."""
    style = ttk.Style(root)
    style.theme_use("clam")
    for style_name, settings in input_style_settings().items():
        style.configure(style_name, **settings["configure"])
        style.map(style_name, **settings["map"])


def configure_plot_toolbar(toolbar):
    """Apply the application's light palette to Matplotlib's Tk toolbar."""
    toolbar.configure(bg=PLOT_BACKGROUND)
    buttons = tuple(toolbar._buttons.values())
    for child in toolbar.winfo_children():
        if child in buttons:
            child.configure(
                bg=INPUT_BACKGROUND,
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


def create_styled_button(parent, *, text, command, width=18):
    """Use the simulator's white buttons with blue hover feedback."""
    button = tk.Button(
        parent,
        text=text,
        command=command,
        width=width,
        font=FONT,
        bg="#ffffff",
        fg=TEXT_COLOR,
        activebackground="#b8d8e8",
        activeforeground=TEXT_COLOR,
        disabledforeground="#7a8790",
        relief=tk.SOLID,
        borderwidth=1,
        highlightthickness=0,
        padx=10,
        pady=3,
    )
    button.bind("<Enter>", lambda _: button.configure(bg="#b8d8e8"))
    button.bind("<Leave>", lambda _: button.configure(bg="#ffffff"))
    return button


def set_styled_button_palette(button, *, background, hover_background):
    """Update a styled button's resting and hover colours together."""
    button.configure(bg=background, activebackground=hover_background)
    button.bind(
        "<Enter>",
        lambda _: (
            button.configure(bg=hover_background)
            if button.cget("state") != "disabled"
            else None
        ),
    )
    button.bind(
        "<Leave>",
        lambda _: button.configure(bg=background),
    )


def centered_window_position(
    screen_width, screen_height, window_width, window_height, *, vertical_offset=0
):
    """Return the direct screen-centered position used by the ship simulator."""
    return (
        (screen_width - window_width) // 2,
        (screen_height - window_height) // 2 - vertical_offset,
    )


def configure_window(root):
    """Keep the window inside the screen and apply the simulator's light palette."""
    root.title("Bayesian CTRV — Ship Trajectory Predictor")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    left, top = centered_window_position(
        screen_width,
        screen_height,
        MAIN_WINDOW_WIDTH,
        MAIN_WINDOW_HEIGHT,
        vertical_offset=MAIN_WINDOW_VERTICAL_OFFSET,
    )
    root.geometry(f"{MAIN_WINDOW_WIDTH}x{MAIN_WINDOW_HEIGHT}+{left}+{top}")
    root.minsize(1000, 620)
    root.configure(bg=APP_BACKGROUND)
    configure_input_styles(root)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)


def create_menu_bar(gui):
    """Expose application actions through the same four menus as the simulator."""
    menu_bar = tk.Menu(gui.root)
    file_menu = tk.Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="Open CSV…", command=gui.open_csv)
    open_csv_menu_index = file_menu.index("end")
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=gui.close)
    menu_bar.add_cascade(label="File", menu=file_menu)

    view_menu = tk.Menu(menu_bar, tearoff=0)
    view_menu.add_checkbutton(
        label="Show settings",
        variable=gui.settings_visible_var,
        command=gui.toggle_settings,
    )
    coordinate_menu = tk.Menu(view_menu, tearoff=0)
    for label, value in (("Local [m]", "m"), ("Local [km]", "km"), ("GPS [°]", "gps")):
        coordinate_menu.add_radiobutton(
            label=label,
            variable=gui.controls.variables["data"]["coordinate_display_mode"],
            value=value,
            command=lambda mode=value: gui.set_coordinate_display_mode(mode),
        )
    view_menu.add_cascade(label="Coordinate display", menu=coordinate_menu)
    view_menu.add_command(label="Plot display…", command=gui.show_plot_display)
    menu_bar.add_cascade(label="View", menu=view_menu)

    settings_menu = tk.Menu(menu_bar, tearoff=0)
    settings_menu.add_command(
        label="Priors…", command=lambda: gui.show_settings_dialog("priors")
    )
    settings_menu.add_command(
        label="Inference parameters…", command=gui.show_inference_settings
    )
    settings_menu.add_command(
        label="Analysis setup…", command=lambda: gui.show_settings_dialog("data")
    )
    menu_bar.add_cascade(label="Settings", menu=settings_menu)
    settings_menu_index = menu_bar.index("end")

    help_menu = tk.Menu(menu_bar, tearoff=0)
    help_menu.add_command(label="Show Help", command=gui.show_help)
    menu_bar.add_cascade(label="Help", menu=help_menu)
    gui._menu_bar = menu_bar
    gui._file_menu = file_menu
    gui._open_csv_menu_index = open_csv_menu_index
    gui._settings_menu_index = settings_menu_index
    gui.root.configure(menu=menu_bar)


def set_analysis_menu_enabled(gui, enabled):
    """Visibly lock menu actions that would change an active inference run."""
    menu_bar = getattr(gui, "_menu_bar", None)
    file_menu = getattr(gui, "_file_menu", None)
    open_csv_menu_index = getattr(gui, "_open_csv_menu_index", None)
    settings_menu_index = getattr(gui, "_settings_menu_index", None)
    if (
        menu_bar is None
        or file_menu is None
        or open_csv_menu_index is None
        or settings_menu_index is None
    ):
        return
    state = "normal" if enabled else "disabled"
    file_menu.entryconfigure(open_csv_menu_index, state=state)
    menu_bar.entryconfigure(settings_menu_index, state=state)
