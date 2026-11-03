"""Desktop presentation matching the ship simulator, without coupling the tools."""

import sys
import tkinter as tk
from tkinter import ttk

from .localization import translate

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
    settings = {
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
    settings["Predictor.TEntry"]["configure"].update(
        insertcolor=TEXT_COLOR,
        insertwidth=2,
    )
    settings["Predictor.TButton"] = {
        "configure": {
            "background": INPUT_BACKGROUND,
            "foreground": TEXT_COLOR,
            "padding": (4, 0),
        },
        "map": {
            "background": [
                ("disabled", DISABLED_INPUT_BACKGROUND),
                ("active", "#b8d8e8"),
            ],
            "foreground": [
                ("disabled", DISABLED_INPUT_TEXT_COLOR),
                ("!disabled", TEXT_COLOR),
            ],
        },
    }
    return settings


def configure_input_styles(root):
    """Keep themed inputs aligned with the application's light palette."""
    style = ttk.Style(root)
    if root.tk.call("tk", "windowingsystem") == "aqua":
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


def create_styled_button(parent, *, text, command, width=18, padx=10, pady=3):
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
        padx=padx,
        pady=pady,
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
    root.resizable(True, True)
    if sys.platform == "win32":

        def lock_resize_border(event):
            if event.widget is root:
                _remove_windows_resize_frame(root)

        root.bind("<Map>", lock_resize_border, add="+")
    root.configure(bg=APP_BACKGROUND)
    configure_input_styles(root)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)


def _remove_windows_resize_frame(root):
    """Remove the Windows sizing border while retaining titlebar button styles."""
    import ctypes
    from ctypes import wintypes

    native = ctypes.WinDLL("user32", use_last_error=True)
    native.GetParent.argtypes = [wintypes.HWND]
    native.GetParent.restype = wintypes.HWND
    suffix = "LongPtrW" if ctypes.sizeof(ctypes.c_void_p) == 8 else "LongW"
    get_style = getattr(native, f"GetWindow{suffix}")
    set_style = getattr(native, f"SetWindow{suffix}")
    get_style.argtypes = [wintypes.HWND, ctypes.c_int]
    get_style.restype = ctypes.c_ssize_t
    set_style.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    set_style.restype = ctypes.c_ssize_t
    native.SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    native.SetWindowPos.restype = wintypes.BOOL

    # winfo_id identifies Tk's client window; its parent owns the native frame.
    handle = native.GetParent(root.winfo_id())
    if not handle:
        raise OSError("Could not find the native Tk window frame.")
    ctypes.set_last_error(0)
    style = get_style(handle, -16)  # GWL_STYLE
    if not style and (error := ctypes.get_last_error()):
        raise ctypes.WinError(error)
    resize_frame = 0x00040000  # WS_THICKFRAME
    if not style & resize_frame:
        return
    if not set_style(handle, -16, style & ~resize_frame):
        raise ctypes.WinError(ctypes.get_last_error())
    # Refresh the frame without changing size, position, focus, or stacking.
    frame_flags = 0x0037  # FRAMECHANGED | NOSIZE | NOMOVE | NOZORDER | NOACTIVATE
    if not native.SetWindowPos(handle, None, 0, 0, 0, 0, frame_flags):
        raise ctypes.WinError(ctypes.get_last_error())


def create_menu_bar(gui):
    """Expose application actions and a separate presentation-language menu."""
    language = getattr(gui, "language", "en")
    if getattr(gui, "_menu_bar", None) is not None:
        # A menu command may still be executing: retain the native menu tree.
        for menu, index, source in gui._menu_label_entries:
            menu.entryconfigure(index, label=translate(source, language))
        return

    label_entries = []

    def add(parent_menu, kind, source, **options):
        getattr(parent_menu, f"add_{kind}")(
            label=translate(source, language), **options
        )
        label_entries.append((parent_menu, parent_menu.index("end"), source))

    menu_bar = tk.Menu(gui.root)
    file_menu = tk.Menu(menu_bar, tearoff=0)
    add(file_menu, "command", "Open CSV…", command=gui.open_csv)
    open_csv_menu_index = file_menu.index("end")
    file_menu.add_separator()
    add(file_menu, "command", "Exit", command=gui.close)
    add(menu_bar, "cascade", "File", menu=file_menu)

    view_menu = tk.Menu(menu_bar, tearoff=0)
    add(
        view_menu,
        "checkbutton",
        "Show settings",
        variable=gui.settings_visible_var,
        command=gui.toggle_settings,
    )
    coordinate_menu = tk.Menu(view_menu, tearoff=0)
    for label, value in (("Local [m]", "m"), ("Local [km]", "km")):
        add(
            coordinate_menu,
            "radiobutton",
            label,
            variable=gui.controls.variables["data"]["coordinate_display_mode"],
            value=value,
            command=lambda mode=value: gui.set_coordinate_display_mode(mode),
        )
    add(view_menu, "cascade", "Coordinate display", menu=coordinate_menu)
    add(view_menu, "command", "Plot display…", command=gui.show_plot_display)
    add(menu_bar, "cascade", "View", menu=view_menu)

    settings_menu = tk.Menu(menu_bar, tearoff=0)
    add(
        settings_menu,
        "command",
        "Priors…",
        command=lambda: gui.show_settings_dialog("priors"),
    )
    add(
        settings_menu,
        "command",
        "Inference parameters…",
        command=gui.show_inference_settings,
    )
    add(
        settings_menu,
        "command",
        "Analysis setup…",
        command=lambda: gui.show_settings_dialog("data"),
    )
    add(menu_bar, "cascade", "Settings", menu=settings_menu)
    settings_menu_index = menu_bar.index("end")

    language_menu = tk.Menu(menu_bar, tearoff=0)
    for label, value in (("English", "en"), ("Deutsch", "de")):
        add(
            language_menu,
            "radiobutton",
            label,
            variable=getattr(gui, "language_var", None),
            value=value,
            command=lambda selected=value: gui.set_language(selected),
        )
    add(menu_bar, "cascade", "Language", menu=language_menu)

    help_menu = tk.Menu(menu_bar, tearoff=0)
    add(help_menu, "command", "Show Help", command=gui.show_help)
    add(menu_bar, "cascade", "Help", menu=help_menu)
    gui._menu_bar = menu_bar
    gui._file_menu = file_menu
    gui._open_csv_menu_index = open_csv_menu_index
    gui._settings_menu_index = settings_menu_index
    gui._settings_menu = settings_menu
    gui._menu_label_entries = tuple(label_entries)
    gui.root.configure(menu=menu_bar)


def set_analysis_menu_enabled(gui, enabled, *, lock_header=False):
    """Lock the header during playback and protect an active run's configuration."""
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
    for index in range(menu_bar.index("end") + 1):
        disabled = lock_header or (index == settings_menu_index and not enabled)
        menu_bar.entryconfigure(index, state="disabled" if disabled else "normal")
