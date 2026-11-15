"""Help window for the standalone ship trajectory GUI."""

import tkinter as tk
from collections.abc import Callable

from . import controls

HELP_CONTENT_MAX_SCREEN_RATIO = 0.65
HELP_WINDOW_WIDTH = 660
HELP_DESCRIPTION_WRAP_LENGTH = 300


def _add_description_rows(parent, rows, left_column_width, background_color):
    """
    Add two-column description rows to a help section.

    Parameters
    ----------
    parent : tk.Widget
        Parent frame where rows are inserted.
    rows : list[tuple[str, str]]
        Left and right text of each row.
    left_column_width : int
        Width of the left label column.
    background_color : str
        Background color of the help window.
    """
    for row, (left_text, description) in enumerate(rows):
        tk.Label(
            parent,
            text=left_text,
            width=left_column_width,
            anchor="nw",
            font=("Arial", 10, "bold"),
            bg=background_color,
            fg="black",
        ).grid(row=row, column=0, sticky="nw", pady=2)

        tk.Label(
            parent,
            text=description,
            anchor="nw",
            justify="left",
            wraplength=HELP_DESCRIPTION_WRAP_LENGTH,
            bg=background_color,
            fg="black",
        ).grid(row=row, column=1, sticky="nw", pady=2)


def _create_scrollable_content(parent, background_color):
    """Create the scrollable content area inside the help window."""
    scroll_container = tk.Frame(parent, bg=background_color)
    scroll_container.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(
        scroll_container,
        bg=background_color,
        borderwidth=0,
        highlightthickness=0,
    )
    scrollbar = tk.Scrollbar(
        scroll_container,
        orient=tk.VERTICAL,
        command=canvas.yview,
    )
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    content_frame = tk.Frame(
        canvas,
        bg=background_color,
        padx=24,
    )
    content_window = canvas.create_window(
        (0, 0),
        window=content_frame,
        anchor="nw",
    )

    content_frame.bind(
        "<Configure>",
        lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.bind(
        "<Configure>",
        lambda event: canvas.itemconfigure(content_window, width=event.width),
    )

    return canvas, content_frame, scrollbar


def _fit_scrollable_content_to_screen(help_window, canvas, content, scrollbar):
    """Limit help content height and hide the scrollbar when it is unnecessary."""
    help_window.update_idletasks()

    content_height = content.winfo_reqheight()
    maximum_height = max(
        240,
        int(help_window.winfo_screenheight() * HELP_CONTENT_MAX_SCREEN_RATIO),
    )

    canvas.configure(
        width=HELP_WINDOW_WIDTH - scrollbar.winfo_reqwidth(),
        height=min(content_height, maximum_height),
    )

    if content_height <= maximum_height:
        scrollbar.pack_forget()


def show_help_window(
    root,
    app_background_color,
    pause_callback: Callable[[], None],
    focus_callback: Callable[[], None],
    keyboard_left_column_width,
    menu_left_column_width,
    button_left_column_width,
    simulation_time_left_column_width,
):
    """
    Show a help window with keyboard shortcuts and basic usage.

    The simulation is paused before opening the help window so the trajectory
    does not keep changing while the user is reading the instructions.
    """
    pause_callback()

    help_window = tk.Toplevel(root)
    help_window.withdraw()
    help_window.title("Help")
    help_window.resizable(False, False)
    help_window.configure(bg=app_background_color)

    # Keep the help window associated with the main window without making it
    # modal, so the main application can still be closed while help is open.
    help_window.transient(root)

    main_frame = tk.Frame(help_window, bg=app_background_color)
    main_frame.pack(fill=tk.BOTH, expand=True)

    header_frame = tk.Frame(main_frame, bg=app_background_color)
    header_frame.pack(fill=tk.X, padx=24, pady=(18, 12))

    tk.Label(
        header_frame,
        text="Simulation Controls",
        font=("Arial", 13, "bold"),
        anchor="w",
        bg=app_background_color,
        fg="black",
    ).pack(fill=tk.X)

    help_canvas, help_content_frame, help_scrollbar = _create_scrollable_content(
        main_frame,
        app_background_color,
    )

    # ==================================================
    # Keyboard controls
    # ==================================================
    keyboard_frame = tk.LabelFrame(
        help_content_frame,
        text="Keyboard shortcuts",
        padx=12,
        pady=10,
        bg=app_background_color,
        fg="black",
    )
    keyboard_frame.pack(fill=tk.X, pady=(0, 12))

    keyboard_shortcuts = [
        ("↑ / ↓", "Increase / decrease speed"),
        ("← / →", "Steer left / right"),
        ("Space", "Start / pause / continue simulation"),
        ("Esc", "Exit full-screen mode"),
        ("Ctrl + S", "Save trajectory data as CSV"),
    ]

    _add_description_rows(
        parent=keyboard_frame,
        rows=keyboard_shortcuts,
        left_column_width=keyboard_left_column_width,
        background_color=app_background_color,
    )

    # ==================================================
    # Menu controls
    # ==================================================
    menu_frame = tk.LabelFrame(
        help_content_frame,
        text="Menu",
        padx=12,
        pady=10,
        bg=app_background_color,
        fg="black",
    )
    menu_frame.pack(fill=tk.X, pady=(0, 16))

    menu_descriptions = [
        ("File → Save CSV", "Pause simulation and save trajectory data"),
        ("File → Exit", "Close the application"),
        ("View", "Choose local meters or local kilometers"),
        (
            "Settings",
            "Movement Parameters configures initial speed, speed limits, turn limits and step sizes before a run or after Reset",
        ),
        ("Help → Show Help", "Open this help window"),
    ]
    if root.tk.call("tk", "windowingsystem") != "aqua":
        menu_descriptions.insert(
            3,
            ("View → Full Screen", "Enter or leave full-screen mode"),
        )

    _add_description_rows(
        parent=menu_frame,
        rows=menu_descriptions,
        left_column_width=menu_left_column_width,
        background_color=app_background_color,
    )

    # ==================================================
    # Button controls
    # ==================================================
    button_frame = tk.LabelFrame(
        help_content_frame,
        text="Buttons",
        padx=12,
        pady=10,
        bg=app_background_color,
        fg="black",
    )
    button_frame.pack(fill=tk.X, pady=(0, 16))

    button_descriptions = [
        ("Start Simulation", "Start the interactive simulation"),
        ("Pause Simulation", "Pause the simulation without clearing data"),
        ("Continue Simulation", "Continue a paused simulation"),
        ("Save CSV", "Pause simulation and save trajectory data"),
        ("Reset", "Clear trajectory and unlock movement parameters"),
        ("Center Steering", "Reset steering to 0 °/s"),
    ]

    _add_description_rows(
        parent=button_frame,
        rows=button_descriptions,
        left_column_width=button_left_column_width,
        background_color=app_background_color,
    )

    # ==================================================
    # Simulation information
    # ==================================================
    simulation_info_frame = tk.LabelFrame(
        help_content_frame,
        text="Simulation Information",
        padx=12,
        pady=10,
        bg=app_background_color,
        fg="black",
    )
    simulation_info_frame.pack(fill=tk.X, pady=(0, 16))

    simulation_info_descriptions = [
        (
            "Time factor",
            "Choose 1×, 5×, 10× or 20× in Actions, also while running. This accelerates simulated time; CSV sampling uses the same simulated-time interval",
        ),
        (
            "Movement Parameters",
            "Apply validates and saves the draft; Cancel or closing the window discards edits. Turn limits are symmetric and arrow keys use the selected step sizes",
        ),
        (
            "While paused",
            "Simulation time stops; exported seconds contain no pause gap",
        ),
        (
            "CSV format",
            "time,x,y plus ground_truth_x/y [m], ground_truth_velocity [m/s], ground_truth_yaw [rad] and ground_truth_turn_rate [rad/s]. Bayesian Predictor uses truth only for display and evaluation. Without measurement noise, x/y equal ground_truth_x/y",
        ),
        (
            "Repeated save",
            "Append only new samples to the same trajectory file, without duplicate timestamps",
        ),
        (
            "CSV interval",
            "Every 2 simulated seconds by default, using original positions without interpolation; the simulation still advances in 0.05 s steps",
        ),
        (
            "After Reset",
            "Choose a new CSV filename for the new simulation run",
        ),
        (
            "Turn Radius",
            "Speed divided by angular velocity; ∞ means straight travel",
        ),
    ]

    _add_description_rows(
        parent=simulation_info_frame,
        rows=simulation_info_descriptions,
        left_column_width=simulation_time_left_column_width,
        background_color=app_background_color,
    )

    def close_help_window():
        help_window.destroy()
        focus_callback()

    footer_frame = tk.Frame(main_frame, bg=app_background_color)
    footer_frame.pack(fill=tk.X, padx=24, pady=(16, 18))

    controls.create_styled_button(
        footer_frame,
        text="OK",
        width=12,
        command=close_help_window,
    ).pack(anchor="e")

    controls.bind_mouse_wheel_to_canvas(help_content_frame, help_canvas)
    _fit_scrollable_content_to_screen(
        help_window,
        help_canvas,
        help_content_frame,
        help_scrollbar,
    )

    # Restore keyboard focus after closing the help window.
    help_window.protocol("WM_DELETE_WINDOW", close_help_window)

    # Show the window only after its final layout has been calculated to avoid
    # a visible resize immediately after opening.
    help_window.update_idletasks()
    help_window.geometry(f"{HELP_WINDOW_WIDTH}x{help_window.winfo_reqheight()}+0+0")
    help_window.deiconify()
    help_window.focus_set()
