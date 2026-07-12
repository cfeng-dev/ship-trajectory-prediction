"""Help window for the interactive ship trajectory GUI."""

import tkinter as tk
from collections.abc import Callable


def center_window_on_screen(window):
    """
    Center a child window on the screen.

    Parameters
    ----------
    window : tk.Toplevel
        Child window to be centered.
    """
    window.update_idletasks()

    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    window_width = window.winfo_width()
    window_height = window.winfo_height()

    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    window.geometry(f"+{x}+{y}")


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
            anchor="w",
            font=("Arial", 10, "bold"),
            bg=background_color,
            fg="black",
        ).grid(row=row, column=0, sticky="w", pady=2)

        tk.Label(
            parent,
            text=description,
            anchor="w",
            bg=background_color,
            fg="black",
        ).grid(row=row, column=1, sticky="w", pady=2)


def show_help_window(
    root,
    app_background_color,
    pause_callback: Callable[[], None],
    focus_callback: Callable[[], None],
    keyboard_left_column_width,
    menu_left_column_width,
    button_left_column_width,
):
    """
    Show a help window with keyboard shortcuts and basic usage.

    The simulation is paused before opening the help window so the trajectory
    does not keep changing while the user is reading the instructions.
    """
    pause_callback()

    help_window = tk.Toplevel(root)
    help_window.title("Help")
    help_window.resizable(False, False)
    help_window.configure(bg=app_background_color)

    # Keep the help window above the main window.
    help_window.transient(root)
    help_window.grab_set()

    main_frame = tk.Frame(
        help_window,
        padx=24,
        pady=18,
        bg=app_background_color,
    )
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(
        main_frame,
        text="Simulation Controls",
        font=("Arial", 13, "bold"),
        anchor="w",
        bg=app_background_color,
        fg="black",
    ).pack(fill=tk.X, pady=(0, 12))

    # ==================================================
    # Keyboard controls
    # ==================================================
    keyboard_frame = tk.LabelFrame(
        main_frame,
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
        main_frame,
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
        ("Help → Show Help", "Open this help window"),
    ]

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
        main_frame,
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
        ("Reset", "Stop simulation and clear trajectory"),
        ("Center Steering", "Reset steering to 0 °/s"),
    ]

    _add_description_rows(
        parent=button_frame,
        rows=button_descriptions,
        left_column_width=button_left_column_width,
        background_color=app_background_color,
    )

    def close_help_window():
        help_window.destroy()
        focus_callback()

    tk.Button(
        main_frame,
        text="OK",
        width=12,
        command=close_help_window,
    ).pack(anchor="e")

    # Restore keyboard focus after closing the help window.
    help_window.protocol("WM_DELETE_WINDOW", close_help_window)

    # Show the help window in the center of the screen.
    center_window_on_screen(help_window)
