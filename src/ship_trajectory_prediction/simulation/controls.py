"""Tkinter controls and plot canvas for the ship trajectory GUI."""

import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def create_styled_button(parent, *, text, command, width):
    """Create a button with the same visual style on Windows and macOS."""
    return tk.Button(
        parent,
        text=text,
        command=command,
        width=width,
        font=("Arial", 10),
        bg="#ffffff",
        fg="#1f2933",
        activebackground="#d8eaf3",
        activeforeground="#1f2933",
        disabledforeground="#7a8790",
        relief=tk.SOLID,
        borderwidth=1,
        highlightthickness=0,
        padx=10,
        pady=3,
    )


def create_gui_widgets(gui):
    """
    Create the main GUI widgets.

    This function builds the left control panel, the separator, and the
    right plot panel. The main ShipTrajectoryGUI object is passed in as
    `gui`, so widget references such as gui.speed_slider can still be used
    by the main GUI logic.
    """
    # Main content frame below the menu bar.
    content_frame = tk.Frame(
        gui.root,
        bg=gui.app_background_color,
    )
    content_frame.pack(fill=tk.BOTH, expand=True)

    create_control_panel(gui, content_frame)
    create_separator(gui, content_frame)
    create_plot_panel(gui, content_frame)


def create_control_panel(gui, parent):
    """
    Create the left control panel.
    """
    control_frame = tk.Frame(
        parent,
        width=gui.control_panel_width,
        bg=gui.control_panel_color,
        padx=14,
        pady=14,
    )
    control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0), pady=10)
    control_frame.pack_propagate(False)

    tk.Label(
        control_frame,
        text="Simulation Control",
        font=("Arial", 12, "bold"),
        bg=gui.control_panel_color,
        fg="black",
    ).pack(pady=(0, 12))

    create_action_section(gui, control_frame)
    create_steering_section(gui, control_frame)
    create_speed_section(gui, control_frame)
    create_status_section(gui, control_frame)


def create_action_section(gui, parent):
    """
    Create the action button section.
    """
    action_frame = tk.LabelFrame(
        parent,
        text="Actions",
        padx=10,
        pady=8,
        bg=gui.control_panel_color,
        fg="black",
    )
    action_frame.pack(fill=tk.X, pady=(0, gui.section_spacing))

    gui.simulation_button = create_styled_button(
        action_frame,
        text="Start Simulation",
        width=18,
        command=gui.toggle_simulation,
    )
    gui.simulation_button.pack(pady=(2, 5))

    create_styled_button(
        action_frame,
        text="Save CSV",
        width=18,
        command=gui.save_csv,
    ).pack(pady=5)

    create_styled_button(
        action_frame,
        text="Reset",
        width=18,
        command=gui.reset,
    ).pack(pady=(5, 2))


def create_steering_section(gui, parent):
    """
    Create the steering control section.
    """
    steering_frame = tk.LabelFrame(
        parent,
        text="Steering",
        padx=10,
        pady=8,
        bg=gui.control_panel_color,
        fg="black",
    )
    steering_frame.pack(fill=tk.X, pady=(0, gui.section_spacing))

    tk.Label(
        steering_frame,
        text="Left  ←   0 °/s   →  Right",
        width=24,
        anchor="center",
        bg=gui.control_panel_color,
        fg="black",
    ).pack(pady=(0, 2))

    gui.steering_slider = tk.Scale(
        steering_frame,
        from_=gui.min_steering_deg_per_second,
        to=gui.max_steering_deg_per_second,
        orient=tk.HORIZONTAL,
        length=180,
        resolution=gui.steering_resolution,
        bg=gui.control_panel_color,
        fg="black",
        highlightbackground=gui.control_panel_color,
    )
    gui.steering_slider.set(0)
    gui.steering_slider.pack(pady=(0, 5))

    create_styled_button(
        steering_frame,
        text="Center Steering",
        width=18,
        command=gui.center_steering,
    ).pack(pady=(2, 0))


def create_speed_section(gui, parent):
    """
    Create the speed control section.
    """
    speed_frame = tk.LabelFrame(
        parent,
        text="Speed",
        padx=10,
        pady=8,
        bg=gui.control_panel_color,
        fg="black",
    )
    speed_frame.pack(fill=tk.X, pady=(0, gui.section_spacing))

    tk.Label(
        speed_frame,
        text="Slow  ←   m/s   →  Fast",
        width=24,
        anchor="center",
        bg=gui.control_panel_color,
        fg="black",
    ).pack(pady=(0, 2))

    gui.speed_slider = tk.Scale(
        speed_frame,
        from_=gui.min_speed,
        to=gui.max_speed,
        orient=tk.HORIZONTAL,
        length=180,
        resolution=gui.speed_resolution,
        digits=3,
        bg=gui.control_panel_color,
        fg="black",
        highlightbackground=gui.control_panel_color,
    )
    gui.speed_slider.set(gui.initial_speed)
    gui.speed_slider.pack(pady=(0, 2))


def create_status_section(gui, parent):
    """
    Create the status display section.
    """
    status_frame = tk.LabelFrame(
        parent,
        text="Status",
        padx=8,
        pady=6,
        bg=gui.control_panel_color,
        fg="black",
    )
    status_frame.pack(pady=(0, 0), fill=tk.X)

    # The value labels do not use a fixed character width.
    # This avoids cutting off long numbers when the position grows.
    gui.simulation_value_label = create_status_value_label(gui, status_frame)
    gui.position_value_label = create_status_value_label(gui, status_frame)
    gui.heading_value_label = create_status_value_label(gui, status_frame)
    gui.omega_value_label = create_status_value_label(gui, status_frame)
    gui.speed_value_label = create_status_value_label(gui, status_frame)
    gui.radius_value_label = create_status_value_label(gui, status_frame)
    gui.time_value_label = create_status_value_label(gui, status_frame)

    status_frame.columnconfigure(0, weight=0)
    status_frame.columnconfigure(1, weight=1)

    status_rows = [
        ("Simulation:", gui.simulation_value_label),
        ("Position:", gui.position_value_label),
        ("Heading:", gui.heading_value_label),
        ("Omega:", gui.omega_value_label),
        ("Speed:", gui.speed_value_label),
        ("Radius:", gui.radius_value_label),
        ("Time:", gui.time_value_label),
    ]

    for row_index, (label_text, value_label) in enumerate(status_rows):
        tk.Label(
            status_frame,
            text=label_text,
            anchor="w",
            width=10,
            font=gui.status_label_font,
            bg=gui.control_panel_color,
            fg="black",
        ).grid(
            row=row_index,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=gui.status_row_padding_y,
        )

        value_label.grid(
            row=row_index,
            column=1,
            sticky="ew",
            pady=gui.status_row_padding_y,
        )


def create_status_value_label(gui, parent):
    """
    Create one status value label.
    """
    return tk.Label(
        parent,
        anchor="w",
        justify="left",
        font=gui.status_value_font,
        bg=gui.control_panel_color,
        fg="black",
    )


def create_separator(gui, parent):
    """
    Create the vertical separator between control panel and plot area.
    """
    separator = tk.Frame(
        parent,
        bg=gui.separator_color,
        width=2,
    )
    separator.pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=10)


def create_plot_panel(gui, parent):
    """
    Create the right plot panel and Matplotlib canvas.
    """
    plot_frame = tk.Frame(
        parent,
        bg=gui.plot_panel_color,
        padx=10,
        pady=10,
    )
    plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=10)

    gui.figure = Figure(
        figsize=(gui.figure_width, gui.figure_height),
        facecolor=gui.figure_background_color,
    )
    gui.figure.patch.set_facecolor(gui.figure_background_color)

    gui.ax = gui.figure.add_subplot(111)
    gui.ax.set_facecolor(gui.plot_background_color)

    gui.canvas = FigureCanvasTkAgg(gui.figure, master=plot_frame)

    canvas_widget = gui.canvas.get_tk_widget()
    canvas_widget.configure(
        bg=gui.plot_panel_color,
        highlightbackground=gui.plot_panel_color,
        highlightthickness=0,
    )
    canvas_widget.pack(fill=tk.BOTH, expand=True)
