"""Tkinter controls and plot canvas for the ship trajectory GUI."""

import tkinter as tk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ship_trajectory_prediction.trajectory.coordinates import (
    METERS_PER_KILOMETER,
    local_to_gps_coordinates,
)


def create_styled_button(parent, *, text, command, width):
    """Create a button with the same visual style on Windows and macOS."""
    normal_color = "#ffffff"
    hover_color = "#b8d8e8"

    button = tk.Button(
        parent,
        text=text,
        command=command,
        width=width,
        font=("Arial", 10),
        bg=normal_color,
        fg="#1f2933",
        activebackground=hover_color,
        activeforeground="#1f2933",
        disabledforeground="#7a8790",
        relief=tk.SOLID,
        borderwidth=1,
        highlightthickness=0,
        padx=10,
        pady=3,
    )
    button.bind("<Enter>", lambda _event: button.configure(bg=hover_color))
    button.bind("<Leave>", lambda _event: button.configure(bg=normal_color))
    return button


def create_menu_bar(gui):
    """Create the desktop-style application menu bar."""
    menu_bar = tk.Menu(gui.root)

    file_menu = tk.Menu(menu_bar, tearoff=0)
    file_menu.add_command(
        label="Save CSV",
        accelerator="Ctrl+S",
        command=gui.save_csv,
    )
    file_menu.add_separator()
    file_menu.add_command(
        label="Exit",
        command=gui.root.destroy,
    )
    menu_bar.add_cascade(label="File", menu=file_menu)

    gui.view_menu = tk.Menu(menu_bar, tearoff=0)
    gui.coordinate_display_var = tk.StringVar(value=gui.coordinate_display_mode)
    coordinate_display_modes = (
        ("Local [m]", "local"),
        ("Local [km]", "km"),
        ("GPS [°]", "gps"),
    )
    for label, mode in coordinate_display_modes:
        gui.view_menu.add_radiobutton(
            label=label,
            variable=gui.coordinate_display_var,
            value=mode,
            command=gui.change_coordinate_display,
        )

    gui.fullscreen_menu_index = None
    if gui.root.tk.call("tk", "windowingsystem") != "aqua":
        gui.view_menu.add_separator()
        gui.view_menu.add_command(
            label="Enter Full Screen",
            command=gui.toggle_fullscreen,
        )
        gui.fullscreen_menu_index = gui.view_menu.index("end")
    menu_bar.add_cascade(label="View", menu=gui.view_menu)

    gui.settings_menu = tk.Menu(menu_bar, tearoff=0)
    gui.settings_menu.add_command(
        label="GPS Start Position...",
        command=gui.show_gps_start_position_dialog,
    )
    gui.gps_position_menu_index = gui.settings_menu.index("end")
    menu_bar.add_cascade(label="Settings", menu=gui.settings_menu)

    help_menu = tk.Menu(menu_bar, tearoff=0)
    help_menu.add_command(
        label="Show Help",
        command=gui.show_help,
    )
    menu_bar.add_cascade(label="Help", menu=help_menu)

    gui.root.config(menu=menu_bar)


def bind_keyboard_controls(gui):
    """Bind keys to steering, simulation controls, and CSV export."""
    gui.root.bind("<Up>", gui.increase_speed)
    gui.root.bind("<Down>", gui.decrease_speed)
    gui.root.bind("<Left>", gui.steer_left)
    gui.root.bind("<Right>", gui.steer_right)
    gui.root.bind("<space>", gui.toggle_simulation_with_keyboard)
    gui.root.bind("<Escape>", gui.exit_fullscreen)

    gui.root.bind("<Control-s>", gui.save_csv_with_keyboard)
    gui.root.bind("<Control-S>", gui.save_csv_with_keyboard)
    gui.root.bind("<Command-s>", gui.save_csv_with_keyboard)
    gui.root.bind("<Command-S>", gui.save_csv_with_keyboard)

    gui.root.focus_set()


def create_gui_widgets(gui):
    """
    Create the main GUI widgets.

    This function builds the left control panel and the right plot panel.
    The main ShipTrajectoryGUI object is passed in as `gui`, so widget
    references such as gui.speed_slider can still be used by the main GUI
    logic.
    """
    # Main content frame below the menu bar.
    content_frame = tk.Frame(
        gui.root,
        bg=gui.app_background_color,
    )
    content_frame.pack(fill=tk.BOTH, expand=True)

    create_control_panel(gui, content_frame)
    create_plot_panel(gui, content_frame)


def create_control_panel(gui, parent):
    """
    Create the scrollable left control panel.
    """
    control_container = tk.Frame(
        parent,
        width=gui.control_panel_width,
        bg=gui.control_panel_color,
    )
    control_container.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0), pady=10)
    control_container.pack_propagate(False)

    control_canvas = tk.Canvas(
        control_container,
        bg=gui.control_panel_color,
        borderwidth=0,
        highlightthickness=0,
    )
    control_scrollbar = tk.Scrollbar(
        control_container,
        orient=tk.VERTICAL,
        command=control_canvas.yview,
    )
    control_canvas.configure(yscrollcommand=control_scrollbar.set)

    control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    control_frame = tk.Frame(
        control_canvas,
        bg=gui.control_panel_color,
        padx=14,
        pady=14,
    )
    control_window = control_canvas.create_window(
        (0, 0),
        window=control_frame,
        anchor="nw",
    )

    control_frame.bind(
        "<Configure>",
        lambda _event: control_canvas.configure(
            scrollregion=control_canvas.bbox("all")
        ),
    )
    control_canvas.bind(
        "<Configure>",
        lambda event: control_canvas.itemconfigure(
            control_window,
            width=event.width,
        ),
    )

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

    bind_mouse_wheel_to_canvas(control_frame, control_canvas)


def bind_mouse_wheel_to_canvas(container, canvas):
    """Scroll a canvas when the pointer is over its content widgets."""

    def scroll_panel_with_wheel(event):
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            direction = -1
        else:
            direction = 1

        canvas.yview_scroll(direction, "units")
        return "break"

    def scroll_panel_with_touchpad(event):
        _delta_x, delta_y = canvas.tk.call(
            "tk::PreciseScrollDeltas",
            event.delta,
        )
        delta_y = float(delta_y)
        scroll_region = canvas.bbox("all")

        if delta_y == 0 or scroll_region is None:
            return "break"

        content_height = scroll_region[3] - scroll_region[1]
        first, last = canvas.yview()

        if content_height <= 0 or last - first >= 1:
            return "break"

        max_first = 1 - (last - first)
        new_first = min(max(first - delta_y / content_height, 0), max_first)
        canvas.yview_moveto(new_first)
        return "break"

    widgets = [container]
    while widgets:
        widget = widgets.pop()
        widget.bind("<MouseWheel>", scroll_panel_with_wheel, add="+")
        widget.bind("<Button-4>", scroll_panel_with_wheel, add="+")
        widget.bind("<Button-5>", scroll_panel_with_wheel, add="+")
        try:
            widget.bind("<TouchpadScroll>", scroll_panel_with_touchpad, add="+")
        except tk.TclError:
            # TouchpadScroll was added in Tk 9.
            pass
        widgets.extend(widget.winfo_children())


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
        ("Turn Radius:", gui.radius_value_label),
        ("Time:", gui.time_value_label),
    ]

    for row_index, (label_text, value_label) in enumerate(status_rows):
        tk.Label(
            status_frame,
            text=label_text,
            anchor="w",
            width=12,
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


def update_status_display(gui):
    """Update the simulation status labels."""
    heading_deg = np.rad2deg(gui.simulator.theta_current)
    omega = gui.get_omega_from_steering()
    omega_deg = np.rad2deg(omega)
    speed = gui.get_speed_from_slider()
    simulation_state = gui.get_simulation_state()

    radius_text = "∞" if abs(omega) < 1e-8 else f"{speed / omega:.2f} m"

    if simulation_state == "running":
        simulation_text = "Running"
        simulation_color = "darkgreen"
    elif simulation_state == "paused":
        simulation_text = "Paused"
        simulation_color = "orange"
    else:
        simulation_text = "Stopped"
        simulation_color = "darkred"

    gui.simulation_value_label.config(
        text=simulation_text,
        fg=simulation_color,
    )

    if gui.coordinate_display_mode == "gps":
        longitude, latitude = local_to_gps_coordinates(
            [gui.simulator.x_current],
            [gui.simulator.y_current],
            reference_longitude=gui.reference_longitude,
            reference_latitude=gui.reference_latitude,
        )
        position_text = f"lon = {longitude[0]:.4f}°\nlat = {latitude[0]:.4f}°"
    elif gui.coordinate_display_mode == "km":
        position_text = (
            f"x = {gui.simulator.x_current / METERS_PER_KILOMETER:.3f} km\n"
            f"y = {gui.simulator.y_current / METERS_PER_KILOMETER:.3f} km"
        )
    else:
        position_text = (
            f"x = {gui.simulator.x_current:.2f} m\ny = {gui.simulator.y_current:.2f} m"
        )

    gui.position_value_label.config(text=position_text)
    gui.heading_value_label.config(text=f"{heading_deg:.1f}°")
    gui.omega_value_label.config(text=f"{omega_deg:.1f}°/s")
    gui.speed_value_label.config(text=f"{speed:.2f} m/s")
    gui.radius_value_label.config(text=radius_text)
    gui.time_value_label.config(text=f"{gui.simulator.current_time:.1f} s")
