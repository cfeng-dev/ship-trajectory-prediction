"""Main-window presentation for the interactive ship trajectory GUI."""

import tkinter as tk

import numpy as np

from ship_trajectory_prediction.simulation.coordinates import (
    METERS_PER_KILOMETER,
    local_to_gps_coordinates,
)


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
