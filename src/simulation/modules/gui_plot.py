"""
@file gui_plot.py
@description Contains plotting helper functions for the interactive ship trajectory GUI.
@date Created on: 09.07.2026
@author C.Feng
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.path import Path


def get_auto_tick_step(axis_range):
    """
    Choose a readable tick distance based on the visible axis range.

    Parameters
    ----------
    axis_range : float
        Current visible axis range in meters.

    Returns
    -------
    tick_step : int
        Tick distance in meters.
    """
    if axis_range <= 120:
        return 10

    if axis_range <= 300:
        return 25

    if axis_range <= 700:
        return 50

    if axis_range <= 1500:
        return 100

    if axis_range <= 3000:
        return 200

    return 500


def create_ship_shape(gui, theta):
    """
    Create a simple ship-like polygon in real-world meter size.

    The shape is not a regular triangle:
    - sharp bow at the front
    - wider middle section
    - narrower stern at the back

    The ship points in the direction of theta.
    """
    length = gui.ship_length
    width = gui.ship_width

    # Ship points for theta = 0.
    # The ship points to the right.
    ship_shape = np.array(
        [
            [0.50 * length, 0.00 * width],  # Bow / front tip
            [0.20 * length, 0.50 * width],  # Front upper side
            [-0.35 * length, 0.40 * width],  # Rear upper side
            [-0.50 * length, 0.20 * width],  # Stern upper corner
            [-0.50 * length, -0.20 * width],  # Stern lower corner
            [-0.35 * length, -0.40 * width],  # Rear lower side
            [0.20 * length, -0.50 * width],  # Front lower side
        ]
    )

    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    return ship_shape @ rotation_matrix.T


def create_ship_legend_marker():
    """
    Create a small custom ship marker for the legend.

    This marker has the same ship-like shape as the real polygon,
    but it is normalized for display inside the legend.
    """
    half_width = 0.35

    vertices = [
        (1.00, 0.00),  # Bow / front tip
        (0.40, half_width),  # Front upper side
        (-0.70, 0.28),  # Rear upper side
        (-1.00, 0.14),  # Stern upper corner
        (-1.00, -0.14),  # Stern lower corner
        (-0.70, -0.28),  # Rear lower side
        (0.40, -half_width),  # Front lower side
        (1.00, 0.00),  # Close polygon
    ]

    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]

    return Path(vertices, codes)


def add_heading_marker(gui):
    """
    Add a ship marker at the current ship position.

    The marker is drawn in real-world meter size.
    """
    x = gui.simulator.x_current
    y = gui.simulator.y_current
    theta = gui.simulator.theta_current

    ship_shape = create_ship_shape(gui, theta)

    # Move ship shape to current position.
    ship_shape[:, 0] += x
    ship_shape[:, 1] += y

    ship_patch = Polygon(
        ship_shape,
        closed=True,
        facecolor=gui.ship_facecolor,
        edgecolor=gui.ship_edgecolor,
        linewidth=gui.ship_linewidth,
        alpha=gui.ship_alpha,
        label="Ship",
        zorder=5,
    )

    gui.ax.add_patch(ship_patch)


def add_ship_center_marker(gui):
    """
    Add a small marker at the ship center.

    This shows the actual simulated position of the ship.
    """
    if not gui.show_ship_center:
        return

    gui.ax.scatter(
        gui.simulator.x_current,
        gui.simulator.y_current,
        s=gui.ship_center_size,
        color=gui.ship_center_color,
        marker="o",
        zorder=6,
    )


def update_legend(gui):
    """
    Update the plot legend with a fixed heading marker.

    The legend icon does not rotate, because it is only a symbol.
    The real heading marker in the plot still rotates with the ship heading.
    """
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=gui.trajectory_color,
            linewidth=gui.trajectory_linewidth,
            label="Trajectory",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=gui.start_position_color,
            linestyle="None",
            markersize=8,
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker=create_ship_legend_marker(),
            markerfacecolor=gui.ship_facecolor,
            markeredgecolor=gui.ship_edgecolor,
            color=gui.ship_edgecolor,
            linestyle="None",
            markersize=14,
            label="Ship",
        ),
    ]

    gui.ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        handlelength=1.8,
        handletextpad=0.8,
        borderpad=0.5,
    )


def update_axis_limits(gui):
    """
    Update plot axis limits.

    axis_mode = "auto":
        Show the full trajectory automatically with a square axis range.

    axis_mode = "fixed":
        Use fixed axis limits.

    axis_mode = "follow":
        Smoothly follow the current ship position.
    """
    if gui.axis_mode == "auto":
        if gui.simulator.has_data():
            x_values = list(gui.simulator.x_all) + [gui.simulator.x_current]
            y_values = list(gui.simulator.y_all) + [gui.simulator.y_current]

            x_min_data = min(x_values)
            x_max_data = max(x_values)
            y_min_data = min(y_values)
            y_max_data = max(y_values)

            x_center = (x_min_data + x_max_data) / 2
            y_center = (y_min_data + y_max_data) / 2

            x_range = (x_max_data - x_min_data) + 2 * gui.auto_axis_margin
            y_range = (y_max_data - y_min_data) + 2 * gui.auto_axis_margin

            # Use the larger range for both axes.
            # This keeps the visible plot area square in meters.
            plot_range = max(x_range, y_range)

            x_min = x_center - plot_range / 2
            x_max = x_center + plot_range / 2
            y_min = y_center - plot_range / 2
            y_max = y_center + plot_range / 2
        else:
            x_center = (gui.axis_x_min + gui.axis_x_max) / 2
            y_center = (gui.axis_y_min + gui.axis_y_max) / 2

            x_range = gui.axis_x_max - gui.axis_x_min
            y_range = gui.axis_y_max - gui.axis_y_min

            plot_range = max(x_range, y_range)

            x_min = x_center - plot_range / 2
            x_max = x_center + plot_range / 2
            y_min = y_center - plot_range / 2
            y_max = y_center + plot_range / 2

    elif gui.axis_mode == "follow":
        ship_x = gui.simulator.x_current
        ship_y = gui.simulator.y_current

        # Smoothly move the camera toward the current ship position.
        gui.camera_x += gui.camera_smoothness * (ship_x - gui.camera_x)
        gui.camera_y += gui.camera_smoothness * (ship_y - gui.camera_y)

        x_min = gui.camera_x - gui.view_width / 2
        x_max = gui.camera_x + gui.view_width / 2
        y_min = gui.camera_y - gui.view_height / 2
        y_max = gui.camera_y + gui.view_height / 2

    else:
        x_min = gui.axis_x_min
        x_max = gui.axis_x_max
        y_min = gui.axis_y_min
        y_max = gui.axis_y_max

    gui.ax.set_xlim(x_min, x_max)
    gui.ax.set_ylim(y_min, y_max)

    # Choose tick step automatically so the axis labels stay readable.
    x_axis_range = x_max - x_min
    y_axis_range = y_max - y_min
    tick_step = get_auto_tick_step(max(x_axis_range, y_axis_range))

    x_tick_min = np.floor(x_min / tick_step) * tick_step
    x_tick_max = np.ceil(x_max / tick_step) * tick_step

    y_tick_min = np.floor(y_min / tick_step) * tick_step
    y_tick_max = np.ceil(y_max / tick_step) * tick_step

    gui.ax.set_xticks(
        np.arange(
            x_tick_min,
            x_tick_max + tick_step,
            tick_step,
        )
    )
    gui.ax.set_yticks(
        np.arange(
            y_tick_min,
            y_tick_max + tick_step,
            tick_step,
        )
    )


def update_plot(gui):
    """
    Update the trajectory plot.
    """
    gui.ax.clear()

    # Reapply plot background after clearing.
    gui.ax.set_facecolor(gui.plot_background_color)

    if gui.simulator.has_data():
        # Add the current position to the displayed trajectory.
        # This makes the blue line connect to the current heading marker.
        x_plot = list(gui.simulator.x_all) + [gui.simulator.x_current]
        y_plot = list(gui.simulator.y_all) + [gui.simulator.y_current]

        gui.ax.plot(
            x_plot,
            y_plot,
            color=gui.trajectory_color,
            linewidth=gui.trajectory_linewidth,
            label="Trajectory",
        )

        gui.ax.scatter(
            gui.simulator.x_all[0],
            gui.simulator.y_all[0],
            color=gui.start_position_color,
            label="Start",
            zorder=4,
        )

        # Show current ship shape and heading direction.
        add_heading_marker(gui)

        # Show the simulated ship center.
        add_ship_center_marker(gui)

    gui.ax.set_xlabel("x [m]")
    gui.ax.set_ylabel("y [m]", rotation=0, labelpad=15, va="center")
    gui.ax.set_title("Ship Trajectory Simulation")

    # Keep 1 meter on x-axis equal to 1 meter on y-axis.
    gui.ax.set_aspect("equal", adjustable="box")

    gui.ax.grid(
        True,
        color=gui.grid_color,
        alpha=gui.grid_alpha,
        linewidth=gui.grid_linewidth,
    )

    # Update axis range and ticks depending on axis_mode.
    update_axis_limits(gui)

    if gui.simulator.has_data():
        update_legend(gui)

    gui.canvas.draw()
