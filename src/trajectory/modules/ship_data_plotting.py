"""
@file ship_data_plotting.py
@description Provides helper functions to plot real ship trajectory data.
@date Created on: 09.07.2026
@author C.Feng
"""

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrowPatch

# ==================================================
# Plot settings
# ==================================================
TRAJECTORY_FIGURE_SIZE = (8, 6)
SPEED_FIGURE_SIZE = (10, 6)

DEFAULT_ARROW_STEP = 18  # 18 points = about 3 minutes if data interval is 10 s

TRAJECTORY_COLOR = "tab:blue"
START_COLOR = "black"
END_COLOR = "green"
DIRECTION_COLOR = "darkorange"

TRAJECTORY_MARKER_SIZE = 3
START_MARKER_SIZE = 45
END_MARKER_SIZE = 110

TRAJECTORY_LINE_WIDTH = 1.8
DIRECTION_LINE_WIDTH = 1.5

END_ALPHA = 0.45
ARROW_MUTATION_SCALE = 14


class HandlerDirectionArrow(HandlerPatch):
    """
    Custom legend handler for direction arrows.
    """

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        center_y = ydescent + height / 2

        arrow = FancyArrowPatch(
            (xdescent, center_y),
            (xdescent + width, center_y),
            arrowstyle="->",
            color=orig_handle.get_edgecolor(),
            linewidth=orig_handle.get_linewidth(),
            mutation_scale=ARROW_MUTATION_SCALE,
            transform=trans,
        )

        return [arrow]


def plot_ship_trajectory(data, arrow_step=DEFAULT_ARROW_STEP):
    """
    Plot the ship trajectory using GPS longitude and latitude.

    The plot shows:
    - the trajectory line
    - start point
    - end point
    - optional direction arrows along the path

    Parameters
    ----------
    data : pandas.DataFrame
        Ship trajectory data containing gps_longitude and gps_latitude.
    arrow_step : int or None, optional
        Distance between direction arrows in number of data points.
        If None, no direction arrows are plotted.
    """
    if data.empty:
        raise ValueError("The input data is empty.")

    if arrow_step is not None and arrow_step <= 0:
        raise ValueError("arrow_step must be a positive integer or None.")

    longitude = data["gps_longitude"].to_numpy()
    latitude = data["gps_latitude"].to_numpy()

    plt.figure(figsize=TRAJECTORY_FIGURE_SIZE)

    # Trajectory line
    trajectory_line = plt.plot(
        longitude,
        latitude,
        color=TRAJECTORY_COLOR,
        marker="o",
        markersize=TRAJECTORY_MARKER_SIZE,
        linewidth=TRAJECTORY_LINE_WIDTH,
        label="Ship trajectory",
    )[0]

    # Start point
    start_marker = plt.scatter(
        longitude[0],
        latitude[0],
        s=START_MARKER_SIZE,
        color=START_COLOR,
        marker="o",
        label="Start",
        zorder=4,
    )

    # End point
    end_marker = plt.scatter(
        longitude[-1],
        latitude[-1],
        s=END_MARKER_SIZE,
        color=END_COLOR,
        alpha=END_ALPHA,
        marker="X",
        label="End",
        zorder=5,
    )

    legend_handles = [
        trajectory_line,
        start_marker,
        end_marker,
    ]

    legend_handler_map = {}

    # Direction arrows
    if arrow_step is not None:
        for index in range(0, len(longitude) - 1, arrow_step):
            start = (longitude[index], latitude[index])
            end = (longitude[index + 1], latitude[index + 1])

            dx = end[0] - start[0]
            dy = end[1] - start[1]

            if dx == 0 and dy == 0:
                continue

            plt.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "->",
                    "color": DIRECTION_COLOR,
                    "linewidth": DIRECTION_LINE_WIDTH,
                    "mutation_scale": ARROW_MUTATION_SCALE,
                },
            )

        direction_handle = FancyArrowPatch(
            (0, 0),
            (1, 0),
            arrowstyle="->",
            color=DIRECTION_COLOR,
            linewidth=DIRECTION_LINE_WIDTH,
            mutation_scale=ARROW_MUTATION_SCALE,
            label="Direction",
        )

        legend_handles.append(direction_handle)
        legend_handler_map[FancyArrowPatch] = HandlerDirectionArrow()

    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.title("Ship Trajectory with Direction")
    plt.grid(True)
    plt.axis("equal")

    ax = plt.gca()
    ax.ticklabel_format(useOffset=False, style="plain")

    plt.legend(
        handles=legend_handles,
        handler_map=legend_handler_map,
    )

    plt.tight_layout()
    plt.show()


def plot_ship_speeds(data):
    """
    Plot ship speed signals over time.

    Parameters
    ----------
    data : pandas.DataFrame
        Ship trajectory data containing time, gps_speed,
        shaft_speed, and thruster_speed.
    """
    if data.empty:
        raise ValueError("The input data is empty.")

    plt.figure(figsize=SPEED_FIGURE_SIZE)

    plt.plot(data["time"], data["gps_speed"], label="GPS speed")
    plt.plot(data["time"], data["shaft_speed"], label="Shaft speed")
    plt.plot(data["time"], data["thruster_speed"], label="Thruster speed")

    plt.xlabel("Time")
    plt.ylabel("Speed")
    plt.title("Ship Speed Signals Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
