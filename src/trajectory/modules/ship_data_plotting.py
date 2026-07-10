"""
@file ship_data_plotting.py
@description Provides helper functions to plot real ship trajectory data.
@date Created on: 09.07.2026
@author C.Feng
"""

import matplotlib.pyplot as plt
import numpy as np
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
EARTH_RADIUS_KM = 6371.0088
MIN_AXIS_MARGIN_METERS = 15


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


def plot_ship_trajectory(data, arrow_step=DEFAULT_ARROW_STEP, coordinate_unit="gps"):
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
    coordinate_unit : {"gps", "km", "m"}, optional
        Coordinate representation. ``"gps"`` plots longitude and latitude in
        degrees. ``"km"`` and ``"m"`` plot local east and north distances
        from the first trajectory point.
    """
    if data.empty:
        raise ValueError("The input data is empty.")

    if coordinate_unit not in {"gps", "km", "m"}:
        raise ValueError("coordinate_unit must be 'gps', 'km', or 'm'.")

    if arrow_step is not None and arrow_step <= 0:
        raise ValueError("arrow_step must be a positive integer or None.")

    longitude = data["gps_longitude"].to_numpy()
    latitude = data["gps_latitude"].to_numpy()

    if coordinate_unit in {"km", "m"}:
        reference_longitude = np.radians(longitude[0])
        reference_latitude = np.radians(latitude[0])
        longitude_radians = np.radians(longitude)
        latitude_radians = np.radians(latitude)
        mean_latitude = (latitude_radians + reference_latitude) / 2

        x_coordinates_km = (
            EARTH_RADIUS_KM
            * (longitude_radians - reference_longitude)
            * np.cos(mean_latitude)
        )
        y_coordinates_km = EARTH_RADIUS_KM * (
            latitude_radians - reference_latitude
        )

        if coordinate_unit == "m":
            x_coordinates = x_coordinates_km * 1000
            y_coordinates = y_coordinates_km * 1000
            axis_unit = "m"
        else:
            x_coordinates = x_coordinates_km
            y_coordinates = y_coordinates_km
            axis_unit = "km"

        x_label = f"x [{axis_unit}]"
        y_label = f"y [{axis_unit}]"
    else:
        x_coordinates = longitude
        y_coordinates = latitude
        x_label = "Longitude [deg]"
        y_label = "Latitude [deg]"

    plt.figure(figsize=TRAJECTORY_FIGURE_SIZE)

    # Trajectory line
    trajectory_line = plt.plot(
        x_coordinates,
        y_coordinates,
        color=TRAJECTORY_COLOR,
        marker="o",
        markersize=TRAJECTORY_MARKER_SIZE,
        linewidth=TRAJECTORY_LINE_WIDTH,
        label="Ship trajectory",
    )[0]

    # Start point
    start_marker = plt.scatter(
        x_coordinates[0],
        y_coordinates[0],
        s=START_MARKER_SIZE,
        color=START_COLOR,
        marker="o",
        label="Start",
        zorder=4,
    )

    # End point
    end_marker = plt.scatter(
        x_coordinates[-1],
        y_coordinates[-1],
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
        for index in range(0, len(x_coordinates) - 1, arrow_step):
            start = (x_coordinates[index], y_coordinates[index])
            end = (x_coordinates[index + 1], y_coordinates[index + 1])

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

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Ship Trajectory with Direction")
    plt.grid(True)
    plt.axis("equal")

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    if coordinate_unit in {"km", "m"}:
        x_center = (x_coordinates.min() + x_coordinates.max()) / 2
        y_center = (y_coordinates.min() + y_coordinates.max()) / 2
        data_range = max(np.ptp(x_coordinates), np.ptp(y_coordinates))
        minimum_margin = MIN_AXIS_MARGIN_METERS if coordinate_unit == "m" else 0.015
        margin = max(data_range * 0.05, minimum_margin)
        plot_range = max(data_range + 2 * margin, 2 * minimum_margin)

        ax.set_xlim(x_center - plot_range / 2, x_center + plot_range / 2)
        ax.set_ylim(y_center - plot_range / 2, y_center + plot_range / 2)

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
