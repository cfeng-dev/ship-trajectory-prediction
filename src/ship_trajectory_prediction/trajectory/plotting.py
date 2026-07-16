"""Plotting utilities for real ship trajectory data."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrowPatch

from ship_trajectory_prediction.coordinates import (
    calculate_speed_from_gps,
    gps_to_local_coordinates,
)

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
        # Convert GPS coordinates to a local Cartesian coordinate system.
        # The first GPS point is used as the origin (x=0, y=0). For the
        # relatively short trajectories considered here, the equirectangular
        # approximation provides local east/north distances:
        #
        #   x = R * (longitude - reference_longitude) * cos(mean_latitude)
        #   y = R * (latitude  - reference_latitude)
        #
        # All angles must be in radians. The cosine term compensates for the
        # decreasing east-west distance of one longitude degree toward the
        # poles. Positive x points east and positive y points north.
        x_coordinates, y_coordinates = gps_to_local_coordinates(
            longitude,
            latitude,
            unit=coordinate_unit,
        )
        x_label = f"x [{coordinate_unit}]"
        y_label = f"y [{coordinate_unit}]"
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
    Plot ship speed signals and speed calculated from GPS positions over time.

    The calculated ship speed is the distance between two consecutive GPS
    positions divided by their actual time difference. With the current data,
    this time difference is normally 10 seconds.

    Parameters
    ----------
    data : pandas.DataFrame
        Ship trajectory data containing time, gps_speed,
        shaft_speed, and thruster_speed.
    """
    if data.empty:
        raise ValueError("The input data is empty.")

    calculated_speed_kmh = calculate_speed_from_gps(data, unit="km/h")

    figure, (speed_axis, propulsion_axis) = plt.subplots(
        2,
        1,
        figsize=SPEED_FIGURE_SIZE,
        sharex=True,
    )

    speed_axis.plot(data["time"], data["gps_speed"], label="GPS speed")
    speed_axis.plot(
        data["time"],
        calculated_speed_kmh,
        label="Calculated from GPS positions",
        linewidth=2,
    )
    speed_axis.set_ylabel("Ship speed [km/h]")
    speed_axis.set_title("Ship Speed Over Time")
    speed_axis.grid(True)
    speed_axis.legend()

    propulsion_axis.plot(data["time"], data["shaft_speed"], label="Shaft speed")
    propulsion_axis.plot(data["time"], data["thruster_speed"], label="Thruster speed")
    propulsion_axis.set_xlabel("Time")
    propulsion_axis.set_ylabel("Propulsion speed")
    propulsion_axis.grid(True)
    propulsion_axis.legend()

    figure.tight_layout()
    plt.show()
