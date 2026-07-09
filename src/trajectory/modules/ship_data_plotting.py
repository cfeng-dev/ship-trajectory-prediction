"""
@file ship_data_plotting.py
@description Provides helper functions to plot real ship trajectory data.
@date Created on: 09.07.2026
@author C.Feng
"""

import matplotlib.pyplot as plt


def plot_ship_trajectory(data, arrow_step=20):
    """
    Plot the ship trajectory using GPS longitude and latitude.

    The plot shows:
    - the trajectory line
    - start point
    - end point
    - direction arrows along the path

    Parameters
    ----------
    data : pandas.DataFrame
        Ship trajectory data containing gps_longitude and gps_latitude.
    arrow_step : int, optional
        Distance between direction arrows in number of data points.
    """
    if data.empty:
        raise ValueError("The input data is empty.")

    longitude = data["gps_longitude"].to_numpy()
    latitude = data["gps_latitude"].to_numpy()

    plt.figure(figsize=(8, 6))

    # Trajectory line
    plt.plot(
        longitude,
        latitude,
        color="tab:blue",
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="Ship trajectory",
    )

    # Start point
    plt.scatter(
        longitude[0],
        latitude[0],
        s=45,
        color="black",
        marker="o",
        label="Start",
        zorder=4,
    )

    # End point
    plt.scatter(
        longitude[-1],
        latitude[-1],
        s=110,
        color="green",
        alpha=0.45,
        marker="X",
        label="End",
        zorder=5,
    )

    # Direction arrows
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
                "color": "darkorange",
                "linewidth": 1.5,
                "mutation_scale": 14,
            },
        )

    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.title("Ship Trajectory with Direction")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")

    ax = plt.gca()
    ax.ticklabel_format(useOffset=False, style="plain")

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

    plt.figure(figsize=(10, 6))
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
