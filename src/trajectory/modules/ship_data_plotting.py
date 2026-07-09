"""
@file ship_data_plotting.py
@description Provides helper functions to plot real ship trajectory data.
@date Created on: 09.07.2026
@author C.Feng
"""

import matplotlib.pyplot as plt


def plot_ship_trajectory(data):
    """
    Plot the ship trajectory using GPS longitude and latitude.

    Parameters
    ----------
    data : pandas.DataFrame
        Ship trajectory data containing gps_longitude and gps_latitude.
    """
    if data.empty:
        raise ValueError("The input data is empty.")

    plt.figure(figsize=(8, 6))
    plt.plot(
        data["gps_longitude"],
        data["gps_latitude"],
        marker="o",
        linestyle="-",
        label="Ship trajectory",
    )

    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.title("Ship Trajectory")
    plt.grid(True)
    plt.legend()
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
