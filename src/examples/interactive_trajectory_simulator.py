"""
@file interactive_trajectory_simulator.py
@description Lets the user interactively steer a simple 2D ship trajectory and saves the simulated data.
@date Created on: 03.07.2026
@author C.Feng
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from trajectory_models import (
    add_observation_noise,
    save_trajectory_data,
    simulate_curved_trajectory,
    simulate_straight_trajectory,
)


def append_segment(
    t_all,
    x_all,
    y_all,
    segment_types,
    t_segment,
    x_segment,
    y_segment,
    segment_type,
    current_time,
):
    """
    Append a new trajectory segment to the full trajectory.

    The first point of each new segment is removed to avoid duplicate points
    at the transition between two segments.
    """
    if len(t_all) > 0:
        t_segment = t_segment[1:]
        x_segment = x_segment[1:]
        y_segment = y_segment[1:]

    t_shifted = t_segment + current_time

    t_all.extend(t_shifted)
    x_all.extend(x_segment)
    y_all.extend(y_segment)
    segment_types.extend([segment_type] * len(t_segment))

    return t_all, x_all, y_all, segment_types


def plot_current_trajectory(x_all, y_all):
    """
    Plot the current true trajectory.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(x_all, y_all, marker="o", markersize=3, label="Current trajectory")
    plt.scatter(x_all[0], y_all[0], color="black", label="Start position")
    plt.scatter(x_all[-1], y_all[-1], color="red", label="Current position")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interactive Ship Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    # ==================================================
    # Simulation parameters
    # ==================================================
    x_current = 0.0
    y_current = 0.0
    theta_current = np.deg2rad(0)

    v = 1.0
    radius = 10.0
    sigma = 0.2

    segment_duration = 5.0
    points_per_segment = 30

    # Lists for complete trajectory
    t_all = []
    x_all = []
    y_all = []
    segment_types = []

    current_time = 0.0

    print("Interactive 2D Ship Trajectory Simulator")
    print("----------------------------------------")
    print("Commands:")
    print("  s = drive straight")
    print("  l = turn left")
    print("  r = turn right")
    print("  p = plot current trajectory")
    print("  q = finish and save CSV")
    print()

    while True:
        command = input("Enter command (s/l/r/p/q): ").strip().lower()

        if command == "q":
            break

        if command == "p":
            if len(x_all) > 0:
                plot_current_trajectory(x_all, y_all)
            else:
                print("No trajectory data yet.")
            continue

        if command not in ["s", "l", "r"]:
            print("Invalid command. Please use s, l, r, p, or q.")
            continue

        # Local time for this segment
        t_segment = np.linspace(0, segment_duration, points_per_segment)

        if command == "s":
            x_segment, y_segment = simulate_straight_trajectory(
                t=t_segment,
                x0=x_current,
                y0=y_current,
                v=v,
                theta=theta_current,
            )
            segment_type = "straight"

            # Heading does not change during straight motion
            theta_new = theta_current

        elif command == "l":
            x_segment, y_segment = simulate_curved_trajectory(
                t=t_segment,
                x0=x_current,
                y0=y_current,
                v=v,
                radius=radius,
                theta=theta_current,
            )
            segment_type = "left_curve"

            # Positive angular velocity for left turn
            omega = v / radius
            theta_new = theta_current + omega * segment_duration

        else:  # command == "r"
            x_segment, y_segment = simulate_curved_trajectory(
                t=t_segment,
                x0=x_current,
                y0=y_current,
                v=v,
                radius=-radius,
                theta=theta_current,
            )
            segment_type = "right_curve"

            # Negative angular velocity for right turn
            omega = v / (-radius)
            theta_new = theta_current + omega * segment_duration

        # Append segment to complete trajectory
        t_all, x_all, y_all, segment_types = append_segment(
            t_all=t_all,
            x_all=x_all,
            y_all=y_all,
            segment_types=segment_types,
            t_segment=t_segment,
            x_segment=x_segment,
            y_segment=y_segment,
            segment_type=segment_type,
            current_time=current_time,
        )

        # Update current state
        x_current = x_segment[-1]
        y_current = y_segment[-1]
        theta_current = theta_new
        current_time = t_all[-1]

        print(
            f"Added {segment_type}. "
            f"Current position: x={x_current:.2f}, y={y_current:.2f}, "
            f"heading={np.rad2deg(theta_current):.1f} deg"
        )

    if len(x_all) == 0:
        print("No trajectory was simulated. Nothing saved.")
        return

    # Convert lists to arrays
    t = np.array(t_all)
    x_true = np.array(x_all)
    y_true = np.array(y_all)

    # Add observation noise after the full trajectory is complete
    x_obs, y_obs = add_observation_noise(
        x=x_true,
        y=y_true,
        sigma=sigma,
        random_seed=42,
    )

    # Create DataFrame
    trajectory_df = pd.DataFrame(
        {
            "t": t,
            "x_true": x_true,
            "y_true": y_true,
            "x_obs": x_obs,
            "y_obs": y_obs,
            "trajectory_type": segment_types,
            "v": v,
            "sigma": sigma,
            "radius": radius,
        }
    )

    # Save simulated trajectory
    save_trajectory_data(
        df=trajectory_df,
        filename="interactive_ship_trajectory.csv",
    )

    # Plot final trajectory
    plt.figure(figsize=(8, 6))

    plt.plot(
        x_true,
        y_true,
        label="True trajectory",
    )

    plt.scatter(
        x_obs,
        y_obs,
        s=15,
        label="Noisy observations",
    )

    plt.scatter(
        x_true[0],
        y_true[0],
        color="black",
        label="Start position",
    )

    plt.scatter(
        x_true[-1],
        y_true[-1],
        color="red",
        label="End position",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interactive Simulated Ship Trajectory with Observation Noise")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
