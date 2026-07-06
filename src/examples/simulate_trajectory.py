"""
@file simulate_trajectory.py
@description Runs simple 2D ship trajectory simulations and saves synthetic data.
@date Created on: 03.07.2026
@author C.Feng
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trajectory_models import (
    add_observation_noise,
    create_trajectory_dataframe,
    save_trajectory_data,
    simulate_curved_trajectory,
    simulate_straight_trajectory,
)


def main():
    # ==================================================
    # Simulation parameters
    # ==================================================
    x0 = 0.0
    y0 = 0.0
    v = 1.0
    sigma = 0.2

    theta = np.deg2rad(20)
    radius = 10.0

    # Time settings
    t_straight = np.linspace(0, 10, 50)
    t_curve = np.linspace(0, 10, 50)

    # ==================================================
    # First segment: straight trajectory
    # ==================================================
    x_straight, y_straight = simulate_straight_trajectory(
        t=t_straight,
        x0=x0,
        y0=y0,
        v=v,
        theta=theta,
    )

    # Last point of straight trajectory becomes start point of curve
    x_curve_start = x_straight[-1]
    y_curve_start = y_straight[-1]

    # ==================================================
    # Second segment: curved trajectory
    # ==================================================
    x_curve, y_curve = simulate_curved_trajectory(
        t=t_curve,
        x0=x_curve_start,
        y0=y_curve_start,
        v=v,
        radius=radius,
        theta=theta,
    )

    # Remove first curve point to avoid duplicate point at transition
    x_curve = x_curve[1:]
    y_curve = y_curve[1:]
    t_curve_shifted = t_curve[1:] + t_straight[-1]

    # ==================================================
    # Combine both segments into one trajectory
    # ==================================================
    t = np.concatenate([t_straight, t_curve_shifted])
    x_true = np.concatenate([x_straight, x_curve])
    y_true = np.concatenate([y_straight, y_curve])

    # Add observation noise to the complete trajectory
    x_obs, y_obs = add_observation_noise(
        x=x_true,
        y=y_true,
        sigma=sigma,
        random_seed=42,
    )

    # Mark which part of the trajectory each point belongs to
    trajectory_type = ["straight"] * len(t_straight) + ["curved"] * len(t_curve_shifted)

    # Create DataFrame
    trajectory_df = pd.DataFrame(
        {
            "t": t,
            "x_true": x_true,
            "y_true": y_true,
            "x_obs": x_obs,
            "y_obs": y_obs,
            "trajectory_type": trajectory_type,
            "v": v,
            "sigma": sigma,
            "theta": theta,
            "radius": radius,
        }
    )

    # Save combined trajectory
    save_trajectory_data(
        df=trajectory_df,
        filename="combined_straight_curve_trajectory.csv",
    )

    # ==================================================
    # Plot trajectory
    # ==================================================
    plt.figure(figsize=(8, 6))

    plt.plot(
        x_true,
        y_true,
        label="True combined trajectory",
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
        x_curve_start,
        y_curve_start,
        color="black",
        marker="x",
        label="Transition point",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Simulated Combined Ship Trajectory with Observation Noise")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
