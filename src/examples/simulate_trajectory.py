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
    # Time points
    t = np.linspace(0, 20, 100)

    # Common parameters
    x0 = 0.0
    y0 = 0.0
    v = 1.0

    # Observation noise parameter
    sigma = 0.2

    # ==================================================
    # Straight trajectory
    # ==================================================
    theta = np.deg2rad(20)

    x_straight, y_straight = simulate_straight_trajectory(
        t=t,
        x0=x0,
        y0=y0,
        v=v,
        theta=theta,
    )

    x_straight_obs, y_straight_obs = add_observation_noise(
        x=x_straight,
        y=y_straight,
        sigma=sigma,
        random_seed=42,
    )

    straight_df = create_trajectory_dataframe(
        t=t,
        x_true=x_straight,
        y_true=y_straight,
        x_obs=x_straight_obs,
        y_obs=y_straight_obs,
        trajectory_type="straight",
        v=v,
        sigma=sigma,
        theta=theta,
        radius=None,
    )

    save_trajectory_data(
        df=straight_df,
        filename="straight_trajectory.csv",
    )

    # ==================================================
    # Curved trajectory
    # ==================================================
    radius = 10.0

    x_curve, y_curve = simulate_curved_trajectory(
        t=t,
        x0=x0,
        y0=y0,
        v=v,
        radius=radius,
    )

    x_curve_obs, y_curve_obs = add_observation_noise(
        x=x_curve,
        y=y_curve,
        sigma=sigma,
        random_seed=43,
    )

    curve_df = create_trajectory_dataframe(
        t=t,
        x_true=x_curve,
        y_true=y_curve,
        x_obs=x_curve_obs,
        y_obs=y_curve_obs,
        trajectory_type="curved",
        v=v,
        sigma=sigma,
        theta=None,
        radius=radius,
    )

    save_trajectory_data(
        df=curve_df,
        filename="curved_trajectory.csv",
    )

    # Save both trajectories in one file
    all_data_df = pd.concat(
        [straight_df, curve_df],
        ignore_index=True,
    )

    save_trajectory_data(
        df=all_data_df,
        filename="simulated_ship_trajectories.csv",
    )

    # ==================================================
    # Plot trajectories
    # ==================================================
    plt.figure(figsize=(8, 6))

    plt.plot(x_straight, y_straight, label="True straight trajectory")
    plt.plot(x_curve, y_curve, label="True curved trajectory")

    plt.scatter(
        x_straight_obs,
        y_straight_obs,
        s=15,
        label="Noisy straight observations",
    )
    plt.scatter(
        x_curve_obs,
        y_curve_obs,
        s=15,
        label="Noisy curved observations",
    )

    plt.scatter(x0, y0, color="black", label="Start position")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Simulated Ship Trajectories with Observation Noise")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
