"""
@file simulate_trajectory.py
@description Simulates simple 2D ship trajectories and saves synthetic data with observation noise.
@date Created on: 03.07.2026
@author C.Feng
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory for simulated data
DATA_DIR = PROJECT_ROOT / "data" / "simulated"


def simulate_straight_trajectory(t, x0, y0, v, theta):
    """
    Simulate a straight ship trajectory.

    Parameters
    ----------
    t : np.ndarray
        Time points.
    x0 : float
        Initial x-position.
    y0 : float
        Initial y-position.
    v : float
        Constant speed.
    theta : float
        Heading angle in radians.

    Returns
    -------
    x : np.ndarray
        Simulated x-positions.
    y : np.ndarray
        Simulated y-positions.
    """
    x = x0 + v * t * np.cos(theta)
    y = y0 + v * t * np.sin(theta)

    return x, y


def simulate_curved_trajectory(t, x0, y0, v, radius):
    """
    Simulate a curved ship trajectory using circular motion.

    Parameters
    ----------
    t : np.ndarray
        Time points.
    x0 : float
        Initial x-position.
    y0 : float
        Initial y-position.
    v : float
        Constant speed.
    radius : float
        Turning radius of the ship.

    Returns
    -------
    x : np.ndarray
        Simulated x-positions.
    y : np.ndarray
        Simulated y-positions.
    """
    omega = v / radius  # Angular velocity

    x = x0 + radius * np.sin(omega * t)
    y = y0 + radius * (1 - np.cos(omega * t))

    return x, y


def add_observation_noise(x, y, sigma, random_seed=None):
    """
    Add Gaussian observation noise to simulated trajectory positions.

    Parameters
    ----------
    x : np.ndarray
        True x-positions.
    y : np.ndarray
        True y-positions.
    sigma : float
        Standard deviation of the Gaussian observation noise.
    random_seed : int, optional
        Random seed for reproducible results.

    Returns
    -------
    x_obs : np.ndarray
        Noisy observed x-positions.
    y_obs : np.ndarray
        Noisy observed y-positions.
    """
    rng = np.random.default_rng(random_seed)

    x_obs = x + rng.normal(0, sigma, size=len(x))
    y_obs = y + rng.normal(0, sigma, size=len(y))

    return x_obs, y_obs


def create_trajectory_dataframe(
    t,
    x_true,
    y_true,
    x_obs,
    y_obs,
    trajectory_type,
    v,
    sigma,
    theta=None,
    radius=None,
):
    """
    Create a DataFrame containing simulated trajectory data.

    Parameters
    ----------
    t : np.ndarray
        Time points.
    x_true : np.ndarray
        True x-positions without noise.
    y_true : np.ndarray
        True y-positions without noise.
    x_obs : np.ndarray
        Observed x-positions with noise.
    y_obs : np.ndarray
        Observed y-positions with noise.
    trajectory_type : str
        Type of trajectory, e.g. "straight" or "curved".
    v : float
        Constant speed.
    sigma : float
        Observation noise standard deviation.
    theta : float, optional
        Heading angle in radians.
    radius : float, optional
        Turning radius.

    Returns
    -------
    df : pd.DataFrame
        Simulated trajectory data.
    """
    df = pd.DataFrame(
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

    return df


def save_trajectory_data(df, filename):
    """
    Save simulated trajectory data as a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Simulated trajectory data.
    filename : str
        Name of the output CSV file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DATA_DIR / filename
    df.to_csv(output_path, index=False)

    print(f"Saved simulated data to: {output_path}")


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

    # True trajectories without noise
    plt.plot(x_straight, y_straight, label="True straight trajectory")
    plt.plot(x_curve, y_curve, label="True curved trajectory")

    # Noisy observations
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

    # Start position
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
