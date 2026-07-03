"""
@file simulate_trajectory.py
@description Simulates simple ship trajectories: straight motion and curved motion with observation noise.
@date Created on: 03.07.2026
@author C.Feng
"""

import numpy as np
import matplotlib.pyplot as plt


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


def add_observation_noise(x, y, noise_std, random_seed=None):
    """
    Add Gaussian observation noise to simulated trajectory positions.

    Parameters
    ----------
    x : np.ndarray
        True x-positions.
    y : np.ndarray
        True y-positions.
    noise_std : float
        Standard deviation of the Gaussian noise.
    random_seed : int, optional
        Random seed for reproducible results.

    Returns
    -------
    x_noisy : np.ndarray
        Noisy observed x-positions.
    y_noisy : np.ndarray
        Noisy observed y-positions.
    """
    rng = np.random.default_rng(random_seed)

    x_noisy = x + rng.normal(0, noise_std, size=len(x))
    y_noisy = y + rng.normal(0, noise_std, size=len(y))

    return x_noisy, y_noisy


def main():
    # Time points
    t = np.linspace(0, 20, 100)

    # Common parameters
    x0 = 0.0
    y0 = 0.0
    v = 1.0

    # Observation noise parameter
    noise_std = 0.2

    # Straight trajectory
    theta = np.deg2rad(20)
    x_straight, y_straight = simulate_straight_trajectory(
        t=t,
        x0=x0,
        y0=y0,
        v=v,
        theta=theta,
    )

    # Add observation noise to straight trajectory
    x_straight_noisy, y_straight_noisy = add_observation_noise(
        x=x_straight,
        y=y_straight,
        noise_std=noise_std,
        random_seed=42,
    )

    # Curved trajectory
    radius = 10.0
    x_curve, y_curve = simulate_curved_trajectory(
        t=t,
        x0=x0,
        y0=y0,
        v=v,
        radius=radius,
    )

    # Add observation noise to curved trajectory
    x_curve_noisy, y_curve_noisy = add_observation_noise(
        x=x_curve,
        y=y_curve,
        noise_std=noise_std,
        random_seed=43,
    )

    # Plot trajectories
    plt.figure(figsize=(8, 6))

    # True trajectories without noise
    plt.plot(x_straight, y_straight, label="True straight trajectory")
    plt.plot(x_curve, y_curve, label="True curved trajectory")

    # Noisy observations
    plt.scatter(
        x_straight_noisy,
        y_straight_noisy,
        s=15,
        label="Noisy straight observations",
    )
    plt.scatter(
        x_curve_noisy,
        y_curve_noisy,
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
