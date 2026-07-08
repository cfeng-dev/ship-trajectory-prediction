"""
@file ship_trajectory_models.py
@description Provides simple 2D ship trajectory simulation utilities.
@date Created on: 03.07.2026
@author C.Feng
"""

import numpy as np


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


def simulate_curved_trajectory(t, x0, y0, v, radius, theta):
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
    theta : float
        Initial heading angle in radians.

    Returns
    -------
    x : np.ndarray
        Simulated x-positions.
    y : np.ndarray
        Simulated y-positions.
    """
    omega = v / radius

    x = x0 + radius * (np.sin(theta + omega * t) - np.sin(theta))
    y = y0 - radius * (np.cos(theta + omega * t) - np.cos(theta))

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
