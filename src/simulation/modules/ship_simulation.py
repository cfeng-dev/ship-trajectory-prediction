"""
@file ship_simulation.py
@description Provides ship trajectory simulation models and a step-based 2D ship simulator.
@date Created on: 08.07.2026
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


class ShipSimulator:
    """
    Step-based simulator for a simple 2D ship trajectory.

    The ship state is updated using a constant speed and a steering-based
    angular velocity.
    """

    def __init__(
        self,
        v=0.5,
        sigma=0.2,
        dt=0.05,
    ):
        """
        Initialize the ship simulator.

        Parameters
        ----------
        v : float
            Constant ship speed.
        sigma : float
            Observation noise standard deviation.
        dt : float
            Simulation time step in seconds.
        """
        self.v = v
        self.sigma = sigma
        self.dt = dt

        self.reset()

    def reset(self):
        """
        Reset the ship state and stored trajectory.
        """
        self.x_current = 0.0
        self.y_current = 0.0
        self.theta_current = 0.0
        self.current_time = 0.0

        self.t_all = []
        self.x_all = []
        self.y_all = []
        self.theta_all = []
        self.omega_all = []
        self.radius_all = []
        self.motor_state_all = []

    def step(self, omega, motor_running):
        """
        Perform one simulation step.

        Parameters
        ----------
        omega : float
            Angular velocity in rad/s.
        motor_running : bool
            Whether the ship motor is currently running.
        """
        if not motor_running:
            return

        # Store current state before updating.
        self.t_all.append(self.current_time)
        self.x_all.append(self.x_current)
        self.y_all.append(self.y_current)
        self.theta_all.append(self.theta_current)
        self.omega_all.append(omega)
        self.motor_state_all.append(motor_running)

        # Turning radius R = v / omega.
        # For straight motion, omega is almost zero, so radius is infinite.
        if abs(omega) < 1e-8:
            radius = np.inf
        else:
            radius = self.v / omega

        self.radius_all.append(radius)

        # Update ship position and heading.
        self.x_current += self.v * np.cos(self.theta_current) * self.dt
        self.y_current += self.v * np.sin(self.theta_current) * self.dt
        self.theta_current += omega * self.dt
        self.current_time += self.dt

    def has_data(self):
        """
        Check whether trajectory data has been generated.

        Returns
        -------
        bool
            True if trajectory data exists, otherwise False.
        """
        return len(self.x_all) > 0
