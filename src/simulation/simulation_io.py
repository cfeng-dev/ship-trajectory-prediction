"""
@file simulation_io.py
@description Handles DataFrame creation and CSV export for simulated trajectory data.
@date Created on: 06.07.2026
@author C.Feng
"""

from pathlib import Path

import numpy as np
import pandas as pd

from trajectory_models import add_observation_noise

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory for simulated data
DATA_DIR = PROJECT_ROOT / "data" / "simulated"


def create_simulation_dataframe(simulator, random_seed=42):
    """
    Create a DataFrame from a completed ship simulation.

    Parameters
    ----------
    simulator : ShipSimulator
        Simulator instance containing stored trajectory data.
    random_seed : int, optional
        Random seed for reproducible observation noise.

    Returns
    -------
    trajectory_df : pd.DataFrame
        Simulated trajectory data including true and noisy observations.
    """
    x_true = np.array(simulator.x_all)
    y_true = np.array(simulator.y_all)

    x_obs, y_obs = add_observation_noise(
        x=x_true,
        y=y_true,
        sigma=simulator.sigma,
        random_seed=random_seed,
    )

    trajectory_df = pd.DataFrame(
        {
            "t": np.array(simulator.t_all),
            "x_true": x_true,
            "y_true": y_true,
            "x_obs": x_obs,
            "y_obs": y_obs,
            "theta": np.array(simulator.theta_all),
            "omega": np.array(simulator.omega_all),
            "radius": np.array(simulator.radius_all),
            "v": simulator.v,
            "sigma": simulator.sigma,
            "motor_running": np.array(simulator.motor_state_all),
        }
    )

    return trajectory_df


def save_trajectory_data(df, filename):
    """
    Save simulated trajectory data as a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Simulated trajectory data.
    filename : str
        Name of the output CSV file.

    Returns
    -------
    output_path : pathlib.Path
        Path to the saved CSV file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DATA_DIR / filename
    df.to_csv(output_path, index=False)

    print(f"Saved simulated data to: {output_path}")

    return output_path
