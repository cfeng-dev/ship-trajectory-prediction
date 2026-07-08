"""
@file ship_io.py
@description Handles DataFrame creation and CSV export for simulated trajectory data.
@date Created on: 06.07.2026
@author C.Feng
"""

from pathlib import Path

import numpy as np
import pandas as pd

from modules.ship_trajectory_models import add_observation_noise

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]

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

    If only a filename is given, the file is saved in data/simulated.
    If a full path is given, the file is saved at that location.

    Parameters
    ----------
    df : pd.DataFrame
        Simulated trajectory data.
    filename : str or pathlib.Path
        Output CSV filename or full output path.

    Returns
    -------
    output_path : pathlib.Path
        Path to the saved CSV file.
    """
    output_path = Path(filename)

    # If only a filename is given, save it in the default data directory.
    if not output_path.is_absolute():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DATA_DIR / output_path
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV without index and round floating-point values to three decimal places.
    df.to_csv(output_path, index=False, float_format="%.3f")

    print(f"Saved simulated data to: {output_path}")

    return output_path
