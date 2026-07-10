"""
@file plot_ship_data.py
@description Loads real ship trajectory data and plots the trajectory and speed signals.
@date Created on: 09.07.2026
@author C.Feng
"""

from pathlib import Path

from trajectory.modules.ship_data_io import read_ship_data, print_ship_data_summary
from trajectory.modules.ship_data_plotting import (
    plot_ship_speeds,
    plot_ship_trajectory,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

# ==================================================
# Data selection
# ==================================================
RUN_ID = 1
START_TIME = None
END_TIME = None


def main():
    """
    Load the ship data and create basic plots.
    """
    ship_data = read_ship_data(
        DATA_FILE,
        run_id=RUN_ID,
        start_time=START_TIME,
        end_time=END_TIME,
    )

    print_ship_data_summary(ship_data)
    plot_ship_trajectory(ship_data)
    plot_ship_speeds(ship_data)


if __name__ == "__main__":
    main()
