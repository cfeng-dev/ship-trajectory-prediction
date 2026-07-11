"""
@file cli.py
@description Command-line entry points for inspecting and plotting ship data.
@date Created on: 09.07.2026
@author C.Feng
"""

from pathlib import Path

from ship_trajectory_prediction.trajectory.io import (
    print_ship_data_summary,
    read_ship_data,
)
from ship_trajectory_prediction.trajectory.plotting import (
    plot_ship_speeds,
    plot_ship_trajectory,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
TRAJECTORY_COORDINATE_UNIT = "km"  # "m", "km", or "gps"


def plot_main():
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
    plot_ship_trajectory(
        ship_data,
        coordinate_unit=TRAJECTORY_COORDINATE_UNIT,
    )
    plot_ship_speeds(ship_data)


def read_main():
    """Load the real ship trajectory data and print basic information."""
    ship_data = read_ship_data(DATA_FILE)

    print(ship_data.head())
    print()
    print_ship_data_summary(ship_data)


if __name__ == "__main__":
    plot_main()
