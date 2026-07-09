"""
@file read_ship_data.py
@description Loads real ship trajectory data and prints a short summary.
@date Created on: 09.07.2026
@author C.Feng
"""

from pathlib import Path

from modules.ship_data_io import read_ship_data, print_ship_data_summary

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)


def main():
    """
    Load the real ship trajectory data and print basic information.
    """
    ship_data = read_ship_data(DATA_FILE)

    print(ship_data.head())
    print()
    print_ship_data_summary(ship_data)


if __name__ == "__main__":
    main()
