"""Explore and visualize recorded ship trajectory data."""

from pathlib import Path

from ship_trajectory_prediction.trajectory.io import (
    print_ship_data_summary,
    read_ship_data,
)
from ship_trajectory_prediction.trajectory.plotting import (
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

# Data selection
RUN_ID = 1
START_TIME = None
END_TIME = None
TRAJECTORY_COORDINATE_UNIT = "km"  # "m", "km", or "gps"


def main() -> None:
    """Load recorded ship data and create exploratory plots."""
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


if __name__ == "__main__":
    main()
