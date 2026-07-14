"""Plot a saved simulated ship trajectory."""

from pathlib import Path

from ship_trajectory_prediction.trajectory.io import (
    read_ship_data,
    resample_trajectory_data,
)
from ship_trajectory_prediction.trajectory.plotting import plot_ship_trajectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "simulated" / "example_simulated_trajectory.csv"

RUN_ID = 0
START_TIME = None
END_TIME = None
RESAMPLE_INTERVAL_SECONDS = 1
TRAJECTORY_COORDINATE_UNIT = "m"  # "m", "km", or "gps"
ARROW_STEP = 3


def main() -> None:
    """Load and plot one saved simulated trajectory."""
    simulated_data = read_ship_data(
        DATA_FILE,
        run_id=RUN_ID,
        start_time=START_TIME,
        end_time=END_TIME,
    )
    simulated_data = resample_trajectory_data(
        simulated_data,
        interval_seconds=RESAMPLE_INTERVAL_SECONDS,
    )
    plot_ship_trajectory(
        simulated_data,
        arrow_step=ARROW_STEP,
        coordinate_unit=TRAJECTORY_COORDINATE_UNIT,
    )


if __name__ == "__main__":
    main()
