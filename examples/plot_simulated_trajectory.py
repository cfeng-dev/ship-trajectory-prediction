"""Load, resample, and plot a saved simulated ship trajectory."""

from pathlib import Path

from ship_trajectory_prediction.trajectory.io import (
    read_ship_data,
    resample_trajectory_data,
)
from ship_trajectory_prediction.trajectory.plotting import plot_ship_trajectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "simulated" / "example_simulated_trajectory.csv"

# Select one simulation run and, optionally, a UTC time range from the CSV.
# Keep START_TIME and END_TIME as None to use the complete selected run.
RUN_ID = 0
START_TIME = None
END_TIME = None

# Keep the first available sample in each regular time interval. No positions
# are interpolated when an interval contains no observation.
RESAMPLE_INTERVAL_SECONDS = 1

# "m" and "km" use the first selected GPS position as the local origin;
# "gps" plots the original longitude and latitude values.
TRAJECTORY_COORDINATE_UNIT = "m"  # "m", "km", or "gps"

# Direction arrows are separated by this number of resampled data points.
# With the one-second interval above, 3 points correspond to about 3 seconds.
ARROW_STEP = 3


def main() -> None:
    """Load and plot one saved simulated trajectory."""
    # Parse timestamps and apply the configured run and time filters.
    simulated_data = read_ship_data(
        DATA_FILE,
        run_id=RUN_ID,
        start_time=START_TIME,
        end_time=END_TIME,
    )

    # Match the temporal resolution used for the public plotting example.
    simulated_data = resample_trajectory_data(
        simulated_data,
        interval_seconds=RESAMPLE_INTERVAL_SECONDS,
    )

    # The plotting helper converts GPS positions to local coordinates when
    # coordinate_unit is "m" or "km".
    plot_ship_trajectory(
        simulated_data,
        arrow_step=ARROW_STEP,
        coordinate_unit=TRAJECTORY_COORDINATE_UNIT,
    )


if __name__ == "__main__":
    main()
