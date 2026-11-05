"""Plot the BR24 reference trajectory over the radar observation window."""

import numpy as np
import pandas as pd

import bayestraj.observations.plotting as plotting
from bayestraj.paths import project_path

INPUT_CSV = project_path("data/raw/BR24_reference_trajectory.csv")
SENSOR_CSV = project_path("data/raw/BR24_sensor_state.csv")
TRAJECTORY_LABEL = "BR24-Ziel-Referenztrajektorie"
DIRECTION_ARROW_INTERVAL_SECONDS = 180.0
POSITION_NOISE_STD_M = 0.0  # Per x/y axis [m]; 0 disables.
POSITION_NOISE_SEED = 2026  # Reproduces the added position noise.


def main() -> None:
    """Plot the original reference positions within the sensor time range."""
    trajectory_data = pd.read_csv(INPUT_CSV)
    sensor_times = pd.to_numeric(pd.read_csv(SENSOR_CSV)["Time"], errors="coerce")
    if sensor_times.empty or not np.isfinite(sensor_times.to_numpy()).all():
        raise ValueError("BR24 sensor times must be finite and non-empty.")
    reference_times = pd.to_numeric(trajectory_data["time"], errors="raise")
    trajectory_data = trajectory_data.loc[
        reference_times.between(sensor_times.min(), sensor_times.max())
    ]
    plotting.plot_processed_trajectory(
        trajectory_data,
        trajectory_label=TRAJECTORY_LABEL,
        direction_arrow_interval_seconds=DIRECTION_ARROW_INTERVAL_SECONDS,
        position_noise_std_m=POSITION_NOISE_STD_M,
        position_noise_seed=POSITION_NOISE_SEED,
    )


if __name__ == "__main__":
    main()
