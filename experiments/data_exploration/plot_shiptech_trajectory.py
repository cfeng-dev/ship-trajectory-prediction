"""Plot the processed Shiptech trajectory."""

import pandas as pd

import bayestraj.observations.paths as paths
import bayestraj.observations.plotting as plotting

INPUT_CSV = paths.data_path("processed/ship_trajectory_run_42.csv")
TRAJECTORY_LABEL = "Shiptech-Trajektorie"
DIRECTION_ARROW_INTERVAL_SECONDS = 180.0
POSITION_NOISE_STD_M = 0.0  # Per x/y axis [m]; 0 disables.
POSITION_NOISE_SEED = 2026  # Reproduces the added position noise.


def main() -> None:
    """Plot the configured processed Shiptech trajectory."""
    trajectory_data = pd.read_csv(INPUT_CSV)
    plotting.plot_processed_trajectory(
        trajectory_data,
        trajectory_label=TRAJECTORY_LABEL,
        direction_arrow_interval_seconds=DIRECTION_ARROW_INTERVAL_SECONDS,
        position_noise_std_m=POSITION_NOISE_STD_M,
        position_noise_seed=POSITION_NOISE_SEED,
    )


if __name__ == "__main__":
    main()
