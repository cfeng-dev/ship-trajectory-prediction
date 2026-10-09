"""Plot the BR24 radar trajectory selected by IMM association."""

import pandas as pd

import bayestraj.observations.paths as paths
import bayestraj.observations.plotting as plotting

INPUT_CSV = paths.data_path("processed/BR24_target_radar_trajectory.csv")
TRAJECTORY_LABEL = "BR24-Radar-Zieltrajektorie"
DIRECTION_ARROW_INTERVAL_SECONDS = 180.0


def main() -> None:
    """Plot the configured associated BR24 radar trajectory."""
    trajectory_data = pd.read_csv(INPUT_CSV)
    plotting.plot_processed_trajectory(
        trajectory_data,
        trajectory_label=TRAJECTORY_LABEL,
        direction_arrow_interval_seconds=DIRECTION_ARROW_INTERVAL_SECONDS,
    )


if __name__ == "__main__":
    main()
