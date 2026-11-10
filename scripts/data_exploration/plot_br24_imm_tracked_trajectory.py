"""Plot the IMM-tracked BR24 trajectory."""

import pandas as pd

import bayestraj.observations.plotting as plotting
from bayestraj.paths import project_path

INPUT_CSV = project_path("data/processed/BR24_imm_tracked_trajectory.csv")
TRAJECTORY_LABEL = "IMM-getrackte BR24-Zieltrajektorie"
SHOW_POINTS = True  # Show point markers along the trajectory.
DIRECTION_ARROW_INTERVAL_SECONDS = 180.0


def main() -> None:
    """Plot the configured IMM-tracked BR24 trajectory."""
    trajectory_data = pd.read_csv(INPUT_CSV)
    plotting.plot_processed_trajectory(
        trajectory_data,
        trajectory_label=TRAJECTORY_LABEL,
        show_points=SHOW_POINTS,
        direction_arrow_interval_seconds=DIRECTION_ARROW_INTERVAL_SECONDS,
    )


if __name__ == "__main__":
    main()
