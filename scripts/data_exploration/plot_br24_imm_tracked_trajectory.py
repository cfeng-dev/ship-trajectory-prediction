"""Plot BR24 IMM positions and the ground truth stored alongside them."""

import bayestraj.observations.plotting as plotting
from bayestraj.paths import project_path

INPUT_CSV = project_path("data/processed/BR24_imm_tracked_trajectory.csv")
SHOW_IMM_TRACK = True
SHOW_REFERENCE = True
TRAJECTORY_LABEL = "IMM-Track"
DIRECTION_ARROW_STEP = 100  # Point interval; None disables direction arrows.


def main() -> None:
    """Plot the BR24 trajectories selected in the header."""
    plotting.plot_br24_imm_trajectory(
        INPUT_CSV,
        show_imm_track=SHOW_IMM_TRACK,
        show_reference=SHOW_REFERENCE,
        trajectory_label=TRAJECTORY_LABEL,
        direction_arrow_step=DIRECTION_ARROW_STEP,
    )


if __name__ == "__main__":
    main()
