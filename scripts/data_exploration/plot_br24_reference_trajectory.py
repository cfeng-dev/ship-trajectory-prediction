"""Plot the BR24 reference trajectory over the radar observation window."""

import bayestraj.observations.plotting as plotting
from bayestraj.observations.br24 import read_br24_reference_positions
from bayestraj.paths import project_path

INPUT_CSV = project_path("data/raw/BR24_reference_trajectory.csv")
SENSOR_CSV = project_path("data/raw/BR24_sensor_state.csv")
TRAJECTORY_LABEL = "BR24-Ziel-Referenztrajektorie"
DIRECTION_ARROW_STEP = 200  # Point interval; None disables direction arrows.
POSITION_NOISE_STD_M = 0.0  # Per x/y axis [m]; 0 disables.
POSITION_NOISE_SEED = 2026  # Reproduces the added position noise.


def main() -> None:
    """Plot the original reference positions within the sensor time range."""
    trajectory_data = read_br24_reference_positions(INPUT_CSV, sensor_csv=SENSOR_CSV)
    plotting.plot_processed_trajectory(
        trajectory_data,
        trajectory_label=TRAJECTORY_LABEL,
        direction_arrow_step=DIRECTION_ARROW_STEP,
        position_noise_std_m=POSITION_NOISE_STD_M,
        position_noise_seed=POSITION_NOISE_SEED,
    )


if __name__ == "__main__":
    main()
