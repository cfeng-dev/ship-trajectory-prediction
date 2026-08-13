"""Run one deterministic CTRV trajectory prediction."""

from ship_trajectory_prediction.evaluation.deterministic_ctrv import (
    DeterministicExperimentConfig,
)
from ship_trajectory_prediction.evaluation.deterministic_ctrv_cli import (
    parse_deterministic_ctrv_prediction_arguments,
)
from ship_trajectory_prediction.evaluation.deterministic_ctrv_prediction import (
    run_deterministic_ctrv_prediction,
)
from ship_trajectory_prediction.paths import project_path

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

EXPERIMENT = DeterministicExperimentConfig(
    run_id=1,  # Trajectory run to predict.
    start_index=0,  # First point of the selected window.
    observation_count=20,  # Position points used for state estimation.
    prediction_count=5,  # Held-out future position points.
    additional_position_noise_std_m=2.0,  # Per x/y axis [m]; 0 disables.
    position_noise_seed=2026,  # Reproduces the added position noise.
)
SPEED_ESTIMATION_POINTS = 5
HEADING_ESTIMATION_SEGMENTS = 5


def main(argv=None):
    """Run the configured deterministic CTRV experiment."""
    arguments = parse_deterministic_ctrv_prediction_arguments(
        description=__doc__,
        experiment=EXPERIMENT,
        argv=argv,
    )
    return run_deterministic_ctrv_prediction(
        data_file=DATA_FILE,
        experiment=EXPERIMENT,
        speed_estimation_points=SPEED_ESTIMATION_POINTS,
        heading_estimation_segments=HEADING_ESTIMATION_SEGMENTS,
        position_noise_std_m=arguments.position_noise_std_m,
        position_noise_seed=arguments.position_noise_seed,
        show_plot=not arguments.no_plot,
    )


if __name__ == "__main__":
    main()
