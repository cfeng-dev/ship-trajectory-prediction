"""Evaluate deterministic CTRV forecasts across rolling windows."""

import bayestraj.forecasting.deterministic_ctrv as config
import bayestraj.observations.paths as paths
import bayestraj.validation.cli as cli
import bayestraj.validation.deterministic_ctrv_workflow as workflow

DATA_FILE = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

EXPERIMENT = config.DeterministicRollingExperimentConfig(
    run_id=102,  # Trajectory run to evaluate.
    window_mode="sliding",  # Fixed "sliding" or growing "expanding" history.
    observation_count=20,  # Position points used for the first estimate.
    prediction_count=3,  # Held-out future points per rolling forecast.
    additional_position_noise_std_m=5.0,  # Per x/y axis [m]; 0 disables.
    position_noise_seed=2026,  # Reproduces route-wide added position noise.
    stride=None,  # Forecast-origin step; None uses prediction_count.
)
MAX_WINDOWS = None  # Optional smoke-test limit; None evaluates every window.
SPEED_ESTIMATION_POINTS = 5  # Recent noisy positions used for the speed fit.
HEADING_ESTIMATION_SEGMENTS = 5  # Recent movements used for heading/turn rate.
SHOW_PLOT = True  # Show the route-wide rolling-evaluation plot.


def main(argv=None):
    """Run the configured deterministic CTRV rolling evaluation."""
    options = cli.parse_deterministic_ctrv_evaluation_arguments(
        description=__doc__,
        experiment=EXPERIMENT,
        speed_estimation_points=SPEED_ESTIMATION_POINTS,
        heading_estimation_segments=HEADING_ESTIMATION_SEGMENTS,
        max_windows=MAX_WINDOWS,
        show_plot=SHOW_PLOT,
        argv=argv,
    )
    return workflow.run_deterministic_ctrv_evaluation(
        data_file=DATA_FILE,
        experiment=EXPERIMENT,
        options=options,
    )


if __name__ == "__main__":
    main()
