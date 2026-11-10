"""Plot the trajectory tracked online from BR24 radar detections."""

import pandas as pd

import bayestraj.observations.plotting as plotting
from bayestraj.inference.br24_radar_tracker import BR24BayesianRadarTracker
from bayestraj.inference.ctrv_smc import SequentialMonteCarloCTRVConfig
from bayestraj.models.bayesian_ctrv import BayesianCTRVPriors
from bayestraj.observations.br24 import read_br24_tracking_input
from bayestraj.paths import project_path

INPUT_CSV = project_path("data/processed/BR24Dataset20131107_with_reference.csv")
TRAJECTORY_LABEL = "Bayesian-CTRV-getrackte BR24-Zieltrajektorie"
DIRECTION_ARROW_STEP = 100  # Point interval; None disables direction arrows.
PRIORS = BayesianCTRVPriors()
SMC_CONFIG = SequentialMonteCarloCTRVConfig()
INFERENCE_SEED = 42


def main() -> None:
    """Track the configured radar cycles and plot the posterior positions."""
    cycles = read_br24_tracking_input(INPUT_CSV)
    initial_cycle = cycles[0]
    if not initial_cycle.detections:
        raise ValueError(
            "The earliest BR24 radar cycle has no detection to initialize."
        )
    tracker = BR24BayesianRadarTracker.initialize(
        initial_cycle,
        initial_detection_id=initial_cycle.detections[0].detection_id,
        priors=PRIORS,
        config=SMC_CONFIG,
        seed=INFERENCE_SEED,
    )
    positions = [tracker.estimated_position]
    for cycle in cycles[1:]:
        tracker.process_cycle(cycle)
        positions.append(tracker.estimated_position)

    trajectory = pd.DataFrame(positions, columns=["x", "y"])
    trajectory.insert(0, "time", [cycle.time_seconds for cycle in cycles])
    plotting.plot_processed_trajectory(
        trajectory,
        trajectory_label=TRAJECTORY_LABEL,
        direction_arrow_step=DIRECTION_ARROW_STEP,
    )


if __name__ == "__main__":
    main()
