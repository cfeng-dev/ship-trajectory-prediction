"""Compare IMM radar-detection choices with one Bayesian inference method.

Only accepted detection IDs are compared; posterior track positions are not.
For a short run: --method smc --max-cycles 20.
"""

import bayestraj.inference.configuration as inference
import bayestraj.validation.br24_associations as checks
from bayestraj.paths import project_path

RAW_INPUT_CSV = project_path("data/processed/BR24Dataset20131107.csv")
IMM_TRACKED_TRAJECTORY_CSV = project_path(
    "data/processed/BR24_imm_tracked_trajectory.csv"
)
OUTPUT_DIR = project_path("results/validation")
INFERENCE_METHOD = "smc"  # "smc", "mcmc", or "vi".
OBSERVATION_COUNT = inference.DEFAULT_CTRV_ROLLING_OBSERVATION_COUNT
INFERENCE_SEED = 42
MAX_CYCLES = None


def main(argv=None):
    """Run the association check with this script's settings."""
    return checks.run_check(
        raw_csv=RAW_INPUT_CSV,
        imm_csv=IMM_TRACKED_TRAJECTORY_CSV,
        output_dir=OUTPUT_DIR,
        inference_method=INFERENCE_METHOD,
        observation_count=OBSERVATION_COUNT,
        seed=INFERENCE_SEED,
        max_cycles=MAX_CYCLES,
        description=__doc__,
        argv=argv,
    )


if __name__ == "__main__":
    main()
