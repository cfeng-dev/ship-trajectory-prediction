"""Export IMM forecasts for 1, 2 and 3 future positions."""

import bayestraj.validation.prediction_horizons as evaluation
from bayestraj.paths import project_path

DATA_FILE = project_path("data/processed/BR24_imm_tracked_trajectory.csv")
IMM_PYTHON = project_path("external/IMM/br24_imm.py")
OUTPUT_DIR = project_path("results/forecast_evaluation")  # Forecast-step folders.
PREDICTION_COUNTS = (1, 2, 3)  # Number of future positions, not seconds.
MAX_WINDOWS = None
FORECAST_SEED = 42


def main(argv=None):
    """Run IMM once at the longest horizon and export each requested prefix."""
    return evaluation.run_imm_prediction_horizon_evaluation(
        data_file=DATA_FILE,
        imm_python=IMM_PYTHON,
        output_dir=OUTPUT_DIR,
        prediction_counts=PREDICTION_COUNTS,
        max_windows=MAX_WINDOWS,
        forecast_seed=FORECAST_SEED,
        argv=argv,
    )


if __name__ == "__main__":
    main()
