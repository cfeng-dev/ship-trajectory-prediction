"""Explore empirical motion distributions for Bayesian CTRV prior design."""

from ship_trajectory_prediction.calibration.motion_priors import (
    collect_motion_prior_samples,
    plot_motion_prior_distributions,
    print_motion_prior_report,
    suggest_prior_scales,
)
from ship_trajectory_prediction.observations import read_ship_data
from ship_trajectory_prediction.paths import project_path

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

# Data selection and preprocessing
# Keep calibration runs separate from the final evaluation. ``None`` selects
# every run and should therefore only be used for a descriptive overview.
CALIBRATION_RUN_IDS = None  # One: 1; IDs: (1, 3); range: range(1, 4); all: None
MIN_COURSE_DISPLACEMENT_METERS = 1.0
MAX_TIME_GAP_SECONDS = 15.0

# Terminal report
PRINT_PER_RUN_SUMMARY = False

# Histogram data display
PLOT_CENTRAL_QUANTILE = 0.995
HISTOGRAM_MODE = "density"  # Choose "density" or "frequency".


def main():
    """Load calibration runs, report robust scales, and show prior plots."""
    data = read_ship_data(DATA_FILE)
    samples = collect_motion_prior_samples(
        data,
        run_ids=CALIBRATION_RUN_IDS,
        min_course_displacement_m=MIN_COURSE_DISPLACEMENT_METERS,
        max_time_gap_s=MAX_TIME_GAP_SECONDS,
    )
    suggestions = suggest_prior_scales(samples)
    print_motion_prior_report(
        samples,
        suggestions,
        data_file=DATA_FILE,
        print_per_run_summary=PRINT_PER_RUN_SUMMARY,
    )
    figures, axes = plot_motion_prior_distributions(
        samples,
        suggestions,
        central_quantile=PLOT_CENTRAL_QUANTILE,
        histogram_mode=HISTOGRAM_MODE,
        show_sequentially=True,
    )
    return samples, suggestions, figures, axes


if __name__ == "__main__":
    main()
