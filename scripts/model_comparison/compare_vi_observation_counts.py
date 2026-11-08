"""Summarize saved VI forecast CSVs and compare observation-count settings.

Run scripts/inference_evaluation/evaluate_vi_observation_counts.py
first to generate the CSVs.
This script computes summary.csv and plots without running inference.
"""

import bayestraj.validation.comparison as comparison
from bayestraj.paths import project_path

VI_RESULTS_DIR = project_path("results/inference_evaluation/bayesian_ctrv_vi")
SHOW_POINTS = True
SHOW_SUMMARY_TABLE = True  # Display all summary metrics in an additional table.
METRIC_Y_SCALE = "linear"  # "linear" or "symlog" for metric curves.
PLOT_LANGUAGE = "de"  # "de" or "en".


def main():
    """Display metric curves from the completed VI sweep without inference."""
    try:
        return comparison.run_observation_count_comparison(
            vi_dir=VI_RESULTS_DIR,
            show_points=SHOW_POINTS,
            show_summary_table=SHOW_SUMMARY_TABLE,
            metric_y_scale=METRIC_Y_SCALE,
            plot_language=PLOT_LANGUAGE,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
