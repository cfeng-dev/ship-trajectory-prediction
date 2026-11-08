"""Summarize saved MCMC forecast CSVs and compare observation-count settings.

Run scripts/inference_evaluation/evaluate_mcmc_observation_counts.py
first to generate the CSVs.
This script computes summary.csv and plots without running inference.
"""

import bayestraj.validation.comparison as comparison
from bayestraj.paths import project_path

MCMC_RESULTS_DIR = project_path("results/inference_evaluation/bayesian_ctrv_mcmc")
SHOW_POINTS = True
METRIC_Y_SCALE = "linear"  # "linear" or "symlog" for metric curves.
PLOT_LANGUAGE = "de"  # "de" or "en".


def main():
    """Display metric curves from completed sweeps without running inference."""
    try:
        return comparison.run_observation_count_comparison(
            mcmc_dir=MCMC_RESULTS_DIR,
            show_points=SHOW_POINTS,
            metric_y_scale=METRIC_Y_SCALE,
            plot_language=PLOT_LANGUAGE,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
