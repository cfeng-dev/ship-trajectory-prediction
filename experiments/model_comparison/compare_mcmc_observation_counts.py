"""Read the saved MCMC sweep summary and compare observation-count settings.

Run experiments/inference_evaluation/evaluate_mcmc_observation_counts.py
first to generate the CSVs.
This script only reads and displays the MCMC results.
"""

import bayestraj.validation.comparison as comparison
from bayestraj.paths import project_path

MCMC_SUMMARY_CSV = project_path("results/bayesian_ctrv_mcmc_evaluation/summary.csv")
SHOW_POINTS = True
PLOT_LANGUAGE = "de"  # "de" or "en".
OUTPUT_DIR = None  # Optional directory for PNG copies of the displayed figures.


def main():
    """Display metric curves from completed sweeps without running inference."""
    try:
        return comparison.run_observation_count_comparison(
            mcmc_csv=MCMC_SUMMARY_CSV,
            show_points=SHOW_POINTS,
            plot_language=PLOT_LANGUAGE,
            output_dir=OUTPUT_DIR,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
