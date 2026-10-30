"""Compare saved Bayesian CTRV and IMM evaluation CSVs without running models.

Accuracy, density scores and runtime are shown per common forecast origin,
with averages across origins. Coverage bars show the final percentage of all
matched forecast positions inside the joint 50%/90% regions, with lines marking
the nominal coverage.
"""

import bayestraj.validation.comparison as comparison
from bayestraj.paths import project_path

BAYESIAN_CSV = project_path("results/bayesian_ctrv_smc_evaluation.csv")
IMM_CSV = project_path("results/imm_evaluation.csv")


def main():
    """Compare the configured CSV files and display the metric plots."""
    try:
        bayesian, imm = comparison.read_comparison_csvs(BAYESIAN_CSV, IMM_CSV)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    return comparison.run_comparison(bayesian, imm)


if __name__ == "__main__":
    main()
