"""Compare saved Bayesian CTRV and IMM evaluation CSVs without running models.

Each figure shows a metric per common forecast origin, with the same horizons
and reference positions. Dashed lines and the terminal table show averages of
these per-origin metrics, not a pooled RMSE across all forecast positions.
"""

from pathlib import Path

import bayestraj.validation.comparison as comparison

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BAYESIAN_CSV = PROJECT_ROOT / "results/BR24_bayesian_ctrv_forecast_evaluation.csv"
IMM_CSV = PROJECT_ROOT / "results/BR24_imm_forecast_evaluation.csv"


def main():
    """Compare the configured CSV files and display the metric plots."""
    try:
        bayesian, imm = comparison.read_comparison_csvs(BAYESIAN_CSV, IMM_CSV)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    return comparison.run_comparison(bayesian, imm)


if __name__ == "__main__":
    main()
