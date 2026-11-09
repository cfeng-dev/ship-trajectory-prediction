"""Compare saved Bayesian CTRV inference methods and IMM without running models.

Accuracy, density scores and runtime are shown per common forecast origin,
with optional mean lines. Coverage bars show the final percentage of all
matched forecast positions inside the joint 50%/90% regions, with lines marking
the nominal coverage.
"""

import bayestraj.validation.comparison as comparison
from bayestraj.paths import project_path

# Set a CSV path to None to exclude that model; select at least two models.
IMM_CSV = project_path("results/evaluation/imm.csv")
BAYESIAN_SMC_CSV = project_path("results/evaluation/bayesian_ctrv_smc.csv")
BAYESIAN_SEQ_VI_CSV = project_path("results/evaluation/bayesian_ctrv_seq_vi.csv")
BAYESIAN_VI_CSV = project_path("results/evaluation/bayesian_ctrv_vi_05.csv")
BAYESIAN_MCMC_CSV = project_path("results/evaluation/bayesian_ctrv_mcmc_03.csv")

SHOW_POINTS = False  # Mark forecast origins on metric curves.
SHOW_MEAN_LINES = False  # Show mean metric values as dashed lines.
SHOW_SUMMARY_TABLE = True  # Display all summary metrics in an additional table.
METRIC_Y_SCALE = "linear"  # "linear" or "symlog" for metric curves.
PLOT_LANGUAGE = "de"  # "de" or "en" for plot labels.


def main():
    """Compare the configured CSV files and display the metric plots."""
    try:
        models = comparison.read_model_comparison_csvs(
            {
                "IMM_CSV": IMM_CSV,
                "BAYESIAN_SMC_CSV": BAYESIAN_SMC_CSV,
                "BAYESIAN_SEQ_VI_CSV": BAYESIAN_SEQ_VI_CSV,
                "BAYESIAN_VI_CSV": BAYESIAN_VI_CSV,
                "BAYESIAN_MCMC_CSV": BAYESIAN_MCMC_CSV,
            }
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    return comparison.run_model_comparison(
        models,
        show_points=SHOW_POINTS,
        show_mean_lines=SHOW_MEAN_LINES,
        show_summary_table=SHOW_SUMMARY_TABLE,
        metric_y_scale=METRIC_Y_SCALE,
        plot_language=PLOT_LANGUAGE,
    )


if __name__ == "__main__":
    main()
