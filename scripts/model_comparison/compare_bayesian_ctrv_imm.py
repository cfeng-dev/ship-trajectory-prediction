"""Compare all saved Bayesian CTRV and IMM forecast CSVs in one directory.

Accuracy, density scores and runtime are shown per common forecast origin,
with optional mean lines. Coverage bars show the final percentage of all
matched forecast positions inside the joint 50%/90% regions, with lines marking
the nominal coverage.
"""

import bayestraj.validation.comparison as comparison
from bayestraj.paths import project_path

INPUT_DIR = project_path("results/evaluation")  # Read forecast CSVs directly here.

SHOW_POINTS = False  # Mark forecast origins on metric curves.
SHOW_MEAN_LINES = False  # Show mean metric values as dashed lines.
SHOW_SUMMARY_TABLE = True  # Display all summary metrics in an additional table.
METRIC_Y_SCALE = "linear"  # "linear" or "symlog" for metric curves.
PLOT_LANGUAGE = "de"  # "de" or "en" for plot labels.


def main():
    """Discover forecast CSVs in INPUT_DIR and display the metric plots."""
    try:
        models = comparison.read_model_comparison_directory(INPUT_DIR)
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
