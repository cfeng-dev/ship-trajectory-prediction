"""Compare all forecasts saved in the numbered forecast_step_* directories."""

import bayestraj.validation.prediction_horizons as comparison
from bayestraj.paths import project_path

INPUT_DIR = project_path("results/forecast_evaluation")
SHOW_POINTS = True
SHOW_SUMMARY_TABLE = True
METRIC_Y_SCALE = "linear"  # "linear" or "symlog".
PLOT_LANGUAGE = "de"  # "de" or "en".


def main():
    """Create forecast_steps_summary.csv and plots without running inference."""
    try:
        return comparison.run_prediction_horizon_comparison(
            INPUT_DIR,
            show_points=SHOW_POINTS,
            show_summary_table=SHOW_SUMMARY_TABLE,
            metric_y_scale=METRIC_Y_SCALE,
            plot_language=PLOT_LANGUAGE,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
