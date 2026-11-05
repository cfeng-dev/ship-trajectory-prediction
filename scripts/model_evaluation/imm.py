"""Evaluate local Python BR24 IMM forecasts across rolling radar origins."""

import bayestraj.validation.imm_workflow as evaluation
from bayestraj.paths import project_path

IMM_PYTHON = project_path("external/IMM/br24_imm.py")
DATA_FILE = project_path("data/processed/BR24Dataset20131107_with_reference.csv")
OUTPUT_CSV = project_path("results/evaluation/imm.csv")

# Match scripts/model_evaluation/bayesian_ctrv.py for a direct comparison.
PREDICTION_COUNT = 3
MAX_WINDOWS = None
SHOW_ROLLING_DIAGNOSTICS_PLOT = True


def main(argv=None):
    """Run rolling evaluation using this script's header settings."""
    return evaluation.run_evaluation(
        data_file=DATA_FILE,
        imm_python=IMM_PYTHON,
        output_csv=OUTPUT_CSV,
        prediction_count=PREDICTION_COUNT,
        max_windows=MAX_WINDOWS,
        show_rolling_diagnostics_plot=SHOW_ROLLING_DIAGNOSTICS_PLOT,
        description=__doc__,
        argv=argv,
    )


if __name__ == "__main__":
    main()
