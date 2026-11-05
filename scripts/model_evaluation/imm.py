"""Evaluate local Python BR24 IMM forecasts across rolling radar origins."""

import bayestraj.validation.cli as cli
import bayestraj.validation.imm_workflow as workflow
import bayestraj.validation.runtime_plotting as runtime_plotting
from bayestraj.paths import project_path

IMM_PYTHON = project_path("external/IMM/br24_imm.py")
DATA_FILE = project_path("data/processed/BR24Dataset20131107_with_reference.csv")
OUTPUT_CSV = project_path("results/evaluation/imm.csv")

# Match scripts/model_evaluation/bayesian_ctrv.py for a direct comparison.
PREDICTION_COUNT = 3
MAX_WINDOWS = None
SHOW_ROLLING_DIAGNOSTICS_PLOT = True


def main(argv=None):
    """Run rolling multi-step forecasts from the optional local Python IMM."""
    options = cli.parse_imm_evaluation_arguments(
        description=__doc__,
        prediction_count=PREDICTION_COUNT,
        max_windows=MAX_WINDOWS,
        argv=argv,
    )
    predictions, summary = workflow.run_imm_evaluation(
        data_file=DATA_FILE,
        imm_python=IMM_PYTHON,
        options=options,
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved per-forecast IMM metrics: {OUTPUT_CSV}")
    if options.show_plot and SHOW_ROLLING_DIAGNOSTICS_PLOT:
        runtime_plotting.plot_imm_rolling_diagnostics(predictions)
    return predictions, summary


if __name__ == "__main__":
    main()
