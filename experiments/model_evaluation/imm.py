"""Evaluate local Python BR24 IMM forecasts across rolling radar origins."""

import bayestraj.validation.cli as cli
import bayestraj.validation.imm_workflow as workflow
import bayestraj.validation.runtime_plotting as runtime_plotting
from bayestraj.paths import project_path

IMM_PYTHON = project_path("external/IMM/br24_imm.py")
DATA_FILE = project_path("data/processed/BR24Dataset20131107_with_reference.csv")
OUTPUT_CSV = project_path("results/imm_evaluation.csv")

# Match experiments/model_evaluation/bayesian_ctrv.py for a direct comparison.
INITIAL_OBSERVATION_COUNT = 2
PREDICTION_COUNT = 3
STRIDE = 1  # Forecast-origin step: one new forecast per radar cycle.
MAX_WINDOWS = None
CREDIBLE_INTERVAL = 0.9
GAUSSIAN_PLOT_DRAWS = 1_000
PLOT_SEED = 42
SHOW_ROLLING_PLOT = True
SHOW_ROLLING_DIAGNOSTICS_PLOT = True


def main(argv=None):
    """Run rolling multi-step forecasts from the optional local Python IMM."""
    options = cli.parse_imm_evaluation_arguments(
        description=__doc__,
        observation_count=INITIAL_OBSERVATION_COUNT,
        prediction_count=PREDICTION_COUNT,
        stride=STRIDE,
        max_windows=MAX_WINDOWS,
        show_plot=SHOW_ROLLING_PLOT,
        argv=argv,
    )
    predictions, summary = workflow.run_imm_evaluation(
        data_file=DATA_FILE,
        imm_python=IMM_PYTHON,
        options=options,
        credible_interval=CREDIBLE_INTERVAL,
        gaussian_plot_draws=GAUSSIAN_PLOT_DRAWS,
        plot_seed=PLOT_SEED,
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved per-forecast IMM metrics: {OUTPUT_CSV}")
    if options.show_plot and SHOW_ROLLING_DIAGNOSTICS_PLOT:
        runtime_plotting.plot_imm_rolling_diagnostics(predictions)
    return predictions, summary


if __name__ == "__main__":
    main()
