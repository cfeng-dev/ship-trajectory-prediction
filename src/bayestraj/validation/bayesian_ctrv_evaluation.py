"""Execute rolling Bayesian CTRV evaluations using script-owned settings."""

from collections.abc import Sequence
from pathlib import Path

import bayestraj.forecasting.bayesian_ctrv as ctrv_config
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.validation.bayesian_ctrv_workflow as workflow
import bayestraj.validation.cli as cli
import bayestraj.validation.runtime_plotting as runtime_plotting


def run_evaluation(
    *,
    data_file: str | Path,
    output_csv: str | Path,
    experiment: ctrv_config.RollingExperimentConfig,
    priors: bayesian_model.BayesianCTRVPriors,
    fullrank_grad_samples,
    credible_interval,
    max_windows,
    plot_each_window,
    sample_trajectories_per_forecast,
    show_time_labels,
    show_rolling_diagnostics_plot,
    description: str | None = None,
    argv: Sequence[str] | None = None,
):
    """Evaluate and export forecasts using settings supplied by the entrypoint.

    The optional `{inference_method}` placeholder in `output_csv` follows the
    actual selection, including command-line overrides. Literal paths are also
    accepted for custom output filenames.
    """
    options = cli.parse_bayesian_ctrv_evaluation_arguments(
        description=description,
        experiment=experiment,
        priors=priors,
        max_windows=max_windows,
        plot_each_window=plot_each_window,
        argv=argv,
    )
    predictions, summary = workflow.run_bayesian_ctrv_evaluation(
        data_file=data_file,
        experiment=experiment,
        priors=priors,
        fullrank_grad_samples=fullrank_grad_samples,
        credible_interval=credible_interval,
        sample_trajectories_per_forecast=sample_trajectories_per_forecast,
        options=options,
        show_time_labels=show_time_labels,
    )
    output_csv = Path(
        str(output_csv).replace("{inference_method}", options.inference_method)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False)
    print(f"\nSaved per-forecast Bayesian CTRV metrics: {output_csv}")
    if show_rolling_diagnostics_plot:
        runtime_plotting.plot_bayesian_ctrv_rolling_diagnostics(
            predictions,
            inference_selection=options.inference_method,
        )
    return predictions, summary
