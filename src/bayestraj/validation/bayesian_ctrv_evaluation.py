"""Execute rolling Bayesian CTRV evaluations using script-owned settings."""

from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import bayestraj.forecasting.bayesian_ctrv as ctrv_config
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.validation.bayesian_ctrv_workflow as workflow
import bayestraj.validation.cli as cli
import bayestraj.validation.runtime_plotting as runtime_plotting


def run_evaluation(
    *,
    data_file: str | Path,
    output_csv: str | Path | None,
    experiment: ctrv_config.RollingExperimentConfig,
    priors: bayesian_model.BayesianCTRVPriors,
    fullrank_grad_samples,
    credible_interval,
    max_windows,
    plot_each_window,
    sample_trajectories_per_forecast,
    show_time_labels,
    show_rolling_diagnostics_plot,
    vi_config=None,
    sequential_vi_config=None,
    description: str | None = None,
    argv: Sequence[str] | None = None,
):
    """Evaluate and export forecasts using settings supplied by the entrypoint.

    The optional `{inference_method}` placeholder in `output_csv` follows the
    actual selection, including command-line overrides. Sequential VI uses
    `seq_vi` in filenames. Literal paths are also accepted for custom output
    filenames. None or an empty string skips export with a terminal warning.
    Export failures do not prevent diagnostic plots.
    """
    options = cli.parse_bayesian_ctrv_evaluation_arguments(
        description=description,
        experiment=experiment,
        priors=priors,
        vi_config=vi_config,
        max_windows=max_windows,
        plot_each_window=plot_each_window,
        argv=argv,
    )
    output_path = _csv_output_path(output_csv, options.inference_method)
    predictions, summary = workflow.run_bayesian_ctrv_evaluation(
        data_file=data_file,
        experiment=experiment,
        priors=priors,
        vi_config=vi_config,
        sequential_vi_config=sequential_vi_config,
        fullrank_grad_samples=fullrank_grad_samples,
        credible_interval=credible_interval,
        sample_trajectories_per_forecast=sample_trajectories_per_forecast,
        options=options,
        show_time_labels=show_time_labels,
    )
    if output_path is not None:
        _export_csv(predictions, output_path)
    if show_rolling_diagnostics_plot:
        runtime_plotting.plot_bayesian_ctrv_rolling_diagnostics(
            predictions,
            inference_selection=options.inference_method,
        )
    return predictions, summary


def _csv_output_path(value: str | Path | None, inference_method: str) -> Path | None:
    """Resolve an optional CSV destination without creating any files."""
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError("no output file configured")
        if not isinstance(value, (str, Path)):
            raise TypeError("expected a .csv path or None")
        filename_method = (
            "seq_vi" if inference_method == "sequential_vi" else inference_method
        )
        path = Path(str(value).replace("{inference_method}", filename_method))
        if path.suffix.lower() != ".csv":
            raise ValueError("expected a .csv filename")
        if path.is_dir():
            raise ValueError("the output path names a directory")
        return path
    except (OSError, ValueError, TypeError) as error:
        print(f"Warning: OUTPUT_CSV={value!r}: {error}. CSV export skipped.")
        return None


def _export_csv(predictions, output_path: Path) -> None:
    """Replace the destination only after writing the complete CSV successfully."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=".bayestraj-csv-", dir=output_path.parent
        ) as work:
            temporary_csv = Path(work) / "metrics.csv"
            predictions.to_csv(temporary_csv, index=False)
            temporary_csv.replace(output_path)
    except (OSError, ValueError) as error:
        print(f"Warning: OUTPUT_CSV={str(output_path)!r}: {error}. CSV export skipped.")
    else:
        print(f"\nSaved per-forecast Bayesian CTRV metrics: {output_path}")
