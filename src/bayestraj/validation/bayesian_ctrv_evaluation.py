"""Execute rolling Bayesian CTRV evaluations using script-owned settings."""

import argparse
import dataclasses
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import bayestraj.forecasting.bayesian_ctrv as ctrv_config
import bayestraj.inference.configuration as inference
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.numeric_validation as numeric_validation
import bayestraj.paths as paths
import bayestraj.validation.bayesian_ctrv_workflow as workflow
import bayestraj.validation.cli as cli
import bayestraj.validation.rolling as rolling
import bayestraj.validation.runtime_plotting as runtime_plotting

EVALUATION_OUTPUT_CSV = "results/evaluation/bayesian_ctrv_{inference_method}.csv"
ASSOCIATIONS_OUTPUT_CSV = "results/associations/bayesian_ctrv_{inference_method}.csv"


class _DefaultOutput:
    """Distinguish source-owned defaults from explicitly disabled exports."""


_DEFAULT_OUTPUT = _DefaultOutput()

# Stable union of shared forecast fields and all method-specific diagnostics.
PREDICTION_CSV_COLUMNS = (
    "window_index",
    "window_start_index",
    "forecast_start_index",
    "target_index",
    "horizon_step",
    "time",
    "horizon_seconds",
    "x_actual",
    "y_actual",
    "x_median",
    "y_median",
    "x_lower",
    "x_upper",
    "y_lower",
    "y_upper",
    "position_error_m",
    "negative_log_predictive_density",
    "energy_score_m",
    "prediction_radius_m",
    "squared_mahalanobis_distance",
    "squared_mahalanobis_radius",
    "radial_covered",
    "joint_covered_50",
    "joint_covered_90",
    "mean_marginal_interval_width_m",
    "observation_count",
    "configured_observation_count",
    "prediction_count",
    "inference_mode",
    "inference_method",
    "model_variant",
    "converged",
    "mcmc_diagnostics_ok",
    "mcmc_rhat_max",
    "position_noise_std_m",
    "position_observation_noise_std_m",
    "position_noise_seed",
    "posterior_speed_at_origin_median_mps",
    "posterior_heading_at_origin_median_rad",
    "posterior_turn_rate_at_origin_median_rad_s",
    "posterior_sigma_position_observation_median_m",
    "posterior_sigma_speed_process_median_mps",
    "posterior_sigma_turn_rate_process_median_rad_s",
    "radar_processed_cycle_count",
    "radar_accepted_measurement_count",
    "radar_fit_status",
    "is_prior_predictive",
    "fit_status",
    "processed_cycle_count",
    "accepted_measurement_count",
    "window_cycle_count",
    "window_accepted_measurement_count",
    "posterior_fit_count",
    "last_fit_time_seconds",
    "last_fit_runtime_seconds",
    "cumulative_fit_runtime_seconds",
    "bootstrap_completed",
    "last_fit_error",
    "particle_effective_sample_size",
    "particle_resample_count",
    "particle_processed_observation_count",
    "particle_count",
    "sequential_vi_bootstrap_runtime_seconds",
    "sequential_vi_latest_update_runtime_seconds",
    "sequential_vi_cumulative_update_runtime_seconds",
    "sequential_vi_processed_observation_count",
    "forecast_origin_time",
    "forecast_origin_seconds",
    "forecast_origin_x_route",
    "forecast_origin_y_route",
    "x_actual_route",
    "y_actual_route",
    "x_median_route",
    "y_median_route",
    "fit_runtime_seconds",
    "window_runtime_seconds",
)
ASSOCIATION_CSV_COLUMNS = ("time", "cycle_counter", "detection_id", "x", "y")


def run_evaluation(
    *,
    data_file: str | Path,
    output_csv: str | Path | None | _DefaultOutput = _DEFAULT_OUTPUT,
    experiment: ctrv_config.RollingExperimentConfig,
    priors: bayesian_model.BayesianCTRVPriors,
    fullrank_grad_samples,
    credible_interval,
    max_windows,
    plot_each_window,
    sample_trajectories_per_forecast,
    show_time_labels,
    show_rolling_diagnostics_plot,
    associations_output_csv: str | Path | None | _DefaultOutput = _DEFAULT_OUTPUT,
    vi_config=None,
    sequential_vi_config=None,
    description: str | None = None,
    argv: Sequence[str] | None = None,
):
    """Evaluate and export forecasts using settings supplied by the entrypoint.

    Omitted output paths use this module's settings, resolved from the project
    root when the evaluation runs. Explicit arguments remain available.
    The optional `{inference_method}` placeholder in `output_csv` follows the
    actual selection, including command-line overrides. Sequential VI uses
    `seq_vi` in filenames. Literal paths are also accepted for custom output
    filenames. None or an empty string skips export with a terminal warning.
    associations_output_csv accepts the same placeholder for BR24 raw choices;
    None disables that export. Disabling output_csv still disables both files.
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
    use_default_output = output_csv is _DEFAULT_OUTPUT
    if use_default_output:
        output_csv = EVALUATION_OUTPUT_CSV
    output_path = _csv_output_path(output_csv, options.inference_method)
    if use_default_output and output_path is not None:
        output_path = paths.project_path(output_path)
    use_default_associations = associations_output_csv is _DEFAULT_OUTPUT
    if use_default_associations:
        associations_output_csv = ASSOCIATIONS_OUTPUT_CSV
    association_path = None
    if output_path is not None and associations_output_csv is not None:
        association_path = _csv_output_path(
            associations_output_csv,
            options.inference_method,
            setting="ASSOCIATIONS_OUTPUT_CSV",
        )
        if use_default_associations and association_path is not None:
            association_path = paths.project_path(association_path)
        if (
            association_path is not None
            and association_path.resolve() == output_path.resolve()
        ):
            print(
                "Warning: ASSOCIATIONS_OUTPUT_CSV must differ from EVALUATION_OUTPUT_CSV. Association CSV export skipped."
            )
            association_path = None
    association_records = [] if association_path is not None else None
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
        association_records=association_records,
    )
    if output_path is not None:
        export_evaluation_csv(predictions, output_path)
    if association_path is not None and association_records:
        export_associations_csv(pd.DataFrame(association_records), association_path)
    if show_rolling_diagnostics_plot:
        runtime_plotting.plot_bayesian_ctrv_rolling_diagnostics(
            predictions,
            inference_selection=options.inference_method,
        )
    return predictions, summary


def run_observation_count_comparison(
    *,
    data_file: str | Path,
    output_dir: str | Path,
    experiment: ctrv_config.RollingExperimentConfig,
    priors: bayesian_model.BayesianCTRVPriors,
    min_observation_count=2,
    max_observation_count=20,
    max_windows=None,
    vi_config=None,
    mcmc_config=None,
    credible_interval=0.9,
    fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
    sample_trajectories_per_forecast=15,
    argv: Sequence[str] | None = None,
):
    """Sweep BR24 batch-window sizes using the existing rolling evaluation.

    Score identical forecast origins from N=2, including the startup phase.
    Each training window grows until it reaches its configured maximum size.
    BR24 window sizes count cycles, not necessarily accepted measurements.
    Each CSV retains the existing actual-count columns and adds the configured
    window size. A summary row per size collects the existing aggregate metrics.
    Export only forecast metrics and the aggregate summary for this comparison.
    """
    if experiment.inference_method not in {"mcmc", "vi"}:
        raise ValueError("Observation-count comparisons support only MCMC and VI.")
    parser = argparse.ArgumentParser(
        description=run_observation_count_comparison.__doc__
    )
    parser.add_argument("--min-observations", type=int, default=min_observation_count)
    parser.add_argument("--max-observations", type=int, default=max_observation_count)
    parser.add_argument("--max-windows", type=int, default=max_windows)
    parser.add_argument("--output-dir", type=Path, default=Path(output_dir))
    arguments = parser.parse_args(argv)
    try:
        lower = numeric_validation.validate_non_negative_integer(
            "min_observation_count", arguments.min_observations
        )
        upper = numeric_validation.validate_non_negative_integer(
            "max_observation_count", arguments.max_observations
        )
        if lower < bayesian_model.MIN_OBSERVATION_COUNT or upper < lower:
            raise ValueError("Require 2 <= min-observations <= max-observations.")
        if arguments.max_windows is not None:
            limit = numeric_validation.validate_non_negative_integer(
                "max_windows", arguments.max_windows
            )
            if limit == 0:
                raise ValueError("max_windows must be positive or None.")
    except ValueError as error:
        parser.error(str(error))

    cycles = workflow.br24_observations.read_br24_tracking_input(data_file)
    windows = rolling.build_online_forecast_specs(
        len(cycles),
        initial_observation_count=workflow.EVALUATION_START_OBSERVATION_COUNT,
        prediction_count=experiment.prediction_count,
        stride=experiment.stride,
    )
    selected = tuple(window.window_index for window in windows)
    if vi_config is None:
        vi_config = inference.create_default_vi_config()
    rows = []
    for count in range(lower, upper + 1):
        current_experiment = dataclasses.replace(experiment, observation_count=count)
        options = cli.BayesianCTRVEvaluationOptions(
            observation_count=count,
            prediction_count=experiment.prediction_count,
            stride=experiment.stride,
            inference_method=experiment.inference_method,
            vi_algorithm=vi_config["algorithm"],
            turn_rate_prior_abs_rate_deg_s=priors.turn_rate_prior_abs_rate_deg_s,
            inference_seed=experiment.inference_seed,
            position_noise_std_m=experiment.position_noise_std_m,
            position_noise_seed=experiment.position_noise_seed,
            require_converged=vi_config["require_converged"],
            max_windows=arguments.max_windows,
            plot_each_window=False,
        )
        print(
            f"\n=== {experiment.inference_method.upper()}: observation_count={count} ==="
        )
        predictions, summary = workflow.run_bayesian_ctrv_evaluation(
            data_file=data_file,
            experiment=current_experiment,
            priors=priors,
            vi_config=vi_config,
            mcmc_config=mcmc_config,
            fullrank_grad_samples=fullrank_grad_samples,
            credible_interval=credible_interval,
            sample_trajectories_per_forecast=sample_trajectories_per_forecast,
            options=options,
            selected_window_indices=selected,
            show_plot=False,
        )
        destination = arguments.output_dir / (
            f"bayesian_ctrv_{experiment.inference_method}_{count:02d}.csv"
        )
        if not export_evaluation_csv(predictions, destination):
            raise OSError(f"Could not export evaluation CSV: {destination}")
        row = {
            field.name: getattr(summary, field.name)
            for field in dataclasses.fields(summary)
            if field.name != "per_horizon_table"
        }
        row.update(
            observation_count=count,
            prediction_count=experiment.prediction_count,
            inference_seed=experiment.inference_seed,
            data_file=str(Path(data_file).resolve()),
            rmse_m=float((predictions["position_error_m"] ** 2).mean() ** 0.5),
            mcmc_rhat_max=float(predictions["mcmc_rhat_max"].max()),
        )
        rows.append(row)

    summary_table = pd.DataFrame(rows)
    destination = arguments.output_dir / "summary.csv"
    if not export_evaluation_csv(summary_table, destination, summary=True):
        raise OSError(f"Could not export summary CSV: {destination}")
    print("\nObservation-count comparison:")
    print(
        summary_table[["observation_count", "ade_m", "fde_m", "rmse_m"]].to_string(
            index=False
        )
    )
    return summary_table


def _csv_output_path(
    value: str | Path | None, inference_method: str, *, setting="EVALUATION_OUTPUT_CSV"
) -> Path | None:
    """Resolve an optional CSV destination without creating any files."""
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError("no output file configured")
        if not isinstance(value, (str, Path)):
            raise TypeError("expected a .csv path or None")
        if "\0" in str(value):
            raise ValueError("CSV paths must not contain null characters")
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
        print(f"Warning: {setting}={value!r}: {error}. CSV export skipped.")
        return None


def export_evaluation_csv(
    table: pd.DataFrame, output_path: str | Path, *, summary: bool = False
) -> bool:
    """Export evaluation results through the shared atomic CSV writer.

    Individual forecast CSVs always use PREDICTION_CSV_COLUMNS in that order.
    Missing method-specific fields are empty cells; existing values and the
    input table are preserved. Unknown columns are rejected rather than lost.
    Pass summary=True for the aggregate summary's separate table format.
    Return False with a warning when normalization or writing fails.
    """
    try:
        path = Path(output_path)
        if not summary:
            unknown = set(table.columns).difference(PREDICTION_CSV_COLUMNS)
            if unknown:
                raise ValueError(f"Unknown prediction CSV columns: {sorted(unknown)}")
            table = table.reindex(columns=PREDICTION_CSV_COLUMNS)
    except (TypeError, ValueError) as error:
        print(
            f"Warning: EVALUATION_OUTPUT_CSV={str(output_path)!r}: {error}. CSV export skipped."
        )
        return False
    return _export_csv(table, path)


def association_csv_path(evaluation_path: str | Path) -> Path:
    """Keep each method/window's association trace next to its forecast CSV."""
    path = Path(evaluation_path)
    stem = path.stem.removesuffix("_evaluation")
    return path.with_name(f"{stem}_associations.csv")


def export_associations_csv(table: pd.DataFrame, output_path: str | Path) -> bool:
    """Export raw radar choices with nullable integer IDs through the atomic writer."""
    try:
        if set(table.columns) != set(ASSOCIATION_CSV_COLUMNS):
            raise ValueError(
                f"Association CSV requires columns {ASSOCIATION_CSV_COLUMNS}."
            )
        normalized = table.loc[:, ASSOCIATION_CSV_COLUMNS].copy()
        normalized["detection_id"] = pd.array(normalized["detection_id"], dtype="Int64")
        path = Path(output_path)
    except (TypeError, ValueError) as error:
        print(
            f"Warning: ASSOCIATIONS_OUTPUT_CSV={str(output_path)!r}: {error}. CSV export skipped."
        )
        return False
    return _export_csv(
        normalized,
        path,
        description="BR24 radar associations",
        setting="ASSOCIATIONS_OUTPUT_CSV",
    )


def _export_csv(
    predictions,
    output_path: Path,
    *,
    description="per-forecast Bayesian CTRV metrics",
    setting="EVALUATION_OUTPUT_CSV",
) -> bool:
    """Replace the destination only after writing the complete CSV successfully."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=".bayestraj-csv-", dir=output_path.parent
        ) as work:
            temporary_csv = Path(work) / "metrics.csv"
            predictions.to_csv(temporary_csv, index=False, na_rep="")
            temporary_csv.replace(output_path)
    except (OSError, ValueError) as error:
        print(f"Warning: {setting}={str(output_path)!r}: {error}. CSV export skipped.")
        return False
    else:
        print(f"\nSaved {description}: {output_path}")
        return True
