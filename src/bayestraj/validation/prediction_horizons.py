"""Evaluate and compare forecast prefixes on identical complete origin cohorts."""

import argparse
import dataclasses
import hashlib
import json
import re
import uuid
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bayestraj.inference.configuration as inference
import bayestraj.numeric_validation as numeric_validation
import bayestraj.validation.bayesian_ctrv_evaluation as evaluation
import bayestraj.validation.bayesian_ctrv_workflow as bayesian_workflow
import bayestraj.validation.cli as cli
import bayestraj.validation.comparison as comparison
import bayestraj.validation.imm_workflow as imm_workflow
import bayestraj.validation.rolling as rolling
import bayestraj.validation.runtime_plotting as runtime_plotting

METHOD_LABELS = {
    "imm": "IMM",
    "smc": "Bayesian CTRV (SMC)",
    "vi": "Bayesian CTRV (VI)",
    "mcmc": "Bayesian CTRV (MCMC)",
}
FORECAST_METADATA_FILENAME = "forecast_steps_metadata.json"
FORECAST_SUMMARY_FILENAME = "forecast_steps_summary.csv"


def _positive_integer(name, value):
    number = numeric_validation.validate_non_negative_integer(name, value)
    if number == 0:
        raise ValueError(f"{name} must contain positive integers.")
    return number


def _prediction_counts(values):
    counts = tuple(_positive_integer("prediction_counts", value) for value in values)
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("prediction_counts must contain distinct positive integers.")
    return tuple(sorted(counts))


def _filename(method, observation_count):
    if method == "imm":
        return "imm.csv"
    suffix = f"_{observation_count:02d}" if method in {"vi", "mcmc"} else ""
    return f"bayesian_ctrv_{method}{suffix}.csv"


def _validated_predictions(table, method):
    variant = "imm" if method == "imm" else "bayesian"
    table = comparison._validate_table(table, variant)
    if not table["inference_method"].eq(method).all():
        raise ValueError(f"Expected inference_method={method!r}.")
    comparison._validate_observation_count_scores(table, method)
    return table


def _require_common_cohort(tables):
    """Match all origins and each available prefix against the longest horizon."""
    reference_name = max(
        tables, key=lambda name: tables[name]["prediction_count"].max()
    )
    reference = tables[reference_name]
    origins = reference["forecast_origin_seconds"].unique()
    for name, table in tables.items():
        if not np.array_equal(origins, table["forecast_origin_seconds"].unique()):
            raise ValueError("Prediction horizons require identical forecast origins.")
        if name != reference_name:
            count = int(table["prediction_count"].max())
            comparison._align_forecasts(
                {
                    reference_name: reference[reference["horizon_step"] <= count],
                    name: table,
                }
            )


def _read_metadata(directory):
    path = directory / FORECAST_METADATA_FILENAME
    if not path.exists():
        return {}
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(content, dict)
            or content.get("version") != 1
            or not isinstance(content.get("evaluations"), dict)
            or not all(
                isinstance(record, dict) for record in content["evaluations"].values()
            )
        ):
            raise ValueError("Expected forecast-step evaluation records.")
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid forecast-step metadata: {path}") from error
    return content["evaluations"]


def _write_metadata(directory, records):
    path = directory / FORECAST_METADATA_FILENAME
    with TemporaryDirectory(prefix=".bayestraj-horizons-", dir=directory) as work:
        temporary = Path(work) / "metadata.json"
        temporary.write_text(
            json.dumps({"version": 1, "evaluations": records}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def run_prediction_horizon_evaluation(
    *,
    data_file,
    output_dir,
    prediction_counts,
    experiment,
    priors,
    max_windows=None,
    credible_interval=0.9,
    vi_config=None,
    mcmc_config=None,
    fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
    argv=None,
):
    """Run one Bayesian CTRV configuration and export shorter forecast prefixes.

    Exclude shortened tail windows from every horizon. Forecast CSVs retain
    their existing model-specific schema. Runtime refers to the maximum-horizon
    evaluation, since shorter prefixes reuse its forecasts. Comparison scripts
    compute aggregate metrics and plots later without rerunning inference.
    """
    parser = argparse.ArgumentParser(
        description=run_prediction_horizon_evaluation.__doc__
    )
    parser.add_argument(
        "--prediction-counts", type=int, nargs="+", default=prediction_counts
    )
    parser.add_argument(
        "--observations", type=int, default=experiment.observation_count
    )
    parser.add_argument("--output-dir", type=Path, default=Path(output_dir))
    parser.add_argument("--max-windows", type=int, default=max_windows)
    cli.add_student_t_scores_argument(parser)
    arguments = parser.parse_args(argv)
    counts = _prediction_counts(arguments.prediction_counts)
    maximum = max(counts)
    method = experiment.inference_method
    if method not in inference.CTRV_ROLLING_INFERENCE_METHODS:
        raise ValueError("Prediction-horizon automation supports SMC, VI and MCMC.")
    observation_count = _positive_integer("observation_count", arguments.observations)
    if method in {"vi", "mcmc"} and observation_count < 2:
        raise ValueError("Batch observation_count must be at least two.")
    if arguments.max_windows is not None:
        _positive_integer("max_windows", arguments.max_windows)
    stride = (
        1
        if experiment.stride is None
        else _positive_integer("stride", experiment.stride)
    )
    if stride > maximum:
        raise ValueError("stride must not exceed the maximum prediction count.")
    if not 0 < credible_interval < 1:
        raise ValueError("credible_interval must lie between zero and one.")
    vi_config = inference.create_default_vi_config() if vi_config is None else vi_config
    directory = arguments.output_dir
    records = _read_metadata(directory)
    print(f"\n=== {method.upper()}: forecast {maximum} future positions ===")
    current = dataclasses.replace(
        experiment,
        observation_count=observation_count,
        prediction_count=maximum,
        stride=stride,
    )
    options = cli.BayesianCTRVEvaluationOptions(
        observation_count=observation_count,
        prediction_count=maximum,
        stride=stride,
        inference_method=method,
        vi_algorithm=vi_config["algorithm"],
        turn_rate_prior_abs_rate_deg_s=priors.turn_rate_prior_abs_rate_deg_s,
        inference_seed=experiment.inference_seed,
        position_noise_std_m=experiment.position_noise_std_m,
        position_noise_seed=experiment.position_noise_seed,
        require_converged=vi_config["require_converged"],
        max_windows=arguments.max_windows,
        student_t_scores=arguments.student_t_scores,
    )
    predictions, _summary = bayesian_workflow.run_bayesian_ctrv_evaluation(
        data_file=data_file,
        experiment=current,
        priors=priors,
        vi_config=vi_config,
        mcmc_config=mcmc_config,
        fullrank_grad_samples=fullrank_grad_samples,
        credible_interval=credible_interval,
        options=options,
    )
    return _export_forecast_steps(
        predictions,
        method=method,
        directory=directory,
        counts=counts,
        observation_count=observation_count,
        records=records,
        context={
            "data_file": str(Path(data_file).resolve()),
            "inference_seed": experiment.inference_seed,
            "position_noise_std_m": experiment.position_noise_std_m,
            "position_noise_seed": experiment.position_noise_seed,
            "credible_interval": credible_interval,
            "stride": stride,
            "priors": dataclasses.asdict(priors),
        },
    )


def run_imm_prediction_horizon_evaluation(
    *,
    data_file,
    imm_python,
    output_dir,
    prediction_counts,
    max_windows=None,
    forecast_seed=42,
    credible_interval=0.9,
    stride=1,
    argv=None,
):
    """Run the existing IMM tracker once and export every requested prefix."""
    parser = argparse.ArgumentParser(
        description=run_imm_prediction_horizon_evaluation.__doc__
    )
    parser.add_argument(
        "--prediction-counts", type=int, nargs="+", default=prediction_counts
    )
    parser.add_argument("--output-dir", type=Path, default=Path(output_dir))
    parser.add_argument("--max-windows", type=int, default=max_windows)
    parser.add_argument("--seed", type=int, default=forecast_seed)
    parser.add_argument("--stride", type=int, default=stride)
    cli.add_student_t_scores_argument(parser)
    arguments = parser.parse_args(argv)
    counts = _prediction_counts(arguments.prediction_counts)
    if arguments.max_windows is not None:
        _positive_integer("max_windows", arguments.max_windows)
    stride = _positive_integer("stride", arguments.stride)
    if stride > max(counts):
        raise ValueError("stride must not exceed the maximum prediction count.")
    forecast_seed = numeric_validation.validate_non_negative_integer(
        "forecast_seed", arguments.seed
    )
    if not 0 < credible_interval < 1:
        raise ValueError("credible_interval must lie between zero and one.")
    directory = arguments.output_dir
    records = _read_metadata(directory)
    options = cli.IMMEvaluationOptions(
        observation_count=2,
        prediction_count=max(counts),
        stride=stride,
        max_windows=arguments.max_windows,
        show_plot=False,
        student_t_scores=arguments.student_t_scores,
    )
    predictions, _summary = imm_workflow.run_imm_evaluation(
        data_file=data_file,
        imm_python=imm_python,
        options=options,
        credible_interval=credible_interval,
        forecast_seed=forecast_seed,
    )
    return _export_forecast_steps(
        predictions,
        method="imm",
        directory=directory,
        counts=counts,
        observation_count=options.observation_count,
        records=records,
        context={
            "data_file": str(Path(data_file).resolve()),
            "inference_seed": forecast_seed,
            "position_noise_std_m": 0.0,
            "position_noise_seed": None,
            "credible_interval": credible_interval,
            "stride": stride,
            "priors": None,
        },
    )


def _export_forecast_steps(
    predictions, *, method, directory, counts, observation_count, records, context
):
    """Keep model-specific CSV fields while sharing prefix and cohort semantics."""
    maximum = max(counts)
    table = _validated_predictions(predictions, method)
    if table["prediction_count"].max() > maximum:
        raise ValueError("Forecasts exceed the configured maximum prediction count.")
    table = table.loc[table["prediction_count"] == maximum].copy()
    if table.empty:
        raise ValueError(
            "No complete forecasts remain for the maximum prediction count."
        )
    forecast_run_id = uuid.uuid4().hex
    outputs = []
    for count in counts:
        prefix = table.loc[table["horizon_step"] <= count].copy()
        prefix["prediction_count"] = count
        relative = Path(f"forecast_step_{count:02d}") / _filename(
            method, observation_count
        )
        destination = directory / relative
        if method == "imm":
            imm_workflow.export_evaluation_csv(prefix, destination)
        elif not evaluation.export_evaluation_csv(prefix, destination):
            raise OSError(f"Could not export forecast CSV: {destination}")
        records[relative.as_posix()] = {
            **context,
            "inference_method": method,
            "observation_count": observation_count,
            "prediction_count": count,
            "runtime_prediction_count": maximum,
            "forecast_run_id": forecast_run_id,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
        _write_metadata(directory, records)
        outputs.append(destination)
    return tuple(outputs)


def _forecast_step_files(directory):
    """Discover CSVs only in the numbered forecast-step directories."""
    if not directory.is_dir():
        raise ValueError(f"Forecast-step directory not found: {directory}")
    folders = {}
    for folder in directory.iterdir():
        match = re.fullmatch(r"forecast_step_(\d+)", folder.name)
        if not folder.is_dir() or match is None:
            continue
        count = _positive_integer("forecast step", int(match[1]))
        if count in folders:
            raise ValueError(
                f"Duplicate forecast step directory for {count} positions."
            )
        folders[count] = folder
    for count, folder in sorted(folders.items()):
        for path in sorted(folder.iterdir(), key=lambda value: value.name.casefold()):
            if (
                path.is_file()
                and path.suffix.lower() == ".csv"
                and path.name.casefold() != "summary.csv"
                and not path.stem.casefold().endswith("_associations")
            ):
                yield count, path


def _align_common_origins(tables):
    """Restrict manual and automated runs to one shared forecast-origin cohort."""
    common = None
    for table in tables.values():
        origins = table["forecast_origin_seconds"].unique()
        common = origins if common is None else np.intersect1d(common, origins)
    if common.size == 0:
        raise ValueError("The forecast-step CSVs have no common forecast origins.")
    aligned = {
        name: table.loc[table["forecast_origin_seconds"].isin(common)].copy()
        for name, table in tables.items()
    }
    _require_common_cohort(aligned)
    omitted = {
        name: table["forecast_origin_seconds"].nunique() - len(common)
        for name, table in tables.items()
    }
    if any(omitted.values()):
        warnings.warn(
            f"Comparing {len(common)} common forecast origins; omitted unmatched origins: "
            + ", ".join(f"{name}={count}" for name, count in omitted.items() if count),
            UserWarning,
            stacklevel=2,
        )
    return aligned


def read_prediction_horizon_evaluations(input_dir):
    """Discover and compare CSVs in forecast_step_* folders, including manual runs.

    Metadata is optional for manually added CSVs. Where available, it protects
    exported prefixes against modification and mixed generations. Every model
    must provide the same forecast steps; metrics use their shared origins.
    """
    directory = Path(input_dir)
    records = _read_metadata(directory)
    tables, tracked, grouped, configurations, ordering, generations = (
        {},
        {},
        {},
        {},
        {},
        {},
    )
    active_records = {}
    for count, path in _forecast_step_files(directory):
        relative = path.relative_to(directory).as_posix()
        record = records.get(relative)
        try:
            raw = pd.read_csv(path, float_precision="round_trip")
            if record is None and not {"model_variant", "inference_method"}.issubset(
                raw.columns
            ):
                continue
            maximum, forecast_run_id = np.nan, None
            if record is not None:
                method = record["inference_method"]
                if record["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError("Forecast CSV no longer matches its metadata.")
            else:
                methods = raw["inference_method"].drop_duplicates()
                if len(methods) != 1:
                    raise ValueError("Each CSV requires one inference_method.")
                method = methods.iloc[0]
            if method not in METHOD_LABELS:
                raise ValueError("Unsupported inference_method.")
            table = _validated_predictions(raw, method)
            if (table["prediction_count"] > count).any():
                raise ValueError(
                    "prediction_count exceeds the forecast step directory."
                )
            if record is not None:
                recorded_count = _positive_integer(
                    "metadata prediction_count", record["prediction_count"]
                )
                observation_count = _positive_integer(
                    "metadata observation_count", record["observation_count"]
                )
                maximum = _positive_integer(
                    "metadata runtime_prediction_count",
                    record["runtime_prediction_count"],
                )
                forecast_run_id = record["forecast_run_id"]
                if not isinstance(forecast_run_id, str) or not forecast_run_id:
                    raise ValueError("Invalid metadata forecast generation.")
                expected = (
                    f"forecast_step_{count:02d}/{_filename(method, observation_count)}"
                )
                if relative != expected or recorded_count != count or count > maximum:
                    raise ValueError(
                        "CSV path or prediction count disagrees with metadata."
                    )
                if not table["prediction_count"].eq(count).all():
                    raise ValueError("prediction_count disagrees with metadata.")
                if method in {"vi", "mcmc"} and (
                    "configured_observation_count" not in table
                    or not table["configured_observation_count"]
                    .eq(observation_count)
                    .all()
                ):
                    raise ValueError(
                        "configured_observation_count disagrees with metadata."
                    )
                tracked[relative] = table
                active_records[relative] = record
            # Ordinary manual evaluations include shortened tail windows.
            table = table.loc[table["prediction_count"] == count].copy()
            if table.empty:
                raise ValueError(
                    "No complete forecasts match this forecast step directory."
                )
            label = runtime_plotting.comparison_model_label(
                METHOD_LABELS[method], table
            )
            observation_count = (
                int(table["configured_observation_count"].iloc[0])
                if method in {"vi", "mcmc"}
                and "configured_observation_count" in table
                and table["configured_observation_count"].notna().any()
                else 0
            )
            if count in configurations.get(label, set()):
                raise ValueError(
                    "Duplicate model configuration for this forecast step."
                )
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ValueError(f"{relative}: {error}") from error
        tables[relative] = table
        configurations.setdefault(label, set()).add(count)
        grouped.setdefault(label, []).append((count, maximum, relative))
        ordering[label] = (tuple(METHOD_LABELS).index(method), observation_count)
        generations.setdefault(label, set()).add(forecast_run_id)
    if not tables:
        raise ValueError(
            f"No forecast CSVs found in forecast_step_* directories: {directory}"
        )
    fields = (
        "data_file",
        "inference_seed",
        "position_noise_std_m",
        "position_noise_seed",
        "credible_interval",
        "stride",
        "priors",
        "runtime_prediction_count",
    )
    for field in fields:
        relevant_records = [
            record
            for record in active_records.values()
            if field not in {"inference_seed", "position_noise_seed", "priors"}
            or record["inference_method"] != "imm"
        ]
        if (
            len(
                {
                    json.dumps(record.get(field), sort_keys=True)
                    for record in relevant_records
                }
            )
            > 1
        ):
            raise ValueError(
                f"Incompatible metadata {field} across prediction horizons."
            )
    if len({tuple(sorted(counts)) for counts in configurations.values()}) != 1:
        raise ValueError("Each model must contain the same prediction counts.")
    if any(None in run_ids and len(run_ids) > 1 for run_ids in generations.values()):
        raise ValueError("Missing metadata for some automated forecast prefixes.")
    if any(len(run_ids) != 1 for run_ids in generations.values()):
        raise ValueError(
            "Mixed forecast generations: regenerate all prediction counts for each model."
        )
    for entries in grouped.values():
        prefixes = {
            relative: tracked[relative]
            for _count, _maximum, relative in entries
            if relative in tracked
        }
        if prefixes:
            _require_common_cohort(prefixes)
    tables = _align_common_origins(tables)
    models = {}
    for label in sorted(grouped, key=ordering.__getitem__):
        entries = grouped[label]
        rows = []
        for count, maximum, relative in sorted(entries, key=lambda entry: entry[0]):
            table = tables[relative]
            summary = rolling.summarize_rolling_predictions(
                table,
                online_inference_methods=(*inference.ONLINE_INFERENCE_METHODS, "imm"),
            )
            row = {
                field.name: getattr(summary, field.name)
                for field in dataclasses.fields(summary)
                if field.name != "per_horizon_table"
            }
            row.update(
                prediction_count=count,
                runtime_prediction_count=maximum,
                mean_horizon_seconds=float(
                    table.loc[table["horizon_step"] == count, "horizon_seconds"].mean()
                ),
                rmse_m=float(np.sqrt(np.mean(table["position_error_m"] ** 2))),
            )
            rows.append(row)
        models[label] = pd.DataFrame(rows)
    return models


def run_prediction_horizon_comparison(
    input_dir,
    *,
    show=True,
    output_dir=None,
    show_points=True,
    show_summary_table=True,
    metric_y_scale="linear",
    plot_language="de",
):
    """Write forecast_steps_summary.csv and plot metrics without running models."""
    models = read_prediction_horizon_evaluations(input_dir)
    summary = pd.DataFrame(
        [
            {"model": label, **row}
            for label, table in models.items()
            for row in table.to_dict("records")
        ]
    )
    destination = Path(input_dir) / FORECAST_SUMMARY_FILENAME
    if not evaluation.export_evaluation_csv(summary, destination, summary=True):
        raise OSError(f"Could not export prediction-horizon summary: {destination}")
    print(
        summary[
            ["model", "prediction_count", *runtime_plotting.PREDICTION_HORIZON_METRICS]
        ].to_string(index=False)
    )
    print(
        "Automated prefixes retain the longest-forecast runtime; manual CSV runtimes remain as recorded."
    )
    figures = runtime_plotting.plot_prediction_horizon_comparison(
        models,
        show=show,
        output_dir=output_dir,
        show_points=show_points,
        show_summary_table=show_summary_table,
        metric_y_scale=metric_y_scale,
        plot_language=plot_language,
    )
    if not show:
        for figure, _axis in figures:
            plt.close(figure)
    return models
