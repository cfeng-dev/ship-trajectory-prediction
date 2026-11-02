"""Compare saved Bayesian CTRV and IMM rolling evaluation results."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bayestraj.validation.runtime_plotting as runtime_plotting

MODEL_CSV_SPECS = {
    "IMM_CSV": ("imm", "imm", "IMM"),
    "BAYESIAN_SMC_CSV": ("bayesian", "smc", "Bayesian CTRV (SMC)"),
    "BAYESIAN_SEQ_VI_CSV": (
        "bayesian",
        "sequential_vi",
        "Bayesian CTRV (Sequential VI)",
    ),
    "BAYESIAN_VI_CSV": ("bayesian", "vi", "Bayesian CTRV (VI)"),
    "BAYESIAN_MCMC_CSV": ("bayesian", "mcmc", "Bayesian CTRV (MCMC)"),
}

NUMERIC_COLUMNS = (
    "window_index",
    "forecast_origin_seconds",
    "horizon_step",
    "horizon_seconds",
    "prediction_count",
    "x_actual",
    "y_actual",
    "x_median",
    "y_median",
    "position_error_m",
    "energy_score_m",
    "negative_log_predictive_density",
    "fit_runtime_seconds",
)


def _validate_table(predictions, model_variant):
    """Require one finite, complete forecast configuration and median errors."""
    predictions = predictions.copy()
    if "log_predictive_density" in predictions:
        legacy = -pd.to_numeric(
            predictions.pop("log_predictive_density"), errors="raise"
        )
        if "negative_log_predictive_density" not in predictions:
            predictions["negative_log_predictive_density"] = legacy
        elif not np.allclose(
            pd.to_numeric(
                predictions["negative_log_predictive_density"], errors="raise"
            ),
            legacy,
        ):
            raise ValueError("Conflicting log predictive density and NLPD columns.")
    required = {
        *NUMERIC_COLUMNS,
        "time",
        "inference_method",
        "model_variant",
        "joint_covered_50",
        "joint_covered_90",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Missing {model_variant} CSV columns: {missing}")
    if predictions.empty:
        raise ValueError(f"The {model_variant} evaluation table is empty.")
    table = predictions.copy().reset_index(drop=True)
    for name in NUMERIC_COLUMNS:
        table[name] = pd.to_numeric(table[name], errors="raise")
    if not np.all(np.isfinite(table[list(NUMERIC_COLUMNS)].to_numpy(dtype=float))):
        raise ValueError(f"The {model_variant} CSV must contain finite numeric values.")
    for column in ("joint_covered_50", "joint_covered_90"):
        if table[column].isna().any() or not table[column].isin([False, True]).all():
            raise ValueError(f"{column} must contain boolean coverage flags or 0/1.")
        table[column] = table[column].astype(bool)
    if (table["joint_covered_50"] & ~table["joint_covered_90"]).any():
        raise ValueError("Joint 50% coverage must be nested inside joint 90% coverage.")
    for name in ("window_index", "horizon_step", "prediction_count"):
        values = table[name].to_numpy(dtype=float)
        lower_bound = 0 if name == "window_index" else 1
        if np.any(values < lower_bound) or np.any(values != np.floor(values)):
            raise ValueError(f"{name} must contain integers >= {lower_bound}.")
    if (table[["position_error_m", "fit_runtime_seconds"]] < 0).any().any():
        raise ValueError("Position errors and runtime must be non-negative.")
    if table["model_variant"].isna().any() or set(table["model_variant"]) != {
        model_variant
    }:
        raise ValueError(f"Expected model_variant={model_variant!r} in its CSV.")
    methods = table["inference_method"]
    if (
        methods.isna().any()
        or methods.nunique() != 1
        or not isinstance(methods.iloc[0], str)
        or not methods.iloc[0].strip()
    ):
        raise ValueError("Each CSV must have exactly one inference_method.")
    if model_variant == "imm" and methods.iloc[0] != "imm":
        raise ValueError("The IMM CSV must have inference_method='imm'.")
    if model_variant == "bayesian" and methods.iloc[0] == "imm":
        raise ValueError("The Bayesian CSV cannot have inference_method='imm'.")
    table["time"] = pd.to_datetime(table["time"], utc=True, format="mixed")
    if table["time"].isna().any():
        raise ValueError("Forecast target times must not be missing.")
    keys = ["forecast_origin_seconds", "horizon_step"]
    if table.duplicated(keys).any():
        raise ValueError("Duplicate forecast origin/horizon rows in an evaluation CSV.")
    if table.groupby("forecast_origin_seconds")["window_index"].nunique().max() != 1:
        raise ValueError("Each forecast origin must belong to exactly one window.")
    for _, window in table.groupby("window_index", sort=False):
        if window["forecast_origin_seconds"].nunique() != 1:
            raise ValueError("Each window must have exactly one forecast origin.")
        if window["fit_runtime_seconds"].nunique() != 1:
            raise ValueError("Each window must have one constant fit_runtime_seconds.")
        if (
            window["prediction_count"].nunique() != 1
            or len(window) != window["prediction_count"].iloc[0]
        ):
            raise ValueError(
                "Each window must contain its complete prediction_count horizons."
            )
        window = window.sort_values("horizon_step")
        if not np.array_equal(window["horizon_step"], np.arange(1, len(window) + 1)):
            raise ValueError(
                "Each window must have consecutive horizon steps starting at 1."
            )
        horizons = window["horizon_seconds"].to_numpy(dtype=float)
        if np.any(horizons <= 0) or np.any(np.diff(horizons) <= 0):
            raise ValueError("Forecast horizons must be positive and increasing.")
    target_seconds = (
        table["time"] - pd.Timestamp(0, unit="s", tz="UTC")
    ).dt.total_seconds()
    if not np.allclose(
        target_seconds,
        table["forecast_origin_seconds"] + table["horizon_seconds"],
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError(
            "Target time must equal forecast origin time plus horizon_seconds."
        )
    expected_errors = np.hypot(
        table["x_median"] - table["x_actual"], table["y_median"] - table["y_actual"]
    )
    if not np.allclose(
        table["position_error_m"], expected_errors, rtol=1e-8, atol=1e-6
    ):
        raise ValueError(
            "position_error_m must match the saved median positions and reference."
        )
    return table.sort_values(keys).reset_index(drop=True)


def prepare_comparison(bayesian, imm):
    """Align complete forecasts by physical time, never by CSV row/window number."""
    models = {
        "Bayesian CTRV": _validate_table(bayesian, "bayesian"),
        "IMM": _validate_table(imm, "imm"),
    }
    return tuple(_align_forecasts(models).values())


def _align_forecasts(models):
    """Use one common cohort across every selected model."""
    if len(models) < 2:
        raise ValueError("Select at least two CSV files for comparison.")
    common = None
    for table in models.values():
        origins = table["forecast_origin_seconds"].unique()
        common = origins if common is None else np.intersect1d(common, origins)
    if common.size == 0:
        raise ValueError("The CSV files have no common forecast origin times.")
    aligned = {
        label: table[table["forecast_origin_seconds"].isin(common)].reset_index(
            drop=True
        )
        for label, table in models.items()
    }
    reference_label, left = next(iter(aligned.items()))
    keys = ["forecast_origin_seconds", "horizon_step"]
    for label, right in aligned.items():
        if label == reference_label:
            continue
        if left[keys].shape != right[keys].shape or not np.array_equal(
            left[keys], right[keys]
        ):
            raise ValueError(
                "Common forecast origins must have identical complete horizons "
                f"({reference_label}, {label})."
            )
        if not np.allclose(
            left["horizon_seconds"], right["horizon_seconds"], rtol=0, atol=1e-6
        ):
            raise ValueError(
                f"Common forecasts have different horizon_seconds ({reference_label}, {label})."
            )
        if (left["time"] != right["time"]).any():
            raise ValueError(
                f"Common forecasts have different target times ({reference_label}, {label})."
            )
        if not np.allclose(
            left[["x_actual", "y_actual"]],
            right[["x_actual", "y_actual"]],
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError(
                f"Common forecasts must use the same reference positions ({reference_label}, {label})."
            )
    omitted = {
        label: table["forecast_origin_seconds"].nunique() - len(common)
        for label, table in models.items()
    }
    if any(omitted.values()):
        warnings.warn(
            f"Comparing {len(common)} common origins; omitted unmatched origins: "
            + ", ".join(f"{label}={count}" for label, count in omitted.items())
            + ".",
            UserWarning,
            stacklevel=2,
        )
    return aligned


def _read_csv(path):
    if not path.is_file():
        raise ValueError(
            f"Evaluation CSV not found: {path}. Generate or supply a saved evaluation CSV; "
            "this comparison never starts either model."
        )
    return pd.read_csv(path, float_precision="round_trip")


def read_comparison_csvs(
    bayesian_csv: str | Path, imm_csv: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and align complete saved forecasts without running either model."""
    return prepare_comparison(_read_csv(Path(bayesian_csv)), _read_csv(Path(imm_csv)))


def read_model_comparison_csvs(csv_paths):
    """Read and align named CSVs in online-then-batch display order."""
    unknown = set(csv_paths).difference(MODEL_CSV_SPECS)
    if unknown:
        raise ValueError(f"Unknown comparison CSV variables: {sorted(unknown)}")
    selected = {
        name: csv_paths[name]
        for name in MODEL_CSV_SPECS
        if csv_paths.get(name) is not None
    }
    if len(selected) < 2:
        raise ValueError("Select at least two CSV files for comparison.")
    models = {}
    for variable, path in selected.items():
        variant, method, label = MODEL_CSV_SPECS[variable]
        try:
            table = _validate_table(_read_csv(Path(path)), variant)
        except ValueError as error:
            raise ValueError(f"{variable}: {error}") from error
        if table["inference_method"].iloc[0] != method:
            raise ValueError(f"{variable} requires inference_method={method!r}: {path}")
        models[label] = table
    return _align_forecasts(models)


def run_comparison(bayesian, imm, *, show=True, output_dir=None):
    """Report and plot already aligned forecasts using the shared metric style."""
    return run_model_comparison(
        {"bayesian_ctrv": bayesian, "imm": imm},
        labels={
            "bayesian_ctrv": f"Bayesian CTRV ({bayesian['inference_method'].iloc[0].upper()})",
            "imm": "IMM",
        },
        title="Bayesian CTRV vs. IMM",
        show=show,
        output_dir=output_dir,
    )


def run_model_comparison(
    models,
    *,
    show=True,
    output_dir=None,
    labels=None,
    title=None,
    show_points=True,
    show_mean_lines=True,
    show_summary_table=False,
    metric_y_scale="linear",
    plot_language="en",
):
    """Report and plot each selected model on the same forecast cohort."""
    if len(models) < 2:
        raise ValueError("Select at least two CSV files for comparison.")
    first = next(iter(models.values()))
    print(f"Matched forecast origins: {first['forecast_origin_seconds'].nunique()}")
    print(f"Matched forecast positions: {len(first)}")
    summary, figures = runtime_plotting.plot_model_comparison(
        models,
        labels=labels,
        title=title,
        show=show,
        output_dir=output_dir,
        show_points=show_points,
        show_mean_lines=show_mean_lines,
        show_summary_table=show_summary_table,
        metric_y_scale=metric_y_scale,
        plot_language=plot_language,
    )
    print("\nMetrics for matched forecasts:")
    display = summary.copy()
    coverage_rows = display["metric"].str.startswith("joint_coverage_")
    for column in models:
        display[column] = [
            f"{value:.1%}" if coverage else f"{value:.3f}"
            for value, coverage in zip(display[column], coverage_rows, strict=True)
        ]
    print(display.to_string(index=False))
    if any(table["model_variant"].iloc[0] == "imm" for table in models.values()):
        print(
            "\nNote: IMM distribution scores use a fused Gaussian; Bayesian CTRV uses posterior samples."
        )
    print(
        "Runtime measurements may cover different operations across inference methods."
    )
    print(
        "This descriptive comparison does not establish statistical significance or an overall winner."
    )
    if not show:
        for figure, _ in figures:
            plt.close(figure)
    return summary
