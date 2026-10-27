"""Compare saved Bayesian CTRV and IMM evaluation CSVs without running models.

Each figure shows a metric per common forecast origin, with the same horizons
and reference positions. Dashed lines and the terminal table show averages of
these per-origin metrics, not a pooled RMSE across all forecast positions.
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bayestraj.validation.runtime_plotting as runtime_plotting

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BAYESIAN_CSV = PROJECT_ROOT / "results/BR24_bayesian_ctrv_forecast_evaluation.csv"
IMM_CSV = PROJECT_ROOT / "results/BR24_imm_forecast_evaluation.csv"
MODEL_COLORS = ("#0072B2", "#E69F00")
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
    "log_predictive_density",
    "fit_runtime_seconds",
)


def _validate_table(predictions, model_variant):
    """Require one finite, complete forecast configuration and median errors."""
    required = {*NUMERIC_COLUMNS, "time", "inference_method", "model_variant"}
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
    bayesian = _validate_table(bayesian, "bayesian")
    imm = _validate_table(imm, "imm")
    common = np.intersect1d(
        bayesian["forecast_origin_seconds"], imm["forecast_origin_seconds"]
    )
    if common.size == 0:
        raise ValueError("The CSV files have no common forecast origin times.")
    aligned = tuple(
        table[table["forecast_origin_seconds"].isin(common)].reset_index(drop=True)
        for table in (bayesian, imm)
    )
    left, right = aligned
    keys = ["forecast_origin_seconds", "horizon_step"]
    if left[keys].shape != right[keys].shape or not np.array_equal(
        left[keys], right[keys]
    ):
        raise ValueError(
            "Common forecast origins must have identical complete horizons."
        )
    if not np.allclose(
        left["horizon_seconds"], right["horizon_seconds"], rtol=0, atol=1e-6
    ):
        raise ValueError("Common forecasts have different horizon_seconds.")
    if (left["time"] != right["time"]).any():
        raise ValueError("Common forecasts have different target times.")
    if not np.allclose(
        left[["x_actual", "y_actual"]],
        right[["x_actual", "y_actual"]],
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError("Common forecasts must use the same reference positions.")
    omitted = [
        table["forecast_origin_seconds"].nunique() - len(common)
        for table in (bayesian, imm)
    ]
    if any(omitted):
        warnings.warn(
            f"Comparing {len(common)} common origins; omitted unmatched origins: "
            f"Bayesian CTRV={omitted[0]}, IMM={omitted[1]}.",
            UserWarning,
            stacklevel=2,
        )
    return aligned


def plot_comparison(bayesian, imm, *, show=True, output_dir=None):
    """Plot already aligned tables with the evaluator's aggregation and style."""
    series = [
        runtime_plotting._rolling_window_metric_series(table)
        for table in (bayesian, imm)
    ]
    elapsed = series[0][0] - series[0][0][0]
    style = runtime_plotting.RUNTIME_PLOT_STYLE
    labels = (f"Bayesian CTRV ({bayesian['inference_method'].iloc[0].upper()})", "IMM")
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    figures, rows = [], []
    for ylabel, _, _, name, _ in runtime_plotting.ROLLING_DIAGNOSTIC_PLOT_STYLES:
        if name == "fit_runtime_seconds":
            ylabel = "Laufzeit [s]"
        figure, axis = plt.subplots(figsize=style.speed_figure_size)
        averages = []
        for (_, values), label, color in zip(series, labels, MODEL_COLORS, strict=True):
            metric = values[name]
            average = float(np.mean(metric))
            averages.append(average)
            axis.plot(
                elapsed, metric, marker="o", markersize=3, color=color, label=label
            )
            axis.axhline(
                average,
                color=color,
                linestyle="--",
                label=f"{label}: Mittel = {average:.3f}",
            )
        axis.set_xlabel("Zeit [s]", fontsize=style.axis_label_font_size)
        axis.set_ylabel(ylabel, fontsize=style.axis_label_font_size)
        axis.tick_params(axis="both", labelsize=style.axis_tick_font_size)
        axis.set_axisbelow(True)
        axis.grid(True, which="major", axis="both", color="0.7", alpha=0.2)
        axis.legend(loc=style.legend_location)
        figure.suptitle(
            "Bayesian CTRV vs. IMM",
            fontsize=style.title_font_size,
            fontweight=style.title_font_weight,
        )
        figure.tight_layout()
        if output_dir is not None:
            figure.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        figures.append((figure, axis))
        rows.append(
            {
                "metric": name,
                "bayesian_ctrv": averages[0],
                "imm": averages[1],
                "preferred": "higher"
                if name == "mean_log_predictive_density"
                else "lower",
            }
        )
    return pd.DataFrame(rows), tuple(figures)


def _read_csv(path):
    if not path.is_file():
        raise ValueError(
            f"Evaluation CSV not found: {path}. Generate or supply a saved evaluation CSV; "
            "this comparison never starts either model."
        )
    return pd.read_csv(path, float_precision="round_trip")


def main(argv=None):
    """Read two evaluation CSVs, report matched metrics and display/save plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bayesian-csv", type=Path, default=BAYESIAN_CSV)
    parser.add_argument("--imm-csv", type=Path, default=IMM_CSV)
    parser.add_argument(
        "--output-dir", type=Path, help="Optional directory for six PNG figures."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Print metrics without opening plot windows; can be combined with --output-dir.",
    )
    options = parser.parse_args(argv)
    try:
        bayesian, imm = prepare_comparison(
            _read_csv(options.bayesian_csv), _read_csv(options.imm_csv)
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Matched forecast origins: {bayesian['forecast_origin_seconds'].nunique()}")
    print(f"Matched forecast positions: {len(bayesian)}")
    summary, figures = plot_comparison(
        bayesian, imm, show=not options.no_show, output_dir=options.output_dir
    )
    print("\nMean metrics per matched forecast origin (preferred: lower / higher):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(
        "\nNote: IMM distribution scores use a fused Gaussian; Bayesian CTRV uses posterior samples."
    )
    print(
        "Runtime measurements may cover different operations across inference methods."
    )
    print(
        "This descriptive comparison does not establish statistical significance or an overall winner."
    )
    if options.no_show:
        for figure, _ in figures:
            plt.close(figure)
    return summary


if __name__ == "__main__":
    main()
