"""Reproducible orchestration and CSV aggregation for thesis benchmarks."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_METHODS = (
    "deterministic",
    "vi_expanding",
    "rbpf",
    "smc",
    "mcmc_expanding",
)

_METHOD_METADATA = {
    "deterministic": ("deterministic", "deterministic", "expanding"),
    "vi_expanding": ("batch", "vi", "expanding"),
    "rbpf": ("online", "rbpf", None),
    "smc": ("online", "smc", None),
    "mcmc_expanding": ("batch", "mcmc", "expanding"),
}

PREDICTIONS_REQUIRED_COLUMNS = (
    "benchmark_case_id",
    "run_id",
    "method",
    "inference_mode",
    "inference_method",
    "window_mode",
    "observation_count",
    "prediction_count",
    "position_noise_std_m",
    "position_noise_seed",
    "inference_seed",
    "window_index",
    "horizon_step",
    "horizon_seconds",
    "position_error_m",
    "energy_score_m",
    "log_predictive_density",
    "joint_covered_50",
    "joint_covered_90",
    "prediction_radius_m",
    "mean_marginal_interval_width_m",
    "converged",
    "mcmc_diagnostics_ok",
    "window_runtime_seconds",
)

RUN_COLUMNS = (
    "benchmark_case_id",
    "run_id",
    "method",
    "inference_mode",
    "inference_method",
    "window_mode",
    "observation_count",
    "prediction_count",
    "position_noise_std_m",
    "position_noise_seed",
    "inference_seed",
    "selected_window_indices",
    "status",
    "forecast_origin_count",
    "predicted_position_count",
    "computation_time_seconds",
    "mean_window_runtime_seconds",
    "vi_convergence_rate",
    "mcmc_diagnostics_pass_rate",
    "mean_particle_effective_sample_size",
    "final_particle_resample_count",
    "error_message",
)

SUMMARY_COLUMNS = (
    "method",
    "inference_mode",
    "inference_method",
    "prediction_count",
    "position_noise_std_m",
    "trajectory_count",
    "forecast_origin_count",
    "predicted_position_count",
    "mean_ade_m",
    "median_position_error_m",
    "mean_maximum_horizon_fde_m",
    "mean_energy_score_m",
    "mean_log_predictive_density",
    "joint_coverage_50",
    "joint_coverage_90",
    "mean_prediction_radius_m",
    "mean_marginal_interval_width_m",
    "mean_runtime_seconds",
    "median_runtime_seconds",
    "total_runtime_seconds",
    "vi_convergence_rate",
    "mcmc_diagnostics_pass_rate",
)

PER_HORIZON_COLUMNS = (
    "method",
    "inference_mode",
    "inference_method",
    "prediction_count",
    "position_noise_std_m",
    "horizon_step",
    "mean_horizon_seconds",
    "predicted_position_count",
    "mean_position_error_m",
    "median_position_error_m",
    "mean_energy_score_m",
    "mean_log_predictive_density",
    "joint_coverage_50",
    "joint_coverage_90",
    "mean_prediction_radius_m",
    "mean_marginal_interval_width_m",
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Validated dimensions shared by all benchmark cases."""

    run_ids: tuple[int, ...]
    observation_count: int
    prediction_counts: tuple[int, ...]
    position_noise_std_m: tuple[float, ...]
    methods: tuple[str, ...]
    position_noise_seed: int
    inference_seed: int
    max_windows: int | None = None
    mcmc_run_ids: tuple[int, ...] = ()
    mcmc_prediction_counts: tuple[int, ...] = ()
    mcmc_window_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "run_ids", _positive_unique_ints(self.run_ids, "run_ids")
        )
        object.__setattr__(
            self,
            "observation_count",
            _positive_integer(self.observation_count, "observation_count"),
        )
        object.__setattr__(
            self,
            "prediction_counts",
            _positive_unique_ints(self.prediction_counts, "prediction_counts"),
        )
        object.__setattr__(
            self,
            "position_noise_std_m",
            _non_negative_unique_floats(
                self.position_noise_std_m,
                "position_noise_std_m",
            ),
        )
        methods = tuple(str(method).strip().lower() for method in self.methods)
        if not methods or any(method not in BENCHMARK_METHODS for method in methods):
            raise ValueError(
                f"methods must be non-empty values from {BENCHMARK_METHODS}."
            )
        if len(set(methods)) != len(methods):
            raise ValueError("methods must not contain duplicates.")
        object.__setattr__(self, "methods", methods)
        object.__setattr__(
            self,
            "position_noise_seed",
            _non_negative_integer(self.position_noise_seed, "position_noise_seed"),
        )
        object.__setattr__(
            self,
            "inference_seed",
            _non_negative_integer(self.inference_seed, "inference_seed"),
        )
        if self.max_windows is not None:
            object.__setattr__(
                self,
                "max_windows",
                _positive_integer(self.max_windows, "max_windows"),
            )
        object.__setattr__(
            self,
            "mcmc_run_ids",
            _optional_positive_unique_ints(self.mcmc_run_ids, "mcmc_run_ids"),
        )
        object.__setattr__(
            self,
            "mcmc_prediction_counts",
            _optional_positive_unique_ints(
                self.mcmc_prediction_counts,
                "mcmc_prediction_counts",
            ),
        )
        object.__setattr__(
            self,
            "mcmc_window_indices",
            _optional_non_negative_unique_ints(
                self.mcmc_window_indices,
                "mcmc_window_indices",
            ),
        )
        if self.mcmc_run_ids and not set(self.mcmc_run_ids).issubset(self.run_ids):
            raise ValueError("mcmc_run_ids must be contained in run_ids.")
        if self.mcmc_prediction_counts and not set(
            self.mcmc_prediction_counts
        ).issubset(self.prediction_counts):
            raise ValueError(
                "mcmc_prediction_counts must be contained in prediction_counts."
            )


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One reproducible run-method-horizon-noise evaluation."""

    run_id: int
    method: str
    observation_count: int
    prediction_count: int
    position_noise_std_m: float
    position_noise_seed: int
    inference_seed: int
    max_windows: int | None
    selected_window_indices: tuple[int, ...] = ()

    @property
    def case_id(self) -> str:
        """Return a stable identifier for output joins and diagnostics."""
        return (
            f"run-{self.run_id}__{self.method}__p-{self.prediction_count}"
            f"__noise-{self.position_noise_std_m:g}__noise-seed-{self.position_noise_seed}"
            f"__inference-seed-{self.inference_seed}"
        )

    @property
    def inference_metadata(self) -> tuple[str, str, str | None]:
        """Return mode, implementation method, and batch history selection."""
        return _METHOD_METADATA[self.method]


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Tables written by one benchmark execution."""

    predictions: pd.DataFrame
    runs: pd.DataFrame
    summary: pd.DataFrame
    per_horizon: pd.DataFrame


def build_benchmark_cases(config: BenchmarkConfig) -> tuple[BenchmarkCase, ...]:
    """Return a stable Cartesian product, with optional MCMC restrictions."""
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig instance.")
    ordered_methods = tuple(
        method for method in BENCHMARK_METHODS if method in config.methods
    )
    cases = []
    for run_id in config.run_ids:
        for prediction_count in config.prediction_counts:
            for noise_std_m in config.position_noise_std_m:
                for method in ordered_methods:
                    if method == "mcmc_expanding" and not _mcmc_case_selected(
                        config,
                        run_id=run_id,
                        prediction_count=prediction_count,
                    ):
                        continue
                    cases.append(
                        BenchmarkCase(
                            run_id=run_id,
                            method=method,
                            observation_count=config.observation_count,
                            prediction_count=prediction_count,
                            position_noise_std_m=noise_std_m,
                            position_noise_seed=config.position_noise_seed,
                            inference_seed=config.inference_seed,
                            max_windows=config.max_windows,
                            selected_window_indices=(
                                config.mcmc_window_indices
                                if method == "mcmc_expanding"
                                else ()
                            ),
                        )
                    )
    return tuple(cases)


def execute_benchmark(
    config: BenchmarkConfig,
    *,
    output_directory: Path,
    case_runner: Callable[[BenchmarkCase], pd.DataFrame],
) -> BenchmarkResult:
    """Run independent cases, record failures, aggregate successes, and save CSVs."""
    if not callable(case_runner):
        raise TypeError("case_runner must be callable.")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    prediction_tables = []
    run_rows = []
    for case in build_benchmark_cases(config):
        started = time.perf_counter()
        try:
            table = _decorate_predictions(case_runner(case), case)
        except Exception as error:  # Keep independent benchmark cases running.
            run_rows.append(_failed_run_row(case, time.perf_counter() - started, error))
            continue
        prediction_tables.append(table)
        run_rows.append(_successful_run_row(case, table, time.perf_counter() - started))

    predictions = (
        pd.concat(prediction_tables, ignore_index=True)
        if prediction_tables
        else pd.DataFrame(columns=PREDICTIONS_REQUIRED_COLUMNS)
    )
    runs = pd.DataFrame(run_rows, columns=RUN_COLUMNS)
    summary = summarize_benchmark_predictions(predictions, runs)
    per_horizon = summarize_benchmark_per_horizon(predictions)
    _write_csv(predictions, output_directory / "predictions.csv")
    _write_csv(runs, output_directory / "runs.csv")
    _write_csv(summary, output_directory / "summary.csv")
    _write_csv(per_horizon, output_directory / "per_horizon.csv")
    return BenchmarkResult(predictions, runs, summary, per_horizon)


def summarize_benchmark_predictions(
    predictions: pd.DataFrame,
    runs: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate successful predictions across trajectories without total ELPD."""
    if predictions.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    group_columns = [
        "method",
        "inference_mode",
        "inference_method",
        "prediction_count",
        "position_noise_std_m",
    ]
    rows = []
    for keys, group in predictions.groupby(group_columns, sort=True, dropna=False):
        maximum_horizon_step = group["horizon_step"].max()
        maximum_horizon = group.loc[group["horizon_step"] == maximum_horizon_step]
        matching_runs = runs.loc[
            (runs["status"] == "success")
            & (runs["method"] == keys[0])
            & (runs["prediction_count"] == keys[3])
            & (runs["position_noise_std_m"] == keys[4])
        ]
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "trajectory_count": int(group["run_id"].nunique()),
                "forecast_origin_count": int(
                    group[["benchmark_case_id", "window_index"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "predicted_position_count": int(len(group)),
                "mean_ade_m": _mean(group["position_error_m"]),
                "median_position_error_m": _median(group["position_error_m"]),
                "mean_maximum_horizon_fde_m": _mean(
                    maximum_horizon["position_error_m"]
                ),
                "mean_energy_score_m": _mean(group["energy_score_m"]),
                "mean_log_predictive_density": _mean(group["log_predictive_density"]),
                "joint_coverage_50": _mean(group["joint_covered_50"]),
                "joint_coverage_90": _mean(group["joint_covered_90"]),
                "mean_prediction_radius_m": _mean(group["prediction_radius_m"]),
                "mean_marginal_interval_width_m": _mean(
                    group["mean_marginal_interval_width_m"]
                ),
                "mean_runtime_seconds": _mean(
                    matching_runs["computation_time_seconds"]
                ),
                "median_runtime_seconds": _median(
                    matching_runs["computation_time_seconds"]
                ),
                "total_runtime_seconds": _sum(
                    matching_runs["computation_time_seconds"]
                ),
                "vi_convergence_rate": _mean(matching_runs["vi_convergence_rate"]),
                "mcmc_diagnostics_pass_rate": _mean(
                    matching_runs["mcmc_diagnostics_pass_rate"]
                ),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def summarize_benchmark_per_horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate existing horizon-level metrics over successful benchmark rows."""
    if predictions.empty:
        return pd.DataFrame(columns=PER_HORIZON_COLUMNS)
    group_columns = [
        "method",
        "inference_mode",
        "inference_method",
        "prediction_count",
        "position_noise_std_m",
        "horizon_step",
    ]
    rows = []
    for keys, group in predictions.groupby(group_columns, sort=True, dropna=False):
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "mean_horizon_seconds": _mean(group["horizon_seconds"]),
                "predicted_position_count": int(len(group)),
                "mean_position_error_m": _mean(group["position_error_m"]),
                "median_position_error_m": _median(group["position_error_m"]),
                "mean_energy_score_m": _mean(group["energy_score_m"]),
                "mean_log_predictive_density": _mean(group["log_predictive_density"]),
                "joint_coverage_50": _mean(group["joint_covered_50"]),
                "joint_coverage_90": _mean(group["joint_covered_90"]),
                "mean_prediction_radius_m": _mean(group["prediction_radius_m"]),
                "mean_marginal_interval_width_m": _mean(
                    group["mean_marginal_interval_width_m"]
                ),
            }
        )
    return pd.DataFrame(rows, columns=PER_HORIZON_COLUMNS)


def _decorate_predictions(table: pd.DataFrame, case: BenchmarkCase) -> pd.DataFrame:
    """Attach benchmark metadata while retaining existing workflow metrics."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("case_runner must return a pandas DataFrame.")
    required = {
        "window_index",
        "horizon_step",
        "horizon_seconds",
        "position_error_m",
        "window_runtime_seconds",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Prediction table is missing required columns: {missing}")
    result = table.copy()
    inference_mode, inference_method, window_mode = case.inference_metadata
    metadata = {
        "benchmark_case_id": case.case_id,
        "run_id": case.run_id,
        "method": case.method,
        "inference_mode": inference_mode,
        "inference_method": inference_method,
        "window_mode": window_mode,
        "observation_count": case.observation_count,
        "prediction_count": case.prediction_count,
        "position_noise_std_m": case.position_noise_std_m,
        "position_noise_seed": case.position_noise_seed,
        "inference_seed": case.inference_seed,
    }
    for name, value in metadata.items():
        result[name] = value
    for name in PREDICTIONS_REQUIRED_COLUMNS:
        if name not in result:
            result[name] = pd.NA
    return result.loc[
        :,
        [
            *PREDICTIONS_REQUIRED_COLUMNS,
            *sorted(set(result.columns).difference(PREDICTIONS_REQUIRED_COLUMNS)),
        ],
    ]


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    """Write stable benchmark tables with compact millimetre-level precision."""
    table.to_csv(path, index=False, float_format="%.3f")


def _successful_run_row(
    case: BenchmarkCase, table: pd.DataFrame, elapsed: float
) -> dict:
    """Build one durable success record from a decorated prediction table."""
    return {
        **_case_metadata(case),
        "status": "success",
        "forecast_origin_count": int(table["window_index"].nunique()),
        "predicted_position_count": int(len(table)),
        "computation_time_seconds": elapsed,
        "mean_window_runtime_seconds": _mean(table["window_runtime_seconds"]),
        "vi_convergence_rate": _mean(table["converged"]),
        "mcmc_diagnostics_pass_rate": _mean(table["mcmc_diagnostics_ok"]),
        "mean_particle_effective_sample_size": _mean(
            table.get("particle_effective_sample_size", pd.Series(dtype=float))
        ),
        "final_particle_resample_count": _last(
            table.get("particle_resample_count", pd.Series(dtype=float))
        ),
        "error_message": pd.NA,
    }


def _failed_run_row(case: BenchmarkCase, elapsed: float, error: Exception) -> dict:
    """Build one failure record without aborting unrelated benchmark cases."""
    return {
        **_case_metadata(case),
        "status": "failed",
        "forecast_origin_count": 0,
        "predicted_position_count": 0,
        "computation_time_seconds": elapsed,
        "mean_window_runtime_seconds": np.nan,
        "vi_convergence_rate": np.nan,
        "mcmc_diagnostics_pass_rate": np.nan,
        "mean_particle_effective_sample_size": np.nan,
        "final_particle_resample_count": np.nan,
        "error_message": f"{type(error).__name__}: {error}",
    }


def _case_metadata(case: BenchmarkCase) -> dict:
    inference_mode, inference_method, window_mode = case.inference_metadata
    return {
        "benchmark_case_id": case.case_id,
        "run_id": case.run_id,
        "method": case.method,
        "inference_mode": inference_mode,
        "inference_method": inference_method,
        "window_mode": window_mode,
        "observation_count": case.observation_count,
        "prediction_count": case.prediction_count,
        "position_noise_std_m": case.position_noise_std_m,
        "position_noise_seed": case.position_noise_seed,
        "inference_seed": case.inference_seed,
        "selected_window_indices": ",".join(map(str, case.selected_window_indices)),
    }


def _mcmc_case_selected(
    config: BenchmarkConfig, *, run_id: int, prediction_count: int
) -> bool:
    return (not config.mcmc_run_ids or run_id in config.mcmc_run_ids) and (
        not config.mcmc_prediction_counts
        or prediction_count in config.mcmc_prediction_counts
    )


def _mean(values: Iterable) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else np.nan


def _median(values: Iterable) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else np.nan


def _sum(values: Iterable) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(numeric.sum()) if not numeric.empty else np.nan


def _last(values: Iterable) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(numeric.iloc[-1]) if not numeric.empty else np.nan


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must contain positive integers.")
    return int(value)


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _positive_unique_ints(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(_positive_integer(value, name) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


def _optional_positive_unique_ints(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(_positive_integer(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


def _optional_non_negative_unique_ints(
    values: Iterable[int], name: str
) -> tuple[int, ...]:
    result = tuple(_non_negative_integer(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


def _non_negative_unique_floats(
    values: Iterable[float], name: str
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(np.isfinite(value) and value >= 0.0 for value in result):
        raise ValueError(f"{name} must contain finite non-negative values.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return result
