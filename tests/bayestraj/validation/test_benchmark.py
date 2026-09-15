"""Tests for reproducible multi-run thesis benchmark orchestration."""

from __future__ import annotations

import pandas as pd
import pytest

from bayestraj.validation.benchmark import (
    BENCHMARK_METHODS,
    PER_HORIZON_COLUMNS,
    PREDICTIONS_REQUIRED_COLUMNS,
    RUN_COLUMNS,
    SUMMARY_COLUMNS,
    BenchmarkConfig,
    build_benchmark_cases,
    execute_benchmark,
)


def _config(**overrides):
    values = {
        "run_ids": (11, 12),
        "observation_count": 5,
        "prediction_counts": (1, 3),
        "position_noise_std_m": (0.0, 5.0),
        "methods": ("deterministic", "vi_expanding"),
        "position_noise_seed": 2026,
        "inference_seed": 42,
        "max_windows": None,
        "mcmc_run_ids": (),
        "mcmc_prediction_counts": (),
        "mcmc_window_indices": (),
    }
    values.update(overrides)
    return BenchmarkConfig(**values)


def _prediction_table(case):
    return pd.DataFrame(
        {
            "window_index": [0, 0, 1, 1],
            "horizon_step": [1, 3, 1, 3],
            "horizon_seconds": [9.0, 29.0, 11.0, 31.0],
            "position_error_m": [1.0, 3.0, 2.0, 4.0],
            "energy_score_m": [0.5, 1.5, 1.0, 2.0],
            "log_predictive_density": [-1.0, -2.0, -1.5, -2.5],
            "joint_covered_50": [True, False, True, False],
            "joint_covered_90": [True, True, True, True],
            "prediction_radius_m": [4.0, 5.0, 6.0, 7.0],
            "mean_marginal_interval_width_m": [2.0, 3.0, 4.0, 5.0],
            "window_runtime_seconds": [0.2, 0.2, 0.4, 0.4],
            "converged": [case.method == "vi_expanding"] * 4,
            "mcmc_diagnostics_ok": [pd.NA] * 4,
        }
    )


def test_configuration_rejects_empty_or_invalid_benchmark_dimensions():
    with pytest.raises(ValueError, match="run_ids"):
        _config(run_ids=())
    with pytest.raises(ValueError, match="methods"):
        _config(methods=("unknown",))
    with pytest.raises(ValueError, match="prediction_counts"):
        _config(prediction_counts=(0,))
    with pytest.raises(ValueError, match="position_noise_std_m"):
        _config(position_noise_std_m=(-1.0,))


def test_case_generation_is_deterministic_and_excludes_unselected_mcmc():
    config = _config(methods=("mcmc_expanding", "rbpf", "deterministic"))

    cases = build_benchmark_cases(config)

    assert tuple(
        method for method in BENCHMARK_METHODS if method in config.methods
    ) == (
        "deterministic",
        "rbpf",
        "mcmc_expanding",
    )
    assert [case.method for case in cases[:3]] == [
        "deterministic",
        "rbpf",
        "mcmc_expanding",
    ]
    assert len(cases) == 2 * 2 * 2 * 3


def test_mcmc_subset_limits_runs_horizons_and_selected_forecast_origins():
    config = _config(
        methods=("vi_expanding", "mcmc_expanding"),
        mcmc_run_ids=(12,),
        mcmc_prediction_counts=(3,),
        mcmc_window_indices=(2, 5),
    )

    cases = build_benchmark_cases(config)

    mcmc_cases = [case for case in cases if case.method == "mcmc_expanding"]
    assert {(case.run_id, case.prediction_count) for case in mcmc_cases} == {(12, 3)}
    assert mcmc_cases[0].selected_window_indices == (2, 5)


def test_execution_preserves_case_metadata_and_writes_stable_csv_schemas(tmp_path):
    config = _config(run_ids=(11,), prediction_counts=(3,), position_noise_std_m=(5.0,))
    received_cases = []

    def runner(case):
        received_cases.append(case)
        return _prediction_table(case)

    result = execute_benchmark(config, output_directory=tmp_path, case_runner=runner)

    assert len(received_cases) == 2
    assert result.runs["status"].tolist() == ["success", "success"]
    assert set(PREDICTIONS_REQUIRED_COLUMNS).issubset(result.predictions.columns)
    assert result.predictions["run_id"].eq(11).all()
    assert result.predictions["position_noise_seed"].eq(2026).all()
    assert result.predictions["inference_seed"].eq(42).all()
    assert list(pd.read_csv(tmp_path / "runs.csv").columns) == list(RUN_COLUMNS)
    assert list(pd.read_csv(tmp_path / "summary.csv").columns) == list(SUMMARY_COLUMNS)
    assert list(pd.read_csv(tmp_path / "per_horizon.csv").columns) == list(
        PER_HORIZON_COLUMNS
    )
    assert (tmp_path / "predictions.csv").is_file()


def test_methods_in_one_case_receive_identical_noise_configuration(tmp_path):
    config = _config(run_ids=(11,), prediction_counts=(3,), position_noise_std_m=(5.0,))
    received_cases = []

    def runner(case):
        received_cases.append(case)
        return _prediction_table(case)

    execute_benchmark(config, output_directory=tmp_path, case_runner=runner)

    assert {
        (
            case.run_id,
            case.prediction_count,
            case.position_noise_std_m,
            case.position_noise_seed,
        )
        for case in received_cases
    } == {(11, 3, 5.0, 2026)}


def test_failed_case_is_recorded_and_later_cases_continue(tmp_path):
    config = _config(run_ids=(11,), prediction_counts=(1,), position_noise_std_m=(0.0,))

    def runner(case):
        if case.method == "deterministic":
            raise RuntimeError("least squares failed")
        return _prediction_table(case)

    result = execute_benchmark(config, output_directory=tmp_path, case_runner=runner)

    assert result.runs["status"].tolist() == ["failed", "success"]
    assert "least squares failed" in result.runs.loc[0, "error_message"]
    assert result.predictions["method"].unique().tolist() == ["vi_expanding"]


def test_summary_and_per_horizon_aggregate_prediction_metrics(tmp_path):
    config = _config(run_ids=(11,), prediction_counts=(3,), position_noise_std_m=(5.0,))

    result = execute_benchmark(
        config,
        output_directory=tmp_path,
        case_runner=_prediction_table,
    )

    vi_summary = result.summary.query("method == 'vi_expanding'").iloc[0]
    assert vi_summary["mean_ade_m"] == pytest.approx(2.5)
    assert vi_summary["median_position_error_m"] == pytest.approx(2.5)
    assert vi_summary["mean_maximum_horizon_fde_m"] == pytest.approx(3.5)
    assert vi_summary["mean_energy_score_m"] == pytest.approx(1.25)
    assert vi_summary["joint_coverage_90"] == pytest.approx(1.0)

    step_three = result.per_horizon.query(
        "method == 'vi_expanding' and horizon_step == 3"
    ).iloc[0]
    assert step_three["mean_horizon_seconds"] == pytest.approx(30.0)
    assert step_three["mean_position_error_m"] == pytest.approx(3.5)


def test_summary_fde_uses_only_the_maximum_configured_horizon_step(tmp_path):
    config = _config(run_ids=(11,), prediction_counts=(3,), position_noise_std_m=(5.0,))

    def runner(case):
        table = _prediction_table(case)
        return table.loc[table.index != 3].reset_index(drop=True)

    result = execute_benchmark(config, output_directory=tmp_path, case_runner=runner)

    summary = result.summary.query("method == 'vi_expanding'").iloc[0]
    assert summary["mean_maximum_horizon_fde_m"] == pytest.approx(3.0)


def test_csv_outputs_round_floating_point_values_to_three_decimal_places(tmp_path):
    config = _config(run_ids=(11,), prediction_counts=(3,), position_noise_std_m=(5.0,))

    def runner(case):
        table = _prediction_table(case)
        table.loc[0, "position_error_m"] = 1.23456
        return table

    execute_benchmark(config, output_directory=tmp_path, case_runner=runner)

    predictions_csv = (tmp_path / "predictions.csv").read_text(encoding="utf-8")
    assert "1.235" in predictions_csv
    assert "1.23456" not in predictions_csv
