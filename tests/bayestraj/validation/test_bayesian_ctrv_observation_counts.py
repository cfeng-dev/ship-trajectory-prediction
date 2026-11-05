"""Observation-count sweeps reuse the rolling evaluation and CSV exporter."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bayestraj.validation.bayesian_ctrv_evaluation as evaluation
import bayestraj.validation.rolling as rolling

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def sweep_run(tmp_path, monkeypatch):
    monkeypatch.setattr("bayestraj.paths.project_path", lambda path: tmp_path / path)
    calls = []
    monkeypatch.setattr(
        evaluation.workflow.br24_observations,
        "read_br24_tracking_input",
        lambda path: tuple(range(24)),
    )

    def run_workflow(**kwargs):
        calls.append(kwargs)
        options = kwargs["options"]
        windows = rolling.build_online_forecast_specs(
            24,
            initial_observation_count=2,
            prediction_count=options.prediction_count,
            stride=options.stride,
        )
        selected = kwargs["selected_window_indices"]
        windows = [window for window in windows if window.window_index in selected]
        if options.max_windows is not None:
            windows = windows[: options.max_windows]
        assert kwargs.get("association_records") is None, (
            "Sweeps must not record associations"
        )
        predictions = pd.DataFrame(
            [
                {
                    "window_index": window.window_index,
                    "forecast_start_index": window.forecast_start_index,
                    "target_index": window.forecast_start_index + step - 1,
                    "forecast_origin_seconds": float(window.forecast_start_index - 1),
                    "horizon_step": step,
                    "horizon_seconds": float(step),
                    "observation_count": window.observation_count,
                    "configured_observation_count": options.observation_count,
                    "window_cycle_count": min(
                        window.observation_count, options.observation_count
                    ),
                    "window_accepted_measurement_count": min(
                        window.observation_count, options.observation_count
                    )
                    - 1,
                    "position_error_m": float(step),
                    "energy_score_m": 1.5,
                    "prediction_radius_m": 5.0,
                    "radial_covered": True,
                    "joint_covered_50": step == 1,
                    "joint_covered_90": True,
                    "mean_marginal_interval_width_m": 6.0,
                    "negative_log_predictive_density": 2.5,
                    "inference_mode": "batch",
                    "inference_method": options.inference_method,
                    "converged": True if options.inference_method == "vi" else None,
                    "mcmc_diagnostics_ok": (
                        True if options.inference_method == "mcmc" else None
                    ),
                    "mcmc_rhat_max": (
                        1.005 if options.inference_method == "mcmc" else np.nan
                    ),
                    "fit_runtime_seconds": 0.1,
                    "window_runtime_seconds": 0.2,
                }
                for window in windows
                for step in range(1, window.prediction_count + 1)
            ]
        )
        return predictions, rolling.summarize_rolling_predictions(predictions)

    monkeypatch.setattr(
        evaluation.workflow, "run_bayesian_ctrv_evaluation", run_workflow
    )
    return calls


def _script(method):
    return runpy.run_path(
        ROOT / f"scripts/inference_evaluation/evaluate_{method}_observation_counts.py"
    )


@pytest.mark.parametrize("method", ["mcmc", "vi"])
def test_scripts_sweep_two_to_twenty_with_common_origins_and_method_csvs(
    tmp_path, monkeypatch, sweep_run, method
):
    namespace = _script(method)
    assert "ASSOCIATIONS_OUTPUT_DIR" not in namespace
    monkeypatch.setitem(namespace["main"].__globals__, "MAX_WINDOWS", 2)
    monkeypatch.chdir(tmp_path)

    summary = namespace["main"]([])

    assert [call["options"].observation_count for call in sweep_run] == list(
        range(2, 21)
    )
    assert {call["options"].inference_method for call in sweep_run} == {method}
    selected = sweep_run[0]["selected_window_indices"]
    assert selected == tuple(range(22))
    for call in sweep_run:
        assert call["selected_window_indices"] == selected
        assert call["options"].max_windows == 2
        assert call["options"].inference_seed == 42
        assert call["options"].prediction_count == 3
        assert call["options"].position_noise_std_m == 0.0
        assert call["priors"] is namespace["PRIORS"]
        assert call["experiment"].observation_count == call["options"].observation_count
        assert "show_plot" not in call
        assert not hasattr(call["options"], "plot_each_window")

    output = tmp_path / f"results/inference_evaluation/bayesian_ctrv_{method}"
    assert not (tmp_path / "results/associations").exists()
    assert sorted(path.name for path in output.iterdir()) == [
        *(f"bayesian_ctrv_{method}_{count:02d}.csv" for count in range(2, 21)),
        "summary.csv",
    ]
    for count in range(2, 21):
        table = pd.read_csv(output / f"bayesian_ctrv_{method}_{count:02d}.csv")
        assert table.columns.tolist() == list(evaluation.PREDICTION_CSV_COLUMNS)
        assert table["configured_observation_count"].tolist() == [count] * 6
        assert table["observation_count"].tolist() == [2] * 3 + [3] * 3
        assert table["forecast_start_index"].tolist() == [2] * 3 + [3] * 3
        assert "evaluation_start_observation_count" not in table
        assert table["window_cycle_count"].tolist() == [2] * 3 + [min(3, count)] * 3
        assert (
            table["window_accepted_measurement_count"].tolist()
            == [1] * 3 + [min(3, count) - 1] * 3
        )

    saved = pd.read_csv(output / "summary.csv")
    assert saved["observation_count"].tolist() == list(range(2, 21))
    assert "evaluation_start_observation_count" not in saved
    assert "evaluation_start_observation_count" not in summary
    assert saved["window_count"].tolist() == [2] * 19
    assert saved["forecast_count"].tolist() == [6] * 19
    np.testing.assert_allclose(saved["ade_m"], 2.0)
    np.testing.assert_allclose(saved["fde_m"], 3.0)
    np.testing.assert_allclose(saved["rmse_m"], np.sqrt(14 / 3))
    np.testing.assert_allclose(saved["mean_energy_score_m"], 1.5)
    np.testing.assert_allclose(saved["mean_negative_log_predictive_density"], 2.5)
    if method == "mcmc":
        np.testing.assert_allclose(saved["mcmc_rhat_max"], 1.005)
    else:
        assert saved["mcmc_rhat_max"].isna().all()
    assert "per_horizon_table" not in saved
    pd.testing.assert_frame_equal(
        saved, summary.where(summary.notna(), np.nan), check_dtype=False
    )


@pytest.mark.parametrize("method", ["mcmc", "vi"])
def test_sweeps_use_the_same_public_csv_api(tmp_path, monkeypatch, sweep_run, method):
    exports = []
    original_export = evaluation.export_evaluation_csv

    def export(table, path, *, summary=False):
        exports.append((path.name, summary))
        return original_export(table, path, summary=summary)

    monkeypatch.setattr(evaluation, "export_evaluation_csv", export)

    _script(method)["main"](
        ["--min-observations", "2", "--max-observations", "3", "--max-windows", "1"]
    )

    assert exports == [
        (f"bayesian_ctrv_{method}_02.csv", False),
        (f"bayesian_ctrv_{method}_03.csv", False),
        ("summary.csv", True),
    ]


@pytest.mark.parametrize("method", ["mcmc", "vi"])
def test_complete_sweep_has_normal_evaluation_row_count(tmp_path, sweep_run, method):
    summary = _script(method)["main"](
        ["--min-observations", "20", "--max-observations", "20"]
    )

    # A 24-cycle route evaluated from N=2 has 22 origins and horizons 3, ..., 3, 2, 1.
    assert summary["window_count"].tolist() == [22]
    assert summary["forecast_count"].tolist() == [63]
    path = (
        tmp_path
        / f"results/inference_evaluation/bayesian_ctrv_{method}/bayesian_ctrv_{method}_20.csv"
    )
    table = pd.read_csv(path)
    assert len(table) == 63
    assert table["forecast_start_index"].iloc[0] == 2
    assert table["forecast_start_index"].iloc[-1] == 23
    assert table["window_cycle_count"].iloc[0] == 2
    assert table["window_cycle_count"].iloc[-1] == 20


@pytest.mark.parametrize("method", ["mcmc", "vi"])
def test_sweep_cli_can_limit_counts_origins_and_output_directory(
    tmp_path, sweep_run, method
):
    namespace = _script(method)
    destination = tmp_path / "limited"

    summary = namespace["main"](
        [
            "--min-observations",
            "3",
            "--max-observations",
            "4",
            "--max-windows",
            "1",
            "--output-dir",
            str(destination),
        ]
    )

    assert [call["options"].observation_count for call in sweep_run] == [3, 4]
    assert "evaluation_start_observation_count" not in summary
    assert summary["window_count"].tolist() == [1, 1]
    assert (destination / f"bayesian_ctrv_{method}_03.csv").is_file()
    assert (destination / f"bayesian_ctrv_{method}_04.csv").is_file()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--min-observations", "0"],
        ["--min-observations", "1"],
        ["--min-observations", "5", "--max-observations", "4"],
        ["--max-windows", "0"],
    ],
)
def test_invalid_sweep_limits_fail_before_any_inference(sweep_run, arguments):
    with pytest.raises(SystemExit):
        _script("vi")["main"](arguments)
    assert not sweep_run


def test_sweep_can_evaluate_before_the_largest_window_fills(tmp_path, sweep_run):
    summary = _script("vi")["main"](
        ["--min-observations", "24", "--max-observations", "24", "--max-windows", "1"]
    )

    assert "evaluation_start_observation_count" not in summary
    assert summary["observation_count"].tolist() == [24]
    output = (
        tmp_path
        / "results/inference_evaluation/bayesian_ctrv_vi/bayesian_ctrv_vi_24.csv"
    )
    table = pd.read_csv(output)
    assert table["forecast_start_index"].tolist() == [2] * 3
    assert table["window_cycle_count"].tolist() == [2] * 3


def test_sweep_rejects_data_without_future_positions(tmp_path, monkeypatch, sweep_run):
    monkeypatch.setattr(
        evaluation.workflow.br24_observations,
        "read_br24_tracking_input",
        lambda path: tuple(range(2)),
    )
    with pytest.raises(ValueError, match="future"):
        _script("vi")["main"]([])
    assert not sweep_run
    assert not (tmp_path / "results").exists()


def test_sweep_stops_on_csv_export_failure(tmp_path, monkeypatch, sweep_run):
    monkeypatch.setattr(evaluation, "_export_csv", lambda *args: False)

    with pytest.raises(OSError, match="export"):
        _script("vi")["main"](["--max-windows", "1"])

    assert len(sweep_run) == 1
    assert not (
        tmp_path / "results/inference_evaluation/bayesian_ctrv_vi/summary.csv"
    ).exists()
