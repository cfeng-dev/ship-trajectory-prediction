"""CSV-only comparison of matched Bayesian CTRV and IMM forecasts."""

import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/model_evaluation/compare_bayesian_imm.py"


def _table(model):
    errors = [3.0, 4.0, 5.0, 12.0] if model == "bayesian" else [6.0, 8.0, 8.0, 15.0]
    origin = np.array([100.0, 100.0, 120.0, 120.0])
    horizons = np.array([2.5, 5.0, 2.5, 5.0])
    actual_x = np.array([10.0, 12.0, 20.0, 22.0])
    return pd.DataFrame(
        {
            "window_index": [0, 0, 1, 1] if model == "bayesian" else [10, 10, 20, 20],
            "forecast_origin_seconds": origin,
            "horizon_step": [1, 2, 1, 2],
            "horizon_seconds": horizons,
            "prediction_count": 2,
            "time": pd.to_datetime(origin + horizons, unit="s", utc=True),
            "x_actual": actual_x,
            "y_actual": 0.0,
            "x_median": actual_x + errors,
            "y_median": 0.0,
            "position_error_m": errors,
            "energy_score_m": np.array([2.0, 4.0, 6.0, 10.0]) + (model == "imm"),
            "log_predictive_density": np.array([-1.0, -3.0, -5.0, -7.0])
            - (model == "imm"),
            "fit_runtime_seconds": [0.01, 0.01, 0.02, 0.02]
            if model == "bayesian"
            else [0.002, 0.002, 0.004, 0.004],
            "inference_method": "smc" if model == "bayesian" else "imm",
            "model_variant": model,
        }
    )


def test_comparison_aligns_by_origin_and_horizon_not_window_id_or_row_order():
    module = runpy.run_path(SCRIPT)
    bayesian = _table("bayesian").iloc[[3, 0, 2, 1]]
    imm = _table("imm").iloc[[2, 1, 3, 0]]
    original_bayesian, original_imm = bayesian.copy(deep=True), imm.copy(deep=True)

    left, right = module["prepare_comparison"](bayesian, imm)

    assert left["window_index"].tolist() == [0, 0, 1, 1]
    assert right["window_index"].tolist() == [10, 10, 20, 20]
    assert (
        left["horizon_step"].tolist() == right["horizon_step"].tolist() == [1, 2, 1, 2]
    )
    pd.testing.assert_frame_equal(bayesian, original_bayesian)
    pd.testing.assert_frame_equal(imm, original_imm)


def test_comparison_uses_only_complete_common_forecast_origins():
    module = runpy.run_path(SCRIPT)
    imm = _table("imm").iloc[2:]
    with pytest.warns(UserWarning, match="omitted|unmatched"):
        bayesian, matched_imm = module["prepare_comparison"](_table("bayesian"), imm)
    assert bayesian["forecast_origin_seconds"].tolist() == [120, 120]
    assert matched_imm["forecast_origin_seconds"].tolist() == [120, 120]


def test_comparison_accepts_equivalent_target_time_zones_and_resolutions():
    module = runpy.run_path(SCRIPT)
    bayesian = _table("bayesian")
    bayesian["time"] = bayesian["time"].dt.tz_convert("Europe/Berlin").dt.as_unit("us")
    left, right = module["prepare_comparison"](bayesian, _table("imm"))
    assert (left["time"] == right["time"]).all()


def test_comparison_does_not_truncate_different_complete_forecast_horizons():
    module = runpy.run_path(SCRIPT)
    imm = _table("imm").iloc[[0, 2]].copy()
    imm["prediction_count"] = 1
    with pytest.raises(ValueError, match="identical complete horizons"):
        module["prepare_comparison"](_table("bayesian"), imm)


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("empty", "empty"),
        ("missing", "Missing"),
        ("nonfinite", "finite"),
        ("negative", "non-negative"),
        ("wrong_model", "model_variant"),
        ("mixed_method", "inference_method"),
        ("nonstring_method", "inference_method"),
        ("wrong_method", "inference_method"),
        ("duplicate", "Duplicate|duplicate"),
        ("incomplete", "horizon|prediction_count"),
        ("inconsistent_runtime", "runtime"),
        ("incorrect_error", "median|position_error"),
        ("inconsistent_origin", "window|origin"),
        ("fractional_step", "integer|horizon"),
    ],
)
def test_comparison_rejects_invalid_tables_before_plotting(problem, message):
    module = runpy.run_path(SCRIPT)
    table = _table("bayesian")
    if problem == "empty":
        table = table.iloc[:0]
    elif problem == "missing":
        table = table.drop(columns="energy_score_m")
    elif problem == "nonfinite":
        table.loc[0, "position_error_m"] = np.inf
    elif problem == "negative":
        table.loc[0, "position_error_m"] = -1
    elif problem == "wrong_model":
        table["model_variant"] = "imm"
    elif problem == "mixed_method":
        table.loc[0, "inference_method"] = "mcmc"
    elif problem == "nonstring_method":
        table["inference_method"] = 5
    elif problem == "wrong_method":
        table["inference_method"] = "imm"
    elif problem == "duplicate":
        table = pd.concat([table, table.iloc[:1]], ignore_index=True)
    elif problem == "incomplete":
        table = table.drop(index=1)
    elif problem == "inconsistent_runtime":
        table.loc[0, "fit_runtime_seconds"] = 99
    elif problem == "incorrect_error":
        table.loc[0, "position_error_m"] = 99
    elif problem == "inconsistent_origin":
        table.loc[0, "forecast_origin_seconds"] = 99
    elif problem == "fractional_step":
        table["horizon_step"] = table["horizon_step"].astype(float)
        table.loc[0, "horizon_step"] = 1.5
    existing_figures = plt.get_fignums()
    with pytest.raises(ValueError, match=message):
        module["prepare_comparison"](table, _table("imm"))
    assert plt.get_fignums() == existing_figures


@pytest.mark.parametrize("problem", ["reference", "horizons", "time", "no_common"])
def test_comparison_rejects_incompatible_forecasts(problem):
    module = runpy.run_path(SCRIPT)
    imm = _table("imm")
    if problem == "reference":
        imm["x_actual"] += 100
        imm["x_median"] += 100
    elif problem == "horizons":
        imm["horizon_seconds"] += 0.5
        imm["time"] = pd.to_datetime(
            imm["forecast_origin_seconds"] + imm["horizon_seconds"], unit="s", utc=True
        )
    elif problem == "time":
        imm["time"] = pd.to_datetime(
            imm["forecast_origin_seconds"] + imm["horizon_seconds"] + 10,
            unit="s",
            utc=True,
        )
    else:
        imm["forecast_origin_seconds"] += 100
        imm["time"] = pd.to_datetime(
            imm["forecast_origin_seconds"] + imm["horizon_seconds"], unit="s", utc=True
        )
    with pytest.raises(ValueError, match="reference|horizon|time|common"):
        module["prepare_comparison"](_table("bayesian"), imm)


def test_comparison_plots_six_paired_metrics_and_correct_window_means(monkeypatch):
    module = runpy.run_path(SCRIPT)
    left, right = module["prepare_comparison"](_table("bayesian"), _table("imm"))
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    summary, figures = module["plot_comparison"](left, right)
    try:
        assert len(figures) == len(shown) == 6
        indexed = summary.set_index("metric")
        assert indexed.loc["ade_m", "bayesian_ctrv"] == 6.0
        assert indexed.loc["ade_m", "imm"] == 9.25
        assert indexed.loc["fde_m", "bayesian_ctrv"] == 8.0
        assert indexed.loc["mean_log_predictive_density", "bayesian_ctrv"] == -4.0
        assert indexed.loc["mean_log_predictive_density", "preferred"] == "higher"
        assert indexed.loc["fit_runtime_seconds", "bayesian_ctrv"] == 0.015
        expected = (
            ([np.sqrt(12.5), np.sqrt(84.5)], [np.sqrt(50), np.sqrt(144.5)]),
            ([-2, -6], [-3, -7]),
            ([3.5, 8.5], [7, 11.5]),
            ([4, 12], [8, 15]),
            ([3, 8], [4, 9]),
            ([0.01, 0.02], [0.002, 0.004]),
        )
        for (figure, axis), (bayesian_values, imm_values) in zip(
            figures, expected, strict=True
        ):
            assert figure._suptitle.get_text() == "Bayesian CTRV vs. IMM"
            assert figure._suptitle.get_fontweight() == "bold"
            assert figure._suptitle.get_fontsize() == 13
            assert axis.get_xlabel() == "Zeit [s]"
            assert len(axis.lines) == 4
            np.testing.assert_allclose(axis.lines[0].get_xdata(), [0, 20])
            np.testing.assert_allclose(axis.lines[0].get_ydata(), bayesian_values)
            np.testing.assert_allclose(axis.lines[2].get_ydata(), imm_values)
            assert axis.lines[0].get_color() != axis.lines[2].get_color()
            assert (
                axis.lines[1].get_linestyle() == axis.lines[3].get_linestyle() == "--"
            )
            np.testing.assert_allclose(
                axis.lines[1].get_ydata(), np.mean(bayesian_values)
            )
            np.testing.assert_allclose(axis.lines[3].get_ydata(), np.mean(imm_values))
    finally:
        for figure, _ in figures:
            plt.close(figure)


def test_comparison_cli_reads_csv_and_saves_plots_without_running_models(
    tmp_path, monkeypatch, capsys
):
    module = runpy.run_path(SCRIPT)
    from imm_predictor import player

    from bayestraj.validation import bayesian_ctrv_workflow

    def forbidden(*args, **kwargs):
        pytest.fail("CSV comparison must never run a model")

    monkeypatch.setattr(
        bayesian_ctrv_workflow, "run_bayesian_ctrv_evaluation", forbidden
    )
    monkeypatch.setattr(player, "_load_optional_imm_tracker", forbidden)
    monkeypatch.setattr(plt, "show", forbidden)
    bayesian_csv, imm_csv = tmp_path / "bayesian.csv", tmp_path / "imm.csv"
    _table("bayesian").to_csv(bayesian_csv, index=False)
    _table("imm").to_csv(imm_csv, index=False)
    originals = [path.read_bytes() for path in (bayesian_csv, imm_csv)]
    output = tmp_path / "plots"
    previous_figures = plt.get_fignums()
    summary = module["main"](
        [
            "--bayesian-csv",
            str(bayesian_csv),
            "--imm-csv",
            str(imm_csv),
            "--output-dir",
            str(output),
            "--no-show",
        ]
    )
    assert len(summary) == 6
    assert len(list(output.glob("*.png"))) == 6
    assert all(path.stat().st_size > 1_000 for path in output.glob("*.png"))
    assert [path.read_bytes() for path in (bayesian_csv, imm_csv)] == originals
    assert plt.get_fignums() == previous_figures
    report = capsys.readouterr().out
    assert "Matched forecast origins: 2" in report
    assert "Gaussian" in report and "lower" in report and "higher" in report


def test_comparison_reports_missing_csv_without_creating_outputs(tmp_path, capsys):
    module = runpy.run_path(SCRIPT)
    with pytest.raises(SystemExit) as result:
        module["main"](["--bayesian-csv", str(tmp_path / "absent.csv"), "--no-show"])
    assert result.value.code == 2
    assert "CSV" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("show_diagnostics", [False, True])
def test_bayesian_evaluator_exports_predictions_before_diagnostic_plots(
    tmp_path, monkeypatch, show_diagnostics
):
    module = runpy.run_path(ROOT / "experiments/model_evaluation/bayesian_ctrv.py")
    namespace = module["main"].__globals__
    predictions = _table("bayesian")
    original = predictions.copy(deep=True)
    summary = object()
    output = tmp_path / "results" / "bayesian.csv"
    monkeypatch.setitem(namespace, "OUTPUT_CSV", output)
    monkeypatch.setitem(namespace, "SHOW_ROLLING_DIAGNOSTICS_PLOT", show_diagnostics)
    monkeypatch.setattr(
        namespace["workflow"],
        "run_bayesian_ctrv_evaluation",
        lambda **kwargs: (predictions, summary),
    )

    def plot(table, *, inference_selection):
        assert output.is_file()
        np.testing.assert_array_equal(
            pd.read_csv(output)["position_error_m"], [3, 4, 5, 12]
        )

    monkeypatch.setattr(
        namespace["runtime_plotting"], "plot_bayesian_ctrv_rolling_diagnostics", plot
    )
    returned, returned_summary = module["main"]([])
    assert returned is predictions and returned_summary is summary
    assert output.is_file()
    exported = pd.read_csv(output)
    exported["time"] = pd.to_datetime(exported["time"], format="mixed", utc=True)
    pd.testing.assert_frame_equal(exported, original)
    pd.testing.assert_frame_equal(predictions, original)
