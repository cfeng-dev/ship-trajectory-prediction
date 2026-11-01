"""CSV-only comparison of matched Bayesian CTRV and IMM forecasts."""

import runpy
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import bayestraj.validation.bayesian_ctrv_evaluation as evaluation
from bayestraj.validation import comparison, runtime_plotting

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/model_comparison/compare_bayesian_ctrv_imm.py"
MODEL_CSVS = (
    ("IMM_CSV", "imm", "IMM"),
    ("BAYESIAN_SMC_CSV", "smc", "Bayesian CTRV (SMC)"),
    ("BAYESIAN_VI_CSV", "vi", "Bayesian CTRV (VI)"),
    ("BAYESIAN_MCMC_CSV", "mcmc", "Bayesian CTRV (MCMC)"),
    ("BAYESIAN_SEQ_VI_CSV", "sequential_vi", "Bayesian CTRV (Sequential VI)"),
)


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
            "joint_covered_50": [True, False, False, True]
            if model == "bayesian"
            else [False, False, True, False],
            "joint_covered_90": [True, True, False, True]
            if model == "bayesian"
            else [True, False, True, True],
        }
    )


def _write_model_csvs(directory):
    paths = {}
    for index, (variable, method, _label) in enumerate(MODEL_CSVS):
        table = _table("imm" if method == "imm" else "bayesian")
        table["inference_method"] = method
        if method != "imm":
            table["position_error_m"] += index - 1
            table["x_median"] = table["x_actual"] + table["position_error_m"]
        # Labels must come from the configured variable, not the filename.
        path = directory / f"run_{index}.csv"
        table.to_csv(path, index=False)
        paths[variable] = path
    return paths


def _configure_pair(monkeypatch, namespace, bayesian_csv, imm_csv):
    for variable, _method, _label in MODEL_CSVS:
        monkeypatch.setitem(namespace, variable, None)
    monkeypatch.setitem(namespace, "BAYESIAN_SMC_CSV", bayesian_csv)
    monkeypatch.setitem(namespace, "IMM_CSV", imm_csv)


def test_named_csvs_load_all_five_methods_without_modifying_files(tmp_path):
    paths = _write_model_csvs(tmp_path)
    originals = {key: path.read_bytes() for key, path in paths.items()}

    models = comparison.read_model_comparison_csvs(paths)

    assert list(models) == [label for _, _, label in MODEL_CSVS]
    for variable, method, label in MODEL_CSVS:
        assert models[label]["inference_method"].unique().tolist() == [method]
        assert paths[variable].read_bytes() == originals[variable]


def test_named_comparison_supports_bayesian_only_subsets_and_none_paths(tmp_path):
    paths = _write_model_csvs(tmp_path)
    paths.update(IMM_CSV=None, BAYESIAN_MCMC_CSV=None, BAYESIAN_SEQ_VI_CSV=None)

    models = comparison.read_model_comparison_csvs(paths)

    assert list(models) == ["Bayesian CTRV (SMC)", "Bayesian CTRV (VI)"]


@pytest.mark.parametrize("enabled", [0, 1])
def test_named_comparison_requires_at_least_two_models_before_reading(
    tmp_path, enabled
):
    paths = {
        variable: tmp_path / "missing.csv" if index < enabled else None
        for index, (variable, _, _) in enumerate(MODEL_CSVS)
    }
    with pytest.raises(ValueError, match="at least two"):
        comparison.read_model_comparison_csvs(paths)


def test_named_comparison_rejects_metadata_that_disagrees_with_the_variable(tmp_path):
    paths = _write_model_csvs(tmp_path)
    paths["BAYESIAN_SMC_CSV"] = paths["BAYESIAN_VI_CSV"]

    with pytest.raises(ValueError, match="BAYESIAN_SMC_CSV.*smc"):
        comparison.read_model_comparison_csvs(paths)


def test_named_comparison_uses_one_global_intersection_for_all_models(tmp_path):
    paths = _write_model_csvs(tmp_path)
    vi = pd.read_csv(paths["BAYESIAN_VI_CSV"]).iloc[2:]
    vi.to_csv(paths["BAYESIAN_VI_CSV"], index=False)

    with pytest.warns(UserWarning, match="omitted|unmatched"):
        models = comparison.read_model_comparison_csvs(paths)

    assert all(
        table["forecast_origin_seconds"].tolist() == [120, 120]
        for table in models.values()
    )


def test_named_comparison_rejects_pairwise_overlap_without_a_global_cohort(tmp_path):
    paths = _write_model_csvs(tmp_path)
    for variable, rows in (
        ("BAYESIAN_SMC_CSV", slice(0, 2)),
        ("BAYESIAN_VI_CSV", slice(2, 4)),
    ):
        table = pd.read_csv(paths[variable]).iloc[rows]
        table.to_csv(paths[variable], index=False)

    with pytest.raises(ValueError, match="common"):
        comparison.read_model_comparison_csvs(paths)


@pytest.mark.parametrize("problem", ["reference", "horizon", "incomplete"])
def test_named_comparison_checks_the_last_model_as_well_as_the_first_pair(
    tmp_path, problem
):
    paths = _write_model_csvs(tmp_path)
    path = paths["BAYESIAN_SEQ_VI_CSV"]
    table = pd.read_csv(path)
    if problem == "reference":
        table["x_actual"] += 1.0
        table["x_median"] += 1.0
    elif problem == "horizon":
        table["horizon_seconds"] += 0.25
        table["time"] = pd.to_datetime(
            table["forecast_origin_seconds"] + table["horizon_seconds"],
            unit="s",
            utc=True,
        )
    else:
        table = table[table["horizon_step"] == 1].copy()
        table["prediction_count"] = 1
    table.to_csv(path, index=False)

    with pytest.raises(ValueError, match="reference|horizon"):
        comparison.read_model_comparison_csvs(paths)


def test_all_models_share_eight_plots_and_keep_their_labels_and_colors(tmp_path):
    paths = _write_model_csvs(tmp_path)
    models = comparison.read_model_comparison_csvs(paths)
    labels = list(models)
    output_dir = tmp_path / "plots"
    summary, figures = runtime_plotting.plot_model_comparison(
        models, show=False, output_dir=output_dir
    )
    try:
        assert len(figures) == 8
        assert [axis.get_ylabel() for _, axis in figures[:6]] == [
            "Root mean squared error [m]",
            "Negative log predictive density",
            "Average displacement error [m]",
            "Final displacement error [m]",
            "Energy Score [m]",
            "Laufzeit [s]",
        ]
        assert list(summary) == ["metric", *labels, "preferred"]
        indexed = summary.set_index("metric")
        assert indexed.loc["ade_m", labels].tolist() == [9.25, 6.0, 7.0, 8.0, 9.0]
        for _, axis in figures[:6]:
            curves = axis.lines[::2]
            assert [line.get_label() for line in curves] == labels
            assert len({line.get_color() for line in curves}) == 5
            np.testing.assert_array_equal(curves[0].get_xdata(), [0, 20])
        for (_figure, axis), nominal in zip(figures[-2:], (50, 90), strict=True):
            assert len(axis.patches) == 5
            assert axis.get_ylim() == (0.0, 100.0)
            assert axis.lines[0].get_label() == f"Nominal {nominal}%"
        assert len(list(output_dir.glob("*.png"))) == 8
        print(f"COMPARISON_PREVIEW={output_dir / 'ade_m.png'}")
    finally:
        for figure, _ in figures:
            plt.close(figure)


@pytest.mark.parametrize("show_points", [False, True])
@pytest.mark.parametrize("show_mean_lines", [False, True])
@pytest.mark.parametrize("metric_y_scale", ["linear", "symlog"])
def test_run_button_compares_all_five_csvs_without_running_models_or_writing_outputs(
    tmp_path, monkeypatch, show_points, show_mean_lines, metric_y_scale
):
    from bayestraj.validation import imm_workflow

    paths = _write_model_csvs(tmp_path)
    originals = {path: path.read_bytes() for path in paths.values()}
    namespace = runpy.run_path(SCRIPT)["main"].__globals__
    for variable, path in paths.items():
        monkeypatch.setitem(namespace, variable, path)
    monkeypatch.setitem(namespace, "SHOW_POINTS", show_points)
    monkeypatch.setitem(namespace, "SHOW_MEAN_LINES", show_mean_lines)
    monkeypatch.setitem(namespace, "METRIC_Y_SCALE", metric_y_scale)

    def forbidden(*args, **kwargs):
        pytest.fail("CSV comparison must not run inference")

    monkeypatch.setattr(evaluation.workflow, "run_bayesian_ctrv_evaluation", forbidden)
    monkeypatch.setattr(imm_workflow, "run_imm_evaluation", forbidden)
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    previous = set(plt.get_fignums())
    try:
        summary = namespace["main"]()

        assert len(shown) == 8
        assert all(
            figure.axes[0].get_yscale() == metric_y_scale for figure in shown[:6]
        )
        assert list(summary) == [
            "metric",
            *[label for _, _, label in MODEL_CSVS],
            "preferred",
        ]
        labels = list(summary.columns[1:-1])
        assert summary.set_index("metric").loc["ade_m", labels].tolist() == [
            9.25,
            6.0,
            7.0,
            8.0,
            9.0,
        ]
        for figure in shown[:6]:
            axis = figure.axes[0]
            lower, upper = axis.get_ylim()
            assert lower <= 0 <= upper
            assert 0 in axis.get_yticks()
            curves = [line for line in axis.lines if line.get_label() in labels]
            assert [line.get_label() for line in curves] == labels
            assert all(
                line.get_marker() == ("o" if show_points else "None") for line in curves
            )
            means = [line for line in axis.lines if line.get_linestyle() == "--"]
            assert len(means) == (5 if show_mean_lines else 0)
            legend = [text.get_text() for text in axis.get_legend().get_texts()]
            assert any("Mittel =" in text for text in legend) == show_mean_lines
        for figure, nominal in zip(shown[-2:], (50, 90), strict=True):
            axis = figure.axes[0]
            assert len(axis.patches) == 5
            assert len(axis.lines) == 1
            assert axis.get_yscale() == "linear"
            np.testing.assert_allclose(axis.lines[0].get_ydata(), nominal)
        assert set(tmp_path.iterdir()) == set(paths.values())
        assert {path: path.read_bytes() for path in paths.values()} == originals
    finally:
        for number in set(plt.get_fignums()) - previous:
            plt.close(number)


def test_symlog_metrics_keep_negative_zero_and_extreme_scores_visible(tmp_path):
    bayesian, imm = _table("bayesian"), _table("imm")
    bayesian["log_predictive_density"] = [2.0, 2.0, 0.0, 0.0]
    imm["log_predictive_density"] = [-5.0, -5.0, -3800.0, -3800.0]
    imm["fit_runtime_seconds"] = [0.002, 0.002, 10000.0, 10000.0]
    left, right = comparison.prepare_comparison(bayesian, imm)
    originals = [table.copy(deep=True) for table in (left, right)]
    summary, figures = runtime_plotting.plot_model_comparison(
        {"SMC": left, "IMM": right}, show=False, metric_y_scale="symlog"
    )
    try:
        figure, axis = figures[1]
        assert axis.get_yscale() == "symlog"
        assert axis.get_ylabel() == "Negative log predictive density"
        assert all("(symlog)" not in ax.get_ylabel() for _, ax in figures)
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), [-2.0, 0.0])
        np.testing.assert_array_equal(axis.lines[2].get_ydata(), [5.0, 3800.0])
        indexed = summary.set_index("metric")
        assert indexed.loc["mean_negative_log_predictive_density", "SMC"] == -1.0
        assert indexed.loc["mean_negative_log_predictive_density", "IMM"] == 1902.5
        figure.canvas.draw()
        typical_pixels = axis.transData.transform([[0, 0], [0, 5]])[:, 1]
        assert typical_pixels[1] - typical_pixels[0] > 20
        lower, upper = axis.get_ylim()
        assert lower <= -2.0 and upper >= 3800.0
        assert all(ax.get_yscale() == "symlog" for _, ax in figures[:6])
        assert all(ax.get_yscale() == "linear" for _, ax in figures[6:])
        runtime_figure, runtime_axis = figures[5]
        runtime_figure.canvas.draw()
        runtime_pixels = runtime_axis.transData.transform([[0, 0], [0, 0.002]])[:, 1]
        assert runtime_pixels[1] - runtime_pixels[0] > 10
        np.testing.assert_array_equal(
            runtime_axis.lines[2].get_ydata(), [0.002, 10000.0]
        )
        figure.savefig(tmp_path / "nlpd_symlog.png", dpi=120)
        for table, original in zip((left, right), originals, strict=True):
            pd.testing.assert_frame_equal(table, original)
    finally:
        for figure, _ in figures:
            plt.close(figure)


@pytest.mark.parametrize("metric_y_scale", ["linear", "symlog"])
def test_zero_axis_reference_preserves_negative_density_scores(metric_y_scale):
    bayesian, imm = _table("bayesian"), _table("imm")
    bayesian["log_predictive_density"] = [2.0, 2.0, 6.0, 6.0]
    imm["log_predictive_density"] = [3.0, 3.0, 7.0, 7.0]
    left, right = comparison.prepare_comparison(bayesian, imm)
    _, figures = runtime_plotting.plot_model_comparison(
        {"SMC": left, "IMM": right}, show=False, metric_y_scale=metric_y_scale
    )
    try:
        axis = figures[1][1]
        lower, upper = axis.get_ylim()
        assert lower <= -7.0 and upper >= 0.0
        assert 0 in axis.get_yticks()
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), [-2.0, -6.0])
        np.testing.assert_array_equal(axis.lines[2].get_ydata(), [-3.0, -7.0])
    finally:
        for figure, _ in figures:
            plt.close(figure)


def test_invalid_metric_scale_is_rejected_before_creating_plots():
    previous = plt.get_fignums()
    with pytest.raises(ValueError, match="metric_y_scale.*linear.*symlog"):
        runtime_plotting.plot_model_comparison(
            {"SMC": _table("bayesian"), "IMM": _table("imm")}, metric_y_scale="log"
        )
    assert plt.get_fignums() == previous


def test_comparison_aligns_by_origin_and_horizon_not_window_id_or_row_order():
    bayesian = _table("bayesian").iloc[[3, 0, 2, 1]]
    imm = _table("imm").iloc[[2, 1, 3, 0]]
    original_bayesian, original_imm = bayesian.copy(deep=True), imm.copy(deep=True)

    left, right = comparison.prepare_comparison(bayesian, imm)

    assert left["window_index"].tolist() == [0, 0, 1, 1]
    assert right["window_index"].tolist() == [10, 10, 20, 20]
    assert (
        left["horizon_step"].tolist() == right["horizon_step"].tolist() == [1, 2, 1, 2]
    )
    pd.testing.assert_frame_equal(bayesian, original_bayesian)
    pd.testing.assert_frame_equal(imm, original_imm)


@pytest.mark.parametrize("legacy_model", ["bayesian", "imm", None])
def test_comparison_normalizes_old_and_new_density_columns_exactly_once(legacy_model):
    tables = [_table("bayesian"), _table("imm")]
    for table in tables:
        table.loc[0, "log_predictive_density"] = 1.5
    expected = [-table["log_predictive_density"].copy() for table in tables]
    for model, table, values in zip(("bayesian", "imm"), tables, expected, strict=True):
        if model != legacy_model:
            table["negative_log_predictive_density"] = values
            table.drop(columns="log_predictive_density", inplace=True)
    originals = [table.copy(deep=True) for table in tables]

    for table, values in zip(
        comparison.prepare_comparison(*tables), expected, strict=True
    ):
        np.testing.assert_allclose(table["negative_log_predictive_density"], values)
        assert "log_predictive_density" not in table
    for table, original in zip(tables, originals, strict=True):
        pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize("consistent", [True, False])
def test_comparison_checks_duplicated_density_conventions(consistent):
    table = _table("bayesian")
    table["negative_log_predictive_density"] = -table["log_predictive_density"]
    if consistent:
        left, _ = comparison.prepare_comparison(table, _table("imm"))
        np.testing.assert_allclose(
            left["negative_log_predictive_density"], [1, 3, 5, 7]
        )
    else:
        table.loc[0, "negative_log_predictive_density"] = 100.0
        with pytest.raises(ValueError, match="Conflicting"):
            comparison.prepare_comparison(table, _table("imm"))


def test_comparison_uses_only_complete_common_forecast_origins():
    imm = _table("imm").iloc[2:]
    with pytest.warns(UserWarning, match="omitted|unmatched"):
        bayesian, matched_imm = comparison.prepare_comparison(_table("bayesian"), imm)
    assert bayesian["forecast_origin_seconds"].tolist() == [120, 120]
    assert matched_imm["forecast_origin_seconds"].tolist() == [120, 120]


def test_comparison_accepts_equivalent_target_time_zones_and_resolutions():
    bayesian = _table("bayesian")
    bayesian["time"] = bayesian["time"].dt.tz_convert("Europe/Berlin").dt.as_unit("us")
    left, right = comparison.prepare_comparison(bayesian, _table("imm"))
    assert (left["time"] == right["time"]).all()


def test_comparison_does_not_truncate_different_complete_forecast_horizons():
    imm = _table("imm").iloc[[0, 2]].copy()
    imm["prediction_count"] = 1
    with pytest.raises(ValueError, match="identical complete horizons"):
        comparison.prepare_comparison(_table("bayesian"), imm)


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
        comparison.prepare_comparison(table, _table("imm"))
    assert plt.get_fignums() == existing_figures


@pytest.mark.parametrize("problem", ["reference", "horizons", "time", "no_common"])
def test_comparison_rejects_incompatible_forecasts(problem):
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
        comparison.prepare_comparison(_table("bayesian"), imm)


def test_comparison_plots_eight_paired_metrics_and_correct_window_means(monkeypatch):
    left, right = comparison.prepare_comparison(_table("bayesian"), _table("imm"))
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    summary, figures = runtime_plotting.plot_bayesian_imm_comparison(left, right)
    try:
        assert len(figures) == len(shown) == 8
        indexed = summary.set_index("metric")
        assert indexed.loc["ade_m", "bayesian_ctrv"] == 6.0
        assert indexed.loc["ade_m", "imm"] == 9.25
        assert indexed.loc["fde_m", "bayesian_ctrv"] == 8.0
        assert (
            indexed.loc["mean_negative_log_predictive_density", "bayesian_ctrv"] == 4.0
        )
        assert (indexed.iloc[:6]["preferred"] == "lower").all()
        assert indexed.loc["joint_coverage_50", "preferred"] == "near 50%"
        assert indexed.loc["joint_coverage_90", "preferred"] == "near 90%"
        assert indexed.loc["joint_coverage_50", "bayesian_ctrv"] == 0.5
        assert indexed.loc["joint_coverage_50", "imm"] == 0.25
        assert indexed.loc["joint_coverage_90", "bayesian_ctrv"] == 0.75
        assert indexed.loc["joint_coverage_90", "imm"] == 0.75
        assert figures[1][1].get_ylabel() == "Negative log predictive density"
        assert indexed.loc["fit_runtime_seconds", "bayesian_ctrv"] == 0.015
        expected = (
            ([np.sqrt(12.5), np.sqrt(84.5)], [np.sqrt(50), np.sqrt(144.5)]),
            ([2, 6], [3, 7]),
            ([3.5, 8.5], [7, 11.5]),
            ([4, 12], [8, 15]),
            ([3, 8], [4, 9]),
            ([0.01, 0.02], [0.002, 0.004]),
        )
        for (figure, axis), (bayesian_values, imm_values) in zip(
            figures[:6], expected, strict=True
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
        for (_, axis), nominal, values in zip(
            figures[6:], (50, 90), ([50, 25], [75, 75]), strict=True
        ):
            assert axis.get_xlabel() == ""
            assert axis.get_ylabel() == f"Joint {nominal}% coverage [%]"
            np.testing.assert_allclose(axis.get_ylim(), [0, 100])
            assert [tick.get_text() for tick in axis.get_xticklabels()] == [
                "Bayesian CTRV (SMC)",
                "IMM",
            ]
            np.testing.assert_allclose(
                [bar.get_height() for bar in axis.patches], values
            )
            assert [text.get_text() for text in axis.texts] == [
                f"{value:.1f}%" for value in values
            ]
            assert len(axis.lines) == 1
            np.testing.assert_allclose(axis.lines[0].get_ydata(), nominal)
            assert axis.lines[0].get_linestyle() == ":"
            assert axis.lines[0].get_label() == f"Nominal {nominal}%"
    finally:
        for figure, _ in figures:
            plt.close(figure)


@pytest.mark.parametrize(
    "problem", ["missing", "nonboolean", "missing_value", "nonnested"]
)
def test_comparison_rejects_invalid_coverage_flags_before_plotting(problem):
    table = _table("bayesian")
    if problem == "missing":
        table = table.drop(columns="joint_covered_50")
    elif problem == "nonnested":
        table.loc[0, "joint_covered_90"] = False
    else:
        table["joint_covered_50"] = table["joint_covered_50"].astype(object)
        table.loc[0, "joint_covered_50"] = (
            "False" if problem == "nonboolean" else np.nan
        )
    previous_figures = plt.get_fignums()
    with pytest.raises(ValueError, match="joint_covered|coverage"):
        comparison.prepare_comparison(table, _table("imm"))
    assert plt.get_fignums() == previous_figures


def test_final_coverage_counts_all_matched_predictions_with_different_horizon_lengths():
    tables = [_table(model).drop(index=3) for model in ("bayesian", "imm")]
    for table in tables:
        table["prediction_count"] = table.groupby("window_index")[
            "horizon_step"
        ].transform("size")
    left, right = comparison.prepare_comparison(*tables)
    summary, figures = runtime_plotting.plot_bayesian_imm_comparison(
        left, right, show=False
    )
    try:
        row = summary.set_index("metric").loc["joint_coverage_50"]
        assert row["bayesian_ctrv"] == pytest.approx(1 / 3)
        assert row["imm"] == pytest.approx(1 / 3)
        np.testing.assert_allclose(
            [bar.get_height() for bar in figures[6][1].patches], [100 / 3] * 2
        )
        assert [text.get_text() for text in figures[6][1].texts] == ["33.3%", "33.3%"]
    finally:
        for figure, _ in figures:
            plt.close(figure)


def test_comparison_script_reads_csv_without_running_models_or_saving_plots(
    tmp_path, monkeypatch, capsys
):
    module = runpy.run_path(SCRIPT)
    namespace = module["main"].__globals__
    monkeypatch.setattr("sys.argv", [str(SCRIPT)])
    from imm_predictor import player

    from bayestraj.validation import bayesian_ctrv_workflow

    def forbidden(*args, **kwargs):
        pytest.fail("CSV comparison must not run models")

    monkeypatch.setattr(
        bayesian_ctrv_workflow, "run_bayesian_ctrv_evaluation", forbidden
    )
    monkeypatch.setattr(player, "_load_optional_imm_tracker", forbidden)
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    bayesian_csv, imm_csv = tmp_path / "bayesian.csv", tmp_path / "imm.csv"
    _table("bayesian").to_csv(bayesian_csv, index=False)
    _table("imm").to_csv(imm_csv, index=False)
    originals = [path.read_bytes() for path in (bayesian_csv, imm_csv)]
    _configure_pair(monkeypatch, namespace, bayesian_csv, imm_csv)
    previous_figures = set(plt.get_fignums())
    try:
        summary = module["main"]()

        assert len(summary) == 8
        assert len(shown) == 8
        assert set(tmp_path.iterdir()) == {bayesian_csv, imm_csv}
        assert [path.read_bytes() for path in (bayesian_csv, imm_csv)] == originals
        assert len(set(plt.get_fignums()) - previous_figures) == 8
        report = capsys.readouterr().out
        assert "Matched forecast origins: 2" in report
        assert "Gaussian" in report and "near 50%" in report and "near 90%" in report
        assert "50.0%" in report and "25.0%" in report
        assert "mean_negative_log_predictive_density" in report
    finally:
        for figure_number in set(plt.get_fignums()) - previous_figures:
            plt.close(figure_number)


@pytest.mark.parametrize("show", [False, True])
def test_shared_comparison_reads_csvs_and_preserves_reports_and_figure_lifetime(
    tmp_path, monkeypatch, capsys, show
):
    bayesian_csv, imm_csv = tmp_path / "bayesian.csv", tmp_path / "imm.csv"
    _table("bayesian").iloc[::-1].to_csv(bayesian_csv, index=False)
    _table("imm").iloc[::-1].to_csv(imm_csv, index=False)
    original_csvs = [path.read_bytes() for path in (bayesian_csv, imm_csv)]
    previous_figures = set(plt.get_fignums())
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))

    try:
        bayesian, imm = comparison.read_comparison_csvs(str(bayesian_csv), imm_csv)
        summary = comparison.run_comparison(bayesian, imm, show=show)

        assert bayesian["forecast_origin_seconds"].tolist() == [100, 100, 120, 120]
        assert imm["window_index"].tolist() == [10, 10, 20, 20]
        assert len(summary) == 8
        indexed = summary.set_index("metric")
        assert indexed.loc["ade_m", "bayesian_ctrv"] == 6.0
        assert indexed.loc["ade_m", "imm"] == 9.25
        assert indexed.loc["fde_m", "bayesian_ctrv"] == 8.0
        assert indexed.loc["fde_m", "imm"] == 11.5
        assert indexed.loc["fit_runtime_seconds", "bayesian_ctrv"] == 0.015
        assert indexed.loc["fit_runtime_seconds", "imm"] == 0.003
        assert len(shown) == (8 if show else 0)
        assert len(set(plt.get_fignums()) - previous_figures) == (8 if show else 0)
        assert [path.read_bytes() for path in (bayesian_csv, imm_csv)] == original_csvs
        assert set(tmp_path.iterdir()) == {bayesian_csv, imm_csv}
        report = capsys.readouterr().out
        assert "Matched forecast origins: 2" in report
        assert "Matched forecast positions: 4" in report
        assert "Metrics for matched forecasts" in report
        assert "Gaussian" in report and "posterior samples" in report
    finally:
        for figure_number in set(plt.get_fignums()) - previous_figures:
            plt.close(figure_number)


def test_comparison_script_shows_plots_using_configured_csvs(
    tmp_path, monkeypatch, capsys
):
    module = runpy.run_path(SCRIPT)
    namespace = module["main"].__globals__
    monkeypatch.setattr("sys.argv", [str(SCRIPT)])
    bayesian_csv, imm_csv = (
        tmp_path / "configured_bayesian.csv",
        tmp_path / "configured_imm.csv",
    )
    _table("bayesian").to_csv(bayesian_csv, index=False)
    _table("imm").to_csv(imm_csv, index=False)
    _configure_pair(monkeypatch, namespace, bayesian_csv, imm_csv)
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    previous_figures = set(plt.get_fignums())

    try:
        summary = module["main"]()

        indexed = summary.set_index("metric")
        assert indexed.loc["ade_m", "Bayesian CTRV (SMC)"] == 6.0
        assert indexed.loc["ade_m", "IMM"] == 9.25
        assert len(shown) == 8
        assert "Matched forecast positions: 4" in capsys.readouterr().out
        assert set(tmp_path.iterdir()) == {bayesian_csv, imm_csv}
    finally:
        for figure_number in set(plt.get_fignums()) - previous_figures:
            plt.close(figure_number)


def test_comparison_reports_missing_csv_without_creating_outputs(tmp_path, monkeypatch):
    module = runpy.run_path(SCRIPT)
    missing_csv = tmp_path / "absent.csv"
    monkeypatch.setattr("sys.argv", [str(SCRIPT)])
    _configure_pair(
        monkeypatch,
        module["main"].__globals__,
        tmp_path / "another_missing.csv",
        missing_csv,
    )
    with pytest.raises(SystemExit) as result:
        module["main"]()
    assert "Evaluation CSV not found" in str(result.value.code)
    assert str(missing_csv) in str(result.value.code)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("show_diagnostics", [False, True])
def test_bayesian_evaluator_exports_predictions_before_diagnostic_plots(
    tmp_path, monkeypatch, show_diagnostics
):
    module = runpy.run_path(ROOT / "experiments/model_evaluation/bayesian_ctrv.py")
    settings = module["main"].__globals__
    monkeypatch.setitem(
        settings, "EXPERIMENT", replace(settings["EXPERIMENT"], inference_method="smc")
    )
    predictions = _table("bayesian")
    original = predictions.copy(deep=True)
    summary = object()
    output = tmp_path / "results/bayesian_ctrv_smc_evaluation.csv"
    monkeypatch.setitem(settings, "OUTPUT_CSV", output)
    monkeypatch.setitem(settings, "SHOW_ROLLING_DIAGNOSTICS_PLOT", show_diagnostics)
    monkeypatch.setattr(
        evaluation.workflow,
        "run_bayesian_ctrv_evaluation",
        lambda **kwargs: (predictions, summary),
    )
    plotted = []

    def plot(table, *, inference_selection):
        assert output.is_file()
        np.testing.assert_array_equal(
            pd.read_csv(output)["position_error_m"], [3, 4, 5, 12]
        )
        plotted.append(inference_selection)

    monkeypatch.setattr(
        evaluation.runtime_plotting, "plot_bayesian_ctrv_rolling_diagnostics", plot
    )
    returned, returned_summary = module["main"]([])
    assert returned is predictions and returned_summary is summary
    assert plotted == (["smc"] if show_diagnostics else [])
    assert output.is_file()
    exported = pd.read_csv(output)
    exported["time"] = pd.to_datetime(exported["time"], format="mixed", utc=True)
    pd.testing.assert_frame_equal(exported, original)
    pd.testing.assert_frame_equal(predictions, original)
