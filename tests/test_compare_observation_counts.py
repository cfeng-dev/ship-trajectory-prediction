"""Compare observation counts by recomputing summaries from forecast CSVs."""

import hashlib
import json
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bayestraj.validation import bayesian_ctrv_evaluation as evaluation
from bayestraj.validation import comparison, runtime_plotting

ROOT = Path(__file__).resolve().parents[1]
METRICS = tuple(runtime_plotting.OBSERVATION_COUNT_METRICS)
MCMC_METRICS = (*METRICS, "mcmc_rhat_max")


def _predictions(method, count):
    rows = []
    for window, horizons, origin in [(0, 3, 10.0), (1, 1, 11.0)]:
        for step in range(1, horizons + 1):
            error = float(count * (step if window == 0 else 4))
            rows.append(
                dict(
                    window_index=window,
                    forecast_origin_seconds=origin,
                    horizon_step=step,
                    horizon_seconds=float(step),
                    time=pd.Timestamp(origin + step, unit="s", tz="UTC"),
                    prediction_count=horizons,
                    observation_count=window + 2,
                    configured_observation_count=count,
                    x_actual=origin + step,
                    y_actual=0.0,
                    x_median=origin + step + error,
                    y_median=0.0,
                    position_error_m=error,
                    energy_score_m=error / 2,
                    negative_log_predictive_density=step - 0.5,
                    prediction_radius_m=5.0,
                    radial_covered=True,
                    joint_covered_50=step == 1,
                    joint_covered_90=True,
                    mean_marginal_interval_width_m=6.0,
                    inference_mode="batch",
                    inference_method=method,
                    model_variant="bayesian",
                    is_prior_predictive=False,
                    converged=window == 0 if method == "vi" else None,
                    mcmc_diagnostics_ok=window == 0 if method == "mcmc" else None,
                    mcmc_rhat_max=(1.002 if window == 0 else 1.010) + count / 1000
                    if method == "mcmc"
                    else np.nan,
                    fit_runtime_seconds=(window + 1) * count / 10,
                    window_runtime_seconds=(1 if window == 0 else 3) * float(count),
                )
            )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def no_inference(monkeypatch):
    def fail(**kwargs):
        pytest.fail("CSV comparison must never start inference")

    monkeypatch.setattr(evaluation.workflow, "run_bayesian_ctrv_evaluation", fail)


@pytest.fixture
def result_dirs(tmp_path):
    directories = {}
    for method in ["mcmc", "vi"]:
        directory = tmp_path / f"results/inference_evaluation/bayesian_ctrv_{method}"
        directory.mkdir(parents=True)
        for count in [4, 2, 3]:
            assert evaluation.export_evaluation_csv(
                _predictions(method, count),
                directory / f"bayesian_ctrv_{method}_{count:02d}.csv",
            )
        (directory / "summary.csv").write_text("obsolete summary", encoding="utf-8")
        directories[method] = directory
    return directories


def _read(directories):
    return comparison.read_observation_count_evaluations(
        directories["mcmc"], directories["vi"]
    )


def _edit(directories, column, value, method="vi", count=3):
    path = directories[method] / f"bayesian_ctrv_{method}_{count:02d}.csv"
    table = pd.read_csv(path)
    table[column] = value
    table.to_csv(path, index=False)


def _metadata(directory, method):
    records = {}
    for count in [2, 3, 4]:
        path = directory / f"bayesian_ctrv_{method}_{count:02d}.csv"
        records[path.name] = dict(
            data_file="BR24.csv",
            inference_seed=42,
            inference_method=method,
            observation_count=count,
            prediction_count=3,
            credible_interval=0.9,
            position_noise_std_m=0.0,
            position_noise_seed=2026,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    path = directory / "metadata.json"
    path.write_text(json.dumps(dict(version=1, evaluations=records)), encoding="utf-8")
    return path, records


def test_reader_recomputes_metrics_and_counts_runtime_once_per_window(result_dirs):
    models = _read(result_dirs)
    assert list(models) == ["MCMC", "VI"]
    for method, table in models.items():
        counts = np.array([2, 3, 4])
        assert table["observation_count"].tolist() == counts.tolist()
        assert table["window_count"].tolist() == [2] * 3
        assert table["forecast_count"].tolist() == [4] * 3
        np.testing.assert_allclose(table["ade_m"], counts * 2.5)
        np.testing.assert_allclose(table["fde_m"], counts * 3.0)
        np.testing.assert_allclose(table["rmse_m"], counts * np.sqrt(7.5))
        np.testing.assert_allclose(table["mean_energy_score_m"], counts * 1.25)
        np.testing.assert_allclose(table["mean_negative_log_predictive_density"], 1.25)
        np.testing.assert_allclose(table["mean_window_runtime_seconds"], counts * 2.0)
        np.testing.assert_allclose(table["mean_fit_runtime_seconds"], counts * 0.15)
        np.testing.assert_allclose(
            table["total_computation_time_seconds"], counts * 4.0
        )
        np.testing.assert_allclose(table["joint_coverage_50"], 0.5)
        np.testing.assert_allclose(table["joint_coverage_90"], 1.0)
        assert table["data_file"].isna().all() and table["inference_seed"].isna().all()
        if method == "MCMC":
            np.testing.assert_allclose(table["mcmc_rhat_max"], 1.010 + counts / 1000)
            np.testing.assert_allclose(table["mcmc_diagnostics_pass_rate"], 0.5)
        else:
            assert table["mcmc_rhat_max"].isna().all()
            np.testing.assert_allclose(table["vi_convergence_rate"], 0.5)


@pytest.mark.parametrize("method", ["mcmc", "vi"])
def test_comparisons_use_the_same_fit_runtime_per_window(result_dirs, method):
    table = _predictions(method, 3)
    _, series = runtime_plotting._rolling_window_metric_series(table)
    sweep = _read(result_dirs)[method.upper()].set_index("observation_count")

    assert "mean_fit_runtime_seconds" in runtime_plotting.OBSERVATION_COUNT_METRICS
    assert (
        "mean_window_runtime_seconds" not in runtime_plotting.OBSERVATION_COUNT_METRICS
    )
    assert sweep.loc[3, "mean_fit_runtime_seconds"] == pytest.approx(
        series["fit_runtime_seconds"].mean()
    )
    # Unequal horizons must not weight one window's runtime more heavily.
    assert sweep.loc[3, "mean_fit_runtime_seconds"] == pytest.approx(0.45)
    assert sweep.loc[3, "mean_window_runtime_seconds"] == pytest.approx(6.0)


@pytest.mark.parametrize("method", ["mcmc", "vi"])
def test_reader_accepts_a_partial_sweep_and_one_method(result_dirs, method):
    directory = result_dirs[method]
    extra = directory / f"bayesian_ctrv_{method}_04.csv"
    extra.rename(directory / "unrelated.csv")
    models = comparison.read_observation_count_evaluations(
        **{f"{method}_dir": directory}
    )
    assert list(models) == [method.upper()]
    assert models[method.upper()]["observation_count"].tolist() == [2, 3]


def test_reader_requires_at_least_one_directory():
    with pytest.raises(ValueError, match="at least one"):
        comparison.read_observation_count_evaluations()


def test_reader_requires_forecast_csvs_even_if_summary_exists(tmp_path):
    (tmp_path / "summary.csv").write_text("old summary", encoding="utf-8")
    with pytest.raises(ValueError, match="forecast CSV"):
        comparison.read_observation_count_evaluations(mcmc_dir=tmp_path)


@pytest.mark.parametrize(
    "column,value",
    [
        ("configured_observation_count", 9),
        ("inference_method", "smc"),
        ("inference_mode", "online"),
        ("model_variant", "imm"),
        ("position_error_m", np.nan),
        ("joint_covered_90", 1.1),
        ("energy_score_m", np.nan),
        ("window_runtime_seconds", -1.0),
        ("mcmc_rhat_max", np.inf),
        ("is_prior_predictive", "invalid"),
        ("converged", "invalid"),
        ("horizon_step", 1.5),
    ],
)
def test_reader_rejects_invalid_forecast_inputs(result_dirs, column, value):
    _edit(result_dirs, column, value)
    with pytest.raises(ValueError):
        _read(result_dirs)


@pytest.mark.parametrize(
    "change", ["origins", "reference", "incomplete", "missing_window"]
)
def test_reader_rejects_different_forecast_cohorts(result_dirs, change):
    path = result_dirs["vi"] / "bayesian_ctrv_vi_03.csv"
    table = pd.read_csv(path)
    if change == "origins":
        table["forecast_origin_seconds"] += 100
        table["time"] = pd.to_datetime(
            table["forecast_origin_seconds"] + table["horizon_seconds"],
            unit="s",
            utc=True,
        )
    elif change == "reference":
        table["x_actual"] += 1
        table["x_median"] += 1
    elif change == "incomplete":
        table = table.iloc[1:]
    else:
        table = table[table["window_index"] == 0]
    table.to_csv(path, index=False)
    with pytest.raises(ValueError):
        _read(result_dirs)


def test_reader_rejects_duplicate_sizes(result_dirs):
    directory = result_dirs["mcmc"]
    (directory / "bayesian_ctrv_mcmc_2.csv").write_bytes(
        (directory / "bayesian_ctrv_mcmc_02.csv").read_bytes()
    )
    with pytest.raises(ValueError, match="Duplicate"):
        _read(result_dirs)


def test_reader_preserves_negative_log_predictive_density(result_dirs):
    _edit(result_dirs, "negative_log_predictive_density", -2.0)
    table = _read(result_dirs)["VI"]
    assert (
        table.loc[
            table["observation_count"] == 3, "mean_negative_log_predictive_density"
        ].iloc[0]
        == -2.0
    )


def test_reader_accepts_boolean_flags_encoded_as_zero_and_one(result_dirs):
    for field, value in [("is_prior_predictive", 0), ("radial_covered", 1)]:
        _edit(result_dirs, field, value)

    table = _read(result_dirs)["VI"]

    assert table.loc[table["observation_count"] == 3, "radial_coverage"].iloc[0] == 1.0


def test_reader_aligns_common_sizes_with_warning(result_dirs):
    path = result_dirs["vi"] / "bayesian_ctrv_vi_03.csv"
    path.rename(path.with_suffix(".backup"))
    with pytest.warns(UserWarning, match="common observation counts"):
        models = _read(result_dirs)
    assert all(
        table["observation_count"].tolist() == [2, 4] for table in models.values()
    )


def test_reader_preserves_verified_provenance(result_dirs):
    for method, directory in result_dirs.items():
        _metadata(directory, method)
    models = _read(result_dirs)
    assert all(
        table["data_file"].tolist() == ["BR24.csv"] * 3 for table in models.values()
    )
    assert all(
        table["inference_seed"].tolist() == [42] * 3 for table in models.values()
    )


@pytest.mark.parametrize(
    "field,value",
    [("data_file", "other.csv"), ("inference_seed", 43), ("prediction_count", 4)],
)
def test_reader_rejects_mixed_run_metadata(result_dirs, field, value):
    path, records = _metadata(result_dirs["vi"], "vi")
    records["bayesian_ctrv_vi_03.csv"][field] = value
    path.write_text(json.dumps(dict(version=1, evaluations=records)), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        _read(result_dirs)


def test_reader_rejects_stale_metadata_after_csv_changes(result_dirs):
    _metadata(result_dirs["vi"], "vi")
    _edit(result_dirs, "negative_log_predictive_density", -2.0)
    with pytest.raises(ValueError, match="metadata"):
        _read(result_dirs)


@pytest.mark.parametrize("method", ["mcmc", "vi"])
def test_prior_predictive_windows_do_not_require_fit_diagnostics(result_dirs, method):
    _edit(result_dirs, "is_prior_predictive", True, method=method)
    _edit(
        result_dirs,
        "mcmc_diagnostics_ok" if method == "mcmc" else "converged",
        np.nan,
        method=method,
    )
    _edit(result_dirs, "mcmc_rhat_max", np.nan, method=method)

    table = _read(result_dirs)[method.upper()]

    diagnostic = (
        "mcmc_diagnostics_pass_rate" if method == "mcmc" else "vi_convergence_rate"
    )
    assert table.loc[table["observation_count"] == 3, diagnostic].isna().all()


@pytest.mark.parametrize("metric_y_scale", ["linear", "symlog"])
@pytest.mark.parametrize("plot_language", ["de", "en"])
def test_plot_uses_recomputed_metrics_and_integer_ticks(
    result_dirs, tmp_path, metric_y_scale, plot_language
):
    models = _read(result_dirs)
    figures = runtime_plotting.plot_observation_count_comparison(
        models,
        show=False,
        output_dir=tmp_path,
        metric_y_scale=metric_y_scale,
        plot_language=plot_language,
    )
    try:
        assert len(figures) == len(MCMC_METRICS)
        for metric, (figure, axis) in zip(MCMC_METRICS, figures, strict=True):
            assert axis.get_xlabel() == (
                "Beobachtungsfenstergröße T [Zeitschritte]"
                if plot_language == "de"
                else "Observation window size T [time steps]"
            )
            selected = {"MCMC": models["MCMC"]} if metric == "mcmc_rhat_max" else models
            curves = axis.lines[: len(selected)]
            assert [line.get_label() for line in curves] == list(selected)
            scale = 100 if metric.startswith("joint_coverage_") else 1
            for line, table in zip(curves, selected.values(), strict=True):
                np.testing.assert_array_equal(line.get_xdata(), [2, 3, 4])
                np.testing.assert_allclose(line.get_ydata(), table[metric] * scale)
                assert (
                    line.get_color()
                    == runtime_plotting.COMPARISON_INFERENCE_COLORS[
                        table["inference_method"].iloc[0]
                    ]
                )
            if scale == 100:
                assert axis.get_ylim() == (0.0, 100.0)
            assert axis.get_yscale() == ("linear" if scale == 100 else metric_y_scale)
            if metric_y_scale == "symlog":
                lower, upper = axis.get_ylim()
                assert lower <= 0 <= upper
                assert 0 in axis.get_yticks()
            width, height = figure.get_size_inches()
            expected_width, expected_height = (
                runtime_plotting.RUNTIME_PLOT_STYLE.speed_figure_size
            )
            assert width == expected_width and height >= expected_height
            figure.canvas.draw()
            assert (tmp_path / f"{metric}.png").is_file()
    finally:
        for figure, _ in figures:
            plt.close(figure)


@pytest.mark.parametrize(
    "values",
    [
        [-2.0, 0.0, 200.0],
        [0.0, 0.0, 0.0],
        [12.0, 13.0, 700.0],
        [-200.0, -100.0, -2.0],
    ],
)
def test_symlog_preserves_negative_and_zero_metrics(result_dirs, values):
    models = _read(result_dirs)
    for table in models.values():
        table["mean_negative_log_predictive_density"] = values
    figures = runtime_plotting.plot_observation_count_comparison(
        models, show=False, metric_y_scale="symlog"
    )
    try:
        axis = figures[4][1]
        assert axis.get_yscale() == "symlog"
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), values)
        axis.figure.canvas.draw()
        assert np.isfinite(axis.get_ylim()).all()
        lower, upper = axis.get_ylim()
        assert lower <= min(0.0, *values)
        assert upper >= max(0.0, *values)
        assert 0 in axis.get_yticks()
    finally:
        for figure, _ in figures:
            plt.close(figure)


def test_plot_rejects_unsupported_metric_y_scale(result_dirs):
    with pytest.raises(ValueError, match="metric_y_scale"):
        runtime_plotting.plot_observation_count_comparison(
            _read(result_dirs), show=False, metric_y_scale="log"
        )


@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("plot_kind", ["models", "observation_counts"])
def test_comparison_figures_finish_layout_before_display(
    result_dirs, monkeypatch, interactive, plot_kind
):
    original_interactive = plt.isinteractive()
    original_subplots = plt.subplots
    created, shown = [], []

    def create_hidden(*args, **kwargs):
        assert not plt.isinteractive(), "Figure creation must not open a GUI window"
        figure, axis = original_subplots(*args, **kwargs)
        created.append(figure)
        return figure, axis

    def show_finished():
        assert plt.isinteractive() == interactive
        figure = plt.gcf()
        assert not figure.stale, "Draw the final layout before showing the window"
        shown.append(figure)

    monkeypatch.setattr(plt, "subplots", create_hidden)
    monkeypatch.setattr(plt, "show", show_finished)
    try:
        plt.interactive(interactive)
        if plot_kind == "models":
            _, figures = runtime_plotting.plot_model_comparison(
                {method.upper(): _predictions(method, 3) for method in ("mcmc", "vi")},
                show_summary_table=True,
            )
        else:
            figures = runtime_plotting.plot_observation_count_comparison(
                _read(result_dirs), show_summary_table=True
            )
        assert shown == created == [figure for figure, _ in figures]
        assert plt.isinteractive() == interactive
    finally:
        for figure in created:
            plt.close(figure)
        plt.interactive(original_interactive)


@pytest.mark.parametrize("missing", [False, True])
def test_plot_skips_rhat_without_defined_mcmc_diagnostics(result_dirs, missing):
    models = _read(result_dirs)
    if missing:
        models["MCMC"] = models["MCMC"].drop(columns="mcmc_rhat_max")
    else:
        models["MCMC"]["mcmc_rhat_max"] = np.nan
    figures = runtime_plotting.plot_observation_count_comparison(models, show=False)
    try:
        assert len(figures) == len(METRICS)
    finally:
        for figure, _ in figures:
            plt.close(figure)


def test_rhat_plot_preserves_missing_values_and_highlights_nearest_one(result_dirs):
    models = _read(result_dirs)
    models["MCMC"]["mcmc_rhat_max"] = [0.98, np.nan, 1.005]
    figures = runtime_plotting.plot_observation_count_comparison(
        models, show=False, show_summary_table=True, metric_y_scale="symlog"
    )
    try:
        _, axis = figures[-2]
        assert "R-hat" in axis.get_ylabel()
        assert [line.get_label() for line in axis.lines] == ["MCMC"]
        np.testing.assert_allclose(axis.lines[0].get_ydata(), [0.98, np.nan, 1.005])
        assert np.isfinite(axis.get_ylim()).all()
        assert 0 in axis.get_yticks()
        cells = figures[-1][1].tables[0].get_celld()
        assert cells[1, 9].get_text().get_text() == "0.980"
        assert cells[3, 9].get_text().get_text() == "1.005"
        assert cells[1, 9].get_text().get_weight() != "bold"
        assert cells[3, 9].get_text().get_weight() == "bold"
    finally:
        for figure, _ in figures:
            plt.close(figure)


def test_singleton_comparison_keeps_integer_observation_count_ticks(result_dirs):
    models = {label: table.iloc[:1] for label, table in _read(result_dirs).items()}
    figures = runtime_plotting.plot_observation_count_comparison(models, show=False)
    try:
        for _, axis in figures:
            ticks = axis.get_xticks()
            np.testing.assert_array_equal(ticks, np.floor(ticks))
    finally:
        for figure, _ in figures:
            plt.close(figure)


@pytest.mark.parametrize("method", ["mcmc", "vi"])
@pytest.mark.parametrize("metric_y_scale", ["linear", "symlog"])
@pytest.mark.parametrize("show_summary_table", [False, True])
def test_comparison_entrypoint_reads_forecasts_and_generates_summary(
    result_dirs, tmp_path, monkeypatch, method, metric_y_scale, show_summary_table
):
    monkeypatch.setattr("bayestraj.paths.project_path", lambda path: tmp_path / path)
    original_read = comparison._read_csv

    def read_selected(path):
        assert path.parent == result_dirs[method] and path.name != "summary.csv"
        return original_read(path)

    monkeypatch.setattr(comparison, "_read_csv", read_selected)
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    script = ROOT / f"scripts/model_comparison/compare_{method}_observation_counts.py"
    try:
        module = runpy.run_path(script)
        monkeypatch.setitem(
            module["main"].__globals__, "METRIC_Y_SCALE", metric_y_scale
        )
        if not show_summary_table:
            monkeypatch.setitem(module["main"].__globals__, "SHOW_SUMMARY_TABLE", False)
        models = module["main"]()
        saved = pd.read_csv(result_dirs[method] / "summary.csv")
        expected = (
            models[method.upper()]
            .astype(object)
            .where(models[method.upper()].notna(), np.nan)
        )
        pd.testing.assert_frame_equal(saved, expected, check_dtype=False)
        assert saved["observation_count"].tolist() == [2, 3, 4]
        metrics = MCMC_METRICS if method == "mcmc" else METRICS
        assert len(shown) == len(metrics) + int(show_summary_table)
        metric_figures = shown[: len(metrics)]
        assert all(
            figure.axes[0].lines[0].get_label() == method.upper()
            for figure in metric_figures
        )
        for metric, figure in zip(metrics, metric_figures, strict=True):
            expected_scale = (
                "linear" if metric.startswith("joint_coverage_") else metric_y_scale
            )
            assert figure.axes[0].get_yscale() == expected_scale
            if metric_y_scale == "symlog":
                lower, upper = figure.axes[0].get_ylim()
                assert lower <= 0 <= upper
                assert 0 in figure.axes[0].get_yticks()
        if show_summary_table:
            cells = shown[-1].axes[0].tables[0].get_celld()
            assert cells[1, 0].get_text().get_text() == f"{method.upper()} (T=2)"
            assert cells[1, 1].get_text().get_text() == "5.000"
            if method == "mcmc":
                assert cells[1, 9].get_text().get_text() == "1.012"
    finally:
        for figure in shown:
            plt.close(figure)


@pytest.mark.parametrize("plot_language", ["de", "en"])
@pytest.mark.parametrize("count", [3, 19])
def test_observation_count_summary_table_preserves_metrics_and_fits_labels(
    result_dirs, tmp_path, plot_language, count
):
    models = {"MCMC": _read(result_dirs)["MCMC"]}
    table = models["MCMC"].reindex(range(count)).ffill()
    table["observation_count"] = np.arange(2, count + 2)
    table["joint_coverage_50"] = [0.2, 0.5] + [0.95] * (count - 2)
    table["joint_coverage_90"] = [0.95, 0.9] + [0.8] * (count - 2)
    models["MCMC"] = table
    figures = runtime_plotting.plot_observation_count_comparison(
        models,
        show=False,
        show_summary_table=True,
        plot_language=plot_language,
        output_dir=tmp_path,
    )
    try:
        assert len(figures) == len(MCMC_METRICS) + 1
        figure, axis = figures[-1]
        assert not axis.axison
        cells = axis.tables[0].get_celld()
        assert len(cells) == (count + 1) * 10
        assert cells[1, 0].get_text().get_text() == "MCMC (T=2)"
        assert cells[count, 0].get_text().get_text() == f"MCMC (T={count + 1})"
        assert cells[1, 1].get_text().get_text() == "5.000"
        assert cells[1, 6].get_text().get_text() == "20.0%"
        assert cells[1, 8].get_text().get_text() == "0.300"
        assert cells[1, 1].get_text().get_weight() == "bold"
        assert cells[2, 6].get_text().get_weight() == "bold"
        assert cells[2, 7].get_text().get_weight() == "bold"
        assert cells[1, 6].get_text().get_weight() != "bold"
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for cell in cells.values():
            cell_box = cell.get_window_extent(renderer)
            text_box = cell.get_text().get_window_extent(renderer)
            assert figure.bbox.contains(cell_box.x0, cell_box.y0)
            assert figure.bbox.contains(cell_box.x1, cell_box.y1)
            assert cell_box.contains(text_box.x0, text_box.y0)
            assert cell_box.contains(text_box.x1, text_box.y1)
        assert (tmp_path / "metrics_summary.png").stat().st_size > 0
    finally:
        for figure, _ in figures:
            plt.close(figure)


def test_failed_validation_keeps_previous_summary_and_does_not_plot(
    result_dirs, monkeypatch
):
    directory = result_dirs["vi"]
    original = (directory / "summary.csv").read_bytes()
    _edit(result_dirs, "position_error_m", np.nan)
    monkeypatch.setattr(
        plt,
        "subplots",
        lambda *args, **kwargs: pytest.fail("Invalid inputs must not plot"),
    )
    with pytest.raises(ValueError):
        comparison.run_observation_count_comparison(vi_dir=directory, show=False)
    assert (directory / "summary.csv").read_bytes() == original


def test_summary_export_failure_stops_before_plotting(result_dirs, monkeypatch):
    directory = result_dirs["mcmc"]
    original = (directory / "summary.csv").read_bytes()
    monkeypatch.setattr(evaluation, "_export_csv", lambda *args: False)
    monkeypatch.setattr(
        plt,
        "subplots",
        lambda *args, **kwargs: pytest.fail("Failed export must not plot"),
    )
    with pytest.raises(OSError, match="summary.csv"):
        comparison.run_observation_count_comparison(mcmc_dir=directory, show=False)
    assert (directory / "summary.csv").read_bytes() == original
