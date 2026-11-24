"""Forecast-prefix exports compare horizons on one complete origin cohort."""

import dataclasses
import importlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backend_bases import ResizeEvent

from bayestraj.forecasting.bayesian_ctrv import RollingExperimentConfig
from bayestraj.models.bayesian_ctrv import BayesianCTRVPriors
from bayestraj.validation import comparison, runtime_plotting

ROOT = Path(__file__).resolve().parents[3]


def _module():
    return importlib.import_module("bayestraj.validation.prediction_horizons")


def _experiment():
    return RollingExperimentConfig(
        observation_count=10,
        prediction_count=3,
        position_noise_std_m=0.0,
        position_noise_seed=2026,
        inference_seed=42,
    )


def _predictions(method, observation_count=10):
    rows = []
    for window, count in enumerate((3, 3, 2, 1)):
        for step in range(1, count + 1):
            actual = float(10 * window + step)
            error = float((window + 1) * step)
            rows.append(
                dict(
                    window_index=window,
                    forecast_origin_seconds=100.0 + 10 * window,
                    horizon_step=step,
                    horizon_seconds=(2.6, 5.4, 8.1)[step - 1],
                    prediction_count=count,
                    observation_count=window + 2,
                    configured_observation_count=(
                        observation_count if method in {"vi", "mcmc"} else np.nan
                    ),
                    x_actual=actual,
                    y_actual=0.0,
                    x_mean=actual + error,
                    y_mean=0.0,
                    x_median=actual + error,
                    y_median=0.0,
                    point_estimate="mean",
                    position_error_m=error,
                    energy_score_m=error / 2,
                    negative_log_predictive_density=error + 1,
                    student_t_energy_score_m=error / 3,
                    student_t_negative_log_predictive_density=error + 0.5,
                    prediction_radius_m=5.0,
                    radial_covered=True,
                    joint_covered_50=step == 1,
                    joint_covered_90=True,
                    mean_marginal_interval_width_m=6.0,
                    inference_mode="batch" if method in {"vi", "mcmc"} else "online",
                    inference_method=method,
                    model_variant="bayesian",
                    converged=True if method == "vi" else None,
                    mcmc_diagnostics_ok=True if method == "mcmc" else None,
                    is_prior_predictive=False,
                    fit_runtime_seconds=0.1 * (window + 1),
                    time=pd.Timestamp("2026-01-01", tz="UTC"),
                )
            )
    return pd.DataFrame(rows)


def _run(tmp_path, monkeypatch, **overrides):
    module = _module()
    calls = []

    def bayesian(**kwargs):
        calls.append(kwargs)
        options = kwargs["options"]
        return configured_predictions(options.inference_method, options), None

    def configured_predictions(method, options):
        table = _predictions(method, options.observation_count)
        table = table.loc[table["horizon_step"] <= options.prediction_count].copy()
        table["prediction_count"] = np.minimum(
            table["prediction_count"], options.prediction_count
        )
        if options.max_windows is not None:
            table = table.loc[table["window_index"] < options.max_windows]
        return table

    monkeypatch.setattr(
        module.bayesian_workflow, "run_bayesian_ctrv_evaluation", bayesian
    )
    methods = overrides.pop("methods", ("vi", "mcmc"))
    arguments = dict(
        data_file=tmp_path / "trajectory.csv",
        output_dir=tmp_path / "results",
        prediction_counts=(1, 2, 3),
        priors=BayesianCTRVPriors(),
        max_windows=None,
        argv=[],
    )
    arguments.update(overrides)
    outputs = []
    for method in methods:
        current = dataclasses.replace(
            _experiment(),
            inference_method=method,
            observation_count={"vi": 4, "mcmc": 7}.get(method, 10),
        )
        outputs.extend(
            module.run_prediction_horizon_evaluation(experiment=current, **arguments)
        )
    return module, tuple(outputs), calls


def _write_manual_forecasts(directory, method, counts=(1, 2, 3)):
    table = _predictions("vi").drop(columns="configured_observation_count")
    table["inference_method"] = method
    table["inference_mode"] = "online"
    table["model_variant"] = "imm" if method == "imm" else "bayesian"
    table["converged"], table["mcmc_diagnostics_ok"] = None, None
    paths = []
    for count in counts:
        prefix = table.loc[table["horizon_step"] <= count].copy()
        prefix["prediction_count"] = np.minimum(prefix["prediction_count"], count)
        folder = directory / f"forecast_step_{count:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        filename = "imm.csv" if method == "imm" else "bayesian_ctrv_smc.csv"
        path = folder / filename
        prefix.to_csv(path, index=False)
        paths.append(path)
    return paths


def test_manual_smc_and_imm_are_discovered_and_aligned_with_batch_exports(
    tmp_path, monkeypatch
):
    module, outputs, _calls = _run(tmp_path, monkeypatch)
    directory = tmp_path / "results"
    paths = [
        *outputs,
        *_write_manual_forecasts(directory, "imm"),
        *_write_manual_forecasts(directory, "smc"),
    ]
    (directory / "forecast_step_01/summary.csv").write_text("not a forecast")
    (directory / "forecast_step_01/run_associations.csv").write_text("not a forecast")
    pd.DataFrame({"time": [1], "x": [2], "y": [3]}).to_csv(
        directory / "forecast_step_01/observations.csv", index=False
    )
    unrelated = directory / "evaluation"
    unrelated.mkdir()
    (unrelated / "invalid.csv").write_text("not a forecast")
    originals = {path: path.read_bytes() for path in paths}

    with pytest.warns(UserWarning, match="common.*origins"):
        models = module.read_prediction_horizon_evaluations(directory)

    assert list(models) == [
        "IMM",
        "Bayesian CTRV (SMC)",
        "Bayesian CTRV (VI, T=4)",
        "Bayesian CTRV (MCMC, T=7)",
    ]
    for table in models.values():
        assert table["prediction_count"].tolist() == [1, 2, 3]
        assert table["window_count"].tolist() == [2, 2, 2]
        assert table["forecast_count"].tolist() == [2, 4, 6]
    assert models["IMM"]["runtime_prediction_count"].isna().all()
    assert models["Bayesian CTRV (SMC)"]["runtime_prediction_count"].isna().all()
    assert {path: path.read_bytes() for path in paths} == originals


def test_manual_forecasts_work_without_any_metadata(tmp_path):
    _write_manual_forecasts(tmp_path, "smc")
    _write_manual_forecasts(tmp_path, "imm")
    with pytest.warns(UserWarning, match="common.*origins"):
        models = _module().read_prediction_horizon_evaluations(tmp_path)
    assert list(models) == ["IMM", "Bayesian CTRV (SMC)"]
    assert all(table["window_count"].eq(2).all() for table in models.values())


def test_different_automated_run_lengths_use_common_origins_between_models(
    tmp_path, monkeypatch
):
    module, _outputs, _calls = _run(tmp_path, monkeypatch)
    _run(tmp_path, monkeypatch, methods=("vi",), max_windows=1)
    with pytest.warns(UserWarning, match="common.*origins"):
        models = module.read_prediction_horizon_evaluations(tmp_path / "results")
    assert len(models) == 2
    assert all(table["window_count"].eq(1).all() for table in models.values())
    assert all(
        table["prediction_count"].tolist() == [1, 2, 3] for table in models.values()
    )


@pytest.mark.parametrize("problem", ["count", "reference", "horizon", "no_common"])
def test_manual_forecast_validation_preserves_folder_and_cohort_checks(
    tmp_path, problem
):
    _write_manual_forecasts(tmp_path, "imm")
    paths = _write_manual_forecasts(tmp_path, "smc")
    for path in paths:
        table = pd.read_csv(path)
        if problem == "count":
            if path.parent.name != "forecast_step_02":
                continue
            table = _predictions("vi").assign(
                inference_method="smc", inference_mode="online"
            )
        elif problem == "reference":
            table["x_actual"] += 1
            table["x_mean"] += 1
            table["x_median"] += 1
        elif problem == "horizon":
            table["horizon_seconds"] += 0.1
        else:
            table["forecast_origin_seconds"] += 10000
        table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="prediction_count|reference|horizon|common"):
        _module().read_prediction_horizon_evaluations(tmp_path)


def test_models_require_the_same_available_forecast_steps(tmp_path):
    _write_manual_forecasts(tmp_path, "imm")
    _write_manual_forecasts(tmp_path, "smc", counts=(1, 3))
    with pytest.raises(ValueError, match="same prediction counts"):
        _module().read_prediction_horizon_evaluations(tmp_path)


def test_partial_metadata_cannot_hide_mixed_automated_prefixes(tmp_path, monkeypatch):
    module, _outputs, _calls = _run(tmp_path, monkeypatch)
    path = tmp_path / "results/forecast_steps_metadata.json"
    content = json.loads(path.read_text())
    content["evaluations"].pop("forecast_step_01/bayesian_ctrv_vi_04.csv")
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="metadata"):
        module.read_prediction_horizon_evaluations(tmp_path / "results")


@pytest.mark.parametrize("value", [True, 1.0])
def test_metadata_prediction_counts_require_integers(tmp_path, monkeypatch, value):
    module, _outputs, _calls = _run(tmp_path, monkeypatch)
    path = tmp_path / "results/forecast_steps_metadata.json"
    content = json.loads(path.read_text())
    content["evaluations"]["forecast_step_01/bayesian_ctrv_vi_04.csv"][
        "prediction_count"
    ] = value
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="metadata prediction_count"):
        module.read_prediction_horizon_evaluations(tmp_path / "results")


def test_duplicate_manual_configurations_are_not_merged_silently(tmp_path):
    paths = _write_manual_forecasts(tmp_path, "smc")
    pd.read_csv(paths[0]).to_csv(paths[0].parent / "second.CSV", index=False)
    with pytest.raises(ValueError, match="Duplicate model configuration"):
        _module().read_prediction_horizon_evaluations(tmp_path)


def test_forecast_workflows_preserve_unrelated_results_metadata_and_summary(
    tmp_path, monkeypatch
):
    directory = tmp_path / "results"
    directory.mkdir()
    metadata = directory / "metadata.json"
    summary = directory / "summary.csv"
    metadata.write_text("unrelated metadata")
    summary.write_text("unrelated summary")
    module, _outputs, _calls = _run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runtime_plotting,
        "plot_prediction_horizon_comparison",
        lambda *args, **kwargs: (),
    )
    module.run_prediction_horizon_comparison(directory, show=False)
    assert metadata.read_text() == "unrelated metadata"
    assert summary.read_text() == "unrelated summary"
    assert (directory / "forecast_steps_metadata.json").is_file()
    assert (directory / "forecast_steps_summary.csv").is_file()


def test_evaluation_runs_each_model_once_and_exports_three_compatible_prefixes(
    tmp_path, monkeypatch
):
    module, outputs, calls = _run(tmp_path, monkeypatch)

    assert len(calls) == 2
    assert {call["options"].prediction_count for call in calls} == {3}
    assert all(call["options"].stride == 1 for call in calls)
    assert len(outputs) == 6
    filenames = {
        "bayesian_ctrv_vi_04.csv",
        "bayesian_ctrv_mcmc_07.csv",
    }
    for count in (1, 2, 3):
        directory = tmp_path / "results" / f"forecast_step_{count:02d}"
        assert {path.name for path in directory.iterdir()} == filenames
        models = comparison.read_model_comparison_directory(directory)
        assert len(models) == 2
        for table in models.values():
            assert len(table) == 2 * count
            assert table["prediction_count"].eq(count).all()
            assert table["forecast_origin_seconds"].unique().tolist() == [100, 110]
            assert table["horizon_step"].unique().tolist() == list(range(1, count + 1))
            assert "time" not in table and "forecast_origin_time" not in table
    metadata = json.loads(
        (tmp_path / "results/forecast_steps_metadata.json").read_text()
    )
    assert len(metadata["evaluations"]) == 6
    assert all(
        record["runtime_prediction_count"] == 3
        for record in metadata["evaluations"].values()
    )
    assert not (tmp_path / "results/forecast_steps_summary.csv").exists()
    assert module is _module()


def test_comparison_recomputes_metrics_without_rerunning_models(tmp_path, monkeypatch):
    module, outputs, calls = _run(tmp_path, monkeypatch)
    originals = {path: path.read_bytes() for path in outputs}

    def forbidden(**kwargs):
        pytest.fail("Saved forecast comparison must not run inference")

    monkeypatch.setattr(
        module.bayesian_workflow, "run_bayesian_ctrv_evaluation", forbidden
    )
    shown = set(plt.get_fignums())
    models = module.run_prediction_horizon_comparison(
        tmp_path / "results", show=False, show_summary_table=True
    )

    assert len(calls) == 2
    assert set(plt.get_fignums()) == shown
    assert len(models) == 2
    assert list(models) == [
        "Bayesian CTRV (VI, T=4)",
        "Bayesian CTRV (MCMC, T=7)",
    ]
    for table in models.values():
        assert table["prediction_count"].tolist() == [1, 2, 3]
        assert table["window_count"].tolist() == [2, 2, 2]
        assert table["forecast_count"].tolist() == [2, 4, 6]
        np.testing.assert_allclose(table["ade_m"], [1.5, 2.25, 3])
        np.testing.assert_allclose(table["fde_m"], [1.5, 3, 4.5])
        np.testing.assert_allclose(table["rmse_m"], np.sqrt([2.5, 6.25, 70 / 6]))
        np.testing.assert_allclose(table["joint_coverage_50"], [1, 0.5, 1 / 3])
        np.testing.assert_allclose(table["mean_horizon_seconds"], [2.6, 5.4, 8.1])
        assert table["runtime_prediction_count"].eq(3).all()
    assert {path: path.read_bytes() for path in outputs} == originals
    saved = pd.read_csv(tmp_path / "results/forecast_steps_summary.csv")
    assert len(saved) == 6 and saved["prediction_count"].unique().tolist() == [1, 2, 3]


@pytest.mark.parametrize("counts", [(), (0,), (-1,), (True,), (1.5,), (1, 1)])
def test_invalid_horizon_counts_fail_before_inference(tmp_path, monkeypatch, counts):
    with pytest.raises(ValueError, match="prediction_counts"):
        _run(tmp_path, monkeypatch, prediction_counts=counts)
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "problem", ["hash", "cohort", "reference", "horizon", "missing"]
)
def test_saved_comparison_rejects_stale_or_incompatible_forecasts(
    tmp_path, monkeypatch, problem
):
    module, outputs, _calls = _run(tmp_path, monkeypatch)
    path = next(
        path
        for path in outputs
        if "forecast_step_02" in str(path) and path.name == "bayesian_ctrv_vi_04.csv"
    )
    table = pd.read_csv(path)
    if problem == "missing":
        table = table.drop(columns="horizon_seconds")
    elif problem == "cohort":
        table = table[table["forecast_origin_seconds"] == 100]
    elif problem == "reference":
        table["x_actual"] += 1
        table["x_mean"] += 1
        table["x_median"] += 1
    elif problem == "horizon":
        table["horizon_seconds"] += 0.1
    else:
        table["position_error_m"] += 1
    table.to_csv(path, index=False)
    if problem != "hash":
        import hashlib

        metadata_path = tmp_path / "results/forecast_steps_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["evaluations"]["forecast_step_02/bayesian_ctrv_vi_04.csv"][
            "sha256"
        ] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata|origins|reference|horizon|columns"):
        module.read_prediction_horizon_evaluations(tmp_path / "results")


@pytest.mark.parametrize("language", ["de", "en"])
def test_horizon_plots_use_position_counts_and_keep_coverage_nominal_lines(
    tmp_path, monkeypatch, language
):
    module, _outputs, _calls = _run(tmp_path, monkeypatch)
    models = module.read_prediction_horizon_evaluations(tmp_path / "results")
    figures = runtime_plotting.plot_prediction_horizon_comparison(
        models, show=False, show_summary_table=True, plot_language=language
    )
    try:
        assert len(figures) == 8
        for _figure, axis in figures[:7]:
            assert "Position" in axis.get_xlabel() or "position" in axis.get_xlabel()
            for line in axis.lines[: len(models)]:
                np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
        for (_figure, axis), nominal in zip(figures[5:7], (50, 90), strict=True):
            np.testing.assert_allclose(axis.lines[-1].get_ydata(), nominal)
        assert all(
            "Runtime" not in axis.get_ylabel() and "Laufzeit" not in axis.get_ylabel()
            for _figure, axis in figures[:7]
        )
        text = runtime_plotting.COMPARISON_PLOT_TEXT[language]
        for figure, axis in figures:
            captions = [
                item for item in figure.texts if text["horizon_note"] in item.get_text()
            ]
            assert len(captions) == 1
            assert text["window_note"] in captions[0].get_text()
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            bounds = captions[0].get_window_extent(renderer)
            assert figure.bbox.contains(bounds.x0, bounds.y0)
            assert figure.bbox.contains(bounds.x1, bounds.y1)
            if axis.axison:
                assert axis.xaxis.label.get_window_extent(renderer).y0 > bounds.y1 + 2
            else:
                for cell in axis.tables[0].get_celld().values():
                    assert cell.get_window_extent(renderer).y0 > bounds.y1 + 2
    finally:
        for figure, _axis in figures:
            plt.close(figure)


@pytest.mark.parametrize("language", ["de", "en"])
@pytest.mark.parametrize("model_count", [2, 4])
@pytest.mark.parametrize("prediction_counts", [(1, 2, 3), (1, 2, 3, 5, 7, 10)])
def test_forecast_summary_configuration_labels_fit_their_cells(
    tmp_path, language, model_count, prediction_counts
):
    models = _forecast_summary_models(model_count, prediction_counts)
    specs = [
        (label, table["inference_method"].iloc[0]) for label, table in models.items()
    ]
    original = {label: table.copy(deep=True) for label, table in models.items()}
    figures = runtime_plotting.plot_prediction_horizon_comparison(
        models, show=False, show_summary_table=True, plot_language=language
    )
    try:
        figure, axis = figures[-1]
        cells = axis.tables[0].get_celld()
        assert len(cells) == (1 + len(prediction_counts) * model_count) * 8
        assert cells[1, 1].get_text().get_text() == "9.830"
        for index, (_label, method) in enumerate(specs):
            for position, count in enumerate(prediction_counts, start=1):
                cell = cells[len(prediction_counts) * index + position, 0]
                label = cell.get_text().get_text()
                assert method.upper() in label
                assert f"H={count}" in label
                if method in {"vi", "mcmc"}:
                    assert f"T={4 if method == 'vi' else 7}" in label
        for width in (figure.get_figwidth(), 9.0):
            figure.set_size_inches(width, figure.get_figheight(), forward=False)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            for cell in cells.values():
                bounds = cell.get_window_extent(renderer)
                text_bounds = cell.get_text().get_window_extent(renderer)
                assert bounds.x0 <= text_bounds.x0 < text_bounds.x1 <= bounds.x1
                assert bounds.y0 + 3 <= text_bounds.y0 < text_bounds.y1 <= bounds.y1 - 3
                assert cell.get_text().get_fontsize() >= 10
        preview = tmp_path / f"forecast_summary_{language}_{model_count}.png"
        figure.savefig(preview, dpi=120, bbox_inches="tight")
        print(f"FORECAST_TABLE_PREVIEW={preview}")
        for label, table in models.items():
            pd.testing.assert_frame_equal(table, original[label])
    finally:
        for figure, _axis in figures:
            plt.close(figure)


def _forecast_summary_models(model_count, prediction_counts):
    specs = [
        ("Bayesian CTRV (VI, T=4)", "vi"),
        ("Bayesian CTRV (MCMC, T=7)", "mcmc"),
    ]
    if model_count == 4:
        specs = [("IMM", "imm"), ("Bayesian CTRV (SMC)", "smc"), *specs]
    return {
        label: pd.DataFrame(
            {
                "prediction_count": [1, 2, 3],
                "inference_method": method,
                "ade_m": [9.830, 10.443, 11.342],
                "fde_m": [9.830, 11.056, 13.139],
                "rmse_m": [12.005, 12.644, 13.594],
                "mean_energy_score_m": [7.179, 7.592, 8.246],
                "mean_negative_log_predictive_density": [6.828, 6.979, 7.184],
                "joint_coverage_50": [0.384, 0.446, 0.499],
                "joint_coverage_90": [0.775, 0.843, 0.881],
            }
        )
        .iloc[np.arange(len(prediction_counts)) % 3]
        .reset_index(drop=True)
        .assign(prediction_count=prediction_counts)
        for label, method in specs
    }


@pytest.mark.parametrize("window_width", [1000, 1886])
@pytest.mark.parametrize("screen_height", [1080, 3000])
def test_forecast_table_keeps_readable_rows_in_a_short_tk_window(
    tmp_path, monkeypatch, window_width, screen_height
):
    models = _forecast_summary_models(4, (1, 2, 3, 5, 7, 10))
    figures = runtime_plotting.plot_prediction_horizon_comparison(models, show=False)
    try:
        figure, axis = figures[-1]
        canvas = figure.canvas

        def backend_resize(event):
            figure.set_size_inches(
                event.width / figure.dpi, event.height / figure.dpi, forward=False
            )
            ResizeEvent("resize_event", canvas)._process()
            canvas.draw()

        bindings = {"<Configure>": backend_resize}
        widget = Mock()
        widget.winfo_screenheight.return_value = screen_height
        widget.bind.side_effect = lambda name, callback: bindings.update(
            {name: callback}
        )
        scrollbar_factory = Mock()
        monkeypatch.setitem(
            sys.modules,
            "tkinter",
            SimpleNamespace(ttk=SimpleNamespace(Scrollbar=scrollbar_factory)),
        )
        monkeypatch.setattr(canvas, "get_tk_widget", lambda: widget, raising=False)
        monkeypatch.setattr(canvas, "resize", backend_resize, raising=False)
        monkeypatch.setattr(plt, "show", lambda: None)

        runtime_plotting._show_comparison_figure(figure)
        bindings["<Configure>"](SimpleNamespace(width=window_width, height=829))

        renderer = canvas.get_renderer()
        for cell in axis.tables[0].get_celld().values():
            bounds = cell.get_window_extent(renderer)
            text_bounds = cell.get_text().get_window_extent(renderer)
            assert bounds.y0 + 3 <= text_bounds.y0 < text_bounds.y1 <= bounds.y1 - 3
            assert cell.get_text().get_fontsize() >= 10
        assert figure.bbox.height > 829
        scrollbar_factory.assert_called_once()
        assert scrollbar_factory.call_args.kwargs["command"] == widget.yview
        assert widget.configure.call_args.kwargs["scrollregion"][3] > 829
        bindings["<MouseWheel>"](SimpleNamespace(delta=-120))
        widget.yview_scroll.assert_called_once_with(1, "units")
        preview = tmp_path / "forecast_table_resized.png"
        figure.savefig(preview, dpi=100)
        assert plt.imread(preview).shape[0] > 829
    finally:
        for figure, _axis in figures:
            plt.close(figure)


def test_forecast_table_scrolls_with_a_real_hidden_tk_canvas(monkeypatch):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk display is unavailable: {error}")
    root.withdraw()

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    figures = []
    try:
        models = _forecast_summary_models(4, (1, 2, 3, 5, 7, 10))
        figures = runtime_plotting.plot_prediction_horizon_comparison(
            models, show=False
        )
        figure, axis = figures[-1]
        canvas = FigureCanvasTkAgg(figure, master=root)
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)
        monkeypatch.setattr(plt, "show", lambda: None)

        runtime_plotting._show_comparison_figure(figure)
        root.geometry("1886x829")
        root.update_idletasks()
        widget.event_generate("<Configure>", width=1886, height=829)
        root.update_idletasks()
        canvas.draw()

        assert canvas._comparison_table_scrollbar.winfo_exists()
        assert figure.bbox.height > 829
        widget.yview_moveto(1)
        assert widget.yview()[0] > 0
        renderer = canvas.get_renderer()
        for cell in axis.tables[0].get_celld().values():
            bounds = cell.get_window_extent(renderer)
            text_bounds = cell.get_text().get_window_extent(renderer)
            assert bounds.y0 + 3 <= text_bounds.y0 < text_bounds.y1 <= bounds.y1 - 3
    finally:
        root.destroy()
        for figure, _axis in figures:
            plt.close(figure)


@pytest.mark.parametrize("method,count", [("vi", 4), ("mcmc", 7), ("smc", 10)])
def test_scripts_keep_settings_in_headers_and_delegate_to_shared_workflows(
    tmp_path, monkeypatch, method, count
):
    module = _module()
    evaluation = runpy.run_path(
        ROOT / f"scripts/forecast_evaluation/evaluate_{method}_forcast_steps.py"
    )
    calls = []
    monkeypatch.setattr(
        module,
        "run_prediction_horizon_evaluation",
        lambda **kwargs: calls.append(kwargs),
    )
    assert evaluation["OUTPUT_DIR"] == ROOT / "results/forecast_evaluation"
    monkeypatch.setitem(evaluation["main"].__globals__, "OUTPUT_DIR", tmp_path)
    evaluation["main"]([])
    assert evaluation["PREDICTION_COUNTS"] == (1, 2, 3, 5, 7, 10)
    assert calls[0]["prediction_counts"] == (1, 2, 3, 5, 7, 10)
    assert calls[0]["output_dir"] == tmp_path
    assert calls[0]["experiment"].inference_method == method
    assert calls[0]["experiment"].observation_count == count
    assert not {"IMM_PYTHON", "METHODS", "BATCH_OBSERVATION_COUNTS"}.intersection(
        evaluation
    )
    assert "methods" not in calls[0] and "imm_python" not in calls[0]
    compare = runpy.run_path(ROOT / "scripts/model_comparison/compare_forcast_steps.py")
    assert compare["INPUT_DIR"] == ROOT / "results/forecast_evaluation"
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_cli_limits_observations_counts_windows_and_changes_the_output_directory(
    tmp_path, monkeypatch
):
    directory = tmp_path / "custom"
    _module_value, outputs, calls = _run(
        tmp_path,
        monkeypatch,
        methods=("vi",),
        argv=[
            "--observations",
            "9",
            "--prediction-counts",
            "1",
            "2",
            "--max-windows",
            "1",
            "--output-dir",
            str(directory),
            "--no-student-t-scores",
        ],
    )
    assert len(calls) == 1
    assert calls[0]["options"].inference_method == "vi"
    assert calls[0]["options"].observation_count == 9
    assert calls[0]["options"].max_windows == 1
    assert calls[0]["options"].student_t_scores is False
    assert len(outputs) == 2
    assert all(path.parent.parent == directory for path in outputs)
    assert all(path.name == "bayesian_ctrv_vi_09.csv" for path in outputs)
    assert pd.read_csv(outputs[-1])["prediction_count"].tolist() == [2, 2]
    assert not (tmp_path / "results").exists()


def test_export_failure_stops_without_creating_metadata(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(
        module.evaluation, "export_evaluation_csv", lambda *args, **kwargs: False
    )
    with pytest.raises(OSError, match="Could not export forecast CSV"):
        _run(tmp_path, monkeypatch, methods=("vi",))
    assert not (tmp_path / "results/forecast_steps_metadata.json").exists()


def test_partial_reruns_cannot_mix_old_and_new_forecast_generations(
    tmp_path, monkeypatch
):
    module, _outputs, _calls = _run(tmp_path, monkeypatch)
    config = {**module.inference.create_default_vi_config(), "algorithm": "fullrank"}
    _run(
        tmp_path,
        monkeypatch,
        methods=("vi",),
        prediction_counts=(2, 3),
        vi_config=config,
    )

    with pytest.raises(ValueError, match="generation"):
        module.read_prediction_horizon_evaluations(tmp_path / "results")

    # Regenerating all horizons for the method restores one consistent prefix set.
    _run(tmp_path, monkeypatch, methods=("vi",), vi_config=config)
    assert len(module.read_prediction_horizon_evaluations(tmp_path / "results")) == 2


def test_smc_forecast_steps_reuse_one_online_evaluation(tmp_path, monkeypatch):
    module, outputs, calls = _run(tmp_path, monkeypatch, methods=("smc",))
    assert len(calls) == 1
    assert calls[0]["options"].inference_method == "smc"
    assert calls[0]["options"].prediction_count == 3
    assert len(outputs) == 3
    assert all(path.name == "bayesian_ctrv_smc.csv" for path in outputs)
    table = module.read_prediction_horizon_evaluations(tmp_path / "results")[
        "Bayesian CTRV (SMC)"
    ]
    assert table["prediction_count"].tolist() == [1, 2, 3]
    assert table["window_count"].tolist() == [2, 2, 2]


def _run_imm(tmp_path, monkeypatch, **overrides):
    module = _module()
    calls = []
    predictions = _predictions("vi").drop(columns="configured_observation_count")
    predictions["inference_method"], predictions["model_variant"] = "imm", "imm"
    predictions["inference_mode"] = "online"
    predictions["converged"], predictions["mcmc_diagnostics_ok"] = None, None
    predictions["cov_xx"], predictions["cov_xy"], predictions["cov_yy"] = 2.0, 0.25, 3.0
    predictions["mode_cv"], predictions["mode_ctrv"], predictions["mode_ca"] = (
        1.0,
        0.0,
        0.0,
    )

    def evaluate(**kwargs):
        calls.append(kwargs)
        return predictions.copy(deep=True), None

    monkeypatch.setattr(module.imm_workflow, "run_imm_evaluation", evaluate)
    arguments = dict(
        data_file=tmp_path / "trajectory.csv",
        imm_python=tmp_path / "backend.py",
        output_dir=tmp_path / "results",
        prediction_counts=(1, 2, 3),
        max_windows=None,
        forecast_seed=42,
        argv=[],
    )
    arguments.update(overrides)
    outputs = module.run_imm_prediction_horizon_evaluation(**arguments)
    return module, outputs, calls


def test_imm_script_keeps_header_settings_and_delegates_to_shared_workflow(
    tmp_path, monkeypatch
):
    module = _module()
    script = runpy.run_path(
        ROOT / "scripts/forecast_evaluation/evaluate_imm_forcast_steps.py"
    )
    calls = []
    monkeypatch.setattr(
        module,
        "run_imm_prediction_horizon_evaluation",
        lambda **kwargs: calls.append(kwargs),
    )
    assert script["OUTPUT_DIR"] == ROOT / "results/forecast_evaluation"
    monkeypatch.setitem(script["main"].__globals__, "OUTPUT_DIR", tmp_path)
    script["main"]([])
    assert script["PREDICTION_COUNTS"] == (1, 2, 3, 5, 7, 10)
    assert calls[0]["output_dir"] == tmp_path
    assert calls[0]["imm_python"] == ROOT / "external/IMM/br24_imm.py"
    assert calls[0]["forecast_seed"] == 42
    assert "experiment" not in calls[0] and "priors" not in calls[0]
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_imm_exports_prefixes_once_and_preserves_covariances_and_modes(
    tmp_path, monkeypatch
):
    module, outputs, calls = _run_imm(tmp_path, monkeypatch)
    assert len(calls) == 1
    assert calls[0]["options"].prediction_count == 3
    assert calls[0]["options"].observation_count == 2
    assert calls[0]["options"].show_plot is False
    assert calls[0]["forecast_seed"] == 42
    assert len(outputs) == 3
    for count, path in enumerate(outputs, start=1):
        assert path.name == "imm.csv"
        table = pd.read_csv(path)
        assert len(table) == 2 * count
        assert table["prediction_count"].eq(count).all()
        assert table["cov_xy"].eq(0.25).all()
        assert table["mode_cv"].eq(1).all()
        assert "time" not in table
        assert "gaussian_energy_score_m" in table
    assert len(module.read_prediction_horizon_evaluations(tmp_path / "results")) == 1


def test_all_four_automated_models_share_forecast_step_comparison(
    tmp_path, monkeypatch
):
    module, _outputs, _calls = _run(
        tmp_path, monkeypatch, methods=("vi", "mcmc", "smc")
    )
    _run_imm(tmp_path, monkeypatch)
    models = module.read_prediction_horizon_evaluations(tmp_path / "results")
    assert list(models) == [
        "IMM",
        "Bayesian CTRV (SMC)",
        "Bayesian CTRV (VI, T=4)",
        "Bayesian CTRV (MCMC, T=7)",
    ]
    assert all(table["window_count"].eq(2).all() for table in models.values())
    assert all(
        table["runtime_prediction_count"].eq(3).all() for table in models.values()
    )


def test_real_smc_step_exports_work_without_an_imm_backend(tmp_path, monkeypatch):
    module = _module()
    data_file = tmp_path / "trajectory.csv"
    pd.DataFrame(
        {
            "time": np.arange(8, dtype=float),
            "x": np.arange(8, dtype=float),
            "y": np.zeros(8),
        }
    ).to_csv(data_file, index=False)
    monkeypatch.setattr(
        module.imm_workflow,
        "_load_optional_imm_tracker",
        lambda path: pytest.fail("SMC must not load the optional IMM backend"),
    )
    outputs = module.run_prediction_horizon_evaluation(
        data_file=data_file,
        output_dir=tmp_path / "results",
        prediction_counts=(1, 2, 3),
        experiment=_experiment(),
        priors=BayesianCTRVPriors(),
        max_windows=2,
        argv=["--no-student-t-scores"],
    )
    assert len(outputs) == 3
    table = module.read_prediction_horizon_evaluations(tmp_path / "results")[
        "Bayesian CTRV (SMC)"
    ]
    assert table["window_count"].tolist() == [2, 2, 2]
    assert table["forecast_count"].tolist() == [2, 4, 6]
    assert (
        np.isfinite(
            table[
                [
                    "ade_m",
                    "fde_m",
                    "rmse_m",
                    "mean_energy_score_m",
                    "mean_negative_log_predictive_density",
                ]
            ]
        )
        .all()
        .all()
    )
