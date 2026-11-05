"""CSV-only comparison of MCMC and VI observation-count sweeps."""

import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bayestraj.validation import comparison, runtime_plotting

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "ade_m",
    "fde_m",
    "rmse_m",
    "mean_energy_score_m",
    "mean_negative_log_predictive_density",
    "joint_coverage_50",
    "joint_coverage_90",
    "mean_window_runtime_seconds",
)


def _summary(method):
    table = pd.DataFrame(
        {
            "observation_count": [4, 2, 3],
            "inference_mode": "batch",
            "inference_method": method,
            "data_file": "BR24.csv",
            "prediction_count": 3,
            "window_count": 5,
            "forecast_count": 15,
            "ade_m": [2.0, 4.0, 3.0],
            "fde_m": [3.0, 5.0, 4.0],
            "rmse_m": [2.5, 4.5, 3.5],
            "mean_energy_score_m": [1.5, 3.0, 2.5],
            "mean_negative_log_predictive_density": [-0.5, 1.0, 0.0],
            "joint_coverage_50": [0.5, 0.4, 0.6],
            "joint_coverage_90": [0.9, 0.8, 1.0],
            "mean_window_runtime_seconds": [0.4, 0.2, 0.3],
        }
    )
    if method == "vi":
        table[["ade_m", "fde_m", "rmse_m", "mean_energy_score_m"]] += 0.3
        table["mean_negative_log_predictive_density"] += 0.2
        table[["joint_coverage_50", "joint_coverage_90"]] -= 0.1
        table["mean_window_runtime_seconds"] /= 4
    return table


@pytest.fixture
def summary_csvs(tmp_path):
    paths = {}
    for method in ("mcmc", "vi"):
        path = tmp_path / f"results/bayesian_ctrv_{method}_evaluation/summary.csv"
        path.parent.mkdir(parents=True)
        _summary(method).to_csv(path, index=False)
        paths[method] = path
    return paths


def _read(paths):
    return comparison.read_observation_count_summary_csvs(paths["mcmc"], paths["vi"])


def test_summary_reader_sorts_counts_and_preserves_saved_metrics(summary_csvs):
    models = _read(summary_csvs)
    assert list(models) == ["MCMC", "VI"]
    for method, table in models.items():
        expected = (
            _summary(method.lower())
            .sort_values("observation_count")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(table, expected)


@pytest.mark.parametrize("method", ["mcmc", "vi"])
def test_summary_reader_accepts_one_method(summary_csvs, method):
    models = comparison.read_observation_count_summary_csvs(
        **{f"{method}_csv": summary_csvs[method]}
    )
    assert list(models) == [method.upper()]
    expected = _summary(method).sort_values("observation_count").reset_index(drop=True)
    pd.testing.assert_frame_equal(models[method.upper()], expected)


def test_summary_reader_requires_at_least_one_csv():
    with pytest.raises(ValueError, match="at least one"):
        comparison.read_observation_count_summary_csvs()


def test_summary_reader_still_checks_optional_legacy_start_metadata(summary_csvs):
    for method, path in summary_csvs.items():
        table = _summary(method)
        table["evaluation_start_observation_count"] = 20
        table.to_csv(path, index=False)

    models = _read(summary_csvs)

    assert all(
        "evaluation_start_observation_count" not in table for table in models.values()
    )


def test_summary_reader_aligns_common_counts_with_a_warning(summary_csvs):
    _summary("vi").iloc[:2].to_csv(summary_csvs["vi"], index=False)
    with pytest.warns(UserWarning, match="common observation counts"):
        models = _read(summary_csvs)
    assert models["MCMC"]["observation_count"].tolist() == [2, 4]
    assert models["VI"]["observation_count"].tolist() == [2, 4]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("observation_count", 2),
        ("observation_count", 1.5),
        ("ade_m", np.nan),
        ("mean_window_runtime_seconds", -1.0),
        ("joint_coverage_90", 1.1),
        ("inference_method", "smc"),
        ("data_file", "different.csv"),
        ("prediction_count", 4),
        ("evaluation_start_observation_count", 21),
        ("evaluation_start_observation_count", 20),
        ("window_count", 6),
        ("forecast_count", 18),
    ],
)
def test_summary_reader_rejects_invalid_or_incomparable_results(
    summary_csvs, column, value
):
    table = _summary("vi")
    table[column] = value
    table.to_csv(summary_csvs["vi"], index=False)
    with pytest.raises(ValueError):
        _read(summary_csvs)


def test_missing_summary_does_not_start_inference(tmp_path, monkeypatch):
    import bayestraj.validation.bayesian_ctrv_evaluation as evaluation

    def fail(**kwargs):
        pytest.fail("Reading summaries must never run inference")

    monkeypatch.setattr(evaluation.workflow, "run_bayesian_ctrv_evaluation", fail)
    with pytest.raises(ValueError, match="not found"):
        comparison.read_observation_count_summary_csvs(
            tmp_path / "missing.csv", tmp_path / "vi.csv"
        )


def test_plot_uses_observation_count_and_saved_metric_values(summary_csvs, tmp_path):
    models = _read(summary_csvs)
    figures = runtime_plotting.plot_observation_count_comparison(
        models, show=False, output_dir=tmp_path, plot_language="en"
    )
    try:
        assert len(figures) == len(METRICS)
        for metric, (figure, axis) in zip(METRICS, figures, strict=True):
            assert "observation_count" in axis.get_xlabel()
            assert [line.get_label() for line in axis.lines[:2]] == ["MCMC", "VI"]
            scale = 100.0 if metric.startswith("joint_coverage_") else 1.0
            for line, table in zip(axis.lines[:2], models.values(), strict=True):
                np.testing.assert_array_equal(line.get_xdata(), [2, 3, 4])
                np.testing.assert_allclose(line.get_ydata(), scale * table[metric])
            if scale == 100:
                nominal = float(metric.rsplit("_", 1)[1])
                np.testing.assert_allclose(axis.lines[2].get_ydata(), nominal)
                assert axis.get_ylim() == (0.0, 100.0)
            assert (
                axis.lines[0].get_color()
                == runtime_plotting.COMPARISON_INFERENCE_COLORS["mcmc"]
            )
            assert (
                axis.lines[1].get_color()
                == runtime_plotting.COMPARISON_INFERENCE_COLORS["vi"]
            )
            width, height = figure.get_size_inches()
            expected_width, expected_height = (
                runtime_plotting.RUNTIME_PLOT_STYLE.speed_figure_size
            )
            assert width == expected_width
            assert height >= expected_height
            figure.canvas.draw()
            assert (tmp_path / f"{metric}.png").is_file()
    finally:
        for figure, _ in figures:
            plt.close(figure)


def test_singleton_comparison_keeps_integer_observation_count_ticks(summary_csvs):
    models = {label: table.iloc[:1] for label, table in _read(summary_csvs).items()}
    figures = runtime_plotting.plot_observation_count_comparison(models, show=False)
    try:
        for _, axis in figures:
            ticks = axis.get_xticks()
            np.testing.assert_array_equal(ticks, np.floor(ticks))
    finally:
        for figure, _ in figures:
            plt.close(figure)


@pytest.mark.parametrize("method", ["mcmc", "vi"])
def test_comparison_script_only_reads_its_method_csv_and_displays_plots(
    summary_csvs, tmp_path, monkeypatch, method
):
    import bayestraj.validation.bayesian_ctrv_evaluation as evaluation

    monkeypatch.setattr("bayestraj.paths.project_path", lambda path: tmp_path / path)

    def fail(**kwargs):
        pytest.fail("The comparison script must never run inference")

    monkeypatch.setattr(evaluation.workflow, "run_bayesian_ctrv_evaluation", fail)
    read_csv = comparison._read_csv

    def read_selected_csv(path):
        assert path == summary_csvs[method]
        return read_csv(path)

    monkeypatch.setattr(comparison, "_read_csv", read_selected_csv)
    displayed = []
    monkeypatch.setattr(plt, "show", lambda: displayed.append(plt.gcf()))
    try:
        script = (
            ROOT
            / f"experiments/model_comparison/compare_{method}_observation_counts.py"
        )
        models = runpy.run_path(script, run_name="__main__")
        assert "main" in models
        assert len(displayed) == len(METRICS)
        for figure in displayed:
            assert figure.axes[0].lines[0].get_label() == method.upper()
            assert not any(
                line.get_label() == ("VI" if method == "mcmc" else "MCMC")
                for line in figure.axes[0].lines
            )
    finally:
        for figure in displayed:
            plt.close(figure)
