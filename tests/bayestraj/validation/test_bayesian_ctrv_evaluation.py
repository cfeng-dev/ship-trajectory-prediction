"""One configurable evaluation script supports every inference method."""

import importlib
import runpy
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/model_evaluation/bayesian_ctrv.py"


def test_entrypoint_keeps_scoring_defaults_without_trajectory_plot_settings(
    evaluation_run,
):
    _, calls = evaluation_run
    namespace = runpy.run_path(SCRIPT)

    namespace["main"]([])

    assert not {
        "CREDIBLE_INTERVAL",
        "PLOT_EACH_WINDOW",
        "SAMPLE_TRAJECTORIES_PER_FORECAST",
        "SHOW_TIME_LABELS",
    }.intersection(namespace)
    assert namespace["SHOW_ROLLING_DIAGNOSTICS_PLOT"] is True
    assert calls[0]["credible_interval"] == 0.9
    assert calls[0]["options"].stride == 1
    assert not {
        "sample_trajectories_per_forecast",
        "show_time_labels",
        "show_plot",
    }.intersection(calls[0])
    assert not hasattr(calls[0]["options"], "plot_each_window")


def test_removed_window_plot_flag_is_rejected_before_evaluation(evaluation_run):
    _, calls = evaluation_run

    with pytest.raises(SystemExit, match="2"):
        runpy.run_path(SCRIPT)["main"](["--plot-each-window"])

    assert calls == []


@pytest.fixture
def evaluation_run(tmp_path, monkeypatch):
    evaluation = importlib.import_module(
        "bayestraj.validation.bayesian_ctrv_evaluation"
    )
    monkeypatch.setattr("bayestraj.paths.project_path", lambda path: tmp_path / path)
    monkeypatch.setattr(
        evaluation.runtime_plotting,
        "plot_bayesian_ctrv_rolling_diagnostics",
        lambda *args, **kwargs: None,
    )
    calls = []

    def run_workflow(**kwargs):
        calls.append(kwargs)
        records = kwargs.get("association_records")
        if records is not None and Path(kwargs["data_file"]).name.startswith("BR24"):
            records.extend(
                [
                    {
                        "time": 10.0,
                        "cycle_counter": 1,
                        "detection_id": 7,
                        "x": 110.0,
                        "y": -20.0,
                    },
                    {
                        "time": 11.0,
                        "cycle_counter": 2,
                        "detection_id": None,
                        "x": None,
                        "y": None,
                    },
                ]
            )
        return pd.DataFrame(
            {
                "inference_method": [kwargs["options"].inference_method],
                "position_error_m": [2.5],
            }
        ), None

    monkeypatch.setattr(
        evaluation.workflow, "run_bayesian_ctrv_evaluation", run_workflow
    )
    return evaluation, calls


@pytest.mark.parametrize(
    ("method", "canonical_method", "csv_name"),
    [
        (
            "smc",
            "smc",
            "bayesian_ctrv_smc.csv",
        ),
        (
            "vi",
            "vi",
            "bayesian_ctrv_vi.csv",
        ),
        (
            "mcmc",
            "mcmc",
            "bayesian_ctrv_mcmc.csv",
        ),
        (
            "sequential_vi",
            "sequential_vi",
            "bayesian_ctrv_seq_vi.csv",
        ),
        (
            "seq_vi",
            "sequential_vi",
            "bayesian_ctrv_seq_vi.csv",
        ),
    ],
)
def test_run_button_uses_header_method_and_exports_its_own_csv(
    tmp_path, monkeypatch, evaluation_run, method, canonical_method, csv_name
):
    evaluation, calls = evaluation_run
    namespace = runpy.run_path(SCRIPT)
    monkeypatch.setitem(
        namespace["main"].__globals__,
        "EXPERIMENT",
        replace(namespace["EXPERIMENT"], inference_method=method),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", [str(SCRIPT)])

    namespace["main"]()

    assert len(calls) == 1
    assert calls[0]["experiment"].inference_method == canonical_method
    assert calls[0]["options"].inference_method == canonical_method
    assert calls[0]["options"].prediction_count == 3
    assert calls[0]["options"].stride == 1
    assert calls[0]["data_file"] == (
        tmp_path / "data/processed/BR24Dataset20131107_with_reference.csv"
    )
    assert "VI_CONFIG" not in namespace
    assert "SEQUENTIAL_VI_CONFIG" not in namespace
    assert calls[0]["vi_config"] is None
    assert calls[0]["sequential_vi_config"] is None
    output = tmp_path / "results/evaluation" / csv_name
    associations = tmp_path / "results/associations" / csv_name
    assert set((tmp_path / "results").rglob("*.csv")) == {output, associations}
    trace = pd.read_csv(associations)
    assert trace.columns.tolist() == list(evaluation.ASSOCIATION_CSV_COLUMNS)
    assert trace["cycle_counter"].tolist() == [1, 2]
    assert trace["detection_id"].iloc[0] == 7
    assert trace.loc[1, ["detection_id", "x", "y"]].isna().all()
    exported = pd.read_csv(output)
    assert exported.columns.tolist() == list(evaluation.PREDICTION_CSV_COLUMNS)
    pd.testing.assert_frame_equal(
        exported[["inference_method", "position_error_m"]],
        pd.DataFrame(
            {"inference_method": [canonical_method], "position_error_m": [2.5]}
        ),
    )


def test_standard_evaluation_uses_the_public_csv_api(monkeypatch, evaluation_run):
    evaluation, _ = evaluation_run
    exports = []

    def export(table, path, *, summary=False):
        exports.append((path, summary))
        return True

    monkeypatch.setattr(evaluation, "export_evaluation_csv", export)

    runpy.run_path(SCRIPT)["main"]([])

    assert len(exports) == 1
    assert exports[0][1] is False


def test_entrypoint_keeps_output_paths_in_the_shared_source_module(
    tmp_path, monkeypatch, evaluation_run
):
    evaluation, calls = evaluation_run
    namespace = runpy.run_path(SCRIPT)
    assert "EVALUATION_OUTPUT_CSV" not in namespace
    assert "ASSOCIATIONS_OUTPUT_CSV" not in namespace
    monkeypatch.setattr(evaluation, "EVALUATION_OUTPUT_CSV", tmp_path / "metrics.csv")
    monkeypatch.setattr(
        evaluation, "ASSOCIATIONS_OUTPUT_CSV", tmp_path / "detections.csv"
    )

    namespace["main"]([])

    assert len(calls) == 1
    assert set(tmp_path.iterdir()) == {
        tmp_path / "metrics.csv",
        tmp_path / "detections.csv",
    }


@pytest.mark.parametrize(
    "forecasts,associations", [(True, True), (True, False), (False, True)]
)
def test_explicit_api_output_overrides_keep_their_disabled_export_semantics(
    tmp_path, monkeypatch, evaluation_run, forecasts, associations
):
    evaluation, calls = evaluation_run
    namespace = runpy.run_path(SCRIPT)
    run = evaluation.run_evaluation
    output = tmp_path / "custom_forecasts.csv"
    trace = tmp_path / "custom_associations.csv"

    def run_with_overrides(**kwargs):
        return run(
            **kwargs,
            output_csv=output if forecasts else None,
            associations_output_csv=trace if associations else None,
        )

    monkeypatch.setattr(evaluation, "run_evaluation", run_with_overrides)
    namespace["main"]([])

    expected = {output} if forecasts else set()
    if forecasts and associations:
        expected.add(trace)
    assert set(tmp_path.iterdir()) == expected
    assert (calls[0]["association_records"] is not None) == (forecasts and associations)


def test_association_filename_can_be_configured_independently_and_follows_cli_method(
    tmp_path, monkeypatch, evaluation_run
):
    evaluation, calls = evaluation_run
    namespace = runpy.run_path(SCRIPT)
    monkeypatch.setattr(evaluation, "EVALUATION_OUTPUT_CSV", tmp_path / "forecast.csv")
    monkeypatch.setattr(
        evaluation, "ASSOCIATIONS_OUTPUT_CSV", tmp_path / "raw_{inference_method}.csv"
    )

    namespace["main"](["--inference-method", "seq_vi"])

    assert len(calls) == 1
    assert set(tmp_path.iterdir()) == {
        tmp_path / "forecast.csv",
        tmp_path / "raw_seq_vi.csv",
    }
    assert pd.read_csv(tmp_path / "raw_seq_vi.csv")["cycle_counter"].tolist() == [1, 2]


def test_association_export_can_be_disabled_without_disabling_forecast_export(
    tmp_path, monkeypatch, evaluation_run
):
    evaluation, calls = evaluation_run
    namespace = runpy.run_path(SCRIPT)
    monkeypatch.setattr(evaluation, "ASSOCIATIONS_OUTPUT_CSV", None)

    namespace["main"]([])

    assert calls[0]["association_records"] is None
    assert list((tmp_path / "results/evaluation").iterdir()) == [
        tmp_path / "results/evaluation/bayesian_ctrv_smc.csv"
    ]


def test_association_destination_cannot_overwrite_the_forecast_csv(
    tmp_path, monkeypatch, capsys, evaluation_run
):
    evaluation, calls = evaluation_run
    namespace = runpy.run_path(SCRIPT)
    destination = tmp_path / "same.csv"
    monkeypatch.setattr(evaluation, "EVALUATION_OUTPUT_CSV", destination)
    monkeypatch.setattr(evaluation, "ASSOCIATIONS_OUTPUT_CSV", destination)

    namespace["main"]([])

    assert calls[0]["association_records"] is None
    assert pd.read_csv(destination).columns.tolist() == list(
        evaluation.PREDICTION_CSV_COLUMNS
    )
    assert "ASSOCIATIONS_OUTPUT_CSV" in capsys.readouterr().out


@pytest.mark.parametrize("destination", [42, "trace.txt", "bad\0.csv"])
def test_invalid_association_path_keeps_forecast_export(
    tmp_path, monkeypatch, capsys, evaluation_run, destination
):
    evaluation, calls = evaluation_run
    namespace = runpy.run_path(SCRIPT)
    monkeypatch.setattr(evaluation, "ASSOCIATIONS_OUTPUT_CSV", destination)

    namespace["main"]([])

    assert calls[0]["association_records"] is None
    assert list((tmp_path / "results/evaluation").iterdir()) == [
        tmp_path / "results/evaluation/bayesian_ctrv_smc.csv"
    ]
    assert "ASSOCIATIONS_OUTPUT_CSV" in capsys.readouterr().out


def test_cli_override_exports_selected_method_without_overwriting_other_methods(
    tmp_path, monkeypatch, evaluation_run
):
    _, calls = evaluation_run
    namespace = runpy.run_path(SCRIPT)
    monkeypatch.setitem(
        namespace["main"].__globals__,
        "EXPERIMENT",
        replace(namespace["EXPERIMENT"], inference_method="smc"),
    )
    monkeypatch.chdir(tmp_path)

    namespace["main"](["--inference-method", "vi", "--max-windows", "3"])
    namespace["main"]([])

    assert [call["options"].inference_method for call in calls] == ["vi", "smc"]
    assert calls[0]["options"].max_windows == 3
    assert calls[1]["options"].max_windows is None
    assert namespace["main"].__globals__["EXPERIMENT"].inference_method == "smc"
    outputs = set((tmp_path / "results").rglob("*.csv"))
    assert outputs == {
        tmp_path / "results/evaluation/bayesian_ctrv_vi.csv",
        tmp_path / "results/evaluation/bayesian_ctrv_smc.csv",
        tmp_path / "results/associations/bayesian_ctrv_vi.csv",
        tmp_path / "results/associations/bayesian_ctrv_smc.csv",
    }
    assert pd.read_csv(tmp_path / "results/evaluation/bayesian_ctrv_vi.csv")[
        "inference_method"
    ].tolist() == ["vi"]


@pytest.mark.parametrize(
    ("method", "canonical_method", "csv_method"),
    [("mcmc", "mcmc", "mcmc"), ("seq_vi", "sequential_vi", "seq_vi")],
)
def test_cli_override_uses_actual_method_for_csv_and_diagnostics(
    tmp_path, monkeypatch, evaluation_run, method, canonical_method, csv_method
):
    evaluation, calls = evaluation_run
    plotted = []

    def plot(table, *, inference_selection):
        output = tmp_path / f"results/evaluation/bayesian_ctrv_{csv_method}.csv"
        assert output.is_file()
        plotted.append(inference_selection)

    monkeypatch.setattr(
        evaluation.runtime_plotting, "plot_bayesian_ctrv_rolling_diagnostics", plot
    )
    namespace = runpy.run_path(SCRIPT)

    predictions, summary = namespace["main"](["--inference-method", method])

    assert calls[0]["options"].inference_method == canonical_method
    assert predictions["inference_method"].tolist() == [canonical_method]
    assert summary is None
    assert plotted == [canonical_method]
    assert not (tmp_path / "results/evaluation/bayesian_ctrv_smc.csv").exists()


@pytest.mark.parametrize(
    "output", [None, "", "   ", 42, "metrics.txt", ".", "bad\0.csv"]
)
def test_missing_or_invalid_csv_output_warns_and_keeps_evaluation_and_plots(
    tmp_path, monkeypatch, capsys, evaluation_run, output
):
    evaluation, calls = evaluation_run
    namespace = runpy.run_path(SCRIPT)
    monkeypatch.setattr(evaluation, "EVALUATION_OUTPUT_CSV", output)
    monkeypatch.chdir(tmp_path)
    plotted = []
    monkeypatch.setattr(
        evaluation.runtime_plotting,
        "plot_bayesian_ctrv_rolling_diagnostics",
        lambda predictions, **kwargs: plotted.append(predictions),
    )

    predictions, summary = namespace["main"]([])

    assert len(calls) == 1
    assert len(plotted) == 1 and plotted[0] is predictions
    assert summary is None
    assert list(tmp_path.iterdir()) == []
    terminal = capsys.readouterr().out
    assert "Warning:" in terminal and "EVALUATION_OUTPUT_CSV" in terminal
    assert "CSV export skipped" in terminal
    assert "Saved per-forecast" not in terminal


def test_csv_destination_directory_is_rejected_without_writing_files(
    tmp_path, monkeypatch, capsys, evaluation_run
):
    evaluation, calls = evaluation_run
    destination = tmp_path / "metrics.csv"
    destination.mkdir()
    namespace = runpy.run_path(SCRIPT)
    monkeypatch.setattr(evaluation, "EVALUATION_OUTPUT_CSV", destination)

    namespace["main"]([])

    assert len(calls) == 1
    assert list(destination.iterdir()) == []
    assert list(tmp_path.iterdir()) == [destination]
    assert "CSV export skipped" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["mkdir", "write", "replace"])
def test_csv_write_failure_preserves_previous_results_and_still_plots(
    tmp_path, monkeypatch, capsys, evaluation_run, failure
):
    evaluation, calls = evaluation_run
    namespace = runpy.run_path(SCRIPT)
    destination = tmp_path / "metrics.csv"
    original = b"previous,results\n1,2\n"
    destination.write_bytes(original)
    monkeypatch.setattr(evaluation, "EVALUATION_OUTPUT_CSV", destination)
    monkeypatch.setattr(
        evaluation,
        "ASSOCIATIONS_OUTPUT_CSV",
        tmp_path / "associations.csv",
    )
    plotted = []
    monkeypatch.setattr(
        evaluation.runtime_plotting,
        "plot_bayesian_ctrv_rolling_diagnostics",
        lambda predictions, **kwargs: plotted.append(predictions),
    )

    def deny(*args, **kwargs):
        raise PermissionError("destination is not writable")

    if failure == "mkdir":
        monkeypatch.setattr(Path, "mkdir", deny)
    elif failure == "replace":
        monkeypatch.setattr(Path, "replace", deny)
    else:

        def partial_write(table, path, **kwargs):
            Path(path).write_text("partial output", encoding="utf-8")
            raise OSError("disk full")

        monkeypatch.setattr(pd.DataFrame, "to_csv", partial_write)

    predictions, _ = namespace["main"]([])

    assert len(calls) == 1
    assert len(plotted) == 1 and plotted[0] is predictions
    assert destination.read_bytes() == original
    assert list(tmp_path.iterdir()) == [destination]
    terminal = capsys.readouterr().out
    assert "Warning:" in terminal and "CSV export skipped" in terminal
    assert "Saved per-forecast" not in terminal


@pytest.mark.parametrize("method", ["smc", "vi", "mcmc", "sequential_vi"])
def test_script_settings_and_source_output_control_evaluation(
    tmp_path, monkeypatch, evaluation_run, method
):
    evaluation, calls = evaluation_run
    namespace = runpy.run_path(SCRIPT)
    settings = namespace["main"].__globals__
    experiment = replace(
        namespace["EXPERIMENT"],
        inference_method=method,
        observation_count=14,
        prediction_count=7,
        stride=3,
        inference_seed=17,
        position_noise_std_m=1.25,
        position_noise_seed=19,
    )
    priors = replace(
        namespace["PRIORS"],
        speed_prior_upper_mps=6.0,
        turn_rate_prior_abs_rate_deg_s=3.0,
    )
    output = tmp_path / "custom_results" / f"{method}.csv"
    data_file = tmp_path / "custom_data.csv"
    for name, value in {
        "DATA_FILE": data_file,
        "EXPERIMENT": experiment,
        "PRIORS": priors,
        "MAX_WINDOWS": 2,
        "SHOW_ROLLING_DIAGNOSTICS_PLOT": False,
    }.items():
        monkeypatch.setitem(settings, name, value)
    monkeypatch.setattr(evaluation, "EVALUATION_OUTPUT_CSV", output)

    namespace["main"]([])

    call = calls[0]
    assert call["data_file"] == data_file
    assert call["experiment"] is experiment
    assert call["priors"] is priors
    assert call["vi_config"] is None
    assert call["sequential_vi_config"] is None
    assert call["credible_interval"] == 0.9
    assert call["options"].observation_count == 14
    assert call["options"].prediction_count == 7
    assert call["options"].stride == 3
    assert call["options"].inference_seed == 17
    assert call["options"].position_noise_std_m == 1.25
    assert call["options"].position_noise_seed == 19
    assert call["options"].turn_rate_prior_abs_rate_deg_s == 3.0
    assert call["options"].vi_algorithm == "fullrank"
    assert call["options"].require_converged is False
    assert call["options"].max_windows == 2
    assert pd.read_csv(output)["inference_method"].tolist() == [method]


def test_comparison_defines_individual_csv_paths_for_all_five_methods(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    namespace = runpy.run_path(
        ROOT / "scripts/model_comparison/compare_bayesian_ctrv_imm.py"
    )

    assert namespace["IMM_CSV"] == ROOT / "results/evaluation/imm.csv"
    for method in ("smc", "vi", "mcmc", "seq_vi"):
        assert namespace[f"BAYESIAN_{method.upper()}_CSV"] == (
            ROOT / f"results/evaluation/bayesian_ctrv_{method}.csv"
        )
