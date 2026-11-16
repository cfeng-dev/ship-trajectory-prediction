"""Observation-count sweeps reuse the rolling evaluation and CSV exporter."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import bayestraj.validation.bayesian_ctrv_evaluation as evaluation
import bayestraj.validation.comparison as comparison
import bayestraj.validation.rolling as rolling
from bayestraj.inference.particle_utils import SequentialCTRVFit

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("method", ["mcmc", "vi"])
@pytest.mark.parametrize("source", ["associated", "raw"])
def test_sweep_input_formats_preserve_common_origins_reference_and_metadata(
    tmp_path, monkeypatch, method, source
):
    times = np.array([49073.2, 49075.8, 49078.5, 49084.1, 49086.9, 49089.8])
    positions = np.array([10.0, 12.0, 14.0, 18.0, 20.0, 22.0])
    data = pd.DataFrame(
        {
            "time": times,
            "cycle_counter": [2, 3, 4, 6, 7, 8],
            "detection_id": 1,
            "ground_truth_x": positions + 5.0,
            "ground_truth_y": 0.0,
        }
    )
    if source == "associated":
        data["x"], data["y"] = positions, -4.0
        data["cov_xx"], data["cov_xy"], data["cov_yy"] = 2.0, 0.25, 1.0
    else:
        data["sensor_x"], data["sensor_y"], data["sensor_yaw"] = 0.0, 0.0, 0.0
        data["range"], data["azimuth"] = (
            np.hypot(positions, 4.0),
            np.arctan2(-4.0, positions),
        )
        data["std_range"], data["std_azimuth"] = 0.5, 0.02
        data["x_relative"], data["y_relative"] = positions, -4.0
    path = tmp_path / "trajectory.csv"
    data.to_csv(path, index=False)
    namespace = _script(method)
    monkeypatch.setitem(namespace["main"].__globals__, "DATA_FILE", path)
    fits = []

    class Fit(SequentialCTRVFit):
        converged = True
        metadata = SimpleNamespace(cmdstan_config={"max_depth": 5})
        num_draws_sampling = 2

        def diagnose(self):
            return "No problems detected"

        def summary(self):
            return pd.DataFrame(
                {"R_hat": [1.005], "ESS_bulk": [700.0], "ESS_tail": [400.0]},
                index=["speed_state[1]"],
            )

        def method_variables(self):
            return {
                "divergent__": np.array([[0, 1], [0, 0]]),
                "treedepth__": np.array([[4, 5], [5, 4]]),
            }

    def fit_model(window, *, position_observations, **kwargs):
        fits.append(position_observations)
        spread = np.linspace(-0.1, 0.1, 32)
        posterior = Fit(
            {
                "x_at_origin": position_observations.x_meters[-1] + spread,
                "y_at_origin": position_observations.y_meters[-1] + spread,
                "speed_at_origin": np.full(32, 1.0),
                "heading_at_origin": spread,
                "turn_rate_at_origin": spread * 0.01,
                "sigma_position_observation": np.full(32, 0.1),
                "sigma_speed_process": np.full(32, 0.1),
                "sigma_turn_rate_process": np.full(32, 0.001),
            }
        )
        if method == "vi":
            transcript = tmp_path / "vi.txt"
            transcript.write_text(
                "  iter = 100\n  tol_rel_obj = 0.01\n"
                "  iter ELBO delta_ELBO_mean delta_ELBO_med notes\n"
                "100 -123.0 0.3 0.2\n"
                "The maximum number of iterations is reached! "
                "The algorithm may not have converged.\nCOMPLETED.\n",
                encoding="utf-8",
            )
            posterior.runset = SimpleNamespace(stdout_files=[transcript])
            posterior.converged = False
        return posterior

    monkeypatch.setattr(
        evaluation.workflow.batch_inference, "fit_bayesian_ctrv_model", fit_model
    )
    output = tmp_path / "sweep"
    outputs = namespace["main"](
        [
            "--min-observations",
            "2",
            "--max-observations",
            "3",
            "--max-windows",
            "3",
            "--output-dir",
            str(output),
        ]
    )
    assert len(outputs) == 2
    for count, destination in zip([2, 3], outputs, strict=True):
        table = pd.read_csv(destination)
        np.testing.assert_array_equal(
            table["forecast_origin_seconds"].unique(), times[1:4]
        )
        np.testing.assert_allclose(
            table["x_actual"], [19.0, 23.0, 25.0, 23.0, 25.0, 27.0, 25.0, 27.0]
        )
        assert table.groupby("window_index")["window_cycle_count"].first().tolist() == [
            2,
            min(3, count),
            count,
        ]
        assert table["configured_observation_count"].eq(count).all()
        if method == "vi":
            assert table["vi_termination"].eq("iteration_limit").all()
            assert table["vi_elbo_delta_mean"].eq(0.3).all()
            assert table["vi_elbo_delta_median"].eq(0.2).all()
            assert not table["converged"].any()
        else:
            assert table["vi_termination"].isna().all()
            assert table["vi_elbo_delta_mean"].isna().all()
        expected_mcmc = {
            "mcmc_divergences": 1,
            "mcmc_ess_bulk_min": 700.0,
            "mcmc_ess_tail_min": 400.0,
            "mcmc_treedepth_hits": 2,
        }
        for name, value in expected_mcmc.items():
            if method == "mcmc":
                assert table[name].eq(value).all()
            else:
                assert table[name].isna().all()
        np.testing.assert_allclose(
            table.loc[table["window_index"].eq(0), "horizon_seconds"], [2.7, 8.3, 11.1]
        )
    np.testing.assert_allclose(fits[0].x_meters, [10.0, 12.0])
    if source == "associated":
        np.testing.assert_array_equal(
            fits[0].measurement_covariances, [[[2.0, 0.25], [0.25, 1.0]]] * 2
        )
    metadata = evaluation.read_observation_count_metadata(output)
    assert all(
        record["data_file"] == str(path.resolve()) for record in metadata.values()
    )
    assert not (output / "summary.csv").exists()
    summary = comparison.read_observation_count_evaluations(
        **{f"{method}_dir": output}
    )[method.upper()]
    assert summary["observation_count"].tolist() == [2, 3]
    assert summary["window_count"].tolist() == [3, 3]
    assert summary["forecast_count"].tolist() == [8, 8]


@pytest.fixture
def sweep_run(tmp_path, monkeypatch):
    monkeypatch.setattr("bayestraj.paths.project_path", lambda path: tmp_path / path)
    monkeypatch.setattr(
        evaluation.workflow.br24_observations,
        "is_br24_associated_input",
        lambda path: False,
    )
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
                    "time": pd.Timestamp(
                        window.forecast_start_index + step - 1, unit="s", tz="UTC"
                    ),
                    "prediction_count": window.prediction_count,
                    "x_actual": float(window.forecast_start_index + step - 1),
                    "y_actual": 0.0,
                    "x_median": float(window.forecast_start_index + 2 * step - 1),
                    "x_mean": float(window.forecast_start_index + 2 * step - 1),
                    "y_mean": 0.0,
                    "point_estimate": "mean",
                    "y_median": 0.0,
                    "model_variant": "bayesian",
                    "is_prior_predictive": False,
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
                    "nlpd_definition": "gaussian_latent_2d_v1",
                    "evaluation_definition": "gaussian_latent_2d_gl64_runtime_v2",
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
def test_evaluation_exports_forecasts_and_metadata_without_creating_summary(
    tmp_path, sweep_run, method
):
    outputs = _script(method)["main"](
        ["--min-observations", "2", "--max-observations", "3", "--max-windows", "1"]
    )
    directory = tmp_path / f"results/inference_evaluation/bayesian_ctrv_{method}"
    assert not (directory / "summary.csv").exists()
    assert outputs == tuple(
        directory / f"bayesian_ctrv_{method}_{count:02d}.csv" for count in [2, 3]
    )
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert set(metadata["evaluations"]) == {path.name for path in outputs}
    assert all(
        record["inference_seed"] == 42 for record in metadata["evaluations"].values()
    )
    assert all(
        record["nlpd_definition"] == "gaussian_latent_2d_v1"
        for record in metadata["evaluations"].values()
    )
    assert all(
        record["evaluation_definition"] == "gaussian_latent_2d_gl64_runtime_v2"
        for record in metadata["evaluations"].values()
    )


@pytest.mark.parametrize("method", ["mcmc", "vi"])
def test_scripts_sweep_two_to_fifteen_with_common_origins_and_method_csvs(
    tmp_path, monkeypatch, sweep_run, method
):
    namespace = _script(method)
    assert "ASSOCIATIONS_OUTPUT_DIR" not in namespace
    monkeypatch.setitem(namespace["main"].__globals__, "MAX_WINDOWS", 2)
    monkeypatch.chdir(tmp_path)

    outputs = namespace["main"]([])

    assert [call["options"].observation_count for call in sweep_run] == list(
        range(2, 16)
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
        *(f"bayesian_ctrv_{method}_{count:02d}.csv" for count in range(2, 16)),
        "metadata.json",
    ]
    for count in range(2, 16):
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

    assert len(outputs) == 14
    assert not (output / "summary.csv").exists()
    saved = comparison.read_observation_count_evaluations(**{f"{method}_dir": output})[
        method.upper()
    ]
    assert saved["observation_count"].tolist() == list(range(2, 16))
    assert "evaluation_start_observation_count" not in saved
    assert saved["window_count"].tolist() == [2] * 14
    assert saved["forecast_count"].tolist() == [6] * 14
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
    assert saved["inference_seed"].tolist() == [42] * 14
    assert saved["data_file"].tolist() == [str(namespace["DATA_FILE"].resolve())] * 14


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
    ]


@pytest.mark.parametrize("method", ["mcmc", "vi"])
def test_sweep_reuses_single_evaluation_without_reparsing_cli_or_plotting(
    tmp_path, monkeypatch, sweep_run, method
):
    calls = []
    original_run = evaluation.run_evaluation

    def run(**kwargs):
        calls.append(kwargs)
        return original_run(**kwargs)

    def unexpected_call(*args, **kwargs):
        pytest.fail("A sweep must not parse per-run CLI arguments or plot diagnostics.")

    monkeypatch.setattr(evaluation, "run_evaluation", run)
    monkeypatch.setattr(
        evaluation.cli, "parse_bayesian_ctrv_evaluation_arguments", unexpected_call
    )
    monkeypatch.setattr(
        evaluation.runtime_plotting,
        "plot_bayesian_ctrv_rolling_diagnostics",
        unexpected_call,
    )

    _script(method)["main"](
        ["--min-observations", "2", "--max-observations", "3", "--max-windows", "1"]
    )

    assert [call["experiment"].observation_count for call in calls] == [2, 3]
    assert len(sweep_run) == 2
    for count, call in zip([2, 3], calls, strict=True):
        assert call["output_csv"] == (
            tmp_path
            / f"results/inference_evaluation/bayesian_ctrv_{method}"
            / f"bayesian_ctrv_{method}_{count:02d}.csv"
        )
        assert call["options"].observation_count == count
        assert call["options"].max_windows == 1
        assert call["selected_window_indices"] == tuple(range(22))
        assert call["associations_output_csv"] is None
        assert call["show_rolling_diagnostics_plot"] is False
        assert call["require_csv_export"] is True


def test_observation_count_evaluation_preserves_explicit_inference_settings(
    tmp_path, sweep_run
):
    namespace = _script("mcmc")
    mcmc_config = {"chains": 2, "iter_sampling": 75}
    vi_config = {"algorithm": "meanfield", "require_converged": True}

    evaluation.run_observation_count_evaluation(
        data_file=namespace["DATA_FILE"],
        output_dir=tmp_path / "custom",
        experiment=namespace["EXPERIMENT"],
        priors=namespace["PRIORS"],
        min_observation_count=2,
        max_observation_count=2,
        max_windows=1,
        mcmc_config=mcmc_config,
        vi_config=vi_config,
        credible_interval=0.8,
        fullrank_grad_samples=13,
        argv=[],
    )

    assert len(sweep_run) == 1
    call = sweep_run[0]
    assert call["mcmc_config"] is mcmc_config
    assert call["vi_config"] is vi_config
    assert call["options"].vi_algorithm == "meanfield"
    assert call["options"].require_converged is True
    assert call["credible_interval"] == 0.8
    assert call["fullrank_grad_samples"] == 13


@pytest.mark.parametrize("method", ["mcmc", "vi"])
def test_complete_sweep_has_normal_evaluation_row_count(tmp_path, sweep_run, method):
    outputs = _script(method)["main"](
        ["--min-observations", "20", "--max-observations", "20"]
    )

    # A 24-cycle route evaluated from N=2 has 22 origins and horizons 3, ..., 3, 2, 1.
    assert len(outputs) == 1
    path = (
        tmp_path
        / f"results/inference_evaluation/bayesian_ctrv_{method}/bayesian_ctrv_{method}_20.csv"
    )
    table = pd.read_csv(path)
    assert table["window_index"].nunique() == 22
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

    outputs = namespace["main"](
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
    assert len(outputs) == 2
    assert all(pd.read_csv(path)["window_index"].nunique() == 1 for path in outputs)
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
    outputs = _script("vi")["main"](
        ["--min-observations", "24", "--max-observations", "24", "--max-windows", "1"]
    )

    assert len(outputs) == 1 and outputs[0].name == "bayesian_ctrv_vi_24.csv"
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


def test_sweep_stops_on_metadata_export_failure(tmp_path, monkeypatch, sweep_run):
    original_replace = Path.replace

    def replace(path, destination):
        if Path(destination).name == "metadata.json":
            raise OSError("metadata is read-only")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", replace)

    with pytest.raises(OSError, match="metadata"):
        _script("vi")["main"](["--max-windows", "1"])

    assert len(sweep_run) == 1
    destination = tmp_path / "results/inference_evaluation/bayesian_ctrv_vi"
    assert (destination / "bayesian_ctrv_vi_02.csv").is_file()
    assert not (destination / "summary.csv").exists()


def test_partial_sweeps_preserve_other_metadata_and_existing_summary(
    tmp_path, sweep_run
):
    namespace = _script("vi")
    namespace["main"](
        ["--min-observations", "2", "--max-observations", "3", "--max-windows", "1"]
    )
    directory = tmp_path / "results/inference_evaluation/bayesian_ctrv_vi"
    (directory / "summary.csv").write_text("previous derived results", encoding="utf-8")

    namespace["main"](
        ["--min-observations", "4", "--max-observations", "4", "--max-windows", "1"]
    )

    metadata = evaluation.read_observation_count_metadata(directory)
    assert set(metadata) == {f"bayesian_ctrv_vi_{count:02d}.csv" for count in [2, 3, 4]}
    assert (directory / "summary.csv").read_text(
        encoding="utf-8"
    ) == "previous derived results"
    table = comparison.read_observation_count_evaluations(vi_dir=directory)["VI"]
    assert table["observation_count"].tolist() == [2, 3, 4]


def test_rerunning_one_size_preserves_seed_provenance_for_other_sizes(
    tmp_path, monkeypatch, sweep_run
):
    namespace = _script("mcmc")
    namespace["main"](
        ["--min-observations", "2", "--max-observations", "3", "--max-windows", "1"]
    )
    monkeypatch.setitem(
        namespace["main"].__globals__,
        "EXPERIMENT",
        evaluation.dataclasses.replace(namespace["EXPERIMENT"], inference_seed=43),
    )

    namespace["main"](
        ["--min-observations", "2", "--max-observations", "2", "--max-windows", "1"]
    )

    directory = tmp_path / "results/inference_evaluation/bayesian_ctrv_mcmc"
    metadata = evaluation.read_observation_count_metadata(directory)
    assert metadata["bayesian_ctrv_mcmc_02.csv"]["inference_seed"] == 43
    assert metadata["bayesian_ctrv_mcmc_03.csv"]["inference_seed"] == 42
    with pytest.raises(ValueError, match="inference_seed"):
        comparison.read_observation_count_evaluations(mcmc_dir=directory)


def test_invalid_metadata_stops_before_inference(tmp_path, sweep_run):
    directory = tmp_path / "results/inference_evaluation/bayesian_ctrv_vi"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="metadata"):
        _script("vi")["main"]([])

    assert sweep_run == []
