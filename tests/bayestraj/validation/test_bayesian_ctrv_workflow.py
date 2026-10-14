"""Tests for Bayesian CTRV rolling-evaluation presentation rules."""

import numpy as np
import pandas as pd
import pytest

import bayestraj.forecasting.bayesian_ctrv as forecasting
import bayestraj.inference.ctrv_sequential_vi as sequential_vi_model
import bayestraj.inference.ctrv_smc as smc_model
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.validation.bayesian_ctrv_workflow as workflow
import bayestraj.validation.cli as validation_cli
import bayestraj.validation.rolling as rolling
import bayestraj.validation.runtime_plotting as runtime_plotting


def test_online_window_specs_do_not_require_a_batch_window_mode():
    windows = workflow._build_evaluation_windows(
        15,
        online_mode=True,
        initial_observation_count=10,
        prediction_count=3,
        stride=2,
        window_mode=None,
    )

    assert windows == rolling.build_online_forecast_specs(
        15,
        initial_observation_count=10,
        prediction_count=3,
        stride=2,
    )


def test_processed_evaluation_input_keeps_local_positions_and_numeric_time(tmp_path):
    data_file = tmp_path / "processed_trajectory.csv"
    pd.DataFrame(
        {
            "time": [0.0, 2.5, 7.0, 11.0],
            "x": [100.0, 103.0, 107.0, 110.0],
            "y": [-20.0, -18.0, -15.0, -13.0],
        }
    ).to_csv(data_file, index=False)

    trajectory_data = workflow._load_evaluation_trajectory(data_file)
    window = workflow._prepare_evaluation_window(
        trajectory_data,
        observation_count=3,
        prediction_count=1,
        start_index=0,
    )
    route_x, route_y = workflow._prepare_route_coordinates(trajectory_data)
    route_time_seconds = workflow._prepare_route_time_seconds(
        trajectory_data,
    )

    np.testing.assert_allclose(window.x_meters, [100.0, 103.0, 107.0, 110.0])
    np.testing.assert_allclose(window.y_meters, [-20.0, -18.0, -15.0, -13.0])
    np.testing.assert_allclose(route_x, [100.0, 103.0, 107.0, 110.0])
    np.testing.assert_allclose(route_y, [-20.0, -18.0, -15.0, -13.0])
    np.testing.assert_allclose(route_time_seconds, [0.0, 2.5, 7.0, 11.0])


def test_window_metric_summary_includes_rmse_and_mean_log_predictive_density():
    evaluation = type(
        "Evaluation",
        (),
        {
            "ade_m": 10.0,
            "fde_m": 20.0,
            "mean_log_predictive_density": -3.4567,
            "prediction_table": pd.DataFrame({"position_error_m": [3.0, 4.0]}),
        },
    )()

    summary = workflow._window_metric_summary(evaluation)

    assert summary == "ADE=10.00 m, FDE=20.00 m, RMSE=3.54 m, Mean LogPD=-3.457"


def _write_br24_evaluation_input(path, *, reference_x_offset=0.0):
    cycle_count = 5
    times = np.arange(cycle_count, dtype=float)
    ranges = 10.0 + times
    pd.DataFrame(
        {
            "time": times,
            "cycle_counter": np.arange(1, cycle_count + 1),
            "sensor_x": np.zeros(cycle_count),
            "sensor_y": np.zeros(cycle_count),
            "sensor_yaw": np.zeros(cycle_count),
            "detection_id": np.arange(1, cycle_count + 1),
            "range": ranges,
            "azimuth": np.zeros(cycle_count),
            "std_range": np.full(cycle_count, 0.5),
            "std_azimuth": np.zeros(cycle_count),
            "x_relative": ranges,
            "y_relative": np.zeros(cycle_count),
            "ground_truth_x": ranges + reference_x_offset,
            "ground_truth_y": np.zeros(cycle_count),
            "ground_truth_yaw": np.zeros(cycle_count),
            "ground_truth_velocity": np.ones(cycle_count),
        }
    ).to_csv(path, index=False)


def _run_br24_evaluation(
    data_file,
    *,
    inference_method="smc",
    position_noise_std_m=0.0,
):
    experiment = forecasting.RollingExperimentConfig(
        observation_count=2,
        prediction_count=1,
        position_noise_std_m=position_noise_std_m,
        position_noise_seed=2026,
        stride=1,
        inference_method=inference_method,
        inference_seed=42,
    )
    options = validation_cli.BayesianCTRVEvaluationOptions(
        observation_count=2,
        prediction_count=1,
        stride=1,
        inference_method=inference_method,
        vi_algorithm="meanfield",
        turn_rate_prior_abs_rate_deg_s=10.0,
        inference_seed=42,
        position_noise_std_m=position_noise_std_m,
        position_noise_seed=2026,
        require_converged=False,
        max_windows=None,
        plot_each_window=False,
    )
    return workflow.run_bayesian_ctrv_evaluation(
        data_file=data_file,
        experiment=experiment,
        priors=bayesian_model.BayesianCTRVPriors(),
        smc_config=smc_model.SequentialMonteCarloCTRVConfig(
            particle_count=64,
            posterior_draw_count=16,
            predictive_log_density_threshold=None,
        ),
        fullrank_grad_samples=1,
        credible_interval=0.9,
        sample_trajectories_per_forecast=4,
        options=options,
        show_plot=False,
    )


def test_br24_evaluation_tracks_raw_cycles_and_uses_reference_only_for_metrics(
    tmp_path,
):
    input_csv = tmp_path / "BR24Dataset20131107_with_reference.csv"
    changed_reference_csv = tmp_path / "changed_reference.csv"
    _write_br24_evaluation_input(input_csv)
    _write_br24_evaluation_input(changed_reference_csv, reference_x_offset=100.0)

    predictions, summary = _run_br24_evaluation(input_csv)
    changed_predictions, _ = _run_br24_evaluation(changed_reference_csv)

    assert summary.inference_method == "smc"
    assert summary.window_count == 3
    assert predictions["x_actual"].tolist() == [12.0, 13.0, 14.0]
    assert changed_predictions["x_actual"].tolist() == [112.0, 113.0, 114.0]
    np.testing.assert_allclose(
        predictions["x_median"],
        changed_predictions["x_median"],
    )


def test_br24_evaluation_requires_smc_and_unmodified_raw_radar_measurements(tmp_path):
    input_csv = tmp_path / "BR24Dataset20131107_with_reference.csv"
    _write_br24_evaluation_input(input_csv)

    with pytest.raises(ValueError, match="requires the SMC"):
        _run_br24_evaluation(input_csv, inference_method="rbpf")
    with pytest.raises(ValueError, match="artificial position noise"):
        _run_br24_evaluation(input_csv, position_noise_std_m=0.1)


def test_sequential_vi_configuration_reporting_does_not_require_particles(capsys):
    config = sequential_vi_model.SequentialVIConfig(n_bootstrap=7, draws=25)

    workflow._print_online_configuration(
        "sequential_vi",
        particle_filter_config=None,
        sequential_vi_config=config,
    )

    output = capsys.readouterr().out
    assert "Bootstrap observations: 7" in output
    assert "Posterior draws       : 25" in output
    assert "Variational family    : fullrank" in output
    assert "Particles" not in output


def test_rolling_plot_is_hidden_only_for_overlapping_forecast_origins():
    assert workflow._should_plot_rolling_predictions(prediction_count=3, stride=None)
    assert workflow._should_plot_rolling_predictions(prediction_count=3, stride=3)
    assert not workflow._should_plot_rolling_predictions(prediction_count=3, stride=1)
    assert not workflow._should_plot_rolling_predictions(prediction_count=3, stride=2)


def test_explicit_plot_switch_disables_non_overlapping_rolling_plots():
    assert not workflow._should_show_rolling_plot(
        show_plot=False,
        prediction_count=3,
        stride=3,
    )
    assert workflow._should_show_rolling_plot(
        show_plot=True,
        prediction_count=3,
        stride=3,
    )


def test_selected_window_indices_preserve_chronological_online_safe_order():
    windows = rolling.build_rolling_window_specs(
        12,
        initial_observation_count=3,
        prediction_count=2,
        stride=2,
        window_mode="expanding",
    )

    selected = workflow._select_window_specs(windows, (3, 1))

    assert [window.window_index for window in selected] == [1, 3]


def test_selected_window_indices_reject_unknown_forecast_origin():
    windows = rolling.build_rolling_window_specs(
        8,
        initial_observation_count=3,
        prediction_count=2,
        window_mode="expanding",
    )

    try:
        workflow._select_window_specs(windows, (99,))
    except ValueError as error:
        assert "selected_window_indices" in str(error)
    else:
        raise AssertionError("Unknown selected window index must fail clearly.")


def test_online_filter_diagnostics_are_available_for_benchmark_records():
    online_filter = type(
        "OnlineFilter",
        (),
        {
            "last_effective_sample_size": 123.0,
            "effective_sample_size": 100.0,
            "resample_count": 4,
            "processed_observation_count": 9,
        },
    )()
    particle_config = type("ParticleConfig", (), {"particle_count": 500})()

    diagnostics = workflow._online_filter_diagnostics(online_filter, particle_config)

    assert diagnostics == {
        "particle_effective_sample_size": 123.0,
        "particle_resample_count": 4,
        "particle_processed_observation_count": 9,
        "particle_count": 500,
    }


def test_inference_configs_default_to_source_owned_factories():
    vi_config, mcmc_config, rbpf_config, smc_config, sequential_vi_config = (
        workflow._resolve_inference_configs(None, None, None, None)
    )

    assert vi_config["algorithm"] == "fullrank"
    assert mcmc_config["chains"] >= 1
    assert rbpf_config.particle_count > 0
    assert smc_config.particle_count > 0
    assert isinstance(sequential_vi_config, sequential_vi_model.SequentialVIConfig)


def test_fit_runtime_plot_uses_observation_history_for_its_x_axis(monkeypatch):
    predictions = pd.DataFrame(
        {
            "window_index": [1, 0, 1, 0],
            "observation_count": [7, 5, 7, 5],
            "fit_runtime_seconds": [2.0, 1.0, 2.0, 1.0],
            "inference_method": ["vi", "vi", "vi", "vi"],
        }
    )

    class Axis:
        def __init__(self):
            self.plot_arguments = None
            self.plot_keywords = None
            self.xlabel = None
            self.ylabel = None
            self.title = None
            self.title_keywords = None
            self.legend_location = None
            self.tick_parameters = None
            self.grid_called = False

        def plot(self, *arguments, **keywords):
            self.plot_arguments = arguments
            self.plot_keywords = keywords

        def set_xlabel(self, value, **_kwargs):
            self.xlabel = value

        def set_ylabel(self, value, **_kwargs):
            self.ylabel = value

        def set_title(self, value, **keywords):
            self.title = value
            self.title_keywords = keywords

        def grid(self, *_args, **_kwargs):
            self.grid_called = True

        def tick_params(self, **keywords):
            self.tick_parameters = keywords

        def legend(self, *, loc):
            self.legend_location = loc

    axis = Axis()
    figure = type("Figure", (), {"tight_layout": lambda _self: None})()
    monkeypatch.setattr(
        runtime_plotting.plt,
        "subplots",
        lambda *, figsize: (figure, axis),
    )
    monkeypatch.setattr(runtime_plotting.plt, "show", lambda: None)

    returned_figure, returned_axis = (
        runtime_plotting.plot_bayesian_ctrv_inference_runtime(
            predictions,
            inference_selection="vi",
        )
    )

    assert returned_figure is figure
    assert returned_axis is axis
    assert axis.plot_arguments == ([5, 7], [1.0, 2.0])
    assert axis.plot_keywords["label"] == "Inferenzzeit pro Vorhersagefenster"
    assert (
        axis.plot_keywords["color"]
        == runtime_plotting.RUNTIME_PLOT_STYLE.derived_data_color
    )
    assert axis.plot_keywords["markersize"] == 3
    assert axis.xlabel == "Anzahl bisher beobachteter Positionen"
    assert axis.ylabel == "Inferenzzeit pro Vorhersagefenster [s]"
    assert axis.title == "Bayesian-CTRV (VI)"
    assert not axis.grid_called
    assert axis.title_keywords == {"pad": 16, "fontsize": 13, "fontweight": "bold"}
    assert axis.tick_parameters == {"axis": "both", "labelsize": 11}
    assert axis.legend_location == "upper right"
