"""Tests for Bayesian CTRV rolling-evaluation presentation rules."""

import numpy as np
import pandas as pd
import pytest
import trajectory_predictor.dashboard as dashboard

import bayestraj.forecasting.bayesian_ctrv as forecasting
import bayestraj.inference.ctrv_sequential_vi as sequential_vi_model
import bayestraj.inference.ctrv_smc as smc_model
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.validation.bayesian_ctrv_workflow as workflow
import bayestraj.validation.cli as validation_cli
import bayestraj.validation.rolling as rolling
import bayestraj.validation.runtime_plotting as runtime_plotting


@pytest.mark.parametrize("batch_observation_count", [3, 10, 50])
def test_online_window_specs_start_at_two_independently_of_batch_window_size(
    batch_observation_count,
):
    windows = workflow._build_evaluation_windows(
        15,
        online_mode=True,
        batch_observation_count=batch_observation_count,
        prediction_count=3,
        stride=2,
        window_mode=None,
    )

    assert windows == rolling.build_online_forecast_specs(
        15,
        initial_observation_count=2,
        prediction_count=3,
        stride=2,
    )


def test_batch_windows_grow_from_two_then_keep_only_the_last_ten_positions():
    windows = workflow._build_evaluation_windows(
        15,
        online_mode=False,
        batch_observation_count=10,
        prediction_count=3,
        stride=1,
        window_mode="sliding",
    )

    assert [window.forecast_start_index for window in windows] == list(range(2, 15))
    assert [window.observation_count for window in windows] == [
        *range(2, 11),
        10,
        10,
        10,
        10,
    ]
    assert [window.start_index for window in windows] == [0] * 9 + [1, 2, 3, 4]
    assert [window.prediction_count for window in windows[-3:]] == [3, 2, 1]


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


def test_window_metric_summary_includes_rmse_and_mean_negative_log_predictive_density():
    evaluation = type(
        "Evaluation",
        (),
        {
            "ade_m": 10.0,
            "fde_m": 20.0,
            "mean_negative_log_predictive_density": 3.4567,
            "prediction_table": pd.DataFrame({"position_error_m": [3.0, 4.0]}),
        },
    )()

    summary = workflow._window_metric_summary(evaluation)

    assert summary == "ADE=10.00 m, FDE=20.00 m, RMSE=3.54 m, Mean NLPD=3.457"


def _write_br24_evaluation_input(path, *, reference_x_offset=0.0, cycle_count=5):
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
    max_windows=None,
    sequential_vi_config=None,
    prediction_count=1,
):
    experiment = forecasting.RollingExperimentConfig(
        observation_count=10,
        prediction_count=prediction_count,
        position_noise_std_m=position_noise_std_m,
        position_noise_seed=2026,
        stride=1,
        inference_method=inference_method,
        inference_seed=42,
    )
    options = validation_cli.BayesianCTRVEvaluationOptions(
        observation_count=10,
        prediction_count=prediction_count,
        stride=1,
        inference_method=inference_method,
        vi_algorithm="meanfield",
        turn_rate_prior_abs_rate_deg_s=10.0,
        inference_seed=42,
        position_noise_std_m=position_noise_std_m,
        position_noise_seed=2026,
        require_converged=False,
        max_windows=max_windows,
        plot_each_window=False,
    )
    return workflow.run_bayesian_ctrv_evaluation(
        data_file=data_file,
        experiment=experiment,
        priors=bayesian_model.BayesianCTRVPriors(),
        smc_config=smc_model.SequentialMonteCarloCTRVConfig(
            particle_count=64,
            posterior_draw_count=16,
        ),
        sequential_vi_config=sequential_vi_config,
        fullrank_grad_samples=1,
        credible_interval=0.9,
        sample_trajectories_per_forecast=4,
        options=options,
        show_plot=False,
    )


@pytest.mark.parametrize("method", ["vi", "mcmc", "smc", "sequential_vi"])
@pytest.mark.parametrize("radar_input", [True, False])
def test_forecast_metrics_start_at_two_with_batch_fit_or_sequential_prior(
    tmp_path, monkeypatch, method, radar_input
):
    input_csv = tmp_path / "trajectory.csv"
    if radar_input:
        _write_br24_evaluation_input(input_csv)
    else:
        pd.DataFrame(
            {"time": [10.0, 12.8, 16.0, 18.5], "x": [0.0, 1.0, 2.0, 3.0], "y": 0.0}
        ).to_csv(input_csv, index=False)

    fit_counts = []

    def fit_model(window, *, position_observations, **kwargs):
        from bayestraj.inference.particle_utils import SequentialCTRVFit

        fit_counts.append(window.observation_count)
        workflow.bayesian_model.build_stan_data(
            window, position_observations=position_observations
        )
        if method == "sequential_vi":
            pytest.fail("Sequential VI must wait for its configured bootstrap.")
        fit = sequential_vi_model.forecast_ctrv_prior_predictive(
            x_at_origin=position_observations.x_meters[-1],
            y_at_origin=position_observations.y_meters[-1],
            origin_time_seconds=window.time_seconds[window.observation_count - 1],
            future_time_seconds=window.time_seconds[window.prediction_slice],
            priors=kwargs["priors"],
            draw_count=16,
            seed=42,
        )

        class FittedPosterior(SequentialCTRVFit):
            converged = True

            def diagnose(self):
                return "No problems detected"

        names = (
            *bayesian_model.PARAMETER_NAMES,
            "x_at_origin",
            "y_at_origin",
            "x_prediction",
            "y_prediction",
            "x_observation_prediction",
            "y_observation_prediction",
        )
        return FittedPosterior({name: fit.stan_variable(name) for name in names})

    monkeypatch.setattr(workflow.batch_inference, "fit_bayesian_ctrv_model", fit_model)
    monkeypatch.setattr(
        workflow.cmdstan_inference, "variational_converged", lambda fit: True
    )
    predictions, summary = _run_br24_evaluation(
        input_csv, inference_method=method, max_windows=1
    )

    assert summary.window_count == 1
    assert predictions["observation_count"].tolist() == [2]
    assert predictions["forecast_start_index"].tolist() == [2]
    assert predictions["x_actual"].tolist() == [12.0 if radar_input else 2.0]
    assert np.all(np.isfinite(predictions["position_error_m"]))
    assert np.all(np.isfinite(predictions["negative_log_predictive_density"]))
    assert fit_counts == ([2] if method in {"vi", "mcmc"} else [])
    assert predictions["is_prior_predictive"].tolist() == [method == "sequential_vi"]
    if method == "vi":
        assert predictions["converged"].tolist() == [True]
    elif method == "mcmc":
        assert predictions["mcmc_diagnostics_ok"].tolist() == [True]


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_radar_batch_evaluation_starts_early_and_keeps_the_configured_window(
    tmp_path, monkeypatch, method
):
    from bayestraj.inference.particle_utils import SequentialCTRVFit

    input_csv = tmp_path / "radar.csv"
    _write_br24_evaluation_input(input_csv, cycle_count=14)
    fit_positions = []

    class FittedPosterior(SequentialCTRVFit):
        converged = True

        def diagnose(self):
            return "No problems detected"

    def fit_model(window, *, position_observations, **kwargs):
        fit_positions.append(position_observations.x_meters.copy())
        values = np.ones(32)
        return FittedPosterior(
            {
                "x_at_origin": values * position_observations.x_meters[-1],
                "y_at_origin": values * position_observations.y_meters[-1],
                "speed_at_origin": values,
                "heading_at_origin": values * 0.0,
                "turn_rate_at_origin": values * 0.0,
                "sigma_position_observation": values * 0.1,
                "sigma_speed_process": values * 0.1,
                "sigma_turn_rate_process": values * 0.01,
            }
        )

    monkeypatch.setattr(workflow.batch_inference, "fit_bayesian_ctrv_model", fit_model)
    predictions, summary = _run_br24_evaluation(input_csv, inference_method=method)

    assert summary.window_count == 12
    assert predictions["forecast_start_index"].tolist() == list(range(2, 14))
    assert not predictions["is_prior_predictive"].any()
    for origin, positions in zip(range(2, 14), fit_positions, strict=True):
        np.testing.assert_array_equal(
            positions, 10.0 + np.arange(max(0, origin - 10), origin)
        )


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_processed_batch_evaluation_grows_then_slides_without_fitting_future_points(
    tmp_path, monkeypatch, method
):
    input_csv = tmp_path / "positions.csv"
    times = np.cumsum(np.linspace(2.0, 3.0, 14))
    pd.DataFrame({"time": times, "x": np.arange(14), "y": 0.0}).to_csv(
        input_csv, index=False
    )
    fit_observations = []
    fit_seeds = []

    def fit_window(window, *, position_observations, initial_seed, **kwargs):
        fit_observations.append(position_observations.x_meters.copy())
        fit_seeds.append(initial_seed)
        fit = sequential_vi_model.forecast_ctrv_prior_predictive(
            x_at_origin=position_observations.x_meters[-1],
            y_at_origin=position_observations.y_meters[-1],
            origin_time_seconds=window.time_seconds[window.observation_count - 1],
            future_time_seconds=window.time_seconds[window.prediction_slice],
            priors=kwargs["priors"],
            draw_count=32,
            seed=initial_seed,
        )
        return fit, initial_seed

    monkeypatch.setattr(workflow, "_fit_window", fit_window)
    monkeypatch.setattr(
        workflow.cmdstan_inference, "variational_converged", lambda fit: True
    )
    monkeypatch.setattr(workflow, "_mcmc_diagnostics_ok", lambda fit: True)

    predictions, _ = _run_br24_evaluation(input_csv, inference_method=method)

    assert predictions["observation_count"].tolist() == [*range(2, 11), 10, 10, 10]
    assert [len(positions) for positions in fit_observations] == [
        *range(2, 11),
        10,
        10,
        10,
    ]
    for origin, positions in zip(range(2, 14), fit_observations, strict=True):
        np.testing.assert_array_equal(positions, np.arange(max(0, origin - 10), origin))
    assert not predictions["is_prior_predictive"].any()
    assert fit_seeds == [42] * 12


@pytest.mark.parametrize(
    ("radar_input", "position_noise_std_m"),
    [(True, 0.0), (False, 0.0), (False, 1.5)],
)
def test_rolling_forecasts_match_the_tool_at_every_origin(
    tmp_path, radar_input, position_noise_std_m
):
    input_csv = tmp_path / "trajectory.csv"
    if radar_input:
        _write_br24_evaluation_input(input_csv)
    else:
        pd.DataFrame(
            {
                "time": [10.0, 12.8, 16.0, 18.5, 21.3, 25.0],
                "x": np.arange(6, dtype=float),
                "y": np.zeros(6),
            }
        ).to_csv(input_csv, index=False)
    experiment = dashboard.PosteriorDashboardConfig(
        start_index=0,
        maximum_observation_count=None,
        prediction_count=3,
        position_noise_std_m=position_noise_std_m,
        position_noise_seed=2026,
        inference_method="smc",
        inference_seed=42,
    )
    options = {
        "experiment": experiment,
        "priors": bayesian_model.BayesianCTRVPriors(),
        "smc_config": smc_model.SequentialMonteCarloCTRVConfig(
            particle_count=64, posterior_draw_count=16
        ),
    }
    if radar_input:
        trajectory, _, _, load_update = dashboard.create_br24_radar_dashboard_loader(
            input_csv, **options
        )
    else:
        trajectory, _, _, load_update = dashboard.create_posterior_dashboard_loader(
            pd.read_csv(input_csv), vi_config={}, mcmc_config={}, **options
        )

    predictions, _ = _run_br24_evaluation(
        input_csv, position_noise_std_m=position_noise_std_m, prediction_count=3
    )

    for count, rows in predictions.groupby("forecast_start_index", sort=True):
        update = load_update(int(count))
        forecast = update.forecast
        np.testing.assert_array_equal(
            forecast.median_positions,
            rows[["x_median_route", "y_median_route"]].to_numpy(),
        )
        gui_metrics = dashboard.analysis_metrics_at(
            trajectory, {int(count): update}, int(count)
        )
        assert gui_metrics.negative_log_predictive_density == pytest.approx(
            rows["negative_log_predictive_density"].mean()
        )


def test_processed_sequential_vi_keeps_prior_forecasts_until_its_own_bootstrap(
    tmp_path, monkeypatch
):
    input_csv = tmp_path / "positions.csv"
    pd.DataFrame({"time": np.arange(7), "x": np.arange(7), "y": 0.0}).to_csv(
        input_csv, index=False
    )

    def unexpected_initialize(*args, **kwargs):
        pytest.fail("Sequential VI must wait for its configured bootstrap count.")

    monkeypatch.setattr(workflow, "_advance_online_filter", unexpected_initialize)
    predictions, summary = _run_br24_evaluation(
        input_csv,
        inference_method="sequential_vi",
        sequential_vi_config=sequential_vi_model.SequentialVIConfig(n_bootstrap=5),
        max_windows=3,
    )

    assert summary.window_count == 3
    assert predictions["observation_count"].tolist() == [2, 3, 4]
    assert predictions["is_prior_predictive"].all()
    assert predictions["converged"].isna().all()


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
    np.testing.assert_allclose(
        predictions["forecast_origin_seconds"],
        [1.0, 2.0, 3.0],
    )
    assert changed_predictions["x_actual"].tolist() == [112.0, 113.0, 114.0]
    np.testing.assert_allclose(
        predictions["x_median"],
        changed_predictions["x_median"],
    )


@pytest.mark.parametrize("method", ["vi", "mcmc", "smc", "sequential_vi"])
def test_br24_evaluation_dispatches_the_selected_radar_method(
    tmp_path, monkeypatch, method
):
    input_csv = tmp_path / "radar.csv"
    _write_br24_evaluation_input(input_csv)
    tracker_type = workflow.br24_radar_smc.BR24BayesianRadarTracker
    original_initialize = tracker_type.initialize
    configured = []

    class FittedFilter:
        converged = True

        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.last_fit = type(
                "SamplerDiagnostics",
                (),
                {"diagnose": lambda self: "No problems detected"},
            )()

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    def initialize(cycle, **kwargs):
        configured.append(kwargs)
        tracker = original_initialize(
            cycle,
            initial_detection_id=kwargs["initial_detection_id"],
            priors=kwargs["priors"],
            config=kwargs["config"],
            seed=kwargs["seed"],
        )
        tracker.filter = FittedFilter(tracker.filter)
        return tracker

    monkeypatch.setattr(tracker_type, "initialize", initialize)
    predictions, summary = _run_br24_evaluation(input_csv, inference_method=method)

    assert configured[0]["inference_method"] == method
    assert configured[0]["observation_count"] == 10
    assert configured[0]["vi_config"]["algorithm"] == "meanfield"
    assert configured[0]["sequential_vi_config"].n_bootstrap == 3
    assert summary.inference_method == method
    assert set(predictions["inference_method"]) == {method}
    assert predictions["forecast_origin_seconds"].tolist() == [1.0, 2.0, 3.0]


def test_br24_evaluation_requires_unmodified_raw_radar_measurements(tmp_path):
    input_csv = tmp_path / "BR24Dataset20131107_with_reference.csv"
    _write_br24_evaluation_input(input_csv)

    with pytest.raises(ValueError, match="artificial position noise"):
        _run_br24_evaluation(input_csv, position_noise_std_m=0.1)


@pytest.mark.parametrize("method", ["vi", "mcmc"])
def test_radar_prior_forecasts_do_not_count_as_failed_fits(tmp_path, method):
    input_csv = tmp_path / "radar.csv"
    _write_br24_evaluation_input(input_csv)
    predictions, _ = _run_br24_evaluation(input_csv)
    predictions["inference_mode"] = "batch"
    predictions["inference_method"] = method
    predictions["is_prior_predictive"] = [True, False, False]
    column = "converged" if method == "vi" else "mcmc_diagnostics_ok"
    predictions[column] = [None, True, False]

    summary = rolling.summarize_rolling_predictions(predictions)

    rate = (
        summary.vi_convergence_rate
        if method == "vi"
        else summary.mcmc_diagnostics_pass_rate
    )
    assert rate == 0.5


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
    vi_config, mcmc_config, smc_config, sequential_vi_config = (
        workflow._resolve_inference_configs(None, None, None, None)
    )

    assert vi_config["algorithm"] == "fullrank"
    assert mcmc_config["chains"] >= 1
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


@pytest.mark.parametrize("model", ["bayesian", "imm"])
def test_rolling_diagnostics_plot_renders_color_coded_figures_without_axis_titles(
    monkeypatch,
    model,
):
    predictions = pd.DataFrame(
        {
            "window_index": [1, 0, 1, 0, 0],
            "forecast_origin_seconds": [120.0, 100.0, 120.0, 100.0, 100.0],
            "horizon_step": [1, 1, 2, 2, 3],
            "position_error_m": [5.0, 3.0, 12.0, 4.0, 8.0],
            "negative_log_predictive_density": [5.0, 1.0, 7.0, 3.0, 5.0],
            "energy_score_m": [6.0, 2.0, 10.0, 4.0, 6.0],
            "fit_runtime_seconds": [2.0, 1.0, 2.0, 1.0, 1.0],
            "inference_method": ["smc" if model == "bayesian" else "imm"] * 5,
        }
    )
    original = predictions.copy(deep=True)
    shown_figures = []
    monkeypatch.setattr(
        runtime_plotting.plt,
        "show",
        lambda: shown_figures.append(runtime_plotting.plt.gcf()),
    )

    if model == "bayesian":
        figures = runtime_plotting.plot_bayesian_ctrv_rolling_diagnostics(
            predictions,
            inference_selection="smc_expanding",
        )
    else:
        figures = runtime_plotting.plot_imm_rolling_diagnostics(predictions)
    runtime_label = (
        "Inferenzzeit" if model == "bayesian" else "Update-/Forecast-Laufzeit"
    )

    expected_plots = (
        ("ADE [m]", "ADE", "#E69F00", [5.0, 8.5], 6.75, "Mittlerer ADE: 6.75 m"),
        ("FDE [m]", "FDE", "#D55E00", [8.0, 12.0], 10.0, "Mittlerer FDE: 10.00 m"),
        (
            "RMSE [m]",
            "RMSE",
            "#0072B2",
            [np.sqrt(89 / 3), np.sqrt(84.5)],
            7.3195498508,
            "Mittlerer RMSE: 7.32 m",
        ),
        (
            "Energy Score [m]",
            "Energy Score",
            "#009E73",
            [4.0, 8.0],
            6.0,
            "Mittlerer Energy Score: 6.00 m",
        ),
        (
            "NLPD",
            "NLPD",
            "#CC79A7",
            [3.0, 6.0],
            4.5,
            "Mittlere NLPD: 4.500",
        ),
        (
            f"{runtime_label} [s]",
            runtime_label,
            "#56B4E9",
            [1.0, 2.0],
            1.5,
            f"Mittlere {runtime_label}: 1.500 s",
        ),
    )
    assert len(figures) == len(expected_plots)
    assert shown_figures == [figure for figure, _ in figures]
    for (figure, axis), (ylabel, label, color, values, mean, mean_label) in zip(
        figures,
        expected_plots,
        strict=True,
    ):
        assert figure._suptitle.get_fontsize() == 13
        assert figure._suptitle.get_fontweight() == "bold"
        assert figure._suptitle.get_text() == (
            "Bayesian CTRV (SMC, wachsendes Fenster)" if model == "bayesian" else "IMM"
        )
        assert figure.get_size_inches() == pytest.approx(
            runtime_plotting.RUNTIME_PLOT_STYLE.speed_figure_size
        )
        assert axis.get_xlabel() == "Zeit [s]"
        assert axis.get_title() == ""
        assert axis.get_ylabel() == ylabel
        for gridlines in (axis.get_xgridlines(), axis.get_ygridlines()):
            assert gridlines
            assert all(line.get_visible() for line in gridlines)
            assert all(line.get_alpha() == 0.2 for line in gridlines)
            assert all(line.get_color() == "0.7" for line in gridlines)
        assert axis.get_axisbelow() is True
        np.testing.assert_allclose(axis.lines[0].get_xdata(), [0.0, 20.0])
        np.testing.assert_allclose(axis.lines[0].get_ydata(), values)
        assert axis.lines[0].get_color() == color
        assert len(axis.lines) == 1
        assert np.mean(values) == pytest.approx(mean)
        assert [entry.get_text() for entry in axis.get_legend().get_texts()] == [
            label,
        ]
        mean_caption = [text for text in figure.texts if text.get_text() == mean_label]
        assert len(mean_caption) == 1
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        caption_box = mean_caption[0].get_window_extent(renderer)
        xlabel_box = axis.xaxis.label.get_window_extent(renderer)
        assert caption_box.y1 < xlabel_box.y0
        assert figure.bbox.contains(caption_box.x0, caption_box.y0)
        assert figure.bbox.contains(caption_box.x1, caption_box.y1)
        runtime_plotting.plt.close(figure)
    pd.testing.assert_frame_equal(predictions, original)


@pytest.mark.parametrize("model", ["bayesian", "imm"])
@pytest.mark.parametrize(
    "invalid", ["missing_column", "empty", "mixed_methods", "nonfinite", "mixed_times"]
)
def test_rolling_diagnostics_reject_invalid_tables_before_opening_figures(
    model, invalid
):
    predictions = pd.DataFrame(
        {
            "window_index": [0, 0],
            "forecast_origin_seconds": [100.0, 100.0],
            "horizon_step": [1, 2],
            "position_error_m": [3.0, 4.0],
            "negative_log_predictive_density": [1.0, 2.0],
            "energy_score_m": [2.0, 3.0],
            "fit_runtime_seconds": [0.01, 0.01],
            "inference_method": ["smc" if model == "bayesian" else "imm"] * 2,
        }
    )
    if invalid == "missing_column":
        predictions = predictions.drop(columns="energy_score_m")
    elif invalid == "empty":
        predictions = predictions.iloc[:0]
    elif invalid == "mixed_methods":
        predictions.loc[1, "inference_method"] = "other"
    elif invalid == "nonfinite":
        predictions.loc[1, "position_error_m"] = np.nan
    else:
        predictions.loc[1, "forecast_origin_seconds"] = 101.0
    plot = (
        runtime_plotting.plot_bayesian_ctrv_rolling_diagnostics
        if model == "bayesian"
        else runtime_plotting.plot_imm_rolling_diagnostics
    )
    figures_before = runtime_plotting.plt.get_fignums()
    with pytest.raises(ValueError):
        plot(predictions)
    assert runtime_plotting.plt.get_fignums() == figures_before
