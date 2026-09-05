"""Tests for the combined Bayesian CTRV posterior dashboard."""

import importlib
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backend_bases import (
    FigureCanvasBase,
    KeyEvent,
    MouseEvent,
    NavigationToolbar2,
    ResizeEvent,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import bayestraj.inference.configuration as inference
import bayestraj.inference.ctrv_rbpf as rbpf
import bayestraj.inference.ctrv_smc as smc
import bayestraj.models.bayesian_ctrv as bayesian_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("figure_size", [(11, 8), (10, 6), (9.5, 5.5)])
@pytest.mark.parametrize("dpi", [100, 150])
def test_dashboard_labels_clear_adjacent_panels_and_controls(figure_size, dpi):
    dashboard = _load_dashboard_module()
    coordinates = np.arange(4.0)
    figure = Figure(figsize=figure_size, dpi=dpi)
    canvas = FigureCanvasAgg(figure)
    _, navigator = dashboard.create_sequential_posterior_dashboard_figure(
        dashboard.PosteriorDashboardTrajectory(*([coordinates] * 4)),
        bayesian_model.BayesianCTRVPriors(),
        lambda count: dashboard.PosteriorDashboardUpdate(count, _dashboard_samples()),
        maximum_observation_count=4,
        figure=figure,
    )
    try:
        navigator.slider.set_val(1)
        navigator.show_selected_observation_count(None)
        for group_index in (0, 1):
            navigator.group_selector.set_active(group_index)
            canvas.draw()
            _assert_dashboard_vertical_spacing(navigator, canvas.get_renderer())
    finally:
        navigator.disconnect()


def _assert_dashboard_vertical_spacing(navigator, renderer):
    minimum_gap = renderer.points_to_pixels(6)
    for upper, lower in zip(
        navigator.posterior_axes[:-1], navigator.posterior_axes[1:], strict=True
    ):
        gap = (
            upper.xaxis.label.get_window_extent(renderer).y0
            - lower.title.get_window_extent(renderer).y1
        )
        assert gap >= minimum_gap, f"Posterior-panel label gap: {gap:.1f}px"
    label_bottom = (
        navigator.posterior_axes[-1].xaxis.label.get_window_extent(renderer).y0
    )
    controls_top = navigator.group_selector.ax.get_tightbbox(renderer).y1
    assert label_bottom - controls_top >= minimum_gap, (
        f"Posterior/control gap: {label_bottom - controls_top:.1f}px"
    )


def test_dashboard_spacing_tracks_canvas_resize_without_loading_updates():
    dashboard = _load_dashboard_module()
    coordinates = np.arange(4.0)
    figure = Figure(figsize=(11, 8))
    canvas = FigureCanvasAgg(figure)
    requests = []
    _, navigator = dashboard.create_sequential_posterior_dashboard_figure(
        dashboard.PosteriorDashboardTrajectory(*([coordinates] * 4)),
        bayesian_model.BayesianCTRVPriors(),
        figure=figure,
        maximum_observation_count=4,
        request_update=requests.append,
    )
    try:
        for figure_size in ((9.5, 5.5), (14, 9), (10, 6)):
            figure.set_size_inches(*figure_size)
            ResizeEvent("resize_event", canvas)._process()
            canvas.draw()
            _assert_dashboard_vertical_spacing(navigator, canvas.get_renderer())
        assert requests == []
        assert navigator.observation_count == 0
    finally:
        navigator.disconnect()


def test_dashboard_spacing_survives_toolbar_navigation_after_resize():
    dashboard = _load_dashboard_module()
    coordinates = np.arange(4.0)
    figure = Figure(figsize=(11, 8))
    canvas = FigureCanvasAgg(figure)
    toolbar = NavigationToolbar2(canvas)
    _, navigator = dashboard.create_sequential_posterior_dashboard_figure(
        dashboard.PosteriorDashboardTrajectory(*([coordinates] * 4)),
        bayesian_model.BayesianCTRVPriors(),
        figure=figure,
        maximum_observation_count=4,
        request_update=lambda _: None,
    )
    try:
        canvas.draw()
        toolbar.push_current()
        navigator.trajectory_axis.set_xlim(0.5, 2.0)
        toolbar.push_current()
        figure.set_size_inches(9.5, 5.5)
        ResizeEvent("resize_event", canvas)._process()
        for navigate in (toolbar.home, toolbar.forward, toolbar.back):
            navigate()
            canvas.draw()
            _assert_dashboard_vertical_spacing(navigator, canvas.get_renderer())
    finally:
        navigator.disconnect()


def test_embedded_async_navigation_waits_caches_and_preserves_pause():
    dashboard = _load_dashboard_module()
    coordinates = np.arange(4.0)
    trajectory = dashboard.PosteriorDashboardTrajectory(
        coordinates, coordinates, coordinates, coordinates
    )
    requests = []
    supplied_figure = Figure()
    figure, navigator = dashboard.create_sequential_posterior_dashboard_figure(
        trajectory,
        bayesian_model.BayesianCTRVPriors(),
        None,
        maximum_observation_count=4,
        figure=supplied_figure,
        request_update=requests.append,
    )
    assert figure is supplied_figure
    navigator.toggle_playback(None)
    navigator.advance_playback()
    navigator.advance_playback()
    assert navigator.observation_count == 0
    assert navigator.requested_observation_count == 1
    assert requests == [1]
    navigator.handle_key_press(KeyEvent("key_press_event", figure.canvas, key="left"))
    assert not navigator.is_waiting
    assert requests[-1] == 0
    navigator.toggle_playback(None)
    navigator.advance_playback()
    navigator.pause_playback()
    navigator.accept_update(dashboard.PosteriorDashboardUpdate(1, _dashboard_samples()))
    assert navigator.observation_count == 0
    assert not navigator.is_waiting
    navigator.trajectory_axis.set_xlim(0.5, 2.0)
    navigator.toggle_playback(None)
    navigator.advance_playback()
    assert navigator.observation_count == 1
    assert navigator.trajectory_axis.get_xlim() == (0.5, 2.0)
    navigator.slider.set_val(4)
    navigator.show_selected_observation_count(None)
    for count in (2, 3):
        navigator.accept_update(
            dashboard.PosteriorDashboardUpdate(count, _dashboard_samples())
        )
        assert navigator.observation_count == 1
    navigator.accept_update(dashboard.PosteriorDashboardUpdate(4, _dashboard_samples()))
    assert navigator.observation_count == 4
    navigator.disconnect()
    assert not navigator.is_playing


def test_async_playback_stops_after_final_result_not_when_requested():
    dashboard = _load_dashboard_module()
    coordinates = np.arange(2.0)
    _, navigator = dashboard.create_sequential_posterior_dashboard_figure(
        dashboard.PosteriorDashboardTrajectory(*([coordinates] * 4)),
        bayesian_model.BayesianCTRVPriors(),
        maximum_observation_count=1,
        figure=Figure(),
        request_update=lambda _: None,
    )
    navigator.toggle_playback(None)
    navigator.advance_playback()
    assert navigator.is_playing
    assert navigator.is_waiting
    navigator.accept_update(dashboard.PosteriorDashboardUpdate(1, _dashboard_samples()))
    assert navigator.observation_count == 1
    assert not navigator.is_playing
    navigator.disconnect()


def _load_dashboard_module():
    try:
        return importlib.import_module(
            "bayestraj.validation.bayesian_ctrv_posterior_dashboard"
        )
    except ModuleNotFoundError:
        pytest.fail("The Bayesian CTRV posterior dashboard module is missing.")


def _load_dashboard_script():
    script_path = (
        PROJECT_ROOT
        / "experiments"
        / "posterior_analysis"
        / "plot_bayesian_ctrv_posterior_updates.py"
    )
    if not script_path.is_file():
        pytest.fail(f"Posterior dashboard script is missing: {script_path}")
    module_name = f"{script_path.stem}_for_tests"
    specification = importlib.util.spec_from_file_location(module_name, script_path)
    if specification is None or specification.loader is None:
        pytest.fail(f"Cannot load posterior dashboard script: {script_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


class _FakeFit:
    def __init__(self, variables):
        self._variables = variables

    def stan_variable(self, variable_name, **_kwargs):
        return self._variables[variable_name]


class _FakeTimer:
    def __init__(self, interval):
        self.interval = interval
        self.callbacks = []
        self.start_count = 0
        self.stop_count = 0

    def add_callback(self, callback, *args, **kwargs):
        self.callbacks.append((callback, args, kwargs))

    def start(self):
        self.start_count += 1

    def stop(self):
        self.stop_count += 1

    def fire(self):
        for callback, args, kwargs in tuple(self.callbacks):
            callback(*args, **kwargs)


def _dashboard_samples(offset=0.0):
    return {
        "current_speed": np.linspace(1.0 + offset, 2.0 + offset, 20),
        "current_heading": np.linspace(-10.0 + offset, 10.0 + offset, 20),
        "current_turn_rate": np.linspace(-1.0 + offset, 1.0 + offset, 20),
        "position_observation_noise": np.linspace(1.0 + offset, 3.0 + offset, 20),
        "speed_process_noise": np.linspace(0.1 + offset, 0.3 + offset, 20),
        "turn_rate_process_noise": np.linspace(0.2 + offset, 0.4 + offset, 20),
    }


def _dashboard_fit_variables():
    return {
        "speed_at_origin": np.asarray([2.0, 3.0]),
        "heading_at_origin": np.asarray([0.0, np.pi / 2.0]),
        "turn_rate_at_origin": np.asarray([0.0, np.pi / 180.0]),
        "sigma_position_observation": np.asarray([1.0, 2.0]),
        "sigma_speed_process": np.asarray([0.1, 0.2]),
        "sigma_turn_rate_process": np.asarray([np.pi / 180.0, np.pi / 90.0]),
    }


def _dashboard_trajectory_data():
    return pd.DataFrame(
        {
            "time": pd.date_range(
                "2026-01-01",
                periods=4,
                freq="10s",
                tz="UTC",
            ),
            "run_id": 102,
            "gps_latitude": 54.0 + np.arange(4) * 1e-5,
            "gps_longitude": 10.0 + np.arange(4) * 2e-5,
            "gps_speed": np.full(4, 18.0),
        }
    )


@pytest.mark.parametrize("inference_method", ["vi", "mcmc", "rbpf", "smc"])
def test_dashboard_config_accepts_all_ctrv_inference_methods(inference_method):
    dashboard = _load_dashboard_module()

    experiment = dashboard.PosteriorDashboardConfig(
        run_id=102,
        start_index=0,
        maximum_observation_count=20,
        position_noise_std_m=5.0,
        position_noise_seed=2026,
        inference_method=inference_method.upper(),
        inference_seed=42,
    )

    assert experiment.inference_method == inference_method


@pytest.mark.parametrize("inference_method", ["rbpf", "smc"])
def test_dashboard_loader_uses_selected_online_filter_config(inference_method):
    dashboard = _load_dashboard_module()
    rbpf_config = rbpf.SequentialCTRVFilterConfig(
        particle_count=32,
        posterior_draw_count=20,
    )
    smc_config = smc.SequentialMonteCarloCTRVConfig(
        particle_count=48,
        posterior_draw_count=24,
    )
    expected_config = {
        "rbpf": rbpf_config,
        "smc": smc_config,
    }[inference_method]

    experiment = dashboard.PosteriorDashboardConfig(
        run_id=102,
        start_index=0,
        maximum_observation_count=None,
        position_noise_std_m=0.0,
        position_noise_seed=2026,
        inference_method=inference_method,
        inference_seed=42,
    )

    trajectory, maximum_count, minimum_count, load_update = (
        dashboard.create_posterior_dashboard_loader(
            _dashboard_trajectory_data(),
            experiment=experiment,
            priors=bayesian_model.BayesianCTRVPriors(),
            vi_config=inference.create_default_vi_config(),
            mcmc_config=inference.create_default_mcmc_config(),
            rbpf_config=rbpf_config,
            smc_config=smc_config,
        )
    )

    update = load_update(2)

    assert maximum_count == 4
    assert minimum_count == 1
    assert trajectory.reference_x.shape == (4,)
    assert update.observation_count == 2
    assert update.particle_count == expected_config.particle_count
    assert all(
        samples.size == expected_config.posterior_draw_count
        for samples in update.samples_by_parameter.values()
    )


@pytest.mark.parametrize("inference_method", ["vi", "mcmc"])
def test_dashboard_loader_runs_selected_expanding_batch_fit(inference_method):
    dashboard = _load_dashboard_module()
    vi_config = inference.create_default_vi_config()
    mcmc_config = inference.create_default_mcmc_config()
    expected_config = {
        "vi": vi_config,
        "mcmc": mcmc_config,
    }[inference_method]
    fit_calls = []

    def fit_batch_model(window, **options):
        fit_calls.append((window, options))
        return _FakeFit(_dashboard_fit_variables())

    experiment = dashboard.PosteriorDashboardConfig(
        run_id=102,
        start_index=0,
        maximum_observation_count=None,
        position_noise_std_m=0.0,
        position_noise_seed=2026,
        inference_method=inference_method,
        inference_seed=42,
    )

    trajectory, maximum_count, minimum_count, load_update = (
        dashboard.create_posterior_dashboard_loader(
            _dashboard_trajectory_data(),
            experiment=experiment,
            priors=bayesian_model.BayesianCTRVPriors(),
            vi_config=vi_config,
            mcmc_config=mcmc_config,
            rbpf_config=inference.create_default_ctrv_rbpf_config(),
            smc_config=inference.create_default_ctrv_smc_config(),
            fit_batch_model=fit_batch_model,
        )
    )

    update = load_update(3)

    assert maximum_count == 3
    assert minimum_count == 3
    assert trajectory.reference_x.shape == (4,)
    assert len(fit_calls) == 1
    window, options = fit_calls[0]
    assert window.observation_count == 3
    assert options["inference_method"] == inference_method
    assert options["seed"] == 42
    assert options["priors"] == bayesian_model.BayesianCTRVPriors()
    assert options["position_observations"].x_meters == pytest.approx(
        trajectory.observed_x[:3]
    )
    selected_options = {
        name: value
        for name, value in options.items()
        if name
        not in {
            "inference_method",
            "position_observations",
            "priors",
            "seed",
        }
    }
    assert selected_options == expected_config
    assert update.observation_count == 3
    assert update.effective_sample_size is None
    assert update.particle_count is None
    assert update.resample_count is None


def test_dashboard_loader_updates_one_filter_and_extracts_all_parameters():
    dashboard = _load_dashboard_module()
    trajectory_data = _dashboard_trajectory_data()
    variables = _dashboard_fit_variables()
    initializations = []
    updated_times = []
    posterior_seeds = []

    class FakeFilter:
        def __init__(self, config):
            self.config = config
            self.processed_observation_count = 1
            self.resample_count = 1
            self.effective_sample_size = 24.0

        def update(self, time_seconds, _x_observed, _y_observed):
            updated_times.append(time_seconds)
            self.processed_observation_count += 1

        def sample_current_posterior(self, *, seed):
            posterior_seeds.append(seed)
            return _FakeFit(variables)

    rbpf_config = rbpf.SequentialCTRVFilterConfig(
        particle_count=32,
        posterior_draw_count=2,
    )

    def initialize_filter(
        time_seconds,
        x_observed,
        y_observed,
        *,
        priors,
        config,
        seed,
    ):
        initializations.append(
            (time_seconds.copy(), x_observed.copy(), y_observed.copy(), priors, seed)
        )
        return FakeFilter(config)

    trajectory, maximum_count, load_update = (
        dashboard.create_rbpf_posterior_dashboard_loader(
            trajectory_data,
            start_index=0,
            position_noise_std_m=0.0,
            position_noise_seed=2026,
            priors=bayesian_model.BayesianCTRVPriors(),
            rbpf_config=rbpf_config,
            rbpf_seed=42,
            initialize_filter=initialize_filter,
        )
    )

    updates = tuple(load_update(count) for count in (1, 2, 3))

    assert maximum_count == 4
    assert len(initializations) == 1
    assert initializations[0][0] == pytest.approx([0.0])
    assert updated_times == pytest.approx([10.0, 20.0])
    assert posterior_seeds == [42, 42, 42]
    assert trajectory.reference_x.shape == (4,)
    assert trajectory.reference_y.shape == (4,)
    assert trajectory.observed_x == pytest.approx(trajectory.reference_x)
    assert trajectory.observed_y == pytest.approx(trajectory.reference_y)

    final_update = updates[-1]
    assert final_update.observation_count == 3
    assert tuple(final_update.samples_by_parameter) == dashboard.PARAMETER_NAMES
    assert final_update.samples_by_parameter["current_speed"] == pytest.approx(
        [2.0, 3.0]
    )
    assert final_update.samples_by_parameter["current_heading"] == pytest.approx(
        [0.0, 90.0]
    )
    assert final_update.samples_by_parameter["current_turn_rate"] == pytest.approx(
        [0.0, 1.0]
    )
    assert final_update.samples_by_parameter[
        "turn_rate_process_noise"
    ] == pytest.approx([1.0, 2.0])
    assert final_update.effective_sample_size == pytest.approx(24.0)
    assert final_update.particle_count == 32
    assert final_update.resample_count == 1


def test_dashboard_synchronizes_route_and_three_switchable_posterior_axes():
    dashboard = _load_dashboard_module()
    priors = bayesian_model.BayesianCTRVPriors()
    trajectory = dashboard.PosteriorDashboardTrajectory(
        reference_x=[0.0, 1.0, 2.0, 3.0],
        reference_y=[0.0, 1.0, 1.5, 1.8],
        observed_x=[0.1, 1.1, 2.1, 3.1],
        observed_y=[-0.1, 0.9, 1.4, 1.7],
    )
    loaded_counts = []

    def load_update(observation_count):
        loaded_counts.append(observation_count)
        return dashboard.PosteriorDashboardUpdate(
            observation_count,
            _dashboard_samples(offset=0.01 * observation_count),
            effective_sample_size=18.0,
            particle_count=32,
            resample_count=1,
        )

    figure, navigator = dashboard.create_sequential_posterior_dashboard_figure(
        trajectory,
        priors,
        load_update,
        maximum_observation_count=4,
    )

    try:
        assert navigator.observation_count == 0
        assert navigator.parameter_group == "motion"
        assert navigator.parameter_names == (
            "current_speed",
            "current_heading",
            "current_turn_rate",
        )
        assert [line.get_label() for line in navigator.trajectory_axis.lines] == [
            "Aufgezeichnete Trajektorie"
        ]
        assert all(
            [line.get_label() for line in axis.lines] == ["Ausgangs-Prior"]
            for axis in navigator.posterior_axes
        )

        navigator.slider.set_val(3)
        navigator.show_selected_observation_count(None)

        assert navigator.observation_count == 3
        assert loaded_counts == [1, 2, 3]
        assert [line.get_label() for line in navigator.trajectory_axis.lines] == [
            "Aufgezeichnete Trajektorie",
            "Beobachtungen bis N",
            "Aktuelle Position",
        ]
        observed_line = navigator.trajectory_axis.lines[1]
        assert observed_line.get_xdata() == pytest.approx([0.1, 1.1, 2.1])
        assert observed_line.get_ydata() == pytest.approx([-0.1, 0.9, 1.4])
        assert all(
            [line.get_label() for line in axis.lines]
            == ["Ausgangs-Prior", "Posterior-Dichte"]
            for axis in navigator.posterior_axes
        )

        navigator.group_selector.set_active(1)

        assert navigator.parameter_group == "noise"
        assert navigator.parameter_names == (
            "position_observation_noise",
            "speed_process_noise",
            "turn_rate_process_noise",
        )
        assert loaded_counts == [1, 2, 3]
        assert all(
            [line.get_label() for line in axis.lines]
            == ["Ausgangs-Prior", "Posterior-Dichte"]
            for axis in navigator.posterior_axes
        )
    finally:
        plt.close(figure)


def test_dashboard_shows_batch_prior_until_minimum_observation_count():
    dashboard = _load_dashboard_module()
    trajectory = dashboard.PosteriorDashboardTrajectory(
        reference_x=[0.0, 1.0, 2.0, 3.0],
        reference_y=[0.0, 1.0, 1.5, 1.8],
        observed_x=[0.0, 1.0, 2.0, 3.0],
        observed_y=[0.0, 1.0, 1.5, 1.8],
    )
    loaded_counts = []

    def load_update(observation_count):
        loaded_counts.append(observation_count)
        return dashboard.PosteriorDashboardUpdate(
            observation_count,
            _dashboard_samples(),
        )

    figure, navigator = dashboard.create_sequential_posterior_dashboard_figure(
        trajectory,
        bayesian_model.BayesianCTRVPriors(),
        load_update,
        maximum_observation_count=4,
        minimum_posterior_observation_count=3,
    )

    try:
        navigator.slider.set_val(2)
        navigator.show_selected_observation_count(None)

        assert navigator.observation_count == 2
        assert loaded_counts == []
        assert all(
            [line.get_label() for line in axis.lines] == ["Ausgangs-Prior"]
            for axis in navigator.posterior_axes
        )
        assert all(
            "Posterior ab N = 3 verfügbar" in {text.get_text() for text in axis.texts}
            for axis in navigator.posterior_axes
        )

        navigator.slider.set_val(3)
        navigator.show_selected_observation_count(None)

        assert loaded_counts == [3]
        assert all(
            [line.get_label() for line in axis.lines]
            == ["Ausgangs-Prior", "Posterior-Dichte"]
            for axis in navigator.posterior_axes
        )
    finally:
        plt.close(figure)


def test_dashboard_arrow_keys_reuse_cached_stages_and_respect_boundaries():
    dashboard = _load_dashboard_module()
    trajectory = dashboard.PosteriorDashboardTrajectory(
        reference_x=[0.0, 1.0],
        reference_y=[0.0, 1.0],
        observed_x=[0.0, 1.0],
        observed_y=[0.0, 1.0],
    )
    loaded_counts = []

    def load_update(observation_count):
        loaded_counts.append(observation_count)
        return dashboard.PosteriorDashboardUpdate(
            observation_count,
            _dashboard_samples(),
        )

    figure, navigator = dashboard.create_sequential_posterior_dashboard_figure(
        trajectory,
        bayesian_model.BayesianCTRVPriors(),
        load_update,
        maximum_observation_count=2,
    )

    def press(key):
        event = KeyEvent("key_press_event", figure.canvas, key=key)
        figure.canvas.callbacks.process("key_press_event", event)

    try:
        press("left")
        press("right")
        press("right")
        press("right")
        press("left")

        assert navigator.observation_count == 1
        assert navigator.slider.val == pytest.approx(1.0)
        assert loaded_counts == [1, 2]
    finally:
        plt.close(figure)


def test_dashboard_script_runs_the_shared_analysis_without_showing(monkeypatch):
    script = _load_dashboard_script()
    calls = []
    sentinel = object()

    def fake_run(**options):
        calls.append(options)
        return sentinel

    monkeypatch.setattr(
        script.dashboard,
        "run_bayesian_ctrv_posterior_dashboard",
        fake_run,
    )

    result = script.main(["--no-show"])

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0]["data_file"] == script.DATA_FILE
    assert calls[0]["experiment"] is script.EXPERIMENT
    assert script.EXPERIMENT.run_id == 102
    assert script.EXPERIMENT.start_index == 0
    assert script.EXPERIMENT.maximum_observation_count is None
    assert script.EXPERIMENT.position_noise_std_m == pytest.approx(5.0)
    assert script.EXPERIMENT.position_noise_seed == 2026
    assert script.EXPERIMENT.inference_method == "rbpf"
    assert script.EXPERIMENT.inference_seed == 42
    assert calls[0]["priors"] is script.PRIORS
    assert calls[0]["vi_config"] is script.VI_CONFIG
    assert calls[0]["mcmc_config"] is script.MCMC_CONFIG
    assert calls[0]["rbpf_config"] is script.RBPF_CONFIG
    assert calls[0]["smc_config"] is script.SMC_CONFIG
    assert calls[0]["playback_interval_ms"] == 1_000
    assert calls[0]["show_legend"] is True
    assert calls[0]["show"] is False


def test_dashboard_runner_uses_configured_inference_loader(monkeypatch):
    dashboard = _load_dashboard_module()
    trajectory_data = _dashboard_trajectory_data()
    trajectory = dashboard.PosteriorDashboardTrajectory(
        reference_x=[0.0, 1.0, 2.0, 3.0],
        reference_y=[0.0, 1.0, 1.5, 1.8],
        observed_x=[0.0, 1.0, 2.0, 3.0],
        observed_y=[0.0, 1.0, 1.5, 1.8],
    )
    experiment = dashboard.PosteriorDashboardConfig(
        run_id=102,
        start_index=0,
        maximum_observation_count=3,
        position_noise_std_m=0.0,
        position_noise_seed=2026,
        inference_method="vi",
        inference_seed=42,
    )
    priors = bayesian_model.BayesianCTRVPriors()
    vi_config = inference.create_default_vi_config()
    mcmc_config = inference.create_default_mcmc_config()
    rbpf_config = inference.create_default_ctrv_rbpf_config()
    smc_config = inference.create_default_ctrv_smc_config()
    loader_calls = []
    figure_calls = []

    monkeypatch.setattr(
        dashboard.observations_io,
        "read_ship_data",
        lambda _data_file, *, run_id: trajectory_data,
    )

    def create_loader(data, **options):
        loader_calls.append((data, options))
        return trajectory, 3, 3, lambda _count: None

    def create_figure(
        selected_trajectory,
        selected_priors,
        update_loader,
        **options,
    ):
        figure_calls.append(
            (selected_trajectory, selected_priors, update_loader, options)
        )
        return plt.figure(), object()

    monkeypatch.setattr(dashboard, "create_posterior_dashboard_loader", create_loader)
    monkeypatch.setattr(
        dashboard,
        "create_sequential_posterior_dashboard_figure",
        create_figure,
    )

    figure, navigator = dashboard.run_bayesian_ctrv_posterior_dashboard(
        data_file=Path("trajectory.csv"),
        experiment=experiment,
        priors=priors,
        vi_config=vi_config,
        mcmc_config=mcmc_config,
        rbpf_config=rbpf_config,
        smc_config=smc_config,
        playback_interval_ms=750,
        show_legend=False,
        show=False,
    )

    assert len(loader_calls) == 1
    _, loader_options = loader_calls[0]
    assert loader_options == {
        "experiment": experiment,
        "priors": priors,
        "vi_config": vi_config,
        "mcmc_config": mcmc_config,
        "rbpf_config": rbpf_config,
        "smc_config": smc_config,
    }
    assert len(figure_calls) == 1
    assert figure_calls[0][3] == {
        "maximum_observation_count": 3,
        "minimum_posterior_observation_count": 3,
        "show_legend": False,
        "playback_interval_ms": 750,
    }
    assert navigator is not None
    assert not plt.fignum_exists(figure.number)


def test_dashboard_keeps_manual_trajectory_zoom_when_posterior_stage_changes():
    dashboard = _load_dashboard_module()
    trajectory = dashboard.PosteriorDashboardTrajectory(
        reference_x=[0.0, 1.0, 2.0, 3.0],
        reference_y=[0.0, 1.0, 1.5, 1.8],
        observed_x=[0.0, 1.0, 2.0, 3.0],
        observed_y=[0.0, 1.0, 1.5, 1.8],
    )

    def load_update(observation_count):
        return dashboard.PosteriorDashboardUpdate(
            observation_count,
            _dashboard_samples(),
        )

    figure, navigator = dashboard.create_sequential_posterior_dashboard_figure(
        trajectory,
        bayesian_model.BayesianCTRVPriors(),
        load_update,
        maximum_observation_count=4,
    )

    try:
        navigator.slider.set_val(2)
        navigator.show_selected_observation_count(None)
        navigator.trajectory_axis.set_xlim(0.5, 2.5)
        navigator.trajectory_axis.set_ylim(0.4, 1.6)
        figure.canvas.draw()
        zoomed_xlim = navigator.trajectory_axis.get_xlim()
        zoomed_ylim = navigator.trajectory_axis.get_ylim()

        navigator.slider.set_val(3)
        navigator.show_selected_observation_count(None)
        figure.canvas.draw()

        assert navigator.trajectory_axis.get_xlim() == pytest.approx(zoomed_xlim)
        assert navigator.trajectory_axis.get_ylim() == pytest.approx(zoomed_ylim)
    finally:
        plt.close(figure)


def test_dashboard_playback_advances_stops_and_restarts(monkeypatch):
    dashboard = _load_dashboard_module()
    fake_timer = _FakeTimer(interval=750)
    monkeypatch.setattr(
        FigureCanvasBase,
        "new_timer",
        lambda _canvas, *, interval, callbacks=None: fake_timer,
    )
    trajectory = dashboard.PosteriorDashboardTrajectory(
        reference_x=[0.0, 1.0],
        reference_y=[0.0, 1.0],
        observed_x=[0.0, 1.0],
        observed_y=[0.0, 1.0],
    )
    loaded_counts = []

    def load_update(observation_count):
        loaded_counts.append(observation_count)
        return dashboard.PosteriorDashboardUpdate(
            observation_count,
            _dashboard_samples(),
        )

    figure, navigator = dashboard.create_sequential_posterior_dashboard_figure(
        trajectory,
        bayesian_model.BayesianCTRVPriors(),
        load_update,
        maximum_observation_count=2,
        playback_interval_ms=750,
    )

    try:
        assert navigator.is_playing is False
        assert navigator.playback_button.label.get_text() == "Start"
        assert fake_timer.interval == 750

        navigator.toggle_playback(None)
        assert navigator.is_playing is True
        assert navigator.playback_button.label.get_text() == "Pause"
        assert fake_timer.start_count == 1

        fake_timer.fire()
        assert navigator.observation_count == 1
        assert loaded_counts == [1]
        assert navigator.is_playing is True

        fake_timer.fire()
        assert navigator.observation_count == 2
        assert loaded_counts == [1, 2]
        assert navigator.is_playing is False
        assert navigator.playback_button.label.get_text() == "Neu starten"
        assert fake_timer.stop_count == 1

        navigator.toggle_playback(None)
        assert navigator.observation_count == 0
        assert navigator.is_playing is True
        assert navigator.playback_button.label.get_text() == "Pause"
        assert fake_timer.start_count == 2

        navigator.toggle_playback(None)
        assert navigator.is_playing is False
        assert navigator.playback_button.label.get_text() == "Start"
        assert fake_timer.stop_count == 2
    finally:
        plt.close(figure)


@pytest.mark.parametrize("space_key", [" ", "space"])
def test_dashboard_space_key_toggles_and_restarts_playback(monkeypatch, space_key):
    dashboard = _load_dashboard_module()
    fake_timer = _FakeTimer(interval=1_000)
    monkeypatch.setattr(
        FigureCanvasBase,
        "new_timer",
        lambda _canvas, *, interval, callbacks=None: fake_timer,
    )
    trajectory = dashboard.PosteriorDashboardTrajectory(
        reference_x=[0.0, 1.0],
        reference_y=[0.0, 1.0],
        observed_x=[0.0, 1.0],
        observed_y=[0.0, 1.0],
    )

    def load_update(observation_count):
        return dashboard.PosteriorDashboardUpdate(
            observation_count,
            _dashboard_samples(),
        )

    figure, navigator = dashboard.create_sequential_posterior_dashboard_figure(
        trajectory,
        bayesian_model.BayesianCTRVPriors(),
        load_update,
        maximum_observation_count=1,
    )

    def press_space():
        event = KeyEvent("key_press_event", figure.canvas, key=space_key)
        figure.canvas.callbacks.process("key_press_event", event)

    try:
        press_space()
        assert navigator.is_playing is True
        assert navigator.playback_button.label.get_text() == "Pause"
        assert fake_timer.start_count == 1

        fake_timer.fire()
        assert navigator.observation_count == 1
        assert navigator.is_playing is False
        assert navigator.playback_button.label.get_text() == "Neu starten"

        press_space()
        assert navigator.observation_count == 0
        assert navigator.is_playing is True
        assert navigator.playback_button.label.get_text() == "Pause"
        assert fake_timer.start_count == 2

        press_space()
        assert navigator.is_playing is False
        assert navigator.playback_button.label.get_text() == "Start"
        assert fake_timer.stop_count == 2
    finally:
        plt.close(figure)


def test_manual_dashboard_navigation_pauses_playback(monkeypatch):
    dashboard = _load_dashboard_module()
    fake_timer = _FakeTimer(interval=1_000)
    monkeypatch.setattr(
        FigureCanvasBase,
        "new_timer",
        lambda _canvas, *, interval, callbacks=None: fake_timer,
    )
    trajectory = dashboard.PosteriorDashboardTrajectory(
        reference_x=[0.0, 1.0, 2.0],
        reference_y=[0.0, 1.0, 1.5],
        observed_x=[0.0, 1.0, 2.0],
        observed_y=[0.0, 1.0, 1.5],
    )
    loaded_counts = []

    def load_update(observation_count):
        loaded_counts.append(observation_count)
        return dashboard.PosteriorDashboardUpdate(
            observation_count,
            _dashboard_samples(),
        )

    figure, navigator = dashboard.create_sequential_posterior_dashboard_figure(
        trajectory,
        bayesian_model.BayesianCTRVPriors(),
        load_update,
        maximum_observation_count=3,
    )

    try:
        navigator.toggle_playback(None)
        event = KeyEvent("key_press_event", figure.canvas, key="right")
        figure.canvas.callbacks.process("key_press_event", event)

        assert navigator.observation_count == 1
        assert navigator.is_playing is False
        assert navigator.playback_button.label.get_text() == "Start"
        assert fake_timer.stop_count == 1

        navigator.toggle_playback(None)
        navigator.slider.set_val(3)
        navigator.show_selected_observation_count(None)

        assert navigator.observation_count == 3
        assert navigator.is_playing is False
        assert navigator.playback_button.label.get_text() == "Neu starten"
        assert fake_timer.stop_count == 2
        assert loaded_counts == [1, 2, 3]
    finally:
        plt.close(figure)


@pytest.mark.parametrize("playback_interval_ms", [True, 1.5, 0, -1])
def test_dashboard_rejects_invalid_playback_intervals(playback_interval_ms):
    dashboard = _load_dashboard_module()
    trajectory = dashboard.PosteriorDashboardTrajectory(
        reference_x=[0.0, 1.0],
        reference_y=[0.0, 1.0],
        observed_x=[0.0, 1.0],
        observed_y=[0.0, 1.0],
    )

    with pytest.raises(
        ValueError,
        match="playback_interval_ms must be a positive integer",
    ):
        dashboard.create_sequential_posterior_dashboard_figure(
            trajectory,
            bayesian_model.BayesianCTRVPriors(),
            lambda _count: None,
            maximum_observation_count=2,
            playback_interval_ms=playback_interval_ms,
        )


def test_playback_button_click_is_not_handled_as_slider_release(monkeypatch):
    dashboard = _load_dashboard_module()
    fake_timer = _FakeTimer(interval=1_000)
    monkeypatch.setattr(
        FigureCanvasBase,
        "new_timer",
        lambda _canvas, *, interval, callbacks=None: fake_timer,
    )
    trajectory = dashboard.PosteriorDashboardTrajectory(
        reference_x=[0.0, 1.0],
        reference_y=[0.0, 1.0],
        observed_x=[0.0, 1.0],
        observed_y=[0.0, 1.0],
    )
    figure, navigator = dashboard.create_sequential_posterior_dashboard_figure(
        trajectory,
        bayesian_model.BayesianCTRVPriors(),
        lambda observation_count: dashboard.PosteriorDashboardUpdate(
            observation_count,
            _dashboard_samples(),
        ),
        maximum_observation_count=2,
    )

    try:
        x_position, y_position = navigator.playback_button.ax.transAxes.transform(
            (0.5, 0.5)
        )
        press = MouseEvent(
            "button_press_event",
            figure.canvas,
            x_position,
            y_position,
            button=1,
        )
        release = MouseEvent(
            "button_release_event",
            figure.canvas,
            x_position,
            y_position,
            button=1,
        )
        figure.canvas.callbacks.process("button_press_event", press)
        figure.canvas.callbacks.process("button_release_event", release)

        assert navigator.is_playing is True
        assert navigator.playback_button.label.get_text() == "Pause"
        assert fake_timer.start_count == 1
        assert fake_timer.stop_count == 0
    finally:
        plt.close(figure)
