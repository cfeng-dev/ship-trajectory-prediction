"""Tests for the combined Bayesian CTRV posterior dashboard."""

import importlib
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backend_bases import KeyEvent

import bayestraj.inference.ctrv_rbpf as rbpf
import bayestraj.models.bayesian_ctrv as bayesian_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _dashboard_samples(offset=0.0):
    return {
        "current_speed": np.linspace(1.0 + offset, 2.0 + offset, 20),
        "current_heading": np.linspace(-10.0 + offset, 10.0 + offset, 20),
        "current_turn_rate": np.linspace(-1.0 + offset, 1.0 + offset, 20),
        "position_observation_noise": np.linspace(1.0 + offset, 3.0 + offset, 20),
        "speed_process_noise": np.linspace(0.1 + offset, 0.3 + offset, 20),
        "turn_rate_process_noise": np.linspace(0.2 + offset, 0.4 + offset, 20),
    }


def test_dashboard_loader_updates_one_filter_and_extracts_all_parameters():
    dashboard = _load_dashboard_module()
    trajectory_data = pd.DataFrame(
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
    variables = {
        "speed_at_origin": np.asarray([2.0, 3.0]),
        "heading_at_origin": np.asarray([0.0, np.pi / 2.0]),
        "turn_rate_at_origin": np.asarray([0.0, np.pi / 180.0]),
        "sigma_position_observation": np.asarray([1.0, 2.0]),
        "sigma_speed_process": np.asarray([0.1, 0.2]),
        "sigma_turn_rate_process": np.asarray([np.pi / 180.0, np.pi / 90.0]),
    }
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


def test_dashboard_script_runs_the_shared_analysis_without_showing():
    script = _load_dashboard_script()
    calls = []
    sentinel = object()

    def fake_run(**options):
        calls.append(options)
        return sentinel

    script.dashboard.run_bayesian_ctrv_posterior_dashboard = fake_run

    result = script.main(["--no-show"])

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0]["data_file"] == script.DATA_FILE
    assert calls[0]["run_id"] == 102
    assert calls[0]["start_index"] == 0
    assert calls[0]["position_noise_std_m"] == pytest.approx(5.0)
    assert calls[0]["position_noise_seed"] == 2026
    assert calls[0]["priors"] is script.PRIORS
    assert calls[0]["rbpf_config"] is script.RBPF_CONFIG
    assert calls[0]["rbpf_seed"] == 42
    assert calls[0]["show_legend"] is True
    assert calls[0]["show"] is False
