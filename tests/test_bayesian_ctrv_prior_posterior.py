"""Tests for interactive Bayesian CTRV prior-posterior analysis."""

import importlib
import io
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backend_bases import KeyEvent

import bayestraj.models.bayesian_ctrv as bayesian_model


def _load_analysis_module():
    try:
        return importlib.import_module(
            "bayestraj.validation.bayesian_ctrv_prior_posterior"
        )
    except ModuleNotFoundError:
        pytest.fail("The Bayesian CTRV prior-posterior module is missing.")


def test_parameter_specs_map_all_six_priors_to_posterior_variables():
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()

    expected = {
        "current_speed": ("speed_at_origin", None, "m/s"),
        "current_heading": ("heading_at_origin", None, "°"),
        "current_turn_rate": ("turn_rate_at_origin", None, "°/s"),
        "position_observation_noise": (
            "sigma_position_observation",
            None,
            "m",
        ),
        "speed_process_noise": ("sigma_speed_process", None, "m/s"),
        "turn_rate_process_noise": (
            "sigma_turn_rate_process",
            None,
            "°/s",
        ),
    }

    assert analysis.PARAMETER_NAMES == tuple(expected)
    for parameter_name, expected_values in expected.items():
        spec = analysis.build_parameter_spec(parameter_name, priors)
        assert (
            spec.posterior_variable,
            spec.state_index,
            spec.display_unit,
        ) == expected_values


class _FakeFit:
    def __init__(self, variables):
        self._variables = variables

    def stan_variable(self, variable_name, **_kwargs):
        return self._variables[variable_name]


@pytest.mark.parametrize(
    ("parameter_name", "variable_name", "values", "expected"),
    [
        (
            "current_speed",
            "speed_at_origin",
            [1.0, 3.0],
            [1.0, 3.0],
        ),
        (
            "current_heading",
            "heading_at_origin",
            [0.0, np.pi / 2.0],
            [0.0, 90.0],
        ),
        (
            "current_turn_rate",
            "turn_rate_at_origin",
            [0.0, np.pi / 180.0],
            [0.0, 1.0],
        ),
        (
            "position_observation_noise",
            "sigma_position_observation",
            [2.0, 4.0],
            [2.0, 4.0],
        ),
        (
            "speed_process_noise",
            "sigma_speed_process",
            [0.5, 1.5],
            [0.5, 1.5],
        ),
        (
            "turn_rate_process_noise",
            "sigma_turn_rate_process",
            [np.pi / 180.0, np.pi / 90.0],
            [1.0, 2.0],
        ),
    ],
)
def test_extract_posterior_samples_selects_current_state_and_display_units(
    parameter_name,
    variable_name,
    values,
    expected,
):
    analysis = _load_analysis_module()
    spec = analysis.build_parameter_spec(
        parameter_name,
        bayesian_model.BayesianCTRVPriors(),
    )
    fit = _FakeFit({variable_name: np.asarray(values, dtype=float)})

    samples = analysis.extract_posterior_samples(fit, spec)

    assert samples == pytest.approx(expected)


def test_prior_densities_follow_the_configured_distributions_and_units():
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()

    speed = analysis.build_parameter_spec("current_speed", priors)
    heading = analysis.build_parameter_spec("current_heading", priors)
    turn_rate = analysis.build_parameter_spec("current_turn_rate", priors)
    position_noise = analysis.build_parameter_spec(
        "position_observation_noise",
        priors,
    )
    speed_process = analysis.build_parameter_spec("speed_process_noise", priors)
    turn_process = analysis.build_parameter_spec("turn_rate_process_noise", priors)

    assert analysis.evaluate_prior_density(speed, priors, [0.0])[0] == pytest.approx(
        np.sqrt(2.0 / np.pi) / priors.speed_prior_scale
    )
    assert analysis.evaluate_prior_density(
        heading,
        priors,
        [-181.0, -180.0, 0.0, 180.0, 181.0],
    ) == pytest.approx([0.0, 1 / 360.0, 1 / 360.0, 1 / 360.0, 0.0])
    assert analysis.evaluate_prior_density(
        turn_rate,
        priors,
        [0.0],
    )[0] == pytest.approx(
        1.0 / (np.rad2deg(priors.turn_rate_prior_scale) * np.sqrt(2.0 * np.pi))
    )
    assert analysis.evaluate_prior_density(
        position_noise,
        priors,
        [0.0],
    )[0] == pytest.approx(priors.sigma_position_observation_prior_rate)
    assert analysis.evaluate_prior_density(
        speed_process,
        priors,
        [0.0],
    )[0] == pytest.approx(priors.sigma_speed_process_prior_rate)
    assert analysis.evaluate_prior_density(
        turn_process,
        priors,
        [0.0],
    )[0] == pytest.approx(priors.sigma_turn_rate_process_prior_rate * np.pi / 180.0)


def test_terminal_prior_descriptions_are_ascii_safe():
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()

    for parameter_name in analysis.PARAMETER_NAMES:
        spec = analysis.build_parameter_spec(parameter_name, priors)
        analysis._describe_prior(spec, priors).encode("ascii")
        analysis._terminal_display_unit(spec).encode("ascii")


def test_prior_posterior_figure_navigates_in_one_window():
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()
    spec = analysis.build_parameter_spec("current_speed", priors)
    updates = (
        analysis.PosteriorUpdate(3, np.linspace(2.0, 8.0, 100)),
        analysis.PosteriorUpdate(5, np.linspace(3.0, 7.0, 100)),
        analysis.PosteriorUpdate(10, np.linspace(4.0, 6.0, 100)),
        analysis.PosteriorUpdate(20, np.linspace(4.5, 5.5, 100)),
    )

    figure, navigator = analysis.create_prior_posterior_figure(
        spec,
        priors,
        updates,
    )

    try:
        assert len(plt.get_fignums()) == 1
        assert navigator.observation_count == 0
        assert "N = 0" in navigator.axis.get_title()
        assert [line.get_label() for line in navigator.axis.lines] == ["Ausgangs-Prior"]
        assert navigator.slider.val == pytest.approx(0.0)
        assert navigator.slider.valstep == pytest.approx([0.0, 3.0, 5.0, 10.0, 20.0])
        assert not hasattr(navigator, "previous_button")
        assert not hasattr(navigator, "next_button")

        navigator.slider.set_val(3)
        navigator.show_selected_observation_count(None)

        assert navigator.observation_count == 3
        assert "N = 3" in navigator.axis.get_title()
        assert [line.get_label() for line in navigator.axis.lines] == [
            "Ausgangs-Prior",
            "Posterior-Dichte",
        ]

        navigator.slider.set_val(20)
        navigator.show_selected_observation_count(None)

        assert navigator.observation_count == 20

        navigator.slider.set_val(10)
        navigator.show_selected_observation_count(None)
        assert navigator.observation_count == 10
    finally:
        plt.close(figure)


def test_sequential_navigator_loads_jump_point_by_point_and_reuses_updates():
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()
    spec = analysis.build_parameter_spec("current_speed", priors)
    loaded_counts = []

    def load_update(observation_count):
        loaded_counts.append(observation_count)
        return analysis.PosteriorUpdate(
            observation_count,
            np.linspace(1.0, 2.0, 20),
            effective_sample_size=80.0,
            particle_count=100,
            resample_count=2,
        )

    figure, navigator = analysis.create_sequential_prior_posterior_figure(
        spec,
        priors,
        load_update,
        maximum_observation_count=3,
    )

    try:
        assert navigator.observation_count == 0

        navigator.slider.set_val(3)

        assert navigator.observation_count == 0
        assert loaded_counts == []

        navigator.show_selected_observation_count(None)

        assert navigator.observation_count == 3
        assert loaded_counts == [1, 2, 3]
        assert navigator.axis.get_title().splitlines()[-1] == "N = 3"
        assert "RBPF" not in navigator.axis.get_title()
        assert "ESS" not in navigator.axis.get_title()
        assert "Resamplings" not in navigator.axis.get_title()

        navigator.slider.set_val(1)
        navigator.show_selected_observation_count(None)
        assert navigator.observation_count == 1
        assert loaded_counts == [1, 2, 3]

        navigator.slider.set_val(2)
        navigator.show_selected_observation_count(None)
        assert navigator.observation_count == 2
        assert loaded_counts == [1, 2, 3]
    finally:
        plt.close(figure)


def test_arrow_keys_move_slider_one_point_and_respect_boundaries():
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()
    spec = analysis.build_parameter_spec("current_speed", priors)
    updates = (
        analysis.PosteriorUpdate(1, np.linspace(1.0, 2.0, 20)),
        analysis.PosteriorUpdate(2, np.linspace(1.5, 2.5, 20)),
    )
    figure, navigator = analysis.create_prior_posterior_figure(
        spec,
        priors,
        updates,
    )

    def press(key):
        event = KeyEvent("key_press_event", figure.canvas, key=key)
        figure.canvas.callbacks.process("key_press_event", event)

    try:
        press("left")
        assert navigator.observation_count == 0
        assert navigator.slider.val == pytest.approx(0.0)

        press("right")
        assert navigator.observation_count == 1
        assert navigator.slider.val == pytest.approx(1.0)

        press("right")
        press("right")
        assert navigator.observation_count == 2
        assert navigator.slider.val == pytest.approx(2.0)

        press("left")
        assert navigator.observation_count == 1
        assert navigator.slider.val == pytest.approx(1.0)
    finally:
        plt.close(figure)


def test_navigation_slider_leaves_visible_space_below_x_axis_label():
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()
    spec = analysis.build_parameter_spec("current_speed", priors)
    updates = (analysis.PosteriorUpdate(3, np.linspace(2.0, 8.0, 100)),)

    figure, navigator = analysis.create_prior_posterior_figure(
        spec,
        priors,
        updates,
    )

    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        label_bottom = navigator.axis.xaxis.label.get_window_extent(renderer).y0
        slider_top = navigator.slider.ax.get_window_extent(renderer).y1
        gap_points = (label_bottom - slider_top) * 72.0 / figure.dpi

        assert gap_points >= 12.0
    finally:
        plt.close(figure)


def test_current_heading_figure_keeps_prior_boundaries_in_extended_axis():
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()
    spec = analysis.build_parameter_spec("current_heading", priors)
    updates = (analysis.PosteriorUpdate(3, np.linspace(-20.0, 20.0, 100)),)

    figure, navigator = analysis.create_prior_posterior_figure(
        spec,
        priors,
        updates,
    )

    try:
        assert navigator.axis.get_xlim() == pytest.approx((-270.0, 270.0))
        assert navigator.axis.get_xticks().tolist() == [
            -270.0,
            -180.0,
            -90.0,
            0.0,
            90.0,
            180.0,
            270.0,
        ]
    finally:
        plt.close(figure)


def test_legend_can_be_disabled_for_every_navigation_stage():
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()
    spec = analysis.build_parameter_spec("current_speed", priors)
    updates = (analysis.PosteriorUpdate(3, np.linspace(1.0, 2.0, 20)),)

    figure, navigator = analysis.create_prior_posterior_figure(
        spec,
        priors,
        updates,
        show_legend=False,
    )

    try:
        assert navigator.axis.get_legend() is None
        navigator.slider.set_val(3)
        navigator.show_selected_observation_count(None)
        assert navigator.axis.get_legend() is None
    finally:
        plt.close(figure)


def test_heading_summary_uses_circular_center_and_interval():
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()
    spec = analysis.build_parameter_spec("current_heading", priors)
    update = analysis.PosteriorUpdate(
        5,
        np.array([178.0, 179.0, -179.0, -178.0]),
    )

    summary = analysis.summarize_posterior_update(
        spec,
        update,
        credible_interval=0.9,
    )

    assert abs(summary.center) > 170.0
    assert summary.upper - summary.lower < 10.0


def test_rbpf_update_loader_initializes_once_and_processes_one_new_point():
    analysis = _load_analysis_module()
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
    initializations = []
    updated_times = []

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
            assert seed == 42
            return _FakeFit(
                {
                    "heading_at_origin": np.asarray([0.0, np.pi / 2.0]),
                }
            )

    rbpf_config = bayesian_model.SequentialCTRVFilterConfig(
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

    spec, maximum_count, load_update = analysis.create_rbpf_posterior_update_loader(
        trajectory_data,
        parameter_name="current_heading",
        start_index=0,
        position_noise_std_m=0.0,
        position_noise_seed=2026,
        priors=bayesian_model.BayesianCTRVPriors(),
        rbpf_config=rbpf_config,
        rbpf_seed=42,
        initialize_filter=initialize_filter,
    )

    updates = tuple(load_update(count) for count in (1, 2, 3))

    assert spec.parameter_name == "current_heading"
    assert maximum_count == 4
    assert len(initializations) == 1
    assert initializations[0][0] == pytest.approx([0.0])
    assert updated_times == pytest.approx([10.0, 20.0])
    assert tuple(update.observation_count for update in updates) == (1, 2, 3)
    assert updates[-1].samples == pytest.approx([0.0, 90.0])
    assert updates[-1].effective_sample_size == pytest.approx(24.0)
    assert updates[-1].particle_count == 32
    assert updates[-1].resample_count == 1


def test_analysis_runner_loads_one_run_and_shows_one_blocking_window(
    monkeypatch,
):
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()
    rbpf_config = bayesian_model.SequentialCTRVFilterConfig(
        particle_count=100,
        posterior_draw_count=20,
    )
    trajectory_data = pd.DataFrame({"time": ["2026-01-01"], "run_id": [102]})
    spec = analysis.build_parameter_spec("current_speed", priors)
    updates = (
        analysis.PosteriorUpdate(
            1,
            np.linspace(1.0, 2.0, 20),
            effective_sample_size=80.0,
            particle_count=100,
            resample_count=1,
        ),
    )
    figure = object()
    navigator = object()
    captured = {}
    loaded_counts = []
    shown = []
    closed = []

    def fake_read(data_file, *, run_id):
        captured["read"] = (data_file, run_id)
        return trajectory_data

    def fake_create_loader(data, **options):
        captured["loader"] = (data, options)

        def load_update(observation_count):
            loaded_counts.append(observation_count)
            return updates[0]

        return spec, 5, load_update

    def fake_create_figure(
        received_spec,
        received_priors,
        update_loader,
        *,
        maximum_observation_count,
        show_legend,
    ):
        captured["figure"] = (
            received_spec,
            received_priors,
            maximum_observation_count,
            show_legend,
        )
        update_loader(1)
        return figure, navigator

    monkeypatch.setattr(analysis.observations_io, "read_ship_data", fake_read)
    monkeypatch.setattr(
        analysis,
        "create_rbpf_posterior_update_loader",
        fake_create_loader,
    )
    monkeypatch.setattr(
        analysis,
        "create_sequential_prior_posterior_figure",
        fake_create_figure,
    )
    monkeypatch.setattr(analysis.plt, "show", lambda *, block: shown.append(block))
    monkeypatch.setattr(analysis.plt, "close", closed.append)

    output_bytes = io.BytesIO()
    encoded_output = io.TextIOWrapper(output_bytes, encoding="cp950")
    with redirect_stdout(encoded_output):
        result = analysis.run_bayesian_ctrv_prior_posterior_analysis(
            data_file="trajectory.csv",
            run_id=102,
            parameter_name="current_speed",
            start_index=0,
            position_noise_std_m=5.0,
            position_noise_seed=2026,
            priors=priors,
            rbpf_config=rbpf_config,
            rbpf_seed=42,
            credible_interval=0.9,
            show_legend=True,
            show=True,
        )
    encoded_output.flush()
    report = output_bytes.getvalue().decode("cp950")

    assert result == (figure, navigator)
    assert captured["read"] == ("trajectory.csv", 102)
    assert captured["loader"][0].equals(trajectory_data)
    assert captured["loader"][1]["rbpf_config"] is rbpf_config
    assert captured["loader"][1]["rbpf_seed"] == 42
    assert captured["figure"] == (spec, priors, 5, True)
    assert loaded_counts == [1]
    assert shown == [True]
    assert closed == [figure]
    assert "Inferenzmethode       : RBPF" in report
    assert "Beobachtungsstaende" in report
    assert "N=1 bis N=5" in report
    assert "Schieberegler" in report
    assert "Weiter-Klick" not in report
    assert "Posterior nach N=1" in report
    assert "ESS" in report
    assert "Resamplings" in report
    assert "90 %-Intervall" in report
