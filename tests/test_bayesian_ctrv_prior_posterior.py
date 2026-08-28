"""Tests for interactive Bayesian CTRV prior-posterior analysis."""

import importlib
import io
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

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
        "initial_speed": ("speed_state", 0, "m/s"),
        "initial_heading": ("heading_initial", None, "°"),
        "initial_turn_rate": ("turn_rate_state", 0, "°/s"),
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
            "initial_speed",
            "speed_state",
            [[1.0, 9.0], [3.0, 8.0]],
            [1.0, 3.0],
        ),
        (
            "initial_heading",
            "heading_initial",
            [0.0, np.pi / 2.0],
            [0.0, 90.0],
        ),
        (
            "initial_turn_rate",
            "turn_rate_state",
            [[0.0, 1.0], [np.pi / 180.0, 2.0]],
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
def test_extract_posterior_samples_selects_initial_state_and_display_units(
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

    speed = analysis.build_parameter_spec("initial_speed", priors)
    heading = analysis.build_parameter_spec("initial_heading", priors)
    turn_rate = analysis.build_parameter_spec("initial_turn_rate", priors)
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


def test_prior_posterior_figure_navigates_in_one_window():
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()
    spec = analysis.build_parameter_spec("initial_speed", priors)
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
        assert [line.get_label() for line in navigator.axis.lines] == ["Prior-Dichte"]
        assert not navigator.previous_button.active
        assert navigator.next_button.active

        navigator.show_next(None)

        assert navigator.observation_count == 3
        assert "N = 3" in navigator.axis.get_title()
        assert [line.get_label() for line in navigator.axis.lines] == [
            "Prior-Dichte",
            "Posterior-Dichte",
        ]
        assert navigator.previous_button.active
        assert navigator.next_button.active

        for _ in range(3):
            navigator.show_next(None)
        navigator.show_next(None)

        assert navigator.observation_count == 20
        assert not navigator.next_button.active
        assert navigator.previous_button.active

        navigator.show_previous(None)
        assert navigator.observation_count == 10
    finally:
        plt.close(figure)


def test_sequential_navigator_loads_one_new_point_per_click_and_reuses_updates():
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()
    spec = analysis.build_parameter_spec("initial_speed", priors)
    loaded_counts = []

    def load_update(observation_count):
        loaded_counts.append(observation_count)
        return analysis.PosteriorUpdate(
            observation_count,
            np.linspace(1.0, 2.0, 20),
        )

    figure, navigator = analysis.create_sequential_prior_posterior_figure(
        spec,
        priors,
        load_update,
        maximum_observation_count=5,
    )

    try:
        assert navigator.observation_count == 0

        navigator.show_next(None)
        assert navigator.observation_count == 3

        navigator.show_next(None)
        assert navigator.observation_count == 4

        navigator.show_previous(None)
        navigator.show_next(None)
        assert navigator.observation_count == 4
        assert loaded_counts == [3, 4]

        navigator.show_next(None)
        navigator.show_next(None)
        assert navigator.observation_count == 5
        assert loaded_counts == [3, 4, 5]
        assert not navigator.next_button.active
    finally:
        plt.close(figure)


def test_navigation_buttons_leave_visible_space_below_x_axis_label():
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()
    spec = analysis.build_parameter_spec("initial_speed", priors)
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
        button_top = max(
            button.ax.get_window_extent(renderer).y1
            for button in (navigator.previous_button, navigator.next_button)
        )
        gap_points = (label_bottom - button_top) * 72.0 / figure.dpi

        assert gap_points >= 12.0
    finally:
        plt.close(figure)


def test_initial_heading_figure_keeps_prior_boundaries_in_extended_axis():
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()
    spec = analysis.build_parameter_spec("initial_heading", priors)
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
    spec = analysis.build_parameter_spec("initial_speed", priors)
    updates = (analysis.PosteriorUpdate(3, np.linspace(1.0, 2.0, 20)),)

    figure, navigator = analysis.create_prior_posterior_figure(
        spec,
        priors,
        updates,
        show_legend=False,
    )

    try:
        assert navigator.axis.get_legend() is None
        navigator.show_next(None)
        assert navigator.axis.get_legend() is None
    finally:
        plt.close(figure)


def test_heading_summary_uses_circular_center_and_interval():
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()
    spec = analysis.build_parameter_spec("initial_heading", priors)
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


def test_update_loader_reaches_last_available_trajectory_prefix_one_point_at_a_time():
    analysis = _load_analysis_module()
    trajectory_data = pd.DataFrame(
        {
            "time": pd.date_range(
                "2026-01-01",
                periods=7,
                freq="10s",
                tz="UTC",
            ),
            "run_id": 102,
            "gps_latitude": 54.0 + np.arange(7) * 1e-5,
            "gps_longitude": 10.0 + np.arange(7) * 2e-5,
            "gps_speed": np.full(7, 18.0),
        }
    )
    captured = []

    def fake_fit(window, **options):
        captured.append((window, options["position_observations"]))
        draws = np.tile(
            np.linspace(1.0, 2.0, window.observation_count),
            (20, 1),
        )
        return _FakeFit({"speed_state": draws})

    spec, maximum_count, load_update = analysis.create_prior_posterior_update_loader(
        trajectory_data,
        parameter_name="initial_speed",
        start_index=1,
        position_noise_std_m=5.0,
        position_noise_seed=2026,
        priors=bayesian_model.BayesianCTRVPriors(),
        inference_method="vi",
        inference_config={"draws": 20, "require_converged": False},
        inference_seed=42,
        fit_model=fake_fit,
    )

    updates = tuple(load_update(count) for count in range(3, maximum_count + 1))

    assert spec.parameter_name == "initial_speed"
    assert maximum_count == 5
    assert tuple(update.observation_count for update in updates) == (3, 4, 5)
    assert [window.observation_count for window, _ in captured] == [3, 4, 5]
    assert all(
        window.timestamps[0] == trajectory_data["time"].iloc[1]
        for window, _ in captured
    )
    longest_observations = captured[-1][1]
    for window, observations in captured:
        count = window.observation_count
        assert observations.x_meters == pytest.approx(
            longest_observations.x_meters[:count]
        )
        assert observations.y_meters == pytest.approx(
            longest_observations.y_meters[:count]
        )


def test_prefix_updates_refit_consistent_noisy_observation_prefixes(capsys):
    analysis = _load_analysis_module()
    trajectory_data = pd.DataFrame(
        {
            "time": pd.date_range(
                "2026-01-01",
                periods=22,
                freq="10s",
                tz="UTC",
            ),
            "run_id": 102,
            "gps_latitude": 54.0 + np.arange(22) * 1e-5,
            "gps_longitude": 10.0 + np.arange(22) * 2e-5,
            "gps_speed": np.full(22, 18.0),
        }
    )
    captured = []

    def fake_fit(window, **options):
        observations = options["position_observations"]
        captured.append((window, observations, options))
        draws = np.tile(
            np.linspace(1.0, 2.0, window.observation_count),
            (20, 1),
        )
        return _FakeFit({"speed_state": draws})

    priors = bayesian_model.BayesianCTRVPriors()
    spec, updates = analysis.fit_prior_posterior_updates(
        trajectory_data,
        parameter_name="initial_speed",
        observation_counts=(3, 5, 10, 20),
        start_index=1,
        position_noise_std_m=5.0,
        position_noise_seed=2026,
        priors=priors,
        inference_method="vi",
        inference_config={"draws": 20, "require_converged": False},
        inference_seed=42,
        fit_model=fake_fit,
    )

    assert spec.parameter_name == "initial_speed"
    assert tuple(update.observation_count for update in updates) == (3, 5, 10, 20)
    assert [item[0].observation_count for item in captured] == [3, 5, 10, 20]
    assert all(
        item[0].timestamps[0] == trajectory_data["time"].iloc[1] for item in captured
    )
    longest_observations = captured[-1][1]
    for window, observations, options in captured:
        count = window.observation_count
        assert observations.time_seconds == pytest.approx(
            longest_observations.time_seconds[:count]
        )
        assert observations.x_meters == pytest.approx(
            longest_observations.x_meters[:count]
        )
        assert observations.y_meters == pytest.approx(
            longest_observations.y_meters[:count]
        )
        assert options["inference_method"] == "vi"
        assert options["seed"] == 42
        assert options["priors"] is priors
    capsys.readouterr().out.encode("cp950")


def test_analysis_runner_loads_one_run_and_shows_one_blocking_window(
    monkeypatch,
):
    analysis = _load_analysis_module()
    priors = bayesian_model.BayesianCTRVPriors()
    trajectory_data = pd.DataFrame({"time": ["2026-01-01"], "run_id": [102]})
    spec = analysis.build_parameter_spec("initial_speed", priors)
    updates = (analysis.PosteriorUpdate(3, np.linspace(1.0, 2.0, 20)),)
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
        update_loader(3)
        return figure, navigator

    monkeypatch.setattr(analysis.observations_io, "read_ship_data", fake_read)
    monkeypatch.setattr(
        analysis,
        "create_prior_posterior_update_loader",
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
            parameter_name="initial_speed",
            start_index=0,
            position_noise_std_m=5.0,
            position_noise_seed=2026,
            priors=priors,
            inference_method="vi",
            vi_config={"draws": 100},
            mcmc_config={"chains": 4},
            inference_seed=42,
            credible_interval=0.9,
            show_legend=True,
            show=True,
        )
    encoded_output.flush()
    report = output_bytes.getvalue().decode("cp950")

    assert result == (figure, navigator)
    assert captured["read"] == ("trajectory.csv", 102)
    assert captured["loader"][0].equals(trajectory_data)
    assert captured["loader"][1]["inference_config"] == {"draws": 100}
    assert captured["figure"] == (spec, priors, 5, True)
    assert loaded_counts == [3]
    assert shown == [True]
    assert closed == [figure]
    assert "Beobachtungsstaende" in report
    assert "N=3 bis N=5" in report
    assert "Posterior nach N=3" in report
    assert "90 %-Intervall" in report
