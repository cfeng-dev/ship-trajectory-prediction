"""GUI settings are validated without creating a window or starting inference."""

import argparse
import importlib

import numpy as np
import pytest
from trajectory_predictor import cli, settings

from bayestraj.inference.ctrv_sequential_vi import SequentialVIConfig


@pytest.mark.parametrize(
    ("package", "description"),
    [("trajectory_predictor", "trajectory predictor"), ("imm_predictor", "IMM")],
)
def test_command_help_does_not_open_a_window(capsys, package, description):
    main = importlib.import_module(f"{package}.cli").main
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert description in capsys.readouterr().out


@pytest.mark.parametrize("package", ["trajectory_predictor", "imm_predictor"])
@pytest.mark.parametrize("fail", [False, True])
def test_desktop_startup_and_errors_do_not_create_logs(
    tmp_path, monkeypatch, caplog, package, fail
):
    import logging
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    module = importlib.import_module(f"{package}.cli")
    logger = logging.getLogger(package)
    previous_handlers = list(logger.handlers)
    previous_level = logger.level
    root = Mock()
    if fail:
        root.mainloop.side_effect = ValueError("desktop error sentinel")
    app = Mock()
    importer = (
        "_import_trajectory_predictor"
        if package == "trajectory_predictor"
        else "_import_predictor_app"
    )
    monkeypatch.setattr(
        module, "__file__", str(tmp_path / "tools" / package / "cli.py")
    )
    monkeypatch.setattr(module, importer, lambda: lambda root: app)
    monkeypatch.setitem(
        sys.modules, "tkinter", SimpleNamespace(Tk=lambda: root, TclError=RuntimeError)
    )
    try:
        if fail:
            with pytest.raises(ValueError, match="desktop error sentinel"):
                module.main([])
            assert "ValueError: desktop error sentinel" in caplog.text
        else:
            module.main([])
        assert list(tmp_path.iterdir()) == []
        assert logger.handlers == previous_handlers
    finally:
        for handler in list(logger.handlers):
            if handler not in previous_handlers:
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(previous_level)


def test_missing_shared_bayestraj_package_has_a_clear_startup_error(
    monkeypatch, capsys
):
    def missing_predictor_gui():
        raise ModuleNotFoundError("No module named 'bayestraj'", name="bayestraj")

    monkeypatch.setattr(cli, "_import_trajectory_predictor", missing_predictor_gui)

    with pytest.raises(SystemExit) as result:
        cli._load_trajectory_predictor(argparse.ArgumentParser())

    assert result.value.code == 2
    assert "requires the shared bayestraj package" in capsys.readouterr().err


def test_imm_posterior_snapshot_uses_the_loaded_output_fusion(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from imm_predictor import player

    from bayestraj.inference import imm_adapter

    mean = np.array([10.0, 20.0, 0.3, 0.02, 4.0])
    covariance = np.diag([1.0, 2.0, 0.04, 0.01, 0.25])
    state = SimpleNamespace(mode_probabilities=np.array([0.6, 0.3, 0.1]))

    class Tracker:
        def __init__(self):
            self._state = state

    def fuse(actual_state, weights):
        assert actual_state is state
        np.testing.assert_array_equal(weights, [0.6, 0.3, 0.1])
        return mean, covariance

    path = tmp_path / "imm.py"
    path.touch()
    monkeypatch.setattr(player, "IMM_PYTHON", path)
    monkeypatch.setattr(
        imm_adapter.runpy,
        "run_path",
        lambda _path: {"BR24IMMTracker": Tracker, "_mix_to_ctrv": fuse},
    )
    tracker = player._load_optional_imm_tracker()()
    posterior = tracker.posterior_state()
    mean[:] = 99.0
    covariance[:] = 99.0

    np.testing.assert_array_equal(posterior.mean, [10.0, 20.0, 0.3, 0.02, 4.0])
    np.testing.assert_array_equal(
        np.diag(posterior.covariance), [1, 2, 0.04, 0.01, 0.25]
    )
    assert not posterior.mean.flags.writeable
    assert not posterior.covariance.flags.writeable
    np.testing.assert_array_equal(state.mode_probabilities, [0.6, 0.3, 0.1])


@pytest.mark.parametrize("sensor_position", [[0, 0], [100, -50]])
@pytest.mark.parametrize("future_times", [[1.0, 2.0], [0.4, 2.7, 6.0]])
def test_imm_display_adapter_preserves_local_filter_forecasts(
    sensor_position, future_times
):
    import runpy
    from copy import deepcopy

    from imm_predictor import player

    if not player.IMM_PYTHON.is_file():
        pytest.skip("Optional local IMM implementation is unavailable")
    module = runpy.run_path(str(player.IMM_PYTHON))
    state = module["_IMMState"](
        cv_state=np.array([10.0, 3.0, 20.0, 4.0]),
        cv_covariance=np.zeros((4, 4)),
        ctrv_state=np.array([10.0, 20.0, np.arctan2(4, 3), 0.2, 5.0]),
        ctrv_covariance=np.zeros((5, 5)),
        ca_state=np.array([10.0, 3.0, 0.0, 20.0, 4.0, 0.0]),
        ca_covariance=np.zeros((6, 6)),
        mode_probabilities=np.array([0.5, 0.3, 0.2]),
    )
    arguments = dict(
        _last_observation_time_seconds=0.0,
        _sensor_position=np.asarray(sensor_position, dtype=float),
    )
    original = module["BR24IMMTracker"](_state=deepcopy(state), **arguments)
    display = player._load_optional_imm_tracker()(_state=deepcopy(state), **arguments)
    posterior = display.posterior_state()
    mixture = display.posterior_mixture()
    np.testing.assert_array_equal(mixture.weights, [0.5, 0.3, 0.2])
    np.testing.assert_allclose(
        [component.mean[3] for component in mixture.components], [0.0, 0.2, 0.0]
    )
    np.testing.assert_allclose(
        [component.mean[4] for component in mixture.components], [5.0, 5.0, 5.0]
    )
    np.testing.assert_allclose(
        posterior.mean, [10.0, 20.0, np.arctan2(4, 3), 0.06, 5.0]
    )
    assert posterior.covariance[3, 3] == pytest.approx(0.0084)
    expected, actual = original.forecast(future_times), display.forecast(future_times)
    np.testing.assert_array_equal(actual.position_means, expected.position_means)
    np.testing.assert_array_equal(
        actual.position_covariances, expected.position_covariances
    )
    np.testing.assert_array_equal(
        actual.mode_probabilities, expected.mode_probabilities
    )
    future_mixtures = display.forecast_mixtures(future_times)
    assert len(future_mixtures) == len(future_times)
    for index, mixture_at_horizon in enumerate(future_mixtures):
        weights = mixture_at_horizon.weights
        positions = np.array(
            [component.mean[:2] for component in mixture_at_horizon.components]
        )
        mean = weights @ positions
        covariance = sum(
            weight
            * (
                component.covariance[:2, :2]
                + np.outer(position - mean, position - mean)
            )
            for weight, component, position in zip(
                weights, mixture_at_horizon.components, positions, strict=True
            )
        )
        np.testing.assert_array_equal(weights, expected.mode_probabilities[index])
        np.testing.assert_allclose(mean, expected.position_means[index])
        np.testing.assert_allclose(covariance, expected.position_covariances[index])
    after_read = display.forecast(future_times)
    np.testing.assert_array_equal(after_read.position_means, expected.position_means)
    np.testing.assert_array_equal(
        after_read.position_covariances, expected.position_covariances
    )
    np.testing.assert_array_equal(display.posterior_state().mean, posterior.mean)
    np.testing.assert_array_equal(
        display.posterior_state().covariance, posterior.covariance
    )
    np.testing.assert_array_equal(display.posterior_mixture().weights, mixture.weights)
    assert not mixture.weights.flags.writeable
    for before, after in zip(
        mixture.components, display.posterior_mixture().components, strict=True
    ):
        np.testing.assert_array_equal(after.mean, before.mean)
        np.testing.assert_array_equal(after.covariance, before.covariance)


def test_imm_cached_posterior_is_not_replaced_by_later_cycles():
    from types import SimpleNamespace

    from imm_predictor.player import (
        IMMPlaybackStore,
        IMMPlayerConfig,
        IMMStateMixture,
        IMMStatePosterior,
    )

    mean = np.array([10.0, 20.0, 0.3, 0.02, 4.0])
    weights = np.array([0.5, 0.3, 0.2])
    store = IMMPlaybackStore.__new__(IMMPlaybackStore)
    store.config = IMMPlayerConfig(prediction_count=0)
    store.maximum_observation_count = 2
    store.times = np.array([0.0, 1.0])
    store.cycles = [SimpleNamespace(cycle_counter=1), SimpleNamespace(cycle_counter=2)]
    store._tracker = SimpleNamespace(
        last_observation_time_seconds=0.0,
        posterior_state=lambda: IMMStatePosterior(mean, np.eye(5)),
        posterior_mixture=lambda: IMMStateMixture(
            weights, tuple(IMMStatePosterior(mean, np.eye(5)) for _ in range(3))
        ),
    )
    store._processed_cycle_count = 1
    store._tracked_positions = [[10.0, 20.0]]
    store._measurement_covariances = [None]
    store._detection_ids = [1]
    store._nis = [None]
    store._stages = {}
    store._display_states = {}
    store.reference = None
    store._capture_stage(update_time_seconds=0.01)

    mean[4] = 9.0
    weights[:] = [0.2, 0.7, 0.1]
    store._processed_cycle_count = 2
    store._tracked_positions.append([11.0, 21.0])
    store._capture_stage(update_time_seconds=0.01)

    assert store.stage_for(1).posterior.mean[4] == 4.0
    assert store.stage_for(2).posterior.mean[4] == 9.0
    np.testing.assert_array_equal(
        store.stage_for(1).posterior_mixture.weights, [0.5, 0.3, 0.2]
    )
    np.testing.assert_array_equal(
        store.stage_for(2).posterior_mixture.weights, [0.2, 0.7, 0.1]
    )
    assert store.stage_for(1).posterior_mixture.components[0].mean[4] == 4.0
    assert store.stage_for(2).posterior_mixture.components[0].mean[4] == 9.0
    first_display = store.display_state_for(1)
    assert first_display.posterior_medians.speed_mps == 4.0
    assert store.display_state_for(2).posterior_medians.speed_mps == 9.0
    assert store.display_state_for(1) is first_display


@pytest.mark.parametrize(
    "weights",
    [[0, 0, 0], [0.2, 0.2, 0.2], [-0.1, 0.5, 0.6], [np.nan, 0, 1], [0.5, 0.5]],
)
def test_imm_mixture_rejects_invalid_model_probabilities(weights):
    from imm_predictor.player import IMMStateMixture, IMMStatePosterior

    component = IMMStatePosterior(np.zeros(5), np.eye(5))
    with pytest.raises(ValueError):
        IMMStateMixture(weights, (component, component, component))


@pytest.mark.parametrize("has_mixture_reader", [False, True])
@pytest.mark.parametrize("forecast_enabled", [False, True])
def test_imm_forecast_medians_are_cached_without_visualization_samples(
    has_mixture_reader, forecast_enabled
):
    from types import SimpleNamespace

    from imm_predictor.player import (
        IMMPlaybackStore,
        IMMPlayerConfig,
        IMMStateMixture,
        IMMStatePosterior,
    )

    means = np.array([[2.0, 4.0], [3.0, 5.0]])
    forecast = SimpleNamespace(time_seconds=np.array([1.0, 2.0]), position_means=means)
    mixtures = tuple(
        IMMStateMixture(
            [0.75, 0.25, 0],
            tuple(
                IMMStatePosterior([x + step, y + step, 0, 0, 1], np.zeros((5, 5)))
                for x, y in [(0, 0), (8, 16), (99, 99)]
            ),
        )
        for step in (0, 1)
    )

    def read_mixtures(times):
        assert forecast_enabled
        np.testing.assert_array_equal(times, [1, 2])
        return mixtures

    tracker = SimpleNamespace(
        last_observation_time_seconds=0.0, forecast=lambda times: forecast
    )
    if has_mixture_reader:
        tracker.forecast_mixtures = read_mixtures
    store = IMMPlaybackStore.__new__(IMMPlaybackStore)
    store.config = IMMPlayerConfig(
        prediction_count=2 if forecast_enabled else 0, prediction_sample_count=0
    )
    store.maximum_observation_count = 1
    store.times = np.array([0.0, 1.0, 2.0])
    store.cycles = [SimpleNamespace(cycle_counter=1)]
    store._tracker = tracker
    store._processed_cycle_count = 1
    store._tracked_positions = [[0.0, 0.0]]
    store._detection_ids = [1]
    store._measurement_covariances = [None]
    store._nis = [None]
    store._stages = {}
    store._capture_stage(update_time_seconds=0.01)
    stage = store.stage_for(1)
    assert stage.sample_positions is None
    if forecast_enabled:
        np.testing.assert_array_equal(
            stage.forecast_median_positions,
            [[0, 0], [1, 1]] if has_mixture_reader else means,
        )
        assert not stage.forecast_median_positions.flags.writeable
        means[:] = 99
        assert stage.forecast_median_positions[0, 0] != 99
    else:
        assert stage.forecast is None
        assert stage.forecast_median_positions is None


@pytest.fixture
def imm_forecast_case():
    from types import SimpleNamespace

    from imm_predictor.player import IMMPlayerStage

    stage = IMMPlayerStage(
        observation_count=1,
        cycle_counter=1,
        observation_time_seconds=0.0,
        detection_id=1,
        nis=None,
        measurement_covariance=np.eye(2),
        update_time_seconds=0.01,
        tracked_times_seconds=np.array([0.0]),
        tracked_positions=np.array([[999.0, 999.0]]),
        forecast=SimpleNamespace(
            time_seconds=np.array([1.0, 2.0]),
            position_means=np.array([[10.0, 3.0], [4.0, 15.0]]),
            position_covariances=np.repeat((4 * np.eye(2))[None], 2, axis=0),
            mode_probabilities=np.tile([0.7, 0.2, 0.1], (2, 1)),
        ),
        forecast_median_positions=np.array([[2.0, 0.0], [0.0, 5.0]]),
    )
    reference = SimpleNamespace(
        x_meters=np.array([999.0, 2.0, 4.0]),
        y_meters=np.array([999.0, 3.0, 5.0]),
    )
    return stage, reference


@pytest.mark.parametrize("displayed_sample_count", [0, 1, 20])
def test_imm_point_metrics_use_medians_without_changing_distribution_scores(
    imm_forecast_case, displayed_sample_count
):
    from dataclasses import replace

    from imm_predictor.player import _forecast_metrics

    stage, reference = imm_forecast_case
    stage = replace(
        stage, sample_positions=np.full((displayed_sample_count, 3, 2), 1e6)
    )
    result = _forecast_metrics(stage, reference, coverage_stages=(stage,))

    # Reference indices 1 and 2 give median errors 3 m and 4 m, not mean errors 8 m and 10 m.
    assert result.ade_m == 3.5
    assert result.fde_m == 4.0
    assert result.rmse_m == pytest.approx(np.sqrt(12.5))
    original_gaussian_stage = replace(
        stage, forecast_median_positions=stage.forecast.position_means
    )
    gaussian_scores = _forecast_metrics(
        original_gaussian_stage, reference, coverage_stages=(original_gaussian_stage,)
    )
    assert result.energy_score_m == gaussian_scores.energy_score_m
    assert result.predictive_log_density == gaussian_scores.predictive_log_density
    assert (
        result.joint_coverage_50_count == gaussian_scores.joint_coverage_50_count == 0
    )
    assert (
        result.joint_coverage_90_count == gaussian_scores.joint_coverage_90_count == 0
    )
    assert result.joint_coverage_total == gaussian_scores.joint_coverage_total == 2


def test_imm_point_metrics_keep_fused_gaussian_mean_equals_median_fallback(
    imm_forecast_case,
):
    from dataclasses import replace

    from imm_predictor.player import _forecast_metrics

    stage, reference = imm_forecast_case
    stage = replace(stage, forecast_median_positions=None)
    result = _forecast_metrics(stage, reference, coverage_stages=(stage,))
    assert result.ade_m == 9.0
    assert result.fde_m == 10.0
    assert result.rmse_m == pytest.approx(np.sqrt(82.0))


@pytest.mark.parametrize("missing", ["forecast", "reference"])
def test_imm_median_metrics_stay_unavailable_without_forecast_or_reference(
    imm_forecast_case, missing
):
    from dataclasses import replace

    from imm_predictor.player import IMMForecastMetrics, _forecast_metrics

    stage, reference = imm_forecast_case
    if missing == "forecast":
        stage = replace(stage, forecast=None, forecast_median_positions=None)
    else:
        reference = None
    assert _forecast_metrics(
        stage, reference, coverage_stages=(stage,)
    ) == IMMForecastMetrics(None, None, None, None, None, 0, 0, 0)


@pytest.mark.parametrize(
    ("means", "sigmas", "weights", "expected"),
    [
        ([-10, 5, 1000], [0, 2, 100], [0.25, 0.75, 0], 4.138545401409085),
        ([0, 8, -3], [0, 1, 1], [0.6, 0.25, 0.15], 0),
        ([0, -2, 2], [0, 1, 1], [0.3, 0.35, 0.35], 0),
        ([1, 5, 9], [0, 0, 0], [0.2, 0.7, 0.1], 5),
        ([1, 5, 9], [0, 0, 0], [0.5, 0.3, 0.2], 1),
        ([0, 20, 1000], [1, 1, 1], [0.5, 0.5, 0], 10),
        ([0, 600, 1000], [1, 2, 1], [0.5, 0.5, 0], 200),
        ([7.5, -1000, 1000], [2, 0, 0], [1, 0, 0], 7.5),
    ],
)
def test_imm_medians_use_the_full_marginal_including_point_masses(
    means, sigmas, weights, expected
):
    from imm_predictor.player import IMMStateMixture, IMMStatePosterior

    components = tuple(
        IMMStatePosterior([0, 0, 0, 0, mean], np.diag([0, 0, 0, 0, sigma**2]))
        for mean, sigma in zip(means, sigmas, strict=True)
    )
    mixture = IMMStateMixture(weights, components)
    assert mixture.marginal_median(4) == pytest.approx(expected, abs=1e-10)
    np.testing.assert_array_equal(mixture.weights, weights)


def test_imm_heading_median_matches_the_local_branch_used_for_plotting():
    from imm_predictor.player import IMMStateMixture, IMMStatePosterior

    components = tuple(
        IMMStatePosterior(
            [0, 0, np.deg2rad(heading), 0, 0],
            np.diag([0, 0, np.deg2rad(1) ** 2, 0, 0]),
        )
        for heading in (179, -179, 0)
    )
    mixture = IMMStateMixture([0.5, 0.5, 0], components)
    assert abs(np.rad2deg(mixture.marginal_median(2))) == pytest.approx(180)


@pytest.mark.parametrize("available", [False, True])
def test_imm_posterior_medians_use_fused_gaussian_only_as_backend_fallback(available):
    from types import SimpleNamespace

    from imm_predictor.player import IMMStatePosterior, _posterior_medians

    stage = SimpleNamespace(
        posterior_mixture=None,
        posterior=IMMStatePosterior(
            [1, 2, np.deg2rad(30), np.deg2rad(-2), 4], np.eye(5)
        )
        if available
        else None,
    )
    result = _posterior_medians(stage)
    if available:
        assert result.speed_mps == 4
        assert result.heading_degrees == pytest.approx(30)
        assert result.turn_rate_degrees_per_second == pytest.approx(-2)
    else:
        assert result is None


@pytest.mark.parametrize("variance", [-1.0, float("nan")])
def test_imm_posterior_rejects_invalid_covariance(variance):
    from imm_predictor.player import IMMStatePosterior

    covariance = np.eye(5)
    covariance[4, 4] = variance
    with pytest.raises(ValueError):
        IMMStatePosterior(np.zeros(5), covariance)


@pytest.fixture
def form(tmp_path):
    values = settings.default_form_values()
    data_file = tmp_path / "trajectory.csv"
    data_file.touch()
    values["data"]["data_file"] = str(data_file)
    return values


def test_both_predictors_default_to_median_and_keep_mean_plot_only(form):
    from imm_predictor.player import IMMPlayerConfig

    result = settings.parse_settings(form)
    assert result.show_median_forecast is True
    assert result.show_mean_forecast is False
    assert IMMPlayerConfig().show_forecast_median is True
    assert IMMPlayerConfig().show_forecast_mean is False
    form["data"]["show_mean_forecast"] = True
    result = settings.parse_settings(form)
    assert result.show_mean_forecast is True
    assert not hasattr(result.analysis.experiment, "show_mean_forecast")
    form["data"]["show_mean_forecast"] = "yes"
    with pytest.raises(ValueError, match="forecast"):
        settings.parse_settings(form)


@pytest.mark.parametrize("method", ["smc", "sequential_vi", "vi", "mcmc"])
def test_settings_preserve_selected_method_and_units(form, method):
    form["data"]["inference_method"] = method
    form["priors"]["sigma_turn_rate_process_prior_upper_deg_s"] = "7.5"
    result = settings.parse_settings(form)
    assert result.analysis.experiment.inference_method == method
    assert result.analysis.experiment.maximum_observation_count is None
    assert result.analysis.experiment.observation_interval_seconds is None
    assert result.analysis.priors.sigma_turn_rate_process_prior_upper_deg_s == 7.5
    assert result.playback_interval_ms > 0


def test_default_form_values_use_a_100m_follow_ship_view():
    assert settings.default_form_values()["data"]["follow_ship_view_span_m"] == "100.0"


def test_sample_connections_are_off_by_default_and_display_only(form):
    assert form["data"]["connect_sample_trajectories"] is False
    disconnected = settings.parse_settings(form)
    form["data"]["connect_sample_trajectories"] = True
    connected = settings.parse_settings(form)
    assert connected.connect_sample_trajectories is True
    assert connected.analysis == disconnected.analysis


def test_inference_method_display_labels_normalize_to_internal_values():
    assert "rbpf" not in settings.METHOD_DISPLAY_LABELS
    assert settings.METHOD_DISPLAY_LABELS["vi"] == "VI – Variational inference"
    assert settings.METHOD_DISPLAY_LABELS["mcmc"] == "MCMC – Markov chain Monte Carlo"
    assert (
        settings.METHOD_DISPLAY_LABELS["sequential_vi"]
        == "Sequential VI - online variational inference"
    )
    for method, label in settings.METHOD_DISPLAY_LABELS.items():
        assert settings.normalize_inference_method(label) == method
        assert settings.normalize_inference_method(method) == method


def test_gui_inference_method_options_exclude_rbpf():
    assert "rbpf" not in settings.METHODS
    assert "rbpf" not in settings.default_form_values()
    assert settings.METHOD_DISPLAY_OPTIONS == (
        settings.METHOD_DISPLAY_LABELS["smc"],
        settings.METHOD_DISPLAY_LABELS["sequential_vi"],
        settings.METHOD_DISPLAY_LABELS["vi"],
        settings.METHOD_DISPLAY_LABELS["mcmc"],
    )


def test_removed_rbpf_method_is_rejected(form):
    form["data"]["inference_method"] = "rbpf"

    with pytest.raises(ValueError, match="Inference method must be"):
        settings.parse_settings(form)


def test_trajectory_predictor_defaults_to_smc():
    assert settings.default_form_values()["data"]["inference_method"] == "smc"


def test_trajectory_predictor_defaults_to_prediction_funnel():
    assert settings.default_form_values()["data"]["prediction_region_style"] == "funnel"


def test_prediction_region_style_options_prioritize_prediction_funnel():
    assert settings.PREDICTION_REGION_STYLE_DISPLAY_OPTIONS == (
        "Horizon density region",
        "Per-step density regions",
        "Covariance ellipses",
        "Horizon density contours",
        "Per-step density contours",
        "No uncertainty region",
    )


def test_trajectory_predictor_defaults_to_no_added_position_noise():
    assert settings.default_form_values()["data"]["position_noise_std_m"] == "0.0"


def test_settings_do_not_expose_manual_br24_start_selection(form):
    assert "initial_cycle_counter" not in form["data"]
    assert "initial_detection_id" not in form["data"]


def test_settings_do_not_expose_automatic_track_recovery_controls(form):
    defaults = settings.default_form_values()

    for method in ("smc", "sequential_vi"):
        assert "predictive_log_density_threshold" not in defaults[method]
    assert "br24_recovery_nis_gate" not in defaults["smc"]

    result = settings.parse_settings(form)

    assert not hasattr(result.analysis.smc_config, "predictive_log_density_threshold")


def test_trajectory_predictor_defaults_to_br24_tracking_input_with_reference():
    """The GUI starts with BR24 tracking input that includes evaluation labels."""
    data = settings.default_form_values()["data"]

    assert (
        data["data_file"]
        .replace("\\", "/")
        .endswith("data/processed/BR24Dataset20131107_with_reference.csv")
    )
    assert "run_id" not in data


def test_sequential_vi_defaults_and_parsing_preserve_full_config(form):
    defaults = settings.default_form_values()["sequential_vi"]

    assert defaults["n_bootstrap"] == "3"
    assert settings.create_default_sequential_vi_config().n_bootstrap == 10
    assert set(defaults) == set(SequentialVIConfig.__dataclass_fields__)
    form["data"]["inference_method"] = "sequential_vi"
    form["sequential_vi"]["n_bootstrap"] = "6"
    first = settings.parse_settings(form).analysis.sequential_vi_config
    second = settings.parse_settings(form).analysis.sequential_vi_config

    assert isinstance(first, SequentialVIConfig)
    assert first.n_bootstrap == 6
    assert first is not second


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n_bootstrap", "2", "Bootstrap observations"),
        ("algorithm", "meanfield", "fullrank"),
        ("draws", "1", "Posterior draws"),
    ],
)
def test_sequential_vi_settings_report_clear_invalid_fields(
    form,
    field,
    value,
    message,
):
    form["data"]["inference_method"] = "sequential_vi"
    form["sequential_vi"][field] = value

    with pytest.raises(ValueError, match=message):
        settings.parse_settings(form)


def test_trajectory_predictor_defaults_to_two_forecast_steps():
    assert settings.default_form_values()["data"]["prediction_count"] == "2"


def test_process_noise_labels_use_compact_units():
    assert (
        settings.LABELS["sigma_speed_process_prior_upper_mps"]
        == "Speed process: threshold [m/s]"
    )
    assert (
        settings.LABELS["sigma_turn_rate_process_prior_upper_deg_s"]
        == "Turn-rate process: threshold [°/s]"
    )


def test_gui_prior_defaults_match_shared_bayesian_ctrv_configuration():
    assert settings.default_form_values()["priors"] == {
        "speed_prior_upper_mps": "10.0",
        "speed_prior_tail_probability": "0.05",
        "turn_rate_prior_abs_rate_deg_s": "10.0",
        "turn_rate_prior_tail_probability": "0.05",
        "sigma_position_observation_prior_upper_m": "20.0",
        "sigma_position_observation_prior_tail_probability": "0.05",
        "sigma_position_observation_floor_m": "5.0",
        "sigma_speed_process_prior_upper_mps": "2.0",
        "sigma_speed_process_prior_tail_probability": "0.05",
        "sigma_turn_rate_process_prior_upper_deg_s": "2.0",
        "sigma_turn_rate_process_prior_tail_probability": "0.05",
    }


@pytest.mark.parametrize(
    ("group", "field", "value"),
    [
        ("data", "start_index", "-1"),
        ("data", "maximum_observation_count", "2"),
        ("data", "position_noise_std_m", "nan"),
        ("data", "playback_interval_seconds", "0"),
        ("priors", "speed_prior_tail_probability", "1"),
        ("smc", "posterior_draw_count", "1"),
        ("smc", "resample_ess_fraction", "1.1"),
        ("smc", "rejuvenation_scale", "inf"),
    ],
)
def test_invalid_settings_are_rejected(form, group, field, value):
    if group in settings.METHODS:
        form["data"]["inference_method"] = group
    form[group][field] = value
    with pytest.raises(ValueError):
        settings.parse_settings(form)


def test_inactive_method_fields_do_not_block_analysis(form):
    form["vi"]["iter"] = "unfinished edit"
    settings.parse_settings(form)
    form["data"]["inference_method"] = "vi"
    with pytest.raises(ValueError, match="iter"):
        settings.parse_settings(form)


def test_forecast_settings_reach_analysis_and_allow_hiding_samples(form):
    form["data"]["prediction_count"] = "12"
    form["data"]["prediction_sample_count"] = "0"
    result = settings.parse_settings(form)
    assert result.analysis.experiment.prediction_count == 12
    assert result.analysis.experiment.prediction_sample_count == 0
    form["data"]["prediction_count"] = "-1"
    with pytest.raises(ValueError):
        settings.parse_settings(form)


def test_coordinate_display_mode_falls_back_to_metres_when_gps_is_requested(form):
    form["data"]["coordinate_display_mode"] = "gps"

    with pytest.warns(UserWarning, match="Invalid plot coordinate mode"):
        result = settings.parse_settings(form)

    assert settings.COORDINATE_DISPLAY_MODES == ("m", "km")
    assert result.coordinate_display_mode == "m"
    assert not hasattr(result.analysis.experiment, "coordinate_display_mode")


def test_follow_ship_view_span_is_a_presentation_setting(form):
    form["data"]["follow_ship_view_span_m"] = "250"

    result = settings.parse_settings(form)

    assert result.follow_ship_view_span_m == 250.0
    assert not hasattr(result.analysis.experiment, "follow_ship_view_span_m")


def test_batch_options_are_forwarded_without_hidden_overrides(form):
    form["data"]["inference_method"] = "vi"
    form["vi"]["algorithm"] = "fullrank"
    form["vi"]["grad_samples"] = "7"
    form["vi"]["require_converged"] = True
    result = settings.parse_settings(form)
    assert result.analysis.vi_config["algorithm"] == "fullrank"
    assert result.analysis.vi_config["grad_samples"] == 7
    assert result.analysis.vi_config["require_converged"] is True


def test_defaults_are_independent_and_missing_file_is_rejected(form):
    values = settings.default_form_values()
    values["smc"]["particle_count"] = "2"
    assert settings.default_form_values()["smc"]["particle_count"] == "4000"
    form["data"]["data_file"] += ".missing"
    with pytest.raises(ValueError, match="CSV"):
        settings.parse_settings(form)


def test_dialog_validation_is_independent_of_csv_and_other_sections():
    values = settings.default_form_values()
    values["priors"]["speed_prior_upper_mps"] = "25.5"
    result = settings.validate_dialog_values("priors", values["priors"])
    assert result["speed_prior_upper_mps"] == 25.5
    assert values["priors"]["speed_prior_upper_mps"] == "25.5"


@pytest.mark.parametrize("method", ["vi", "mcmc", "smc", "sequential_vi"])
def test_inference_dialog_validates_the_method_being_edited(method):
    values = settings.default_form_values()[method]
    invalid_field = {
        "vi": "draws",
        "mcmc": "chains",
        "sequential_vi": "draws",
    }.get(method, "particle_count")
    values[invalid_field] = "0"
    with pytest.raises(ValueError):
        settings.validate_dialog_values(method, values)


def test_data_options_dialog_preserves_seconds_and_validates_intervals():
    values = {
        "start_index": "2",
        "observation_interval_seconds": "10",
        "maximum_observation_count": "",
        "position_noise_std_m": "1.5",
        "position_noise_seed": "2026",
        "prediction_count": "8",
        "prediction_sample_count": "0",
        "inference_seed": "456",
        "playback_interval_seconds": "1.5",
    }
    result = settings.validate_dialog_values("data", values)
    assert result == {
        "start_index": 2,
        "observation_interval_seconds": 10.0,
        "maximum_observation_count": "",
        "position_noise_std_m": 1.5,
        "position_noise_seed": 2026,
        "prediction_count": 8,
        "prediction_sample_count": 0,
        "inference_seed": 456,
        "playback_interval_seconds": 1.5,
    }
    values["observation_interval_seconds"] = ""
    assert (
        settings.validate_dialog_values("data", values)["observation_interval_seconds"]
        is None
    )
    values["playback_interval_seconds"] = "0"
    with pytest.raises(ValueError):
        settings.validate_dialog_values("data", values)
    values["playback_interval_seconds"] = "1.5"
    values["observation_interval_seconds"] = "0"
    with pytest.raises(ValueError, match="Observation interval"):
        settings.validate_dialog_values("data", values)


def test_parse_settings_converts_playback_seconds_to_timer_milliseconds(form):
    form["data"]["playback_interval_seconds"] = "1.5"

    result = settings.parse_settings(form)

    assert result.playback_interval_ms == 1500


@pytest.mark.parametrize(
    ("display_style", "normalized_style"),
    (
        ("Per-step density contours", "density_contours"),
        ("Per-step density regions", "density_regions"),
        ("Covariance ellipses", "ellipses"),
        ("Horizon density contours", "horizon_density_contours"),
        ("Horizon density region", "funnel"),
        ("No uncertainty region", "none"),
    ),
)
def test_plot_options_dialog_validates_display_switches_independently(
    display_style, normalized_style
):
    values = {
        "show_legend": False,
        "show_reference_trajectory": False,
        "show_observed_trajectory": True,
        "show_current_position": True,
        "show_sample_trajectories": False,
        "connect_sample_trajectories": True,
        "show_median_forecast": True,
        "show_mean_forecast": False,
        "show_prediction_region_50": True,
        "show_prediction_region_90": False,
        "prediction_region_style": display_style,
        "follow_ship_view_span_m": 600.0,
    }

    result = settings.validate_dialog_values("plot", values)

    assert result == {
        **values,
        "prediction_region_style": normalized_style,
    }
