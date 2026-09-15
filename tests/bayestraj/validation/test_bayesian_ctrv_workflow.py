"""Tests for Bayesian CTRV rolling-evaluation presentation rules."""

import bayestraj.validation.bayesian_ctrv_workflow as workflow
import bayestraj.validation.rolling as rolling


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
