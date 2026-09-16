"""Tests for Bayesian CTRV rolling-evaluation presentation rules."""

import pandas as pd

import bayestraj.validation.bayesian_ctrv_workflow as workflow
import bayestraj.validation.rolling as rolling
import bayestraj.validation.runtime_plotting as runtime_plotting


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
    vi_config, mcmc_config, rbpf_config, smc_config = (
        workflow._resolve_inference_configs(None, None, None, None)
    )

    assert vi_config["algorithm"] == "meanfield"
    assert mcmc_config["chains"] >= 1
    assert rbpf_config.particle_count > 0
    assert smc_config.particle_count > 0


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

    returned_figure, returned_axis = runtime_plotting.plot_bayesian_ctrv_inference_runtime(
        predictions,
        inference_selection="vi_expanding",
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
    assert axis.title == "Bayesian-CTRV (VI, Expanding Window)"
    assert not axis.grid_called
    assert axis.title_keywords == {"pad": 16, "fontsize": 13, "fontweight": "bold"}
    assert axis.tick_parameters == {"axis": "both", "labelsize": 11}
    assert axis.legend_location == "upper right"
