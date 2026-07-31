"""Tests for visualizing actual variational posterior draws."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import ship_trajectory_prediction.evaluation.posterior as posterior_module  # noqa: E402
from ship_trajectory_prediction.evaluation.posterior import (  # noqa: E402
    plot_scalar_posterior,
    plot_scalar_posterior_comparison,
    plot_state_credible_band,
    plot_state_posterior_at_time,
    save_bayesian_ctrv_posterior_plots,
)
from ship_trajectory_prediction.models.bayesian_ctrv import (  # noqa: E402
    NOISE_PARAMETER_NAMES,
    VIRunResult,
)


def create_fit(*, draw_count=20, time_count=4):
    """Return one finite CmdStanVB-like fit with scalar and state draws."""
    scalar_draws = np.linspace(0.1, 0.5, draw_count)
    state_time = np.arange(time_count, dtype=float)
    state_draws = np.vstack(
        [state_time + offset for offset in np.linspace(-0.2, 0.2, draw_count)]
    )
    variables = {
        name: scalar_draws + index for index, name in enumerate(NOISE_PARAMETER_NAMES)
    }
    variables.update(
        {
            "x_state": state_draws,
            "y_state": state_draws + 10.0,
            "speed_state": state_draws + 3.0,
            "heading_state": state_draws + 3.0,
            "turn_rate_state": state_draws * 0.001,
        }
    )
    return FakeVariationalFit(**variables)


def test_plot_scalar_posterior_uses_actual_finite_draws():
    """The scalar plot should summarize only the extracted VI draws."""
    fit = create_fit()

    figure, axis = plot_scalar_posterior(
        fit,
        "sigma_position_gps",
        credible_interval=0.8,
        reference_value=0.25,
        bins=5,
    )

    assert isinstance(figure, Figure)
    assert isinstance(axis, Axes)
    assert axis.get_ylabel() == "Density"
    assert len(axis.patches) == 5
    assert sum(
        patch.get_height() * patch.get_width() for patch in axis.patches
    ) == pytest.approx(1.0)
    assert len(axis.lines) == 5
    assert fit.requested_means == [False]
    legend_text = [text.get_text() for text in axis.get_legend().get_texts()]
    assert legend_text == [
        "Variational posterior draws",
        "Posterior mean: 0.3",
        "Posterior median: 0.3",
        "Central 80% credible interval: [0.14, 0.46]",
        "Known reference: 0.25",
    ]
    plt.close(figure)


@pytest.mark.parametrize(
    "draws",
    [
        np.asarray([]),
        np.ones((2, 2)),
        np.asarray([0.1, np.nan]),
        np.asarray([0.1, np.inf]),
    ],
)
def test_plot_scalar_posterior_rejects_invalid_draws(draws):
    """Empty, non-scalar, and non-finite draw sequences must fail."""
    fit = FakeVariationalFit(sigma_position_gps=draws)

    with pytest.raises(ValueError, match="Posterior variable"):
        plot_scalar_posterior(fit, "sigma_position_gps")


@pytest.mark.parametrize("credible_interval", [0.0, -0.1, 1.0, 1.1, np.nan])
def test_posterior_plots_reject_invalid_credible_intervals(credible_interval):
    """Central credible intervals must be strict probabilities."""
    with pytest.raises(ValueError, match="credible_interval"):
        plot_scalar_posterior(
            create_fit(),
            "sigma_position_gps",
            credible_interval=credible_interval,
        )


def test_plot_state_posterior_at_time_uses_one_matrix_column():
    """A zero-based time index should select exactly one latent-state column."""
    fit = create_fit()
    time_values = np.asarray([0.0, 10.0, 20.0, 30.0])

    figure, axis = plot_state_posterior_at_time(
        fit,
        "turn_rate_state",
        2,
        time_values=time_values,
        credible_interval=0.9,
        bins=6,
    )

    assert isinstance(figure, Figure)
    assert isinstance(axis, Axes)
    assert "turn_rate_state" in axis.get_title()
    assert "index 2" in axis.get_title()
    assert "t = 20 s" in axis.get_title()
    assert len(axis.patches) == 6
    plt.close(figure)


@pytest.mark.parametrize("draws", [np.ones(4), np.ones((2, 2, 2)), np.empty((0, 4))])
def test_state_plots_reject_wrong_draw_dimensions(draws):
    """Latent-state draws must be a non-empty draw-by-time matrix."""
    fit = FakeVariationalFit(speed_state=draws)

    with pytest.raises(ValueError, match="number_of_draws"):
        plot_state_posterior_at_time(fit, "speed_state", 0)


@pytest.mark.parametrize("time_index", [-1, 4, 1.5, True])
def test_plot_state_posterior_rejects_invalid_time_index(time_index):
    """The requested latent-state index must be a valid zero-based integer."""
    expected_error = (
        IndexError
        if isinstance(time_index, int) and not isinstance(time_index, bool)
        else ValueError
    )
    with pytest.raises(expected_error, match="time_index"):
        plot_state_posterior_at_time(create_fit(), "speed_state", time_index)


def test_state_credible_band_uses_draw_quantiles_and_optional_references():
    """Median, interval, observations, and truth should remain time-aligned."""
    fit = create_fit()
    time_values = np.asarray([0.0, 10.0, 20.0, 30.0])
    observed_values = np.asarray([3.0, 4.0, 5.0, 6.0])
    reference_values = observed_values + 0.1

    figure, axis = plot_state_credible_band(
        fit,
        "speed_state",
        time_values,
        credible_interval=0.8,
        observed_values=observed_values,
        reference_values=reference_values,
    )

    expected_median = np.median(fit.variables["speed_state"], axis=0)
    assert axis.lines[0].get_ydata() == pytest.approx(expected_median)
    assert axis.lines[1].get_ydata() == pytest.approx(observed_values)
    assert axis.lines[2].get_ydata() == pytest.approx(reference_values)
    assert len(axis.collections) == 1
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Posterior median",
        "Central 80% credible interval",
        "Observed values",
        "Known reference",
    ]
    plt.close(figure)


def test_speed_band_labels_gps_speed_as_an_external_reference():
    """A post-fit GPS-speed overlay must not be called an observation."""
    figure, axis = plot_state_credible_band(
        create_fit(),
        "speed_state",
        [0.0, 10.0, 20.0, 30.0],
        observed_values=[3.0, 4.0, 5.0, 6.0],
        observed_label="External GPS-speed reference",
    )

    legend_text = [text.get_text() for text in axis.get_legend().get_texts()]
    assert "External GPS-speed reference" in legend_text
    assert "Observed values" not in legend_text
    plt.close(figure)


@pytest.mark.parametrize(
    ("option", "values", "message"),
    [
        ("time_values", [0.0, 1.0, 2.0], "time_values"),
        ("observed_values", [1.0, 2.0, 3.0], "observed_values"),
        ("reference_values", [1.0, 2.0, 3.0], "reference_values"),
    ],
)
def test_state_credible_band_rejects_mismatched_lengths(option, values, message):
    """Every optional state overlay must match the latent time dimension."""
    arguments = {
        "time_values": [0.0, 1.0, 2.0, 3.0],
        option: values,
    }

    with pytest.raises(ValueError, match=message):
        plot_state_credible_band(create_fit(), "speed_state", **arguments)


def test_heading_band_converts_unwrapped_quantiles_to_degrees():
    """Heading summaries should not wrap draws before computing quantiles."""
    fit = create_fit()

    figure, axis = plot_state_credible_band(
        fit,
        "heading_state",
        [0.0, 1.0, 2.0, 3.0],
        unit="deg",
    )

    median_degrees = axis.lines[0].get_ydata()
    assert median_degrees == pytest.approx(
        np.degrees(np.median(fit.variables["heading_state"], axis=0))
    )
    assert median_degrees[-1] > 180
    assert np.all(np.diff(median_degrees) > 0)
    assert axis.get_ylabel().endswith("[°]")
    plt.close(figure)


def test_scalar_posterior_comparison_labels_algorithm_and_seed():
    """Each VI approximation should be identifiable without synthetic curves."""
    runs = [
        VIRunResult(
            seed=11,
            algorithm="meanfield",
            fit=create_fit(),
            runtime_seconds=1.0,
        ),
        VIRunResult(
            seed=22,
            algorithm="fullrank",
            fit=create_fit(),
            runtime_seconds=2.0,
        ),
    ]

    figure, axis = plot_scalar_posterior_comparison(
        runs,
        "sigma_turn_rate_process",
        credible_interval=0.9,
        bins=8,
    )

    legend_text = [text.get_text() for text in axis.get_legend().get_texts()]
    assert any("meanfield, seed 11" in text for text in legend_text)
    assert any("fullrank, seed 22" in text for text in legend_text)
    assert all("90% CI=" in text for text in legend_text)
    assert len(axis.patches) == 2
    plt.close(figure)


def test_scalar_posterior_comparison_limits_displayed_runs():
    """Too many overlapping VI histograms should be rejected explicitly."""
    runs = [
        VIRunResult(
            seed=seed,
            algorithm="meanfield",
            fit=create_fit(),
            runtime_seconds=1.0,
        )
        for seed in (1, 2, 3)
    ]

    with pytest.raises(ValueError, match="At most 2"):
        plot_scalar_posterior_comparison(
            runs,
            "sigma_position_gps",
            max_runs=2,
        )


def test_save_bayesian_ctrv_posterior_plots_creates_expected_files(
    tmp_path,
    monkeypatch,
):
    """The convenience exporter should save, return, and close every figure."""
    monkeypatch.setattr(
        plt,
        "show",
        lambda *args, **kwargs: pytest.fail("Posterior plotting must not call show()."),
    )
    fit = create_fit()
    window = FakeWindow()
    plt.close("all")

    generated = save_bayesian_ctrv_posterior_plots(
        fit,
        window,
        tmp_path / "posterior",
        credible_interval=0.9,
        selected_time_indices=(1, 3),
        reference_parameters={"sigma_position_gps": 0.2},
        reference_states={
            "speed_state": np.asarray([3.0, 4.0, 5.0, 6.0]),
            "turn_rate_state": np.asarray([0.0, 0.001, 0.002, 0.003]),
        },
        include_speed_gps_reference=True,
        dpi=80,
    )

    expected_names = {
        *(f"posterior_{name}" for name in NOISE_PARAMETER_NAMES),
        "posterior_speed_state_over_time",
        "posterior_turn_rate_state_over_time",
        "posterior_speed_state_t_001",
        "posterior_turn_rate_state_t_001",
        "posterior_speed_state_t_003",
        "posterior_turn_rate_state_t_003",
    }
    assert generated.keys() == expected_names
    assert len(generated) == 10
    assert all(
        path.is_file() and path.stat().st_size > 0 for path in generated.values()
    )
    assert plt.get_fignums() == []
    assert fit.requested_means == [False] * 10


def test_posterior_library_functions_never_show_automatically(monkeypatch):
    """Reusable plotting functions must leave display control to callers."""
    monkeypatch.setattr(
        plt,
        "show",
        lambda *args, **kwargs: pytest.fail("Posterior plotting must not call show()."),
    )
    fit = create_fit()
    runs = [
        VIRunResult(
            seed=42,
            algorithm="meanfield",
            fit=fit,
            runtime_seconds=1.0,
        )
    ]
    figures = [
        plot_scalar_posterior(fit, "sigma_position_gps")[0],
        plot_state_posterior_at_time(fit, "speed_state", 0)[0],
        plot_state_credible_band(
            fit,
            "speed_state",
            [0.0, 1.0, 2.0, 3.0],
        )[0],
        plot_scalar_posterior_comparison(
            runs,
            "sigma_position_gps",
        )[0],
    ]

    for figure in figures:
        plt.close(figure)


def test_show_bayesian_ctrv_posterior_plots_displays_figures_sequentially(
    monkeypatch,
):
    """The display helper should show and close plots without saving."""
    created_figures = []
    shown_blocks = []
    closed_figures = []

    def fake_plot(kind):
        def create_figure(*args, **kwargs):
            del args, kwargs
            figure = object()
            created_figures.append((kind, figure))
            return figure, object()

        return create_figure

    monkeypatch.setattr(
        posterior_module,
        "plot_scalar_posterior",
        fake_plot("scalar"),
    )
    monkeypatch.setattr(
        posterior_module,
        "plot_state_credible_band",
        fake_plot("band"),
    )
    monkeypatch.setattr(
        posterior_module,
        "plot_state_posterior_at_time",
        fake_plot("time"),
    )
    monkeypatch.setattr(
        posterior_module.plt,
        "show",
        lambda *, block: shown_blocks.append(block),
    )
    monkeypatch.setattr(
        posterior_module.plt,
        "close",
        closed_figures.append,
    )

    posterior_module.show_bayesian_ctrv_posterior_plots(
        object(),
        FakeWindow(),
        selected_time_indices=(0, 3),
    )

    assert [kind for kind, _ in created_figures] == [
        "scalar",
        "scalar",
        "scalar",
        "scalar",
        "band",
        "band",
        "time",
        "time",
        "time",
        "time",
    ]
    assert shown_blocks == [True] * 10
    assert closed_figures == [figure for _, figure in created_figures]


class FakeVariationalFit:
    """Small CmdStanVB-like object exposing all approximate posterior draws."""

    variational_sample = object()

    def __init__(self, **variables):
        self.variables = variables
        self.requested_means = []

    def stan_variable(self, name, *, mean=None):
        """Record that the full variational draws were requested."""
        self.requested_means.append(mean)
        return self.variables[name]


class FakeWindow:
    """Minimal observed and held-out trajectory window for export tests."""

    observation_count = 4
    time_seconds = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    gps_speed_mps = np.asarray([3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    @property
    def observed_slice(self):
        """Return only timestamps used for latent-state inference."""
        return slice(0, self.observation_count)
