"""Tests for Sequential VI comparison plotting data and figures."""

import matplotlib.pyplot as plt
import numpy as np

from bayestraj.inference.ctrv_sequential_vi import (
    DistributionSummary,
    SequentialVIParameterSummary,
    SequentialVIPredictiveDraws,
    SequentialVIResult,
    SequentialVIStateSummary,
    SequentialVIUpdate,
)
from bayestraj.validation import sequential_vi_plotting


def _summary(center: float) -> DistributionSummary:
    return DistributionSummary(center, center, 0.5, center - 1.0, center + 1.0)


def _result() -> SequentialVIResult:
    updates = []
    for index in range(3):
        predictive_state = SequentialVIStateSummary(
            *(_summary(index + offset) for offset in range(5))
        )
        filtered_state = SequentialVIStateSummary(
            *(_summary(index + offset + 0.25) for offset in range(5))
        )
        predictive_parameters = SequentialVIParameterSummary(
            *(_summary(index + offset + 5.0) for offset in range(3))
        )
        filtered_parameters = SequentialVIParameterSummary(
            *(_summary(index + offset + 5.25) for offset in range(3))
        )
        updates.append(
            SequentialVIUpdate(
                observation_index=10 + index,
                time_seconds=20.0 + 2.0 * index,
                x_observed=100.0 + index,
                y_observed=-50.0 - index,
                predictive_state=predictive_state,
                filtered_state=filtered_state,
                predictive_parameters=predictive_parameters,
                filtered_parameters=filtered_parameters,
                predictive_draws=SequentialVIPredictiveDraws(
                    x=np.array([1.0, 2.0]),
                    y=np.array([3.0, 4.0]),
                    sigma_position_observation=np.array([5.0, 6.0]),
                ),
                runtime_seconds=0.2 + index,
                cumulative_runtime_seconds=1.2 + index,
                converged=True,
            )
        )
    return SequentialVIResult(
        bootstrap_runtime_seconds=1.0,
        bootstrap_converged=True,
        bootstrap_draw_count=100,
        updates=tuple(updates),
        processed_observation_count=13,
        completed=True,
        failure_index=None,
        failure_message=None,
    )


def test_extract_sequential_vi_plot_data_aligns_all_update_series() -> None:
    data = sequential_vi_plotting.extract_sequential_vi_plot_data(_result())

    np.testing.assert_array_equal(data.time_seconds, [20.0, 22.0, 24.0])
    assert data.observed_positions.shape == (3, 2)
    assert data.predictive_positions.shape == (3, 2, 3)
    assert data.filtered_positions.shape == (3, 2, 3)
    assert set(data.predictive_motion) == {"speed", "heading", "turn_rate"}
    assert set(data.filtered_parameters) == {
        "sigma_position_observation",
        "sigma_speed_process",
        "sigma_turn_rate_process",
    }
    assert data.filtered_motion["heading"].shape == (3, 3)
    np.testing.assert_array_equal(data.runtime_seconds, [0.2, 1.2, 2.2])
    np.testing.assert_array_equal(
        data.cumulative_runtime_seconds,
        [1.2, 2.2, 3.2],
    )


def test_sequential_vi_plot_functions_return_figures_without_showing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(plt, "show", lambda: (_ for _ in ()).throw(AssertionError()))

    figures = [
        sequential_vi_plotting.plot_trajectory_comparison(_result()),
        sequential_vi_plotting.plot_motion_state_comparison(_result()),
        sequential_vi_plotting.plot_uncertainty_parameter_comparison(_result()),
        sequential_vi_plotting.plot_runtime_comparison(_result()),
    ]

    for figure, axes in figures:
        assert figure.axes
        assert axes is not None
        plt.close(figure)
