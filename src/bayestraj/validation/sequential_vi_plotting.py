"""Focused comparison plots for Sequential VI prediction and filtering."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from bayestraj.inference.ctrv_sequential_vi import (
    DistributionSummary,
    SequentialVIResult,
)
from bayestraj.observations.plotting import DEFAULT_SHIP_DATA_PLOT_STYLE


@dataclass(frozen=True, slots=True)
class SequentialVIPlotData:
    """Aligned numeric series extracted from successful sequential updates."""

    time_seconds: np.ndarray
    observed_positions: np.ndarray
    predictive_positions: np.ndarray
    filtered_positions: np.ndarray
    predictive_motion: dict[str, np.ndarray]
    filtered_motion: dict[str, np.ndarray]
    predictive_parameters: dict[str, np.ndarray]
    filtered_parameters: dict[str, np.ndarray]
    runtime_seconds: np.ndarray
    cumulative_runtime_seconds: np.ndarray


def _summary_row(summary: DistributionSummary) -> tuple[float, float, float]:
    return summary.lower_95, summary.median, summary.upper_95


def _summary_series(updates, group: str, name: str) -> np.ndarray:
    return np.asarray(
        [_summary_row(getattr(getattr(update, group), name)) for update in updates],
        dtype=float,
    )


def extract_sequential_vi_plot_data(
    result: SequentialVIResult,
) -> SequentialVIPlotData:
    """Extract aligned prediction, filtering, parameter, and runtime traces."""
    if not isinstance(result, SequentialVIResult):
        raise TypeError("result must be a SequentialVIResult.")
    updates = tuple(update for update in result.updates if update.error_message is None)
    if not updates:
        raise ValueError("result must contain at least one successful update.")
    motion_names = ("speed", "heading", "turn_rate")
    parameter_names = (
        "sigma_position_observation",
        "sigma_speed_process",
        "sigma_turn_rate_process",
    )
    return SequentialVIPlotData(
        time_seconds=np.asarray([update.time_seconds for update in updates]),
        observed_positions=np.asarray(
            [(update.x_observed, update.y_observed) for update in updates]
        ),
        predictive_positions=np.stack(
            (
                _summary_series(updates, "predictive_state", "x"),
                _summary_series(updates, "predictive_state", "y"),
            ),
            axis=1,
        ),
        filtered_positions=np.stack(
            (
                _summary_series(updates, "filtered_state", "x"),
                _summary_series(updates, "filtered_state", "y"),
            ),
            axis=1,
        ),
        predictive_motion={
            name: _summary_series(updates, "predictive_state", name)
            for name in motion_names
        },
        filtered_motion={
            name: _summary_series(updates, "filtered_state", name)
            for name in motion_names
        },
        predictive_parameters={
            name: _summary_series(updates, "predictive_parameters", name)
            for name in parameter_names
        },
        filtered_parameters={
            name: _summary_series(updates, "filtered_parameters", name)
            for name in parameter_names
        },
        runtime_seconds=np.asarray([update.runtime_seconds for update in updates]),
        cumulative_runtime_seconds=np.asarray(
            [update.cumulative_runtime_seconds for update in updates]
        ),
    )


def _plot_summary(axis, times, values, *, label, color) -> None:
    axis.fill_between(times, values[:, 0], values[:, 2], color=color, alpha=0.14)
    axis.plot(times, values[:, 1], color=color, label=label)


def _style_axis(axis) -> None:
    """Apply the shared standalone-plot typography without changing global style."""
    axis.title.set_fontsize(DEFAULT_SHIP_DATA_PLOT_STYLE.title_font_size)
    axis.title.set_fontweight(DEFAULT_SHIP_DATA_PLOT_STYLE.title_font_weight)
    axis.xaxis.label.set_fontsize(DEFAULT_SHIP_DATA_PLOT_STYLE.axis_label_font_size)
    axis.yaxis.label.set_fontsize(DEFAULT_SHIP_DATA_PLOT_STYLE.axis_label_font_size)
    axis.tick_params(labelsize=DEFAULT_SHIP_DATA_PLOT_STYLE.axis_tick_font_size)


def plot_trajectory_comparison(result: SequentialVIResult):
    """Plot observed, predictive, and filtered position medians."""
    data = extract_sequential_vi_plot_data(result)
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    axis.plot(
        data.observed_positions[:, 0],
        data.observed_positions[:, 1],
        "o-",
        color="#202020",
        label="Beobachtungen",
    )
    axis.plot(
        data.predictive_positions[:, 0, 1],
        data.predictive_positions[:, 1, 1],
        "o-",
        color="#d97706",
        label="Prädiktiver Median",
    )
    axis.plot(
        data.filtered_positions[:, 0, 1],
        data.filtered_positions[:, 1, 1],
        "o-",
        color="#176b87",
        label="Gefilterter Median",
    )
    axis.set(
        xlabel="Ostposition x [m]", ylabel="Nordposition y [m]", title="Sequential VI"
    )
    _style_axis(axis)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure, axis


def plot_motion_state_comparison(result: SequentialVIResult):
    """Plot predictive and filtered speed, heading, and turn-rate summaries."""
    data = extract_sequential_vi_plot_data(result)
    names = ("speed", "heading", "turn_rate")
    labels = ("Geschwindigkeit [m/s]", "Kurswinkel [rad]", "Drehrate [rad/s]")
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(8.0, 7.0))
    for axis, name, label in zip(axes, names, labels, strict=True):
        _plot_summary(
            axis,
            data.time_seconds,
            data.predictive_motion[name],
            label="Prädiktiv",
            color="#d97706",
        )
        _plot_summary(
            axis,
            data.time_seconds,
            data.filtered_motion[name],
            label="Gefiltert",
            color="#176b87",
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].set_title("Bewegungszustand")
    axes[0].legend()
    axes[-1].set_xlabel("Zeit [s]")
    for axis in axes:
        _style_axis(axis)
    figure.tight_layout()
    return figure, axes


def plot_uncertainty_parameter_comparison(result: SequentialVIResult):
    """Plot predictive and filtered CTRV uncertainty parameters."""
    data = extract_sequential_vi_plot_data(result)
    names = tuple(data.filtered_parameters)
    labels = (
        "Beobachtung [m]",
        "Geschwindigkeitsprozess [m/s]",
        "Drehratenprozess [rad/s]",
    )
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(8.0, 7.0))
    for axis, name, label in zip(axes, names, labels, strict=True):
        _plot_summary(
            axis,
            data.time_seconds,
            data.predictive_parameters[name],
            label="Prädiktiv",
            color="#d97706",
        )
        _plot_summary(
            axis,
            data.time_seconds,
            data.filtered_parameters[name],
            label="Gefiltert",
            color="#176b87",
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].set_title("Unsicherheitsparameter")
    axes[0].legend()
    axes[-1].set_xlabel("Zeit [s]")
    for axis in axes:
        _style_axis(axis)
    figure.tight_layout()
    return figure, axes


def plot_runtime_comparison(result: SequentialVIResult):
    """Plot per-update and cumulative Sequential VI runtime."""
    data = extract_sequential_vi_plot_data(result)
    figure, axis = plt.subplots(figsize=(8.0, 4.0))
    axis.plot(
        data.time_seconds,
        data.runtime_seconds,
        "o-",
        color="#176b87",
        label="Aktualisierung",
    )
    axis.plot(
        data.time_seconds,
        data.cumulative_runtime_seconds,
        "o-",
        color="#b42318",
        label="Kumuliert",
    )
    axis.set(xlabel="Zeit [s]", ylabel="Laufzeit [s]", title="Sequential-VI-Laufzeit")
    _style_axis(axis)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure, axis
