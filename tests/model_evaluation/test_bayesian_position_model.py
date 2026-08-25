"""Tests for rolling and synthetic Bayesian Position Model evaluation."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import experiments.model_evaluation.bayesian_position_model as experiment
import experiments.model_evaluation.bayesian_position_model_scenarios as scenarios
import ship_trajectory_prediction.validation.bayesian_position_model_workflow as workflow


def test_main_delegates_k20_default_and_k10_override(monkeypatch):
    """The rolling entry point should expose both approved history options."""
    calls = []
    expected = object()
    monkeypatch.setattr(
        experiment.workflow,
        "run_bayesian_position_evaluation",
        lambda **kwargs: calls.append(kwargs) or expected,
    )

    assert experiment.main([]) is expected
    assert (
        experiment.main(["--history-positions", "10", "--max-windows", "2"]) is expected
    )

    assert calls[0]["options"].history_position_count == 20
    assert calls[1]["options"].history_position_count == 10
    assert calls[1]["options"].max_windows == 2


def test_route_position_noise_is_reproducible():
    """Overlapping rolling windows must see one fixed route perturbation."""
    first = workflow._simulate_route_position_noise(
        12,
        additional_noise_std_m=5.0,
        seed=2026,
    )
    second = workflow._simulate_route_position_noise(
        12,
        additional_noise_std_m=5.0,
        seed=2026,
    )

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert np.any(first[0] != 0.0)


def test_recent_curve_scenario_turns_only_after_first_ten_points():
    """The qualitative diagnostic must contain the requested regime change."""
    straight = scenarios.create_scenario_window(recent_curve=False, seed=100)
    curved = scenarios.create_scenario_window(recent_curve=True, seed=200)
    straight_angles = np.unwrap(
        np.arctan2(np.diff(straight.y_meters), np.diff(straight.x_meters))
    )
    curved_angles = np.unwrap(
        np.arctan2(np.diff(curved.y_meters), np.diff(curved.x_meters))
    )

    assert np.ptp(straight_angles) < 0.15
    assert np.ptp(curved_angles[:9]) < 0.15
    assert curved_angles[-1] - curved_angles[9] > 0.8
    assert curved.prediction_count == 3


def test_scenario_plot_contains_four_k_comparison_panels(monkeypatch):
    """Straight/curve by K=20/K=10 should render in one comparison figure."""
    monkeypatch.setattr(plt, "show", lambda: None)
    results = []
    for scenario_name, recent_curve in (("Straight", False), ("Curve", True)):
        window = scenarios.create_scenario_window(
            recent_curve=recent_curve,
            seed=123 if recent_curve else 321,
        )
        base_x = window.x_meters[window.prediction_slice]
        base_y = window.y_meters[window.prediction_slice]
        draws = 100
        offsets = np.linspace(-0.5, 0.5, draws)[:, None]
        fit = FakeFit(
            x_observation_prediction=base_x[None, :] + offsets,
            y_observation_prediction=base_y[None, :] - offsets,
        )
        evaluation = SimpleNamespace(ade_m=1.0)
        for history_position_count in (20, 10):
            results.append(
                scenarios.ScenarioResult(
                    scenario_name=scenario_name,
                    history_position_count=history_position_count,
                    window=window,
                    fit=fit,
                    evaluation=evaluation,
                )
            )

    figure, axes = scenarios.plot_scenario_comparison(results)

    assert axes.shape == (2, 2)
    assert "K=20" in axes[0, 0].get_title()
    assert "K=10" in axes[0, 1].get_title()
    plt.close(figure)


class FakeFit:
    """Minimal posterior draw container for scenario plotting."""

    variational_sample = object()

    def __init__(self, **variables):
        self.variables = variables

    def stan_variable(self, name, mean=False):
        """Return stored full draws."""
        del mean
        return self.variables[name]
