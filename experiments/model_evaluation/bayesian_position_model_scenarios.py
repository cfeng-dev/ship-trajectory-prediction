"""Compare K=20 and K=10 on straight and recent-curve position scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Patch

import ship_trajectory_prediction.forecasting.inference as inference
import ship_trajectory_prediction.models.bayesian_position_model as position_model
import ship_trajectory_prediction.observations.window as observation_window
import ship_trajectory_prediction.validation.metrics as metrics
import ship_trajectory_prediction.validation.reporting as reporting

OBSERVATION_COUNT = 20
PREDICTION_COUNT = 3
HISTORY_POSITION_COUNTS = (20, 10)
TIME_STEP_SECONDS = 10.0
DISPLACEMENT_LENGTH_METERS = 10.0
CURVE_ROTATION_PER_STEP_RAD = np.deg2rad(5.0)
POSITION_PERTURBATION_STD_M = 0.25
POSTERIOR_SAMPLE_PATH_COUNT = 15
POSTERIOR_DRAWS = 1_000
CREDIBLE_INTERVAL = 0.9

PRIORS = position_model.BayesianPositionModelPriors(
    log_displacement_scale_prior_scale=0.016354,
    rotation_angle_prior_scale=0.016980,
    sigma_displacement_residual_prior_scale=1.989083,
)
VI_CONFIG = inference.create_default_vi_config()


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """One fitted synthetic scenario and its shared position evaluation."""

    scenario_name: str
    history_position_count: int
    window: observation_window.TrajectoryWindowData
    fit: object
    evaluation: metrics.PositionEvaluation


def main():
    """Fit both history lengths to straight and recent-curve scenarios."""
    results = []
    scenarios = (
        ("Ungefähr geradeaus", False, 100),
        ("Geradeaus → letzte Kurve", True, 200),
    )
    for scenario_name, recent_curve, seed in scenarios:
        window = create_scenario_window(recent_curve=recent_curve, seed=seed)
        for history_position_count in HISTORY_POSITION_COUNTS:
            fit = position_model.fit_bayesian_position_model(
                window,
                priors=PRIORS,
                history_position_count=history_position_count,
                inference_method="vi",
                seed=seed + history_position_count,
                draws=POSTERIOR_DRAWS,
                require_converged=False,
                **{
                    key: value
                    for key, value in VI_CONFIG.items()
                    if key not in {"draws", "require_converged", "algorithm"}
                },
            )
            evaluation = metrics.evaluate_position_predictions(
                fit,
                window,
                credible_interval=CREDIBLE_INTERVAL,
                position_variable_names=(
                    "x_observation_prediction",
                    "y_observation_prediction",
                ),
            )
            results.append(
                ScenarioResult(
                    scenario_name=scenario_name,
                    history_position_count=history_position_count,
                    window=window,
                    fit=fit,
                    evaluation=evaluation,
                )
            )
            print(
                f"{scenario_name}, K={history_position_count}: "
                f"ADE={evaluation.ade_m:.3f} m, FDE={evaluation.fde_m:.3f} m"
            )
    plot_scenario_comparison(results)
    return tuple(results)


def create_scenario_window(*, recent_curve: bool, seed: int):
    """Return 20 observed and three held-out regularly sampled positions."""
    total_count = OBSERVATION_COUNT + PREDICTION_COUNT
    x_values = np.zeros(total_count, dtype=float)
    y_values = np.zeros(total_count, dtype=float)
    heading = 0.0
    for point_index in range(1, total_count):
        if recent_curve and point_index >= 10:
            heading += CURVE_ROTATION_PER_STEP_RAD
        x_values[point_index] = x_values[
            point_index - 1
        ] + DISPLACEMENT_LENGTH_METERS * np.cos(heading)
        y_values[point_index] = y_values[
            point_index - 1
        ] + DISPLACEMENT_LENGTH_METERS * np.sin(heading)
    perturbation = np.random.default_rng(seed).normal(
        0.0,
        POSITION_PERTURBATION_STD_M,
        size=(total_count, 2),
    )
    x_values += perturbation[:, 0]
    y_values += perturbation[:, 1]
    timestamps = pd.date_range(
        "2026-01-01",
        periods=total_count,
        freq=pd.to_timedelta(TIME_STEP_SECONDS, unit="s"),
        tz="UTC",
    )
    return observation_window.TrajectoryWindowData(
        timestamps=timestamps,
        time_seconds=np.arange(total_count, dtype=float) * TIME_STEP_SECONDS,
        x_meters=x_values,
        y_meters=y_values,
        reference_longitude=0.0,
        reference_latitude=0.0,
        gps_speed_mps=np.full(total_count, np.nan),
        observation_count=OBSERVATION_COUNT,
    )


def plot_scenario_comparison(results):
    """Plot both scenarios and both local history choices in one figure."""
    result_lookup = {
        (result.scenario_name, result.history_position_count): result
        for result in results
    }
    scenario_names = tuple(dict.fromkeys(result.scenario_name for result in results))
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    sample_generator = np.random.default_rng(42)
    for row_index, scenario_name in enumerate(scenario_names):
        for column_index, history_position_count in enumerate(HISTORY_POSITION_COUNTS):
            result = result_lookup[(scenario_name, history_position_count)]
            axis = axes[row_index, column_index]
            window = result.window
            observed = window.observed_slice
            prediction = window.prediction_slice
            history_start = OBSERVATION_COUNT - history_position_count
            axis.plot(
                window.x_meters[observed],
                window.y_meters[observed],
                color="#8CB6D9",
                linewidth=1.6,
                marker="o",
                markersize=3,
                label="Alle 20 Beobachtungen",
            )
            axis.plot(
                window.x_meters[history_start:OBSERVATION_COUNT],
                window.y_meters[history_start:OBSERVATION_COUNT],
                color="#1F5A85",
                linewidth=2.4,
                marker="o",
                markersize=4,
                label=f"Für Inferenz verwendet (K={history_position_count})",
            )
            ground_truth_x = np.concatenate(
                ([window.x_meters[OBSERVATION_COUNT - 1]], window.x_meters[prediction])
            )
            ground_truth_y = np.concatenate(
                ([window.y_meters[OBSERVATION_COUNT - 1]], window.y_meters[prediction])
            )
            axis.plot(
                ground_truth_x,
                ground_truth_y,
                color="black",
                linestyle="--",
                linewidth=2.0,
                marker="o",
                label="Ground Truth (3 Held-out-Punkte)",
            )
            x_samples = reporting.posterior_variable_samples(
                result.fit, "x_observation_prediction"
            )
            y_samples = reporting.posterior_variable_samples(
                result.fit, "y_observation_prediction"
            )
            sample_indices = sample_generator.choice(
                x_samples.shape[0],
                size=min(POSTERIOR_SAMPLE_PATH_COUNT, x_samples.shape[0]),
                replace=False,
            )
            for path_number, sample_index in enumerate(sample_indices):
                axis.plot(
                    np.concatenate(
                        (
                            [window.x_meters[OBSERVATION_COUNT - 1]],
                            x_samples[sample_index],
                        )
                    ),
                    np.concatenate(
                        (
                            [window.y_meters[OBSERVATION_COUNT - 1]],
                            y_samples[sample_index],
                        )
                    ),
                    color="#D77A61",
                    alpha=0.18,
                    linewidth=1.0,
                    label=("Posterior-Pfade" if path_number == 0 else None),
                )
            median_x = np.median(x_samples, axis=0)
            median_y = np.median(y_samples, axis=0)
            axis.plot(
                np.concatenate(([window.x_meters[OBSERVATION_COUNT - 1]], median_x)),
                np.concatenate(([window.y_meters[OBSERVATION_COUNT - 1]], median_y)),
                color="#B4432F",
                linewidth=2.4,
                marker="o",
                label="Posterior-Median",
            )
            _draw_prediction_regions(axis, x_samples, y_samples)
            axis.set_title(
                f"{scenario_name}\nK={history_position_count}, "
                f"ADE={result.evaluation.ade_m:.2f} m"
            )
            axis.set_xlabel("Ostposition x [m]")
            axis.set_ylabel("Nordposition y [m]")
            axis.grid(True, alpha=0.25)
            axis.set_aspect("equal", adjustable="datalim")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.extend(
        [
            Patch(color="#B4432F", alpha=0.22, label="50-%-Bereich"),
            Patch(color="#B4432F", alpha=0.10, label="90-%-Bereich"),
        ]
    )
    labels.extend(("50-%-Bereich", "90-%-Bereich"))
    figure.legend(handles, labels, loc="outside lower center", ncol=4)
    figure.suptitle(
        "Bayesian Position Model: vollständige gegen lokale Bewegungshistorie",
        fontsize=14,
        fontweight="bold",
    )
    plt.show()
    return figure, axes


def _draw_prediction_regions(axis, x_samples, y_samples):
    """Draw empirical joint 50% and 90% regions at all forecast horizons."""
    for horizon_index in range(x_samples.shape[1]):
        regions = metrics.empirical_covariance_regions(
            x_samples[:, horizon_index],
            y_samples[:, horizon_index],
            probabilities=(0.5, 0.9),
        )
        for probability, alpha in ((0.9, 0.10), (0.5, 0.22)):
            region = regions[probability]
            axis.add_patch(
                Ellipse(
                    xy=region.center,
                    width=region.width,
                    height=region.height,
                    angle=region.angle_degrees,
                    facecolor="#B4432F",
                    edgecolor="#B4432F",
                    alpha=alpha,
                    linewidth=0.7,
                )
            )


if __name__ == "__main__":
    main()
