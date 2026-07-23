"""Compare all Bayesian trajectory models on one shared held-out window."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import pandas as pd

from ship_trajectory_prediction.evaluation.metrics import (
    evaluate_position_predictions,
)
from ship_trajectory_prediction.models.constant_radius import (
    fit_constant_radius_model,
)
from ship_trajectory_prediction.models.constant_turn_rate import (
    fit_constant_turn_rate_model,
)
from ship_trajectory_prediction.models.constant_turn_rate_acceleration import (
    fit_constant_turn_rate_acceleration_model,
)
from ship_trajectory_prediction.models.time_varying_motion import (
    fit_time_varying_motion_model,
)
from ship_trajectory_prediction.models.time_varying_radius import (
    fit_time_varying_radius_model,
)
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import prepare_trajectory_window
from ship_trajectory_prediction.trajectory.io import read_ship_data

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

RUN_ID = 1
START_INDEX = 0
OBSERVATION_COUNT = 20
PREDICTION_COUNT = 5
CREDIBLE_INTERVAL = 0.9


@dataclass(frozen=True)
class ModelSpec:
    """Adapter describing how one model fits the shared trajectory window."""

    name: str
    fit_model: Callable
    fit_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Keep one shared comparison configuration immutable."""
        object.__setattr__(
            self,
            "fit_kwargs",
            MappingProxyType(dict(self.fit_kwargs)),
        )


MODEL_SPECS = (
    ModelSpec(
        name="Constant Radius",
        fit_model=fit_constant_radius_model,
        fit_kwargs={
            "curvature_prior_scale": 0.002,
            "sigma_prior_scale": 20.0,
        },
    ),
    ModelSpec(
        name="Constant Turn Rate",
        fit_model=fit_constant_turn_rate_model,
        fit_kwargs={
            "speed_prior_log_sd": 0.5,
            "heading_prior_scale": 0.5,
            "turn_rate_prior_scale": 0.01,
            "sigma_prior_scale": 20.0,
        },
    ),
    ModelSpec(
        name="Constant Turn Rate and Acceleration",
        fit_model=fit_constant_turn_rate_acceleration_model,
        fit_kwargs={
            "speed_prior_log_sd": 0.5,
            "heading_prior_scale": 0.5,
            "turn_rate_prior_scale": 0.01,
            "acceleration_prior_scale": 0.05,
            "sigma_prior_scale": 20.0,
        },
    ),
    ModelSpec(
        name="Time-Varying Radius",
        fit_model=fit_time_varying_radius_model,
        fit_kwargs={
            "radius_prior_median": 500.0,
            "curvature_initial_prior_scale": 0.002,
            "curvature_rate_prior_scale": 5e-6,
            "sigma_prior_scale": 20.0,
            "integration_substeps": 4,
        },
    ),
    ModelSpec(
        name="Time-Varying Motion",
        fit_model=fit_time_varying_motion_model,
        fit_kwargs={
            "speed_prior_log_sd": 0.5,
            "heading_prior_scale": 0.5,
            "acceleration_initial_scale": 0.1,
            "acceleration_state_scale": 0.02,
            "acceleration_decay_time": 60.0,
            "turn_rate_initial_scale": 0.01,
            "turn_rate_state_scale": 0.003,
            "turn_rate_decay_time": 600.0,
            "sigma_position": 5.0,
            "sigma_speed": 0.2,
        },
    ),
)


def evaluate_models(
    trajectory_data,
    model_specs=MODEL_SPECS,
    *,
    observation_count=OBSERVATION_COUNT,
    prediction_count=PREDICTION_COUNT,
    start_index=START_INDEX,
    credible_interval=CREDIBLE_INTERVAL,
    show_progress=False,
):
    """Fit and evaluate every model on the exact same held-out timestamps."""
    window = prepare_trajectory_window(
        trajectory_data,
        observation_count=observation_count,
        prediction_count=prediction_count,
        start_index=start_index,
    )
    summary_rows = []
    horizon_tables = []

    for model_spec in model_specs:
        print(f"Fitting {model_spec.name} ...")
        fit = model_spec.fit_model(
            window,
            show_progress=show_progress,
            **model_spec.fit_kwargs,
        )
        evaluation = evaluate_position_predictions(
            fit,
            window,
            credible_interval=credible_interval,
        )

        summary_rows.append(
            {
                "model": model_spec.name,
                "ade_m": evaluation.ade_m,
                "fde_m": evaluation.fde_m,
                "radial_coverage": evaluation.radial_coverage,
                "mean_prediction_radius_m": (evaluation.mean_prediction_radius_m),
                "mean_marginal_interval_width_m": (
                    evaluation.mean_marginal_interval_width_m
                ),
            }
        )
        horizon_table = evaluation.prediction_table[
            [
                "horizon_seconds",
                "position_error_m",
                "prediction_radius_m",
                "radial_covered",
            ]
        ].copy()
        horizon_table.insert(0, "model", model_spec.name)
        horizon_tables.append(horizon_table)

    return pd.DataFrame(summary_rows), pd.concat(horizon_tables, ignore_index=True)


def main():
    """Run and print the shared one-window model comparison."""
    trajectory_data = read_ship_data(DATA_FILE, run_id=RUN_ID)
    summary, per_horizon = evaluate_models(trajectory_data)

    print(
        "\nAll models are scored on identical held-out positions; their fitted "
        "inputs and motion assumptions remain model-specific."
    )
    print("\nModel comparison (sorted by ADE):")
    print(summary.sort_values("ade_m").round(3).to_string(index=False))
    print("\nPer-horizon comparison:")
    print(per_horizon.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
