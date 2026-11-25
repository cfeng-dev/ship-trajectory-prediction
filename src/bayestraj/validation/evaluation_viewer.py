"""Read-only, time-aligned spatial and metric data for saved evaluations."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bayestraj.validation import comparison, runtime_plotting

STUDENT_T_COLUMN = "student_t_negative_log_predictive_density"


@dataclass(frozen=True)
class EvaluatedModel:
    """Cache one saved run's point forecasts and per-origin metrics."""

    color: str
    point_estimate: str
    metrics: dict[str, np.ndarray]
    forecasts: tuple[np.ndarray, ...]
    origin_positions: np.ndarray | None


@dataclass(frozen=True)
class EvaluationDataset:
    """A common forecast cohort with a deduplicated evaluation-only reference."""

    models: dict[str, EvaluatedModel]
    origins: np.ndarray
    reference_times: np.ndarray
    reference_positions: np.ndarray
    horizon_seconds: tuple[np.ndarray, ...]
    prediction_counts: np.ndarray
    available_metrics: tuple[str, ...]

    @property
    def elapsed(self):
        return self.origins - self.origins[0]

    def nearest_index(self, elapsed_seconds):
        """Resolve a metric click to the nearest saved physical origin."""
        return int(np.argmin(np.abs(self.elapsed - float(elapsed_seconds))))


def _finite_coordinates(table, columns, label):
    try:
        values = (
            table[list(columns)]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=float)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{label}: missing or invalid coordinate columns {columns}."
        ) from error
    if not np.isfinite(values).all():
        raise ValueError(f"{label}: coordinate geometry must be finite ({columns}).")
    return values


def _reference_route(times, positions):
    """Retain saved target points; never interpolate or infer missing reference."""
    order = np.argsort(times, kind="stable")
    times, positions = times[order], positions[order]
    retained = [0]
    for index in range(1, len(times)):
        previous = retained[-1]
        if np.isclose(times[index], times[previous], rtol=0, atol=1e-6):
            if not np.allclose(
                positions[index], positions[previous], rtol=0, atol=1e-6
            ):
                raise ValueError(
                    "Conflicting reference positions at the same target time."
                )
        else:
            retained.append(index)
    return times[retained], positions[retained]


def load_evaluation_directory(directory: str | Path) -> EvaluationDataset:
    """Reuse the comparison CSV contract and aggregate each complete forecast."""
    tables = comparison.read_model_comparison_directory(directory)
    first = next(iter(tables.values()))
    origins = np.unique(first["forecast_origin_seconds"].to_numpy(dtype=float))
    use_route = all("x_actual_route" in table for table in tables.values())
    if not use_route and any(
        "x_actual_route" in table or "y_actual_route" in table
        for table in tables.values()
    ):
        raise ValueError("All models need matching route coordinate columns.")
    suffix = "_route" if use_route else ""
    actual_columns = (f"x_actual{suffix}", f"y_actual{suffix}")
    reference = _finite_coordinates(first, actual_columns, "Reference")
    target_times = first["forecast_origin_seconds"].to_numpy(dtype=float) + first[
        "horizon_seconds"
    ].to_numpy(dtype=float)
    reference_times, reference_positions = _reference_route(target_times, reference)
    student_t_available = all(
        STUDENT_T_COLUMN in table
        and np.isfinite(
            pd.to_numeric(table[STUDENT_T_COLUMN], errors="coerce").to_numpy(
                dtype=float
            )
        ).all()
        for table in tables.values()
    )
    models = {}
    colors = runtime_plotting._comparison_series_colors(list(tables.values()))
    for (label, table), color in zip(tables.items(), colors, strict=True):
        actual = _finite_coordinates(table, actual_columns, label)
        if not np.allclose(reference, actual, rtol=0, atol=1e-6):
            raise ValueError(f"{label}: models must share reference route coordinates.")
        point_estimate = comparison._point_estimate_name(table)
        columns = (f"x_{point_estimate}{suffix}", f"y_{point_estimate}{suffix}")
        point_positions = _finite_coordinates(table, columns, label)
        route_errors = np.linalg.norm(point_positions - actual, axis=1)
        if not np.allclose(
            route_errors, table["position_error_m"], rtol=1e-8, atol=1e-6
        ):
            raise ValueError(
                f"{label}: route forecasts must match the saved position errors."
            )
        model_origins, metrics = runtime_plotting._rolling_window_metric_series(
            table, student_t_nlpd=student_t_available
        )
        if not np.array_equal(model_origins, origins):
            raise ValueError(f"{label}: metric origins must match forecast origins.")
        groups = tuple(
            group.sort_values("horizon_step")
            for _, group in table.groupby("forecast_origin_seconds", sort=True)
        )
        forecasts = tuple(
            _finite_coordinates(group, columns, label) for group in groups
        )
        origin_columns = ("forecast_origin_x_route", "forecast_origin_y_route")
        positions = None
        if any(column in table for column in origin_columns):
            _finite_coordinates(table, origin_columns, label)
            positions = []
            for group in groups:
                values = _finite_coordinates(group, origin_columns, label)
                if not np.allclose(values, values[0], rtol=0, atol=1e-6):
                    raise ValueError(
                        f"{label}: origin coordinates must be constant within a forecast."
                    )
                positions.append(values[0])
            positions = np.asarray(positions)
        models[label] = EvaluatedModel(
            color, point_estimate, metrics, forecasts, positions
        )
    groups = tuple(
        group for _, group in first.groupby("forecast_origin_seconds", sort=True)
    )
    return EvaluationDataset(
        models=models,
        origins=origins,
        reference_times=reference_times,
        reference_positions=reference_positions,
        horizon_seconds=tuple(
            group["horizon_seconds"].to_numpy(dtype=float) for group in groups
        ),
        prediction_counts=np.array([len(group) for group in groups]),
        available_metrics=tuple(next(iter(models.values())).metrics),
    )
