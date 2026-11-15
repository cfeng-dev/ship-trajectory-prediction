"""Export simulated positions and aligned evaluation-only reference states."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .config import DEFAULT_CSV_EXPORT_INTERVAL_SECONDS

DATA_DIR = paths.default_simulation_data_directory()
TRAJECTORY_COLUMNS = ["time", "x", "y"]
GROUND_TRUTH_COLUMNS = [
    "ground_truth_x",
    "ground_truth_y",
    "ground_truth_velocity",
    "ground_truth_yaw",
    "ground_truth_turn_rate",
]
SIMULATION_COLUMNS = [*TRAJECTORY_COLUMNS, *GROUND_TRUTH_COLUMNS]


@dataclass(frozen=True)
class TrajectorySaveResult:
    """Destination and continuation status of a saved simulation."""

    output_path: Path
    appended: bool


def _validate_trajectory(data):
    """Validate one ordered trajectory or a continuation without resetting time."""
    if data.columns.tolist() not in (TRAJECTORY_COLUMNS, SIMULATION_COLUMNS):
        raise ValueError(
            "Trajectory CSV must contain time,x,y columns, optionally followed "
            "by the complete simulator ground-truth columns."
        )
    trajectory = data.apply(pd.to_numeric, errors="raise").astype(float)
    if trajectory.empty or not np.isfinite(trajectory.to_numpy()).all():
        raise ValueError("Trajectory values must be finite and non-empty.")
    if np.any(np.diff(trajectory["time"].to_numpy()) <= 0.0):
        raise ValueError("Trajectory times must be strictly increasing.")
    return trajectory


def create_simulation_dataframe(
    simulator, *, interval_seconds=DEFAULT_CSV_EXPORT_INTERVAL_SECONDS
):
    """Select original positions and true states on the same fixed time grid."""
    if isinstance(interval_seconds, bool):
        raise ValueError("CSV export interval must be positive and finite.")
    interval = float(interval_seconds)
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("CSV export interval must be positive and finite.")
    heading = np.asarray(simulator.theta_all, dtype=float)
    trajectory = _validate_trajectory(
        pd.DataFrame(
            {
                "time": np.asarray(simulator.t_all, dtype=float),
                "x": np.asarray(simulator.x_all, dtype=float),
                "y": np.asarray(simulator.y_all, dtype=float),
                "ground_truth_x": np.asarray(simulator.x_all, dtype=float),
                "ground_truth_y": np.asarray(simulator.y_all, dtype=float),
                "ground_truth_velocity": np.asarray(simulator.v_all, dtype=float),
                "ground_truth_yaw": np.arctan2(np.sin(heading), np.cos(heading)),
                "ground_truth_turn_rate": np.asarray(simulator.omega_all, dtype=float),
            }
        )
    )
    times = trajectory["time"].to_numpy()
    last_grid_index = (times[-1] - times[0]) / interval
    tolerance = 1e-9 * max(1.0, abs(last_grid_index))
    sample_count = int(np.floor(last_grid_index + tolerance)) + 1
    targets = times[0] + np.arange(sample_count) * interval
    right = np.minimum(np.searchsorted(times, targets), len(times) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(
        np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right
    )
    if not np.allclose(times[indices], targets, rtol=1e-9, atol=1e-9) or np.any(
        np.diff(indices) <= 0
    ):
        raise ValueError(
            "CSV export interval must align with the recorded simulation steps."
        )
    return trajectory.iloc[indices].reset_index(drop=True)


def save_trajectory_data(df, filename, *, append=False):
    """Save one run or append new samples from the same active simulation."""
    trajectory = _validate_trajectory(df)
    output_path = Path(filename).expanduser()
    if not output_path.is_absolute():
        output_path = DATA_DIR / output_path

    has_existing_data = output_path.is_file() and output_path.stat().st_size > 0
    if append:
        if not has_existing_data:
            raise ValueError(
                "Cannot continue a trajectory because its CSV is missing or empty."
            )
        existing = _validate_trajectory(
            pd.read_csv(output_path, float_precision="round_trip")
        )
        if existing.columns.tolist() != trajectory.columns.tolist():
            raise ValueError(
                "Cannot append trajectory data with different CSV columns. "
                "Choose a new filename for the ground-truth export."
            )
        if trajectory["time"].iloc[0] <= existing["time"].iloc[-1]:
            raise ValueError(
                "New trajectory samples must follow the previously saved times."
            )
        mode = "a"
    else:
        if has_existing_data:
            raise FileExistsError(
                "This file already contains a trajectory. Choose a new filename for a new simulation run."
            )
        mode = "w" if output_path.is_file() else "x"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if append:
        with output_path.open("rb") as existing_file:
            existing_file.seek(-1, 2)
            needs_newline = existing_file.read(1) not in {b"\n", b"\r"}
    with output_path.open(mode, encoding="utf-8", newline="") as output_file:
        if needs_newline:
            output_file.write("\n")
        trajectory.to_csv(output_file, header=not append, index=False)
    print(f"Saved simulated trajectory to: {output_path}")
    return TrajectorySaveResult(output_path=output_path, appended=append)
