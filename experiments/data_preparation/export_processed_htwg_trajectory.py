"""Export the HTWG reference trajectory as position-only processed data."""

from pathlib import Path

import numpy as np
import pandas as pd

import bayestraj.observations.paths as paths

INPUT_CSV = paths.data_path("raw/ship_data_htwg.csv")
OUTPUT_CSV = paths.data_path("processed/ship_trajectory_htwg.csv")
POSITION_COLUMNS = ["time", "x", "y"]


def export_htwg_trajectory(input_csv, output_csv):
    """Write the HTWG x/y positions with elapsed seconds from the first sample."""
    source_data = pd.read_csv(input_csv)
    missing_columns = [
        column for column in POSITION_COLUMNS if column not in source_data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    trajectory_data = source_data[POSITION_COLUMNS].copy()
    if len(trajectory_data) < 2:
        raise ValueError("The HTWG trajectory must contain at least two samples.")
    trajectory_data = trajectory_data.apply(pd.to_numeric, errors="coerce")
    if not np.all(np.isfinite(trajectory_data["time"].to_numpy(dtype=float))):
        raise ValueError("time must contain finite numeric values.")

    finite_positions = np.isfinite(
        trajectory_data[["x", "y"]].to_numpy(dtype=float)
    ).all(axis=1)
    trajectory_data = trajectory_data.loc[finite_positions].copy()
    if len(trajectory_data) < 2:
        raise ValueError(
            "The HTWG trajectory must contain at least two finite positions."
        )

    trajectory_data = trajectory_data.sort_values("time").reset_index(drop=True)
    if np.any(np.diff(trajectory_data["time"].to_numpy(dtype=float)) <= 0.0):
        raise ValueError("Source times must be strictly increasing after sorting.")
    trajectory_data["time"] = (
        trajectory_data["time"] - trajectory_data["time"].iloc[0]
    ).round(1)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    trajectory_data.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    """Export the configured HTWG raw trajectory."""
    output_path = export_htwg_trajectory(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
    )
    print(f"Exported processed HTWG trajectory: {output_path}")


if __name__ == "__main__":
    main()
