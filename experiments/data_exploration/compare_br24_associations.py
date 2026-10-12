"""Compare IMM and Bayesian BR24 target-association decisions cycle by cycle."""

from pathlib import Path

import pandas as pd

from bayestraj.inference.br24_radar_smc import BR24BayesianRadarTracker
from bayestraj.inference.ctrv_smc import SequentialMonteCarloCTRVConfig
from bayestraj.models.bayesian_ctrv import BayesianCTRVPriors
from bayestraj.observations.br24 import read_br24_tracking_input
from bayestraj.observations.paths import data_path

RAW_INPUT_CSV = data_path("processed/BR24Dataset20131107.csv")
IMM_TRACKED_TRAJECTORY_CSV = data_path("processed/BR24_imm_tracked_trajectory.csv")
OUTPUT_CSV = data_path("processed/BR24_association_comparison.csv")


def compare_associations(
    *,
    raw_csv: str | Path,
    imm_csv: str | Path,
    output_csv: str | Path,
    priors: BayesianCTRVPriors | None = None,
    config: SequentialMonteCarloCTRVConfig | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Write one IMM-versus-Bayesian association decision for every raw cycle.

    Both trackers start as the MATLAB IMM does: at the lowest cycle counter,
    with the lowest detection ID in that cycle. The IMM association CSV normally
    starts at its second cycle, so its initial selection is reconstructed here.
    """
    cycles = read_br24_tracking_input(raw_csv)
    initial_cycle = cycles[0]
    if not initial_cycle.detections:
        raise ValueError(
            "The earliest BR24 radar cycle has no detection to initialize."
        )
    initial_detection_id = initial_cycle.detections[0].detection_id
    imm_detection_ids = _read_imm_detection_ids(imm_csv, cycles=cycles)
    imm_detection_ids.setdefault(initial_cycle.cycle_counter, initial_detection_id)

    tracker = BR24BayesianRadarTracker.initialize(
        initial_cycle,
        initial_detection_id=initial_detection_id,
        priors=BayesianCTRVPriors() if priors is None else priors,
        config=config,
        seed=seed,
    )
    bayesian_detection_ids = {initial_cycle.cycle_counter: initial_detection_id}
    for cycle in cycles[1:]:
        step = tracker.process_cycle(cycle)
        bayesian_detection_ids[cycle.cycle_counter] = step.detection_id

    comparison = pd.DataFrame(
        {
            "time": [cycle.time_seconds for cycle in cycles],
            "cycle_counter": [cycle.cycle_counter for cycle in cycles],
            "imm_detection_id": pd.array(
                [imm_detection_ids.get(cycle.cycle_counter) for cycle in cycles],
                dtype="Int64",
            ),
            "bayesian_detection_id": pd.array(
                [bayesian_detection_ids[cycle.cycle_counter] for cycle in cycles],
                dtype="Int64",
            ),
        }
    )
    comparison["association_status"] = [
        _association_status(imm_detection_id, bayesian_detection_id)
        for imm_detection_id, bayesian_detection_id in zip(
            comparison["imm_detection_id"],
            comparison["bayesian_detection_id"],
            strict=True,
        )
    ]
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return comparison


def _read_imm_detection_ids(imm_csv, *, cycles) -> dict[int, int]:
    """Read the IMM's accepted detection ID per cycle and validate its domain."""
    imm_data = pd.read_csv(imm_csv)
    required_columns = {"cycle_counter", "detection_id"}
    missing_columns = sorted(required_columns.difference(imm_data.columns))
    if missing_columns:
        raise ValueError(
            "IMM association CSV is missing required columns: "
            f"{', '.join(missing_columns)}."
        )
    if imm_data[["cycle_counter", "detection_id"]].isna().any(axis=None):
        raise ValueError(
            "IMM association cycle counters and detection IDs must be present."
        )
    imm_data = imm_data.loc[:, ["cycle_counter", "detection_id"]].copy()
    for column in ("cycle_counter", "detection_id"):
        values = pd.to_numeric(imm_data[column], errors="raise")
        if not (values == values.astype(int)).all():
            raise ValueError(f"IMM association {column} values must be integers.")
        imm_data[column] = values.astype(int)
    if imm_data["cycle_counter"].duplicated().any():
        raise ValueError("IMM association CSV contains duplicate cycle counters.")
    available_cycles = {cycle.cycle_counter for cycle in cycles}
    unexpected_cycles = sorted(
        set(imm_data["cycle_counter"]).difference(available_cycles)
    )
    if unexpected_cycles:
        raise ValueError(
            "IMM association CSV contains cycle counters absent from BR24 data."
        )
    return dict(
        zip(
            imm_data["cycle_counter"],
            imm_data["detection_id"],
            strict=True,
        )
    )


def _association_status(imm_detection_id, bayesian_detection_id) -> str:
    """Classify agreement, a mutual miss, or one tracker selecting another fix."""
    if pd.isna(imm_detection_id) and pd.isna(bayesian_detection_id):
        return "both_missed"
    if pd.isna(imm_detection_id):
        return "bayesian_only"
    if pd.isna(bayesian_detection_id):
        return "imm_only"
    if imm_detection_id == bayesian_detection_id:
        return "same_detection"
    return "different_detection"


def main() -> None:
    """Create the configured BR24 association-audit CSV and print its summary."""
    comparison = compare_associations(
        raw_csv=RAW_INPUT_CSV,
        imm_csv=IMM_TRACKED_TRAJECTORY_CSV,
        output_csv=OUTPUT_CSV,
    )
    counts = comparison["association_status"].value_counts()
    print(f"Exported BR24 association comparison: {OUTPUT_CSV}")
    for status in (
        "same_detection",
        "different_detection",
        "both_missed",
        "imm_only",
        "bayesian_only",
    ):
        print(f"{status}: {counts.get(status, 0)}")


if __name__ == "__main__":
    main()
