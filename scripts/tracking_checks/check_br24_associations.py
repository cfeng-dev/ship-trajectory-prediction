"""Compare IMM radar-detection choices with one Bayesian inference method.

Only accepted detection IDs are compared; posterior track positions are not.
For a short run: --method smc --max-cycles 20.
"""

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bayestraj import numeric_validation
from bayestraj.inference import configuration as inference
from bayestraj.inference.br24_radar_tracker import BR24BayesianRadarTracker
from bayestraj.inference.ctrv_sequential_vi import SequentialVIConfig
from bayestraj.inference.ctrv_smc import SequentialMonteCarloCTRVConfig
from bayestraj.models.bayesian_ctrv import MIN_OBSERVATION_COUNT, BayesianCTRVPriors
from bayestraj.observations.br24 import (
    br24_measurement_position,
    read_br24_tracking_input,
)
from bayestraj.paths import project_path

RAW_INPUT_CSV = project_path("data/processed/BR24Dataset20131107.csv")
IMM_TRACKED_TRAJECTORY_CSV = project_path(
    "data/processed/BR24_imm_tracked_trajectory.csv"
)
OUTPUT_DIR = project_path("results/validation")
INFERENCE_METHOD = "smc"  # "smc", "mcmc", "vi", or "seq_vi".
OBSERVATION_COUNT = inference.DEFAULT_CTRV_ROLLING_OBSERVATION_COUNT
INFERENCE_SEED = 42
MAX_CYCLES = None


def compare_associations(
    *,
    raw_csv: str | Path,
    imm_csv: str | Path,
    output_csv: str | Path,
    priors: BayesianCTRVPriors | None = None,
    config: SequentialMonteCarloCTRVConfig | None = None,
    seed: int = 42,
    inference_method: str = "smc",
    observation_count: int = inference.DEFAULT_CTRV_ROLLING_OBSERVATION_COUNT,
    vi_config: Mapping[str, Any] | None = None,
    mcmc_config: Mapping[str, Any] | None = None,
    sequential_vi_config: SequentialVIConfig | None = None,
    max_cycles: int | None = None,
    bayesian_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Write one IMM-versus-Bayesian association decision for every raw cycle.

    Both trackers start as the MATLAB IMM does: at the lowest cycle counter,
    with the lowest detection ID in that cycle. The IMM association CSV normally
    starts at its second cycle, so its initial selection is reconstructed here.
    Supply the complete IMM association export for these raw cycles: absent
    rows are interpreted as IMM misses, not as an incomplete recording.
    With bayesian_csv, compare saved decisions on their recorded cycle prefix
    without constructing a tracker or running inference.
    """
    inference_mode, inference_method = inference.normalize_inference_method(
        inference_method
    )
    observation_count = numeric_validation.validate_non_negative_integer(
        "observation_count", observation_count
    )
    if observation_count < MIN_OBSERVATION_COUNT:
        raise ValueError(f"observation_count must be at least {MIN_OBSERVATION_COUNT}.")
    if max_cycles is not None:
        max_cycles = numeric_validation.validate_non_negative_integer(
            "max_cycles", max_cycles
        )
        if max_cycles < 1:
            raise ValueError("max_cycles must be positive.")
    cycles = read_br24_tracking_input(raw_csv)
    initial_cycle = cycles[0]
    if not initial_cycle.detections:
        raise ValueError(
            "The earliest BR24 radar cycle has no detection to initialize."
        )
    initial_detection_id = initial_cycle.detections[0].detection_id
    imm_detection_ids = _read_imm_detection_ids(imm_csv, cycles=cycles)
    if (
        imm_detection_ids.get(initial_cycle.cycle_counter, initial_detection_id)
        != initial_detection_id
    ):
        raise ValueError("IMM initial detection does not match the common BR24 seed.")
    imm_detection_ids.setdefault(initial_cycle.cycle_counter, initial_detection_id)
    if bayesian_csv is not None:
        cycles, bayesian_detection_ids = _read_saved_associations(bayesian_csv, cycles)
        if bayesian_detection_ids[initial_cycle.cycle_counter] != initial_detection_id:
            raise ValueError(
                "Bayesian initial detection does not match the common BR24 seed."
            )
    if max_cycles is not None:
        cycles = cycles[:max_cycles]

    if bayesian_csv is None:
        # Use the same shared defaults and full-rank VI settings as the evaluator.
        vi_options = inference.create_default_vi_config()
        vi_options.update(vi_config or {})
        mcmc_options = inference.create_default_mcmc_config()
        mcmc_options.update(mcmc_config or {})
        _, vi_options = inference.select_inference_config(
            "batch",
            "vi",
            vi_algorithm=vi_options["algorithm"],
            require_converged=vi_options["require_converged"],
            vi_config=vi_options,
            mcmc_config=mcmc_options,
            fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
        )
        tracker = BR24BayesianRadarTracker.initialize(
            initial_cycle,
            initial_detection_id=initial_detection_id,
            priors=BayesianCTRVPriors() if priors is None else priors,
            config=config,
            seed=seed,
            inference_method=inference_method,
            observation_count=observation_count,
            vi_config=vi_options,
            mcmc_config=mcmc_options,
            sequential_vi_config=sequential_vi_config,
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
    comparison["inference_method"] = inference_method
    comparison["configured_observation_count"] = (
        observation_count
        if inference_mode == "batch" and bayesian_csv is None
        else np.nan
    )
    comparison["inference_seed"] = seed if bayesian_csv is None else np.nan
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return comparison


def _read_saved_associations(path, cycles):
    """Require every consumed cycle, including explicit misses, in source order."""
    data = pd.read_csv(path)
    required = {"time", "cycle_counter", "detection_id", "x", "y"}
    if data.empty or not required.issubset(data.columns) or len(data) > len(cycles):
        raise ValueError(
            "Bayesian association CSV must contain recorded radar cycles and raw fixes."
        )
    recorded = cycles[: len(data)]
    if not np.array_equal(
        data["cycle_counter"], [cycle.cycle_counter for cycle in recorded]
    ):
        raise ValueError(
            "Bayesian association cycles must match a complete prefix of the raw input."
        )
    if not np.allclose(
        data["time"], [cycle.time_seconds for cycle in recorded], rtol=0.0, atol=1e-6
    ):
        raise ValueError("Bayesian association times do not match the raw input.")
    identities = {}
    for row, cycle in zip(data.itertuples(index=False), recorded, strict=True):
        if pd.isna(row.detection_id):
            if not pd.isna([row.x, row.y]).all():
                raise ValueError(
                    "A missed Bayesian detection must have empty radar coordinates."
                )
            identity = None
        else:
            if not np.isfinite(row.detection_id) or row.detection_id != int(
                row.detection_id
            ):
                raise ValueError("Bayesian detection IDs must be integers or empty.")
            identity = int(row.detection_id)
            detection = next(
                (d for d in cycle.detections if d.detection_id == identity), None
            )
            if detection is None:
                raise ValueError(
                    f"Bayesian detection {identity} is absent from cycle {cycle.cycle_counter}."
                )
            position = br24_measurement_position(
                sensor_position=cycle.sensor_position,
                range_m=detection.range_m,
                azimuth_rad=detection.azimuth_rad,
            )
            if not np.allclose([row.x, row.y], position, rtol=0.0, atol=1e-6):
                raise ValueError(
                    "Bayesian radar coordinates do not match the selected detection."
                )
        identities[cycle.cycle_counter] = identity
    return recorded, identities


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
    detection_ids = dict(
        zip(
            imm_data["cycle_counter"],
            imm_data["detection_id"],
            strict=True,
        )
    )
    available_detections = {
        cycle.cycle_counter: {d.detection_id for d in cycle.detections}
        for cycle in cycles
    }
    for counter, identity in detection_ids.items():
        if identity not in available_detections[counter]:
            raise ValueError(
                f"IMM detection {identity} is absent from BR24 cycle {counter}."
            )
    return detection_ids


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


def print_association_summary(comparison, *, inference_method):
    """Report exact detection-ID agreement, including mutual misses."""
    if comparison.empty:
        raise ValueError("No BR24 cycles available for comparison.")
    different = ~comparison["association_status"].isin(
        {"same_detection", "both_missed"}
    )
    verdict = "DIFFERENT" if different.any() else "MATCH"
    print(f"\nIMM vs {inference_method.upper()} associations: {verdict}")
    print(
        f"Compared {len(comparison)} cycles (including the common initial detection)."
    )
    counts = comparison["association_status"].value_counts()
    for status in (
        "same_detection",
        "different_detection",
        "both_missed",
        "imm_only",
        "bayesian_only",
    ):
        print(f"{status}: {counts.get(status, 0)}")
    print(
        f"Different association decisions: {different.sum()}/{len(comparison)} cycles"
    )
    if different.any():
        counters = comparison.loc[different, "cycle_counter"].iloc[:10]
        print(f"First differing cycles: {', '.join(str(value) for value in counters)}")


def main(argv=None) -> pd.DataFrame:
    """Run the selected causal tracker and export its association audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        default=INFERENCE_METHOD,
        choices=(*inference.CTRV_ROLLING_INFERENCE_METHODS, "seq_vi"),
    )
    parser.add_argument("--raw-csv", type=Path, default=RAW_INPUT_CSV)
    parser.add_argument("--imm-csv", type=Path, default=IMM_TRACKED_TRAJECTORY_CSV)
    parser.add_argument(
        "--bayesian-csv",
        type=Path,
        help="Read an exported association CSV without running inference.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--observations", type=int, default=OBSERVATION_COUNT)
    parser.add_argument("--seed", type=int, default=INFERENCE_SEED)
    parser.add_argument("--max-cycles", type=int, default=MAX_CYCLES)
    options = parser.parse_args(argv)
    _, method = inference.normalize_inference_method(options.method)
    name = "seq_vi" if method == "sequential_vi" else method
    output = options.output_dir / f"BR24_association_comparison_{name}.csv"
    print(f"\nRunning BR24 association check: {method.upper()}", flush=True)
    comparison = compare_associations(
        raw_csv=options.raw_csv,
        imm_csv=options.imm_csv,
        output_csv=output,
        inference_method=method,
        observation_count=options.observations,
        seed=options.seed,
        max_cycles=options.max_cycles,
        bayesian_csv=options.bayesian_csv,
    )
    print_association_summary(comparison, inference_method=method)
    print(f"Saved association comparison: {output}")
    return comparison


if __name__ == "__main__":
    main()
