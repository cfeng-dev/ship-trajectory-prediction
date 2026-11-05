"""Add time-aligned evaluation labels to the IMM-tracked BR24 trajectory."""

import bayestraj.observations.br24 as br24
from bayestraj.paths import project_path

RADAR_CSV = project_path("data/processed/BR24_imm_tracked_trajectory.csv")
REFERENCE_CSV = project_path("data/raw/BR24_reference_trajectory.csv")
OUTPUT_CSV = project_path(
    "data/processed/BR24_imm_tracked_trajectory_with_reference.csv"
)


def main() -> None:
    """Export the IMM track with interpolated target labels for evaluation."""
    output_path = br24.export_br24_target_radar_with_ground_truth(
        radar_csv=RADAR_CSV,
        reference_csv=REFERENCE_CSV,
        output_csv=OUTPUT_CSV,
    )
    print(f"Exported IMM-tracked BR24 trajectory with reference: {output_path}")


if __name__ == "__main__":
    main()
