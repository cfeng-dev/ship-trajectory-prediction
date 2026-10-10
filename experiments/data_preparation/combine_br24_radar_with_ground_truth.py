"""Add time-aligned evaluation-only ground truth to BR24 radar positions."""

import bayestraj.observations.br24 as br24
import bayestraj.observations.paths as paths

RADAR_CSV = paths.data_path("processed/BR24_target_radar_trajectory.csv")
REFERENCE_CSV = paths.data_path("raw/BR24_target_reference_trajectory.csv")
OUTPUT_CSV = paths.data_path(
    "processed/BR24_target_radar_trajectory_with_ground_truth.csv"
)


def main() -> None:
    """Export radar positions with interpolated target reference for evaluation."""
    output_path = br24.export_br24_target_radar_with_ground_truth(
        radar_csv=RADAR_CSV,
        reference_csv=REFERENCE_CSV,
        output_csv=OUTPUT_CSV,
    )
    print(f"Exported BR24 radar trajectory with ground truth: {output_path}")


if __name__ == "__main__":
    main()
