"""Export the BR24 target trajectory tracked by Reuter's IMM association."""

import bayestraj.observations.br24 as br24
from bayestraj.paths import project_path

RADAR_CSV = project_path("data/raw/BR24_radar_detections.csv")
SENSOR_CSV = project_path("data/raw/BR24_sensor_state.csv")
OUTPUT_CSV = project_path("data/processed/BR24_imm_tracked_trajectory.csv")


def main() -> None:
    """Export the local IMM-tracked BR24 trajectory without ground truth."""
    output_path = br24.export_br24_target_radar_trajectory(
        radar_csv=RADAR_CSV,
        sensor_csv=SENSOR_CSV,
        output_csv=OUTPUT_CSV,
    )
    print(f"Exported IMM-tracked BR24 trajectory: {output_path}")


if __name__ == "__main__":
    main()
