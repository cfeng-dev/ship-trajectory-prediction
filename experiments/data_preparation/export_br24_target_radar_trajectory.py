"""Export BR24 radar observations selected by Reuter's IMM association."""

import bayestraj.observations.br24 as br24
import bayestraj.observations.paths as paths

RADAR_CSV = paths.data_path("raw/BR24_radar_detections.csv")
SENSOR_CSV = paths.data_path("raw/BR24_sensor_state.csv")
OUTPUT_CSV = paths.data_path("processed/BR24_target_radar_trajectory.csv")


def main() -> None:
    """Export the local BR24 target radar trajectory without ground-truth input."""
    output_path = br24.export_br24_target_radar_trajectory(
        radar_csv=RADAR_CSV,
        sensor_csv=SENSOR_CSV,
        output_csv=OUTPUT_CSV,
    )
    print(f"Exported associated BR24 radar trajectory: {output_path}")


if __name__ == "__main__":
    main()
