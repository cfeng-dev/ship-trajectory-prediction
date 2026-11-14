"""Export the IMM-associated BR24 trajectory with evaluation reference labels."""

from pathlib import Path
from tempfile import TemporaryDirectory

import bayestraj.observations.br24 as br24
from bayestraj.paths import project_path

RADAR_CSV = project_path("data/raw/BR24_radar_detections.csv")
SENSOR_CSV = project_path("data/raw/BR24_sensor_state.csv")
REFERENCE_CSV = project_path("data/raw/BR24_reference_trajectory.csv")
OUTPUT_CSV = project_path("data/processed/BR24_imm_tracked_trajectory.csv")


def main() -> None:
    """Export associated fixes and aligned reference columns in one run."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT_CSV.parent) as temporary_directory:
        temporary_csv = Path(temporary_directory) / OUTPUT_CSV.name
        br24.export_br24_target_radar_trajectory(
            radar_csv=RADAR_CSV,
            sensor_csv=SENSOR_CSV,
            output_csv=temporary_csv,
        )
        br24.export_br24_target_radar_with_ground_truth(
            radar_csv=temporary_csv,
            reference_csv=REFERENCE_CSV,
            output_csv=temporary_csv,
        )
        temporary_csv.replace(OUTPUT_CSV)
    print(f"Exported IMM-tracked BR24 trajectory with reference: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
