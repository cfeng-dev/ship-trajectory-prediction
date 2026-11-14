"""Export BR24 tracking input with evaluation-only reference labels."""

from pathlib import Path
from tempfile import TemporaryDirectory

import bayestraj.observations.br24 as br24
from bayestraj.paths import project_path

RADAR_CSV = project_path("data/raw/BR24_radar_detections.csv")
SENSOR_CSV = project_path("data/raw/BR24_sensor_state.csv")
REFERENCE_CSV = project_path("data/raw/BR24_reference_trajectory.csv")
OUTPUT_CSV = project_path("data/processed/BR24Dataset20131107.csv")


def main() -> None:
    """Export raw radar cycles and aligned reference columns in one run."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT_CSV.parent) as temporary_directory:
        temporary_csv = Path(temporary_directory) / OUTPUT_CSV.name
        br24.export_br24_tracking_input(
            radar_csv=RADAR_CSV,
            sensor_csv=SENSOR_CSV,
            output_csv=temporary_csv,
        )
        br24.export_br24_tracking_input_with_reference(
            tracking_csv=temporary_csv,
            reference_csv=REFERENCE_CSV,
            output_csv=temporary_csv,
        )
        temporary_csv.replace(OUTPUT_CSV)
    print(f"Exported BR24 tracking input with reference: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
