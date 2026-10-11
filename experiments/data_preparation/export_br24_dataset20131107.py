"""Flatten BR24 sensor cycles and radar detections into one tracking-input CSV."""

import bayestraj.observations.br24 as br24
import bayestraj.observations.paths as paths

RADAR_CSV = paths.data_path("raw/BR24_radar_detections.csv")
SENSOR_CSV = paths.data_path("raw/BR24_sensor_state.csv")
OUTPUT_CSV = paths.data_path("processed/BR24Dataset20131107.csv")


def main() -> None:
    """Export the model-neutral BR24 tracking input."""
    output_path = br24.export_br24_tracking_input(
        radar_csv=RADAR_CSV,
        sensor_csv=SENSOR_CSV,
        output_csv=OUTPUT_CSV,
    )
    print(f"Exported BR24 tracking input: {output_path}")


if __name__ == "__main__":
    main()
