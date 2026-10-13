"""Export the BR24 reference state data as a position trajectory."""

import bayestraj.observations.io as observations_io
import bayestraj.observations.paths as paths

INPUT_CSV = paths.data_path("raw/BR24_reference_trajectory.csv")
SENSOR_CSV = paths.data_path("raw/BR24_sensor_state.csv")
OUTPUT_CSV = paths.data_path("processed/BR24_reference_trajectory.csv")


def main() -> None:
    """Export the reference trajectory restricted to the BR24 radar window."""
    output_path = observations_io.export_br24_reference_trajectory(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        sensor_csv=SENSOR_CSV,
    )
    print(f"Exported BR24 reference trajectory: {output_path}")


if __name__ == "__main__":
    main()
