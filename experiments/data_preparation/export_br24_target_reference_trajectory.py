"""Export the BR24 target reference state data as a position trajectory."""

import bayestraj.observations.io as observations_io
import bayestraj.observations.paths as paths

INPUT_CSV = paths.data_path("raw/BR24_target_reference_state.csv")
OUTPUT_CSV = paths.data_path("processed/BR24_target_reference_trajectory.csv")


def main() -> None:
    """Export the configured BR24 target reference trajectory."""
    output_path = observations_io.export_br24_target_reference_trajectory(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
    )
    print(f"Exported BR24 target reference trajectory: {output_path}")


if __name__ == "__main__":
    main()
