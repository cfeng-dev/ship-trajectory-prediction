"""Export the Shiptech trajectory as position-only processed data."""

import bayestraj.observations.io as observations_io
import bayestraj.observations.paths as paths

INPUT_CSV = paths.data_path("raw/shiptech_data.csv")
RUN_ID = 42
OUTPUT_CSV = paths.data_path("processed/ship_trajectory_run_42.csv")


def main() -> None:
    """Export the configured Shiptech raw trajectory."""
    output_path = observations_io.export_shiptech_trajectory(
        input_csv=INPUT_CSV,
        run_id=RUN_ID,
        output_csv=OUTPUT_CSV,
    )
    print(f"Exported processed trajectory: {output_path}")


if __name__ == "__main__":
    main()
