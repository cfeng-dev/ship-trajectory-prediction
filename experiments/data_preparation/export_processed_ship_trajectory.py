"""Export one recorded ship run as a generic position-only trajectory CSV."""

import bayestraj.observations.io as observations_io
import bayestraj.observations.paths as paths

INPUT_CSV = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)
RUN_ID = 42
OUTPUT_DIR = paths.data_path("processed")


def main() -> None:
    """Export the configured raw ship run to elapsed seconds and local meters."""
    output_path = observations_io.export_processed_trajectory(
        input_csv=INPUT_CSV,
        run_id=RUN_ID,
        output_dir=OUTPUT_DIR,
    )
    print(f"Exported processed trajectory: {output_path}")


if __name__ == "__main__":
    main()
