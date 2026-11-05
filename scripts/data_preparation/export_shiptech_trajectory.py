"""Export the Shiptech trajectory as position-only processed data."""

import bayestraj.observations.io as observations_io
from bayestraj.paths import project_path

INPUT_CSV = project_path("data/raw/shiptech_data.csv")
RUN_ID = 42
OUTPUT_CSV = project_path("data/processed/ship_trajectory_run_42.csv")


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
