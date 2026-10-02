"""Export the HTWG reference trajectory as position-only processed data."""

import bayestraj.observations.io as observations_io
import bayestraj.observations.paths as paths

INPUT_CSV = paths.data_path("raw/ship_data_htwg.csv")
OUTPUT_CSV = paths.data_path("processed/ship_trajectory_htwg.csv")


def main() -> None:
    """Export the configured HTWG raw trajectory."""
    output_path = observations_io.export_htwg_trajectory(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
    )
    print(f"Exported processed HTWG trajectory: {output_path}")


if __name__ == "__main__":
    main()
