"""Attach evaluation-only target reference labels to the raw BR24 tracking input."""

import bayestraj.observations.br24 as br24
from bayestraj.paths import project_path

TRACKING_CSV = project_path("data/processed/BR24Dataset20131107.csv")
REFERENCE_CSV = project_path("data/raw/BR24_reference_trajectory.csv")
OUTPUT_CSV = project_path("data/processed/BR24Dataset20131107_with_reference.csv")


def main() -> None:
    """Export raw BR24 tracking input with evaluation-only reference labels."""
    output_path = br24.export_br24_tracking_input_with_reference(
        tracking_csv=TRACKING_CSV,
        reference_csv=REFERENCE_CSV,
        output_csv=OUTPUT_CSV,
    )
    print(f"Exported BR24 tracking input with reference: {output_path}")


if __name__ == "__main__":
    main()
