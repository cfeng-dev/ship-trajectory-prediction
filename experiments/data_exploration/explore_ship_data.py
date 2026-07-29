"""Explore and visualize recorded ship trajectory data."""

from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory.io import (
    print_ship_data_summary,
    read_ship_data,
)
from ship_trajectory_prediction.trajectory.plotting import (
    ShipDataPlotStyle,
    plot_ship_curvature,
    plot_ship_speeds,
    plot_ship_trajectory,
)

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

# Data selection
RUN_IDS = (1,)  # One: (1,); selected: (1, 3); range 1-3: range(1, 4); all: None
START_TIME = None
END_TIME = None

# Terminal report
SUMMARY_LABEL_WIDTH = 20

# Plot data settings
TRAJECTORY_COORDINATE_UNIT = "km"  # "m", "km", or "gps"
SPEED_UNIT = "km/h"  # "m/s" or "km/h"
PROPULSION_SPEED_UNIT = "rpm"
MIN_CURVATURE_DISPLACEMENT_METERS = 2.0
MAX_CURVATURE_TIME_GAP_SECONDS = 15.0

# Plot appearance
PLOT_STYLE = ShipDataPlotStyle(
    trajectory_figure_size=(8, 6),
    speed_figure_size=(10, 6),
    curvature_figure_size=(10, 6),
    title_pad=16,
    title_font_size=13,
    title_font_weight="bold",
    axis_label_font_size=13,
    axis_tick_font_size=11,
    time_tick_format="%H:%M",
    time_range_format="%d.%m.%Y %H:%M:%S %Z",
    legend_location="upper right",
    recorded_data_color="#4C78A8",
    derived_data_color="#F58518",
    run_colors=("#4C78A8", "#59A14F", "#B279A2", "#E45756", "#72B7B2"),
    calculated_speed_line_width=1.5,
    calculated_speed_alpha=0.75,
    curvature_line_width=1.5,
    curvature_alpha=0.85,
    stationary_color="#A7A7A7",
    stationary_alpha=0.3,
)


def main() -> None:
    """Load recorded ship data and create exploratory plots."""
    ship_data = read_ship_data(
        DATA_FILE,
        run_id=RUN_IDS,
        start_time=START_TIME,
        end_time=END_TIME,
    )

    print_ship_data_summary(
        ship_data,
        label_width=SUMMARY_LABEL_WIDTH,
        gps_speed_unit=SPEED_UNIT,
        propulsion_speed_unit=PROPULSION_SPEED_UNIT,
    )
    plot_ship_trajectory(
        ship_data,
        coordinate_unit=TRAJECTORY_COORDINATE_UNIT,
        plot_style=PLOT_STYLE,
    )
    plot_ship_speeds(
        ship_data,
        speed_unit=SPEED_UNIT,
        propulsion_speed_unit=PROPULSION_SPEED_UNIT,
        plot_style=PLOT_STYLE,
    )
    plot_ship_curvature(
        ship_data,
        min_displacement_m=MIN_CURVATURE_DISPLACEMENT_METERS,
        max_time_gap_s=MAX_CURVATURE_TIME_GAP_SECONDS,
        plot_style=PLOT_STYLE,
    )


if __name__ == "__main__":
    main()
