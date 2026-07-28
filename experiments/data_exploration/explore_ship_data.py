"""Explore and visualize recorded ship trajectory data."""

from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory.io import (
    print_ship_data_summary,
    read_ship_data,
)
from ship_trajectory_prediction.trajectory.plotting import (
    ShipDataPlotStyle,
    plot_ship_speeds,
    plot_ship_trajectory,
)

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

# Data selection
RUN_ID = 1
START_TIME = None
END_TIME = None

# Plot data settings
TRAJECTORY_COORDINATE_UNIT = "km"  # "m", "km", or "gps"
SPEED_UNIT = "km/h"  # "m/s" or "km/h"
PROPULSION_SPEED_UNIT = "rpm"

# Plot appearance
PLOT_STYLE = ShipDataPlotStyle(
    trajectory_figure_size=(8, 6),
    speed_figure_size=(10, 6),
    title_pad=16,
    title_font_size=13,
    title_font_weight="bold",
    axis_label_font_size=13,
    axis_tick_font_size=11,
    time_tick_format="%H:%M",
    time_range_format="%d.%m.%Y %H:%M:%S %Z",
    recorded_data_color="#4C78A8",
    derived_data_color="#F58518",
    calculated_speed_line_width=1.5,
    calculated_speed_alpha=0.75,
)


def main() -> None:
    """Load recorded ship data and create exploratory plots."""
    ship_data = read_ship_data(
        DATA_FILE,
        run_id=RUN_ID,
        start_time=START_TIME,
        end_time=END_TIME,
    )

    print_ship_data_summary(ship_data)
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


if __name__ == "__main__":
    main()
