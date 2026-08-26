"""Explore and visualize recorded ship trajectory data."""

import numpy as np

import bayestraj.observations.coordinates as coordinates
import bayestraj.observations.io as observations_io
import bayestraj.observations.paths as paths
import bayestraj.observations.plotting as plotting

DATA_FILE = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

# Data selection
RUN_IDS = (1,)  # One: (1,); selected: (1, 3); range 1-3: range(1, 4); all: None
START_TIME = None  # Example: "2026-01-10 08:20:00+00:00"
END_TIME = None

# Terminal report
SUMMARY_LABEL_WIDTH = 20
TURN_RATE_CENTRAL_RANGE = 0.90

# Plot data settings
TRAJECTORY_COORDINATE_UNIT = "m"  # "m", "km", or "gps"
SPEED_UNIT = "m/s"  # "m/s" or "km/h"
PROPULSION_SPEED_UNIT = "rpm"
POSITION_NOISE_STD_M = 5.0  # Per x/y axis [m]; 0 disables.
POSITION_NOISE_SEED = 2026  # Reproduces the added position noise.
MIN_CURVATURE_DISPLACEMENT_METERS = 2.0
MAX_CURVATURE_TIME_GAP_SECONDS = 15.0

# Plot appearance
PLOT_STYLE = plotting.ShipDataPlotStyle(
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
    time_label_line_spacing=2.4,
    show_legend=True,
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
    ship_data = observations_io.read_ship_data(
        DATA_FILE,
        run_id=RUN_IDS,
        start_time=START_TIME,
        end_time=END_TIME,
    )
    ship_data = _add_position_noise(
        ship_data,
        position_noise_std_m=POSITION_NOISE_STD_M,
        seed=POSITION_NOISE_SEED,
    )

    observations_io.print_ship_data_summary(
        ship_data,
        label_width=SUMMARY_LABEL_WIDTH,
        gps_speed_unit=SPEED_UNIT,
        propulsion_speed_unit=PROPULSION_SPEED_UNIT,
    )
    _print_position_noise_setting()
    _print_turn_rate_summary(ship_data)
    plotting.plot_ship_trajectory(
        ship_data,
        coordinate_unit=TRAJECTORY_COORDINATE_UNIT,
        trajectory_label=_trajectory_label(),
        plot_style=PLOT_STYLE,
    )
    plotting.plot_ship_speeds(
        ship_data,
        speed_unit=SPEED_UNIT,
        propulsion_speed_unit=PROPULSION_SPEED_UNIT,
        plot_style=PLOT_STYLE,
    )
    plotting.plot_ship_curvature(
        ship_data,
        min_displacement_m=MIN_CURVATURE_DISPLACEMENT_METERS,
        max_time_gap_s=MAX_CURVATURE_TIME_GAP_SECONDS,
        plot_style=PLOT_STYLE,
    )


def _trajectory_label() -> str:
    """Describe whether the plotted trajectory contains added position noise."""
    if POSITION_NOISE_STD_M > 0:
        return "Verrauschte Schiffstrajektorie"
    return "Schiffstrajektorie"


def _add_position_noise(data, *, position_noise_std_m, seed):
    """Return a copy with reproducible Gaussian noise on GPS positions."""
    if isinstance(position_noise_std_m, bool):
        raise ValueError("position_noise_std_m must be finite and non-negative.")
    try:
        position_noise_std_m = float(position_noise_std_m)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "position_noise_std_m must be finite and non-negative."
        ) from error
    if not np.isfinite(position_noise_std_m) or position_noise_std_m < 0:
        raise ValueError("position_noise_std_m must be finite and non-negative.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    noisy_data = data.copy()
    if noisy_data.empty or position_noise_std_m == 0:
        return noisy_data

    generator = np.random.default_rng(seed)
    if "run_id" in noisy_data.columns:
        run_groups = (
            run_data
            for _, run_data in noisy_data.groupby(
                "run_id",
                sort=False,
                dropna=False,
            )
        )
    else:
        run_groups = (noisy_data,)

    for run_data in run_groups:
        longitude = run_data["gps_longitude"].to_numpy(dtype=float)
        latitude = run_data["gps_latitude"].to_numpy(dtype=float)
        x_meters, y_meters = coordinates.gps_to_local_coordinates(
            longitude,
            latitude,
            unit="m",
        )
        x_meters += generator.normal(
            0.0,
            position_noise_std_m,
            len(run_data),
        )
        y_meters += generator.normal(
            0.0,
            position_noise_std_m,
            len(run_data),
        )
        noisy_longitude, noisy_latitude = coordinates.local_to_gps_coordinates(
            x_meters,
            y_meters,
            reference_longitude=longitude[0],
            reference_latitude=latitude[0],
            unit="m",
        )
        noisy_data.loc[run_data.index, "gps_longitude"] = noisy_longitude
        noisy_data.loc[run_data.index, "gps_latitude"] = noisy_latitude

    return noisy_data


def _print_position_noise_setting() -> None:
    """Print the configured artificial position-noise scenario."""
    description = (
        f"{POSITION_NOISE_STD_M:g} m per x/y axis (seed={POSITION_NOISE_SEED})"
        if POSITION_NOISE_STD_M > 0
        else "disabled"
    )
    print(f"\n{'Added position noise':<{SUMMARY_LABEL_WIDTH}}: {description}")


def _print_turn_rate_summary(data) -> None:
    """Print observed and robust GPS-derived turn-rate ranges."""
    if "run_id" in data.columns:
        run_groups = (
            run_data for _, run_data in data.groupby("run_id", sort=False, dropna=False)
        )
    else:
        run_groups = (data,)

    turn_rate_groups = []
    candidate_count = 0
    for run_data in run_groups:
        ordered_run = run_data.sort_values("time")
        candidate_count += max(len(ordered_run) - 2, 0)
        turn_rate = coordinates.calculate_signed_turn_rate_from_gps(
            ordered_run,
            min_displacement_m=MIN_CURVATURE_DISPLACEMENT_METERS,
            max_time_gap_s=MAX_CURVATURE_TIME_GAP_SECONDS,
        )
        finite_turn_rate = turn_rate[np.isfinite(turn_rate)]
        if finite_turn_rate.size:
            turn_rate_groups.append(finite_turn_rate)

    print("\nGPS-derived turn-rate range:")
    if not turn_rate_groups:
        print("No valid turn-rate samples for the selected filters.")
        return

    turn_rate = np.concatenate(turn_rate_groups)
    tail_probability = (1.0 - TURN_RATE_CENTRAL_RANGE) / 2.0
    central_lower, central_upper = np.quantile(
        turn_rate,
        [tail_probability, 1.0 - tail_probability],
    )
    central_percentage = round(100 * TURN_RATE_CENTRAL_RANGE)
    rows = (
        (
            "Observed [rad/s]",
            f"{turn_rate.min():+.6f} to {turn_rate.max():+.6f}",
        ),
        (
            f"Central {central_percentage}% [rad/s]",
            f"{central_lower:+.6f} to {central_upper:+.6f}",
        ),
        ("Valid samples", f"{len(turn_rate)} of {candidate_count}"),
    )
    label_width = max(SUMMARY_LABEL_WIDTH, *(len(label) for label, _ in rows))
    for label, value in rows:
        print(f"{label:<{label_width}}: {value}")


if __name__ == "__main__":
    main()
