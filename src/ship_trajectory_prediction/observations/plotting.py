"""Plotting utilities for real ship trajectory data."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as mdates
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrowPatch

from ship_trajectory_prediction.coordinates import (
    calculate_gps_distances,
    calculate_signed_curvature_from_gps,
    calculate_speed_from_gps,
    gps_to_local_coordinates,
)
from ship_trajectory_prediction.observations.window import (
    KILOMETERS_PER_HOUR_TO_METERS_PER_SECOND,
)

# ==================================================
# Technical plot settings
# ==================================================
DEFAULT_ARROW_STEP = 18  # 18 points = about 3 minutes if data interval is 10 s

START_COLOR = "black"
END_COLOR = "green"

TRAJECTORY_MARKER_SIZE = 3
START_MARKER_SIZE = 45
END_MARKER_SIZE = 110

TRAJECTORY_LINE_WIDTH = 1.8
DIRECTION_LINE_WIDTH = 1.5

END_ALPHA = 0.45
ARROW_MUTATION_SCALE = 14
MIN_AXIS_MARGIN_METERS = 15


@dataclass(frozen=True, slots=True)
class ShipDataPlotStyle:
    """User-facing appearance settings for ship-data plots."""

    trajectory_figure_size: tuple[float, float] = (8, 6)
    speed_figure_size: tuple[float, float] = (10, 6)
    curvature_figure_size: tuple[float, float] = (10, 6)
    title_pad: float = 16
    title_font_size: float = 13
    title_font_weight: str = "bold"
    axis_label_font_size: float = 13
    axis_tick_font_size: float = 11
    time_tick_format: str = "%H:%M"
    time_range_format: str = "%d.%m.%Y %H:%M:%S %Z"
    time_label_line_spacing: float = 2.4
    legend_location: str = "upper right"
    recorded_data_color: str = "#4C78A8"
    derived_data_color: str = "#F58518"
    run_colors: tuple[str, ...] = (
        "#4C78A8",
        "#59A14F",
        "#B279A2",
        "#E45756",
        "#72B7B2",
    )
    calculated_speed_line_width: float = 1.5
    calculated_speed_alpha: float = 0.75
    curvature_line_width: float = 1.5
    curvature_alpha: float = 0.85
    stationary_color: str = "#A7A7A7"
    stationary_alpha: float = 0.3


DEFAULT_SHIP_DATA_PLOT_STYLE = ShipDataPlotStyle()


class HandlerDirectionArrow(HandlerPatch):
    """
    Custom legend handler for direction arrows.
    """

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        center_y = ydescent + height / 2

        arrow = FancyArrowPatch(
            (xdescent, center_y),
            (xdescent + width, center_y),
            arrowstyle="->",
            color=orig_handle.get_edgecolor(),
            linewidth=orig_handle.get_linewidth(),
            mutation_scale=ARROW_MUTATION_SCALE,
            transform=trans,
        )

        return [arrow]


def plot_ship_trajectory(
    data,
    arrow_step=DEFAULT_ARROW_STEP,
    coordinate_unit="gps",
    plot_style=DEFAULT_SHIP_DATA_PLOT_STYLE,
):
    """
    Plot the ship trajectory using GPS longitude and latitude.

    The plot shows:
    - the trajectory line
    - start point
    - end point
    - optional direction arrows along the path

    Parameters
    ----------
    data : pandas.DataFrame
        Ship trajectory data containing gps_longitude and gps_latitude.
    arrow_step : int or None, optional
        Distance between direction arrows in number of data points.
        If None, no direction arrows are plotted.
    coordinate_unit : {"gps", "km", "m"}, optional
        Coordinate representation. ``"gps"`` plots longitude and latitude in
        degrees. ``"km"`` and ``"m"`` plot local east and north distances
        from the first trajectory point.
    plot_style : ShipDataPlotStyle, optional
        Appearance settings shared with the ship-speed plots.
    """
    if data.empty:
        raise ValueError("The input data is empty.")

    if coordinate_unit not in {"gps", "km", "m"}:
        raise ValueError("coordinate_unit must be 'gps', 'km', or 'm'.")

    if arrow_step is not None and arrow_step <= 0:
        raise ValueError("arrow_step must be a positive integer or None.")

    plot_data = data.copy()
    longitude = plot_data["gps_longitude"].to_numpy()
    latitude = plot_data["gps_latitude"].to_numpy()

    if coordinate_unit in {"km", "m"}:
        # Convert GPS coordinates to a local Cartesian coordinate system.
        # The first GPS point is used as the origin (x=0, y=0). For the
        # relatively short trajectories considered here, the equirectangular
        # approximation provides local east/north distances:
        #
        #   x = R * (longitude - reference_longitude) * cos(mean_latitude)
        #   y = R * (latitude  - reference_latitude)
        #
        # All angles must be in radians. The cosine term compensates for the
        # decreasing east-west distance of one longitude degree toward the
        # poles. Positive x points east and positive y points north.
        plot_data["_plot_x"], plot_data["_plot_y"] = gps_to_local_coordinates(
            longitude,
            latitude,
            unit=coordinate_unit,
        )
        x_label = f"Ostposition x [{coordinate_unit}]"
        y_label = f"Nordposition y [{coordinate_unit}]"
    else:
        plot_data["_plot_x"] = longitude
        plot_data["_plot_y"] = latitude
        x_label = "Längengrad [°]"
        y_label = "Breitengrad [°]"

    plt.figure(figsize=plot_style.trajectory_figure_size)
    run_groups = _trajectory_groups(plot_data)
    trajectory_lines = []
    start_marker = None
    end_marker = None
    legend_handler_map = {}
    for run_index, (run_id, run_data) in enumerate(run_groups):
        x_coordinates = run_data["_plot_x"].to_numpy()
        y_coordinates = run_data["_plot_y"].to_numpy()
        trajectory_color = _run_color(
            run_index,
            run_count=len(run_groups),
            plot_style=plot_style,
        )
        trajectory_lines.append(
            plt.plot(
                x_coordinates,
                y_coordinates,
                color=trajectory_color,
                marker="o",
                markersize=TRAJECTORY_MARKER_SIZE,
                linewidth=TRAJECTORY_LINE_WIDTH,
                label=_run_label("Schiffstrajektorie", run_id, len(run_groups)),
            )[0]
        )

        run_start_marker = plt.scatter(
            x_coordinates[0],
            y_coordinates[0],
            s=START_MARKER_SIZE,
            color=START_COLOR,
            marker="o",
            label="Start" if run_index == 0 else "_nolegend_",
            zorder=4,
        )
        run_end_marker = plt.scatter(
            x_coordinates[-1],
            y_coordinates[-1],
            s=END_MARKER_SIZE,
            color=END_COLOR,
            alpha=END_ALPHA,
            marker="X",
            label="Ende" if run_index == 0 else "_nolegend_",
            zorder=5,
        )
        if run_index == 0:
            start_marker = run_start_marker
            end_marker = run_end_marker

        if arrow_step is not None:
            for index in range(0, len(x_coordinates) - 1, arrow_step):
                start = (x_coordinates[index], y_coordinates[index])
                end = (x_coordinates[index + 1], y_coordinates[index + 1])
                if start == end:
                    continue
                plt.annotate(
                    "",
                    xy=end,
                    xytext=start,
                    arrowprops={
                        "arrowstyle": "->",
                        "color": plot_style.derived_data_color,
                        "linewidth": DIRECTION_LINE_WIDTH,
                        "mutation_scale": ARROW_MUTATION_SCALE,
                    },
                )

    legend_handles = [*trajectory_lines, start_marker, end_marker]
    if arrow_step is not None:
        direction_handle = FancyArrowPatch(
            (0, 0),
            (1, 0),
            arrowstyle="->",
            color=plot_style.derived_data_color,
            linewidth=DIRECTION_LINE_WIDTH,
            mutation_scale=ARROW_MUTATION_SCALE,
            label="Fahrtrichtung",
        )

        legend_handles.append(direction_handle)
        legend_handler_map[FancyArrowPatch] = HandlerDirectionArrow()

    x_coordinates = plot_data["_plot_x"].to_numpy()
    y_coordinates = plot_data["_plot_y"].to_numpy()

    plt.xlabel(x_label, fontsize=plot_style.axis_label_font_size)
    plt.ylabel(y_label, fontsize=plot_style.axis_label_font_size)
    plt.title(
        "Schiffstrajektorie mit Fahrtrichtung",
        pad=plot_style.title_pad,
        fontsize=plot_style.title_font_size,
        fontweight=plot_style.title_font_weight,
    )
    plt.grid(True)
    plt.axis("equal")

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=plot_style.axis_tick_font_size)

    if coordinate_unit in {"km", "m"}:
        x_center = (x_coordinates.min() + x_coordinates.max()) / 2
        y_center = (y_coordinates.min() + y_coordinates.max()) / 2
        data_range = max(np.ptp(x_coordinates), np.ptp(y_coordinates))
        minimum_margin = MIN_AXIS_MARGIN_METERS if coordinate_unit == "m" else 0.015
        margin = max(data_range * 0.05, minimum_margin)
        plot_range = max(data_range + 2 * margin, 2 * minimum_margin)

        ax.set_xlim(x_center - plot_range / 2, x_center + plot_range / 2)
        ax.set_ylim(y_center - plot_range / 2, y_center + plot_range / 2)

    ax.ticklabel_format(useOffset=False, style="plain")

    plt.legend(
        handles=legend_handles,
        handler_map=legend_handler_map,
        loc=plot_style.legend_location,
    )

    plt.tight_layout()
    plt.show()


def plot_ship_speeds(
    data,
    speed_unit="km/h",
    propulsion_speed_unit="rpm",
    plot_style=DEFAULT_SHIP_DATA_PLOT_STYLE,
):
    """
    Plot ship and propulsion speeds in two separate figures.

    The calculated ship speed is the distance between two consecutive GPS
    positions divided by their actual time difference. Recorded ``gps_speed``
    values are interpreted as kilometers per hour and converted for display
    when ``speed_unit`` is ``"m/s"``.

    Parameters
    ----------
    data : pandas.DataFrame
        Ship trajectory data containing time, gps_speed,
        shaft_speed, and thruster_speed.
    speed_unit : {"m/s", "km/h"}, optional
        Display unit for recorded and position-derived ship speed.
    propulsion_speed_unit : str, optional
        Display unit for shaft and thruster speed. No conversion is applied.
    plot_style : ShipDataPlotStyle, optional
        Appearance settings shared with the trajectory plot.

    Returns
    -------
    tuple
        Two figures and their corresponding axes.
    """
    if data.empty:
        raise ValueError("The input data is empty.")
    if speed_unit not in {"m/s", "km/h"}:
        raise ValueError("speed_unit must be 'm/s' or 'km/h'.")

    run_groups = _trajectory_groups(data)
    speed_figure, speed_axis = plt.subplots(figsize=plot_style.speed_figure_size)
    for run_index, (_, run_data) in enumerate(run_groups):
        recorded_gps_speed = np.asarray(run_data["gps_speed"], dtype=float)
        if speed_unit == "m/s":
            recorded_gps_speed = (
                recorded_gps_speed * KILOMETERS_PER_HOUR_TO_METERS_PER_SECOND
            )
        calculated_speed = calculate_speed_from_gps(run_data, unit=speed_unit)
        speed_axis.plot(
            run_data["time"],
            recorded_gps_speed,
            color=plot_style.recorded_data_color,
            label="GPS-Geschwindigkeit" if run_index == 0 else "_nolegend_",
        )
        speed_axis.plot(
            run_data["time"],
            calculated_speed,
            color=plot_style.derived_data_color,
            alpha=plot_style.calculated_speed_alpha,
            label=("Aus GPS-Positionen berechnet" if run_index == 0 else "_nolegend_"),
            linewidth=plot_style.calculated_speed_line_width,
        )
    time_axis_label = _format_time_axis_label(
        data["time"],
        time_range_format=plot_style.time_range_format,
    )
    speed_axis.set_xlabel(
        time_axis_label,
        fontsize=plot_style.axis_label_font_size,
        linespacing=plot_style.time_label_line_spacing,
    )
    speed_axis.set_ylabel(
        f"Schiffsgeschwindigkeit [{speed_unit}]",
        fontsize=plot_style.axis_label_font_size,
    )
    speed_axis.set_title(
        "Schiffsgeschwindigkeit",
        pad=plot_style.title_pad,
        fontsize=plot_style.title_font_size,
        fontweight=plot_style.title_font_weight,
    )
    speed_axis.tick_params(axis="both", labelsize=plot_style.axis_tick_font_size)
    speed_axis.xaxis.set_major_formatter(
        mdates.DateFormatter(plot_style.time_tick_format)
    )
    speed_axis.grid(True)
    speed_axis.legend(loc=plot_style.legend_location)
    speed_figure.tight_layout()
    plt.show()

    propulsion_figure, propulsion_axis = plt.subplots(
        figsize=plot_style.speed_figure_size
    )
    for run_index, (_, run_data) in enumerate(run_groups):
        propulsion_axis.plot(
            run_data["time"],
            run_data["shaft_speed"],
            color=plot_style.recorded_data_color,
            label="Wellendrehzahl" if run_index == 0 else "_nolegend_",
        )
        propulsion_axis.plot(
            run_data["time"],
            run_data["thruster_speed"],
            color=plot_style.derived_data_color,
            label="Strahlruderdrehzahl" if run_index == 0 else "_nolegend_",
        )
    propulsion_axis.set_xlabel(
        time_axis_label,
        fontsize=plot_style.axis_label_font_size,
        linespacing=plot_style.time_label_line_spacing,
    )
    propulsion_axis.set_ylabel(
        f"Antriebsdrehzahl [{propulsion_speed_unit}]",
        fontsize=plot_style.axis_label_font_size,
    )
    propulsion_axis.set_title(
        "Antriebsdrehzahlen",
        pad=plot_style.title_pad,
        fontsize=plot_style.title_font_size,
        fontweight=plot_style.title_font_weight,
    )
    propulsion_axis.tick_params(
        axis="both",
        labelsize=plot_style.axis_tick_font_size,
    )
    propulsion_axis.xaxis.set_major_formatter(
        mdates.DateFormatter(plot_style.time_tick_format)
    )
    propulsion_axis.grid(True)
    propulsion_axis.legend(loc=plot_style.legend_location)
    propulsion_figure.tight_layout()
    plt.show()

    return (speed_figure, propulsion_figure), (speed_axis, propulsion_axis)


def plot_ship_curvature(
    data,
    *,
    min_displacement_m=2.0,
    max_time_gap_s=15.0,
    plot_style=DEFAULT_SHIP_DATA_PLOT_STYLE,
):
    """Plot signed GPS-derived trajectory curvature over the complete run."""
    if data.empty:
        raise ValueError("The input data is empty.")

    run_groups = _trajectory_groups(data)
    figure, axis = plt.subplots(figsize=plot_style.curvature_figure_size)
    for run_index, (run_id, run_data) in enumerate(run_groups):
        curvature = calculate_signed_curvature_from_gps(
            run_data,
            min_displacement_m=min_displacement_m,
            max_time_gap_s=max_time_gap_s,
        )
        curvature_color = (
            plot_style.derived_data_color
            if len(run_groups) == 1
            else _run_color(
                run_index,
                run_count=len(run_groups),
                plot_style=plot_style,
            )
        )
        axis.plot(
            run_data["time"],
            curvature,
            color=curvature_color,
            linewidth=plot_style.curvature_line_width,
            alpha=plot_style.curvature_alpha,
            label=_run_label(
                "Vorzeichenbehaftete Krümmung",
                run_id,
                len(run_groups),
            ),
        )
        _shade_low_motion_periods(
            axis,
            run_data,
            min_displacement_m=min_displacement_m,
            color=plot_style.stationary_color,
            alpha=plot_style.stationary_alpha,
            include_legend_label=run_index == 0,
        )
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axis.set_xlabel(
        _format_time_axis_label(
            data["time"],
            time_range_format=plot_style.time_range_format,
        ),
        fontsize=plot_style.axis_label_font_size,
        linespacing=plot_style.time_label_line_spacing,
    )
    axis.set_ylabel(
        "Vorzeichenbehaftete Krümmung κ [1/m]",
        fontsize=plot_style.axis_label_font_size,
    )
    axis.set_title(
        "Krümmung der Schiffstrajektorie",
        pad=plot_style.title_pad,
        fontsize=plot_style.title_font_size,
        fontweight=plot_style.title_font_weight,
    )
    axis.tick_params(axis="both", labelsize=plot_style.axis_tick_font_size)
    axis.xaxis.set_major_formatter(mdates.DateFormatter(plot_style.time_tick_format))
    axis.grid(True)
    legend_handles, legend_labels = axis.get_legend_handles_labels()
    legend_order = [
        index
        for index, label in enumerate(legend_labels)
        if label != "Stillstand / zu geringe Bewegung"
    ]
    legend_order.extend(
        index
        for index, label in enumerate(legend_labels)
        if label == "Stillstand / zu geringe Bewegung"
    )
    axis.legend(
        [legend_handles[index] for index in legend_order],
        [legend_labels[index] for index in legend_order],
        loc=plot_style.legend_location,
    )
    figure.tight_layout()
    plt.show()
    return figure, axis


def _trajectory_groups(data):
    """Return run-separated trajectory groups without changing row order."""
    if "run_id" not in data.columns:
        return ((None, data),)
    return tuple(
        data.groupby(
            "run_id",
            sort=False,
            dropna=False,
        )
    )


def _run_label(base_label, run_id, run_count):
    """Append a run identifier only when several runs are displayed."""
    if run_count == 1 or run_id is None:
        return base_label
    return f"{base_label} (Run {run_id})"


def _run_color(run_index, *, run_count, plot_style):
    """Return the default color or a stable multi-run palette color."""
    if run_count == 1:
        return plot_style.recorded_data_color
    if not plot_style.run_colors:
        raise ValueError("plot_style.run_colors must not be empty for multiple runs.")
    return plot_style.run_colors[run_index % len(plot_style.run_colors)]


def _format_time_axis_label(time_values, *, time_range_format):
    """Return a compact axis label with the exact observation time range."""
    start_time = time_values.min()
    end_time = time_values.max()
    start_label = start_time.strftime(time_range_format).strip()
    end_label = end_time.strftime(time_range_format).strip()
    return f"Zeit\nStart: {start_label}   Ende: {end_label}"


def _shade_low_motion_periods(
    axis,
    data,
    *,
    min_displacement_m,
    color,
    alpha,
    include_legend_label,
):
    """Shade periods whose GPS displacement is too small for curvature."""
    segment_length = calculate_gps_distances(
        data["gps_longitude"].to_numpy(),
        data["gps_latitude"].to_numpy(),
    )
    low_motion_segment = segment_length < min_displacement_m
    low_motion_sample = np.zeros(len(data), dtype=bool)
    low_motion_sample[:-1] |= low_motion_segment
    low_motion_sample[1:] |= low_motion_segment

    padded_mask = np.pad(low_motion_sample.astype(int), (1, 1))
    transitions = np.diff(padded_mask)
    start_indices = np.flatnonzero(transitions == 1)
    end_indices = np.flatnonzero(transitions == -1) - 1
    for span_index, (start_index, end_index) in enumerate(
        zip(start_indices, end_indices, strict=True)
    ):
        axis.axvspan(
            data["time"].iloc[start_index],
            data["time"].iloc[end_index],
            color=color,
            alpha=alpha,
            label=(
                "Stillstand / zu geringe Bewegung"
                if include_legend_label and span_index == 0
                else "_nolegend_"
            ),
            zorder=0,
        )
