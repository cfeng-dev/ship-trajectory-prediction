"""Configuration values for the interactive ship trajectory GUI."""

from dataclasses import dataclass, field


@dataclass
class GUIConfig:
    """
    Configuration container for the ship trajectory GUI.

    Keeping the visual and simulation settings in one place makes the main
    GUI class shorter and easier to adjust.
    """

    # ==================================================
    # Simulation timing
    # ==================================================

    # Simulation time step in seconds.
    # Each simulation step advances the simulated time by 0.05 s.
    simulation_dt: float = 0.05

    # GUI update interval in milliseconds.
    # The simulation and plot are updated every 50 ms = 0.05 s.
    update_interval_ms: int = 50

    # ==================================================
    # CSV export settings
    # ==================================================
    default_csv_filename: str = "simulated_ship_trajectory.csv"

    # ==================================================
    # Initial simulation values
    # ==================================================
    initial_speed: float = 5.0
    observation_noise_sigma: float = 0.2

    # ==================================================
    # Steering slider settings in degrees per second [°/s]
    # ==================================================
    min_steering_deg_per_second: int = -45
    max_steering_deg_per_second: int = 45
    steering_resolution: int = 1

    # ==================================================
    # Speed slider settings in meters per second [m/s]
    # ==================================================
    min_speed: float = 1.0
    max_speed: float = 10.0
    speed_resolution: float = 0.1

    # ==================================================
    # Keyboard control step sizes
    # ==================================================
    keyboard_speed_step: float = 0.1
    keyboard_steering_step: int = 1

    # ==================================================
    # Ship visualization size in meters [m]
    # ==================================================
    ship_length: float = 10.0
    ship_width: float = 3.0

    # ==================================================
    # Ship visualization colors
    # ==================================================
    ship_facecolor: str = "lightsalmon"
    ship_edgecolor: str = "darkred"
    ship_alpha: float = 0.85
    ship_linewidth: float = 1.5

    # ==================================================
    # Ship center marker style
    # ==================================================
    show_ship_center: bool = True
    ship_center_color: str = "darkred"
    ship_center_size: int = 8

    # ==================================================
    # GUI and plot style colors
    # ==================================================
    app_background_color: str = "#eef7fb"
    control_panel_color: str = "#e3eef4"
    plot_panel_color: str = "#f4f9fc"
    plot_background_color: str = "#eaf6fb"
    trajectory_color: str = "#2a6f97"
    trajectory_linewidth: float = 2.0
    start_position_color: str = "black"
    grid_color: str = "#8fbcd4"
    grid_alpha: float = 0.75
    grid_linewidth: float = 1.0

    # Figure background.
    # None means it will be set to plot_panel_color in __post_init__.
    figure_background_color: str | None = None

    # ==================================================
    # Plot axis settings
    # ==================================================
    # Available modes:
    # "auto"   = show the full trajectory automatically
    # "fixed"  = use fixed axis limits
    # "follow" = smoothly follow the current ship position
    axis_mode: str = "auto"

    # Coordinate representation shown in the plot and status panel:
    # "local" = meters, "km" = kilometers, "gps" = longitude/latitude.
    # The simulation itself always uses local coordinates in meters.
    coordinate_display_mode: str = "local"

    # GPS position represented by the local simulation origin (0, 0).
    reference_longitude: float = 8.312259928385417
    reference_latitude: float = 47.05150553385417

    auto_axis_margin: int = 15
    view_width: int = 100
    view_height: int = 100
    camera_x: float = 0.0
    camera_y: float = 0.0
    camera_smoothness: float = 0.06

    axis_x_min: int = -10
    axis_x_max: int = 50
    axis_y_min: int = -30
    axis_y_max: int = 30
    axis_tick_step: int = 10

    # ==================================================
    # Window and plot size
    # ==================================================
    window_width: int = 1100
    window_height: int = 720

    # Move the main window slightly upward from the exact screen center.
    main_window_vertical_offset: int = 40

    # Keep figure square.
    figure_width: int = 6
    figure_height: int = 6

    # Width of the left control panel in pixels.
    control_panel_width: int = 320

    # ==================================================
    # Status display font and spacing
    # ==================================================
    status_label_font: tuple[str, int, str] = ("Arial", 9, "bold")
    status_value_font: tuple[str, int] = ("Arial", 9)
    status_row_padding_y: int = 1

    # Equal vertical spacing between control panel sections.
    section_spacing: int = 14

    # ==================================================
    # Help window layout
    # ==================================================
    # Change this single value to align the explanation columns in all help sections.
    help_description_column_start: int = 22

    # Separate variables for each help section.
    # They currently use the same value so all descriptions start at the same vertical line.
    help_keyboard_left_column_width: int = field(init=False)
    help_menu_left_column_width: int = field(init=False)
    help_button_left_column_width: int = field(init=False)
    help_simulation_time_left_column_width: int = field(init=False)

    def __post_init__(self):
        """
        Fill values that depend on other configuration values.
        """
        if self.figure_background_color is None:
            self.figure_background_color = self.plot_panel_color

        self.help_keyboard_left_column_width = self.help_description_column_start
        self.help_menu_left_column_width = self.help_description_column_start
        self.help_button_left_column_width = self.help_description_column_start
        self.help_simulation_time_left_column_width = self.help_description_column_start
