"""Interactive GUI for steering, controlling, and exporting a 2D simulation."""

from datetime import datetime, timezone

import numpy as np

from ship_trajectory_prediction.simulation.config import GUIConfig
from ship_trajectory_prediction.simulation.controls import (
    bind_keyboard_controls as bind_gui_keyboard_controls,
)
from ship_trajectory_prediction.simulation.controls import (
    create_gui_widgets,
)
from ship_trajectory_prediction.simulation.core import ShipSimulator
from ship_trajectory_prediction.simulation.dialogs import (
    apply_gps_start_position as apply_gps_position,
)
from ship_trajectory_prediction.simulation.dialogs import (
    save_csv as save_trajectory_csv,
)
from ship_trajectory_prediction.simulation.dialogs import (
    show_gps_start_position_dialog as open_gps_start_position_dialog,
)
from ship_trajectory_prediction.simulation.help import show_help_window
from ship_trajectory_prediction.simulation.plotting import (
    update_plot as draw_ship_plot,
)
from ship_trajectory_prediction.simulation.view import (
    create_menu_bar as build_menu_bar,
)
from ship_trajectory_prediction.simulation.view import (
    update_status_display,
)


class ShipTrajectoryGUI:
    """
    GUI for interactively steering a 2D ship trajectory.

    The ship moves continuously when the simulation is running.
    The steering slider controls the angular velocity.
    The speed slider controls the ship velocity.

    Keyboard controls:
    - Up / Down arrows increase or decrease speed.
    - Left / Right arrows steer the ship.
    - Space starts, pauses, or continues the simulation.
    - Ctrl + S saves the trajectory as CSV.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("2D Ship Trajectory Simulator")

        # Load all adjustable GUI, plot, and simulation settings.
        self.config = GUIConfig()
        self.apply_config()

        # ==================================================
        # Simulation object
        # ==================================================
        self.simulator = ShipSimulator(
            v=self.initial_speed,
            sigma=self.observation_noise_sigma,
            dt=self.simulation_dt,
        )

        # Simulation state:
        # - simulation_running = True means the ship is moving.
        # - simulation_started = True means the simulation was started at least once.
        # This allows three GUI states: Stopped, Running, Paused.
        self.simulation_running = False
        self.simulation_started = False
        self.simulation_start_time = None

        # Remember how much of the current simulation session was saved to
        # each CSV file. Reset starts a new session and clears this mapping.
        self.saved_run_states = {}

        # Initialize camera at the ship start position.
        # This is only used when axis_mode = "follow".
        self.camera_x = self.simulator.x_current
        self.camera_y = self.simulator.y_current

        # ==================================================
        # GUI layout
        # ==================================================
        self.center_main_window_on_screen()
        self.root.configure(bg=self.app_background_color)

        self.create_menu_bar()
        self.create_widgets()
        self.bind_keyboard_controls()
        self.update_simulation_button()
        self.update_status()
        self.update_plot()

        # Start continuous simulation loop.
        self.simulation_loop()

    def apply_config(self):
        """
        Copy configuration values to this GUI object.

        This keeps the rest of the class readable because settings can still be
        accessed as self.window_width, self.trajectory_color, etc.
        """
        for setting_name, setting_value in vars(self.config).items():
            setattr(self, setting_name, setting_value)

    def center_main_window_on_screen(self):
        """
        Center the main application window on the screen.
        """
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width - self.window_width) // 2
        y = (screen_height - self.window_height) // 2 - self.main_window_vertical_offset

        self.root.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")

    def create_menu_bar(self):
        """Create the desktop-style application menu bar."""
        build_menu_bar(self)

    def create_widgets(self):
        """
        Create buttons, sliders, labels, and plot area.
        """
        create_gui_widgets(self)

    def bind_keyboard_controls(self):
        """Bind keyboard controls to the main window."""
        bind_gui_keyboard_controls(self)

    def show_help(self):
        """
        Show a help window with keyboard shortcuts and basic usage.
        """
        show_help_window(
            root=self.root,
            app_background_color=self.app_background_color,
            pause_callback=self.pause_simulation,
            focus_callback=self.root.focus_set,
            keyboard_left_column_width=self.help_keyboard_left_column_width,
            menu_left_column_width=self.help_menu_left_column_width,
            button_left_column_width=self.help_button_left_column_width,
            simulation_time_left_column_width=(
                self.help_simulation_time_left_column_width
            ),
        )

    def increase_speed(self, event=None):
        """
        Increase ship speed using the keyboard.
        """
        new_speed = min(
            self.speed_slider.get() + self.keyboard_speed_step,
            self.max_speed,
        )

        self.speed_slider.set(new_speed)
        self.update_status()

    def decrease_speed(self, event=None):
        """
        Decrease ship speed using the keyboard.
        """
        new_speed = max(
            self.speed_slider.get() - self.keyboard_speed_step,
            self.min_speed,
        )

        self.speed_slider.set(new_speed)
        self.update_status()

    def steer_left(self, event=None):
        """
        Steer the ship to the left using the keyboard.
        """
        new_steering = max(
            self.steering_slider.get() - self.keyboard_steering_step,
            self.min_steering_deg_per_second,
        )

        self.steering_slider.set(new_steering)
        self.update_status()

    def steer_right(self, event=None):
        """
        Steer the ship to the right using the keyboard.
        """
        new_steering = min(
            self.steering_slider.get() + self.keyboard_steering_step,
            self.max_steering_deg_per_second,
        )

        self.steering_slider.set(new_steering)
        self.update_status()

    def toggle_simulation_with_keyboard(self, event=None):
        """
        Start, pause, or continue the simulation using the space key.
        """
        self.toggle_simulation()

        # Prevent the space key from also triggering focused buttons.
        return "break"

    def save_csv_with_keyboard(self, event=None):
        """
        Save the trajectory as CSV using Ctrl + S.
        """
        self.save_csv()

        # Prevent other default Ctrl + S behavior.
        return "break"

    def exit_fullscreen(self, event=None):
        """Leave full-screen mode when Escape is pressed."""
        self.set_fullscreen(False)
        return "break"

    def toggle_fullscreen(self, event=None):
        """Toggle full-screen mode from the keyboard or View menu."""
        is_fullscreen = bool(self.root.attributes("-fullscreen"))
        self.set_fullscreen(not is_fullscreen)
        return "break"

    def set_fullscreen(self, is_fullscreen):
        """Set full-screen mode and update the corresponding menu label."""
        self.root.attributes("-fullscreen", is_fullscreen)

        if self.fullscreen_menu_index is not None:
            menu_label = "Exit Full Screen" if is_fullscreen else "Enter Full Screen"
            self.view_menu.entryconfigure(
                self.fullscreen_menu_index,
                label=menu_label,
            )

    def get_omega_from_steering(self):
        """
        Convert steering slider value from degrees per second to rad/s.

        Returns
        -------
        omega : float
            Angular velocity in rad/s.
        """
        steering_deg_per_second = self.steering_slider.get()

        # Positive slider value means steering to the right.
        # In mathematical coordinates, positive omega turns left,
        # therefore the sign is inverted here.
        omega = -np.deg2rad(steering_deg_per_second)

        return omega

    def get_speed_from_slider(self):
        """
        Get ship velocity from speed slider.

        Returns
        -------
        speed : float
            Ship velocity in m/s.
        """
        speed = self.speed_slider.get()

        return speed

    def simulation_loop(self):
        """
        Continuously run the simulation loop.
        """
        omega = self.get_omega_from_steering()

        # Update ship speed from speed slider.
        self.simulator.v = self.get_speed_from_slider()

        self.simulator.step(
            omega=omega,
            motor_running=self.simulation_running,
        )

        self.update_status()
        self.update_plot()

        self.root.after(self.update_interval_ms, self.simulation_loop)

    def get_simulation_state(self):
        """
        Return the current simulation state as text.

        Returns
        -------
        state : str
            One of "stopped", "running", or "paused".
        """
        if self.simulation_running:
            return "running"

        if self.simulation_started:
            return "paused"

        return "stopped"

    def update_simulation_button(self):
        """
        Update the simulation button text based on the current state.
        """
        state = self.get_simulation_state()

        if state == "running":
            self.simulation_button.config(text="Pause Simulation")
        elif state == "paused":
            self.simulation_button.config(text="Continue Simulation")
        else:
            self.simulation_button.config(text="Start Simulation")

    def toggle_simulation(self):
        """
        Start, pause, or continue the simulation.
        """
        if self.simulation_running:
            self.pause_simulation()
        else:
            self.start_or_continue_simulation()

    def start_or_continue_simulation(self):
        """
        Start the simulation for the first time or continue after pausing.
        """
        if not self.simulation_started:
            self.simulation_start_time = datetime.now(timezone.utc).replace(
                microsecond=0
            )

        self.simulation_running = True
        self.simulation_started = True

        self.update_simulation_button()
        self.update_status()

    def pause_simulation(self):
        """
        Pause the simulation without resetting trajectory data.
        """
        if not self.simulation_started:
            return

        self.simulation_running = False

        self.update_simulation_button()
        self.update_status()

    def center_steering(self):
        """
        Reset steering slider to center.
        """
        self.steering_slider.set(0)

    def change_coordinate_display(self):
        """Switch the position display between meters, kilometers, and GPS."""
        self.coordinate_display_mode = self.coordinate_display_var.get()
        self.update_status()
        self.update_plot()

    def show_gps_start_position_dialog(self):
        """Open a dialog for configuring the GPS position of the local origin."""
        open_gps_start_position_dialog(self)

    def apply_gps_start_position(self, latitude_text, longitude_text, parent=None):
        """Validate and apply the GPS position of the local simulation origin."""
        return apply_gps_position(
            self,
            latitude_text,
            longitude_text,
            parent=parent,
        )

    def save_csv(self):
        """Save the simulated trajectory as CSV."""
        save_trajectory_csv(self)

    def reset(self):
        """
        Reset the simulation and clear the plot.
        """
        self.simulation_running = False
        self.simulation_started = False
        self.simulation_start_time = None
        self.saved_run_states.clear()

        self.steering_slider.set(0)
        self.speed_slider.set(self.initial_speed)

        self.simulator.reset()
        self.simulator.v = self.get_speed_from_slider()

        # Reset camera to the ship start position.
        # This is only used when axis_mode = "follow".
        self.camera_x = self.simulator.x_current
        self.camera_y = self.simulator.y_current

        self.update_simulation_button()
        self.update_status()
        self.update_plot()

    def update_status(self):
        """Update the status display."""
        update_status_display(self)

    def update_plot(self):
        """
        Update the trajectory plot.
        """
        draw_ship_plot(self)
