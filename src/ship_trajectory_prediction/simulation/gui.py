"""Interactive GUI for steering, controlling, and exporting a 2D simulation."""

import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np

from ship_trajectory_prediction.simulation.config import GUIConfig
from ship_trajectory_prediction.simulation.controls import (
    create_gui_widgets,
    create_styled_button,
)
from ship_trajectory_prediction.simulation.core import ShipSimulator
from ship_trajectory_prediction.simulation.help import show_help_window
from ship_trajectory_prediction.simulation.io import (
    DATA_DIR,
    create_simulation_dataframe,
    save_trajectory_data,
)
from ship_trajectory_prediction.simulation.plotting import (
    update_plot as draw_ship_plot,
)
from ship_trajectory_prediction.trajectory.coordinates import (
    METERS_PER_KILOMETER,
    local_to_gps_coordinates,
)


def _parse_gps_start_position(latitude_text, longitude_text):
    """Parse and validate a user-provided GPS start position."""
    try:
        latitude = float(latitude_text)
        longitude = float(longitude_text)
    except ValueError as error:
        raise ValueError("Latitude and longitude must be numeric values.") from error

    if not np.isfinite(latitude) or not np.isfinite(longitude):
        raise ValueError("Latitude and longitude must be finite values.")
    if not -90 < latitude < 90:
        raise ValueError("Latitude must be greater than -90° and less than 90°.")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180° and 180°.")

    return latitude, longitude


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
        """
        Create the desktop-style application menu bar.
        """
        menu_bar = tk.Menu(self.root)

        # ==================================================
        # File menu
        # ==================================================
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(
            label="Save CSV",
            accelerator="Ctrl+S",
            command=self.save_csv,
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Exit",
            command=self.root.destroy,
        )
        menu_bar.add_cascade(label="File", menu=file_menu)

        # ==================================================
        # View menu
        # ==================================================
        view_menu = tk.Menu(menu_bar, tearoff=0)
        self.coordinate_display_var = tk.StringVar(value=self.coordinate_display_mode)
        coordinate_display_modes = (
            ("Local [m]", "local"),
            ("Local [km]", "km"),
            ("GPS [°]", "gps"),
        )
        for label, mode in coordinate_display_modes:
            view_menu.add_radiobutton(
                label=label,
                variable=self.coordinate_display_var,
                value=mode,
                command=self.change_coordinate_display,
            )
        menu_bar.add_cascade(label="View", menu=view_menu)

        # ==================================================
        # Settings menu
        # ==================================================
        self.settings_menu = tk.Menu(menu_bar, tearoff=0)
        self.settings_menu.add_command(
            label="GPS Start Position...",
            command=self.show_gps_start_position_dialog,
        )
        self.gps_position_menu_index = self.settings_menu.index("end")
        menu_bar.add_cascade(label="Settings", menu=self.settings_menu)

        # ==================================================
        # Help menu
        # ==================================================
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(
            label="Show Help",
            command=self.show_help,
        )
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menu_bar)

    def create_widgets(self):
        """
        Create buttons, sliders, labels, and plot area.
        """
        create_gui_widgets(self)

    def bind_keyboard_controls(self):
        """
        Bind keyboard keys to speed, steering, simulation controls, and CSV export.
        """
        self.root.bind("<Up>", self.increase_speed)
        self.root.bind("<Down>", self.decrease_speed)
        self.root.bind("<Left>", self.steer_left)
        self.root.bind("<Right>", self.steer_right)
        self.root.bind("<space>", self.toggle_simulation_with_keyboard)
        self.root.bind("<Escape>", self.exit_fullscreen)

        # Save CSV with Ctrl + S.
        self.root.bind("<Control-s>", self.save_csv_with_keyboard)
        self.root.bind("<Control-S>", self.save_csv_with_keyboard)
        self.root.bind("<Command-s>", self.save_csv_with_keyboard)
        self.root.bind("<Command-S>", self.save_csv_with_keyboard)

        # Make sure the main window can receive keyboard input.
        self.root.focus_set()

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
        self.root.attributes("-fullscreen", False)
        return "break"

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

        self.update_gps_start_position_menu()

    def update_gps_start_position_menu(self):
        """Lock the GPS origin setting while a simulation run is active."""
        menu_state = tk.DISABLED if self.simulation_started else tk.NORMAL
        self.settings_menu.entryconfig(
            self.gps_position_menu_index,
            state=menu_state,
        )

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
        if self.simulation_started:
            messagebox.showwarning(
                "GPS Position Locked",
                "Reset the simulation before changing the GPS start position.",
            )
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("GPS Start Position")
        dialog.resizable(False, False)
        dialog.configure(bg=self.app_background_color)
        dialog.transient(self.root)
        dialog.grab_set()

        form_frame = tk.Frame(
            dialog,
            bg=self.app_background_color,
            padx=18,
            pady=16,
        )
        form_frame.pack(fill=tk.BOTH, expand=True)
        form_frame.columnconfigure(1, weight=1)

        latitude_var = tk.StringVar(value=f"{self.reference_latitude:.8f}")
        longitude_var = tk.StringVar(value=f"{self.reference_longitude:.8f}")

        tk.Label(
            form_frame,
            text="Latitude [°]:",
            anchor="w",
            bg=self.app_background_color,
            fg="black",
        ).grid(row=0, column=0, sticky="w", padx=(0, 10), pady=4)
        latitude_entry = tk.Entry(
            form_frame,
            textvariable=latitude_var,
            width=20,
            justify=tk.RIGHT,
            bg="white",
            fg="black",
            insertbackground="black",
            selectbackground="#007aff",
            selectforeground="white",
            relief=tk.SOLID,
            borderwidth=1,
            highlightthickness=0,
        )
        latitude_entry.grid(row=0, column=1, sticky="ew", pady=4)

        tk.Label(
            form_frame,
            text="Longitude [°]:",
            anchor="w",
            bg=self.app_background_color,
            fg="black",
        ).grid(row=1, column=0, sticky="w", padx=(0, 10), pady=4)
        longitude_entry = tk.Entry(
            form_frame,
            textvariable=longitude_var,
            width=20,
            justify=tk.RIGHT,
            bg="white",
            fg="black",
            insertbackground="black",
            selectbackground="#007aff",
            selectforeground="white",
            relief=tk.SOLID,
            borderwidth=1,
            highlightthickness=0,
        )
        longitude_entry.grid(row=1, column=1, sticky="ew", pady=4)

        button_frame = tk.Frame(form_frame, bg=self.app_background_color)
        button_frame.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="e",
            pady=(14, 0),
        )

        def apply_position():
            if self.apply_gps_start_position(
                latitude_var.get(),
                longitude_var.get(),
                parent=dialog,
            ):
                dialog.destroy()

        create_styled_button(
            button_frame,
            text="Apply",
            width=10,
            command=apply_position,
        ).pack(side=tk.LEFT, padx=(0, 8))
        create_styled_button(
            button_frame,
            text="Cancel",
            width=10,
            command=dialog.destroy,
        ).pack(side=tk.LEFT)

        dialog.bind("<Return>", lambda _event: apply_position())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())

        dialog.update_idletasks()
        x = (
            self.root.winfo_rootx()
            + (self.root.winfo_width() - dialog.winfo_width()) // 2
        )
        y = (
            self.root.winfo_rooty()
            + (self.root.winfo_height() - dialog.winfo_height()) // 2
        )
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
        dialog.focus_set()

    def apply_gps_start_position(self, latitude_text, longitude_text, parent=None):
        """Validate and apply the GPS position of the local simulation origin."""
        if self.simulation_started:
            messagebox.showwarning(
                "GPS Position Locked",
                "Reset the simulation before changing the GPS start position.",
                parent=parent,
            )
            return False

        try:
            latitude, longitude = _parse_gps_start_position(
                latitude_text,
                longitude_text,
            )
        except ValueError as error:
            messagebox.showerror(
                "Invalid GPS Position",
                str(error),
                parent=parent,
            )
            return False

        self.reference_latitude = latitude
        self.reference_longitude = longitude

        self.update_status()
        self.update_plot()

        return True

    def save_csv(self):
        """
        Save the simulated trajectory as CSV.

        The user can choose the output folder and filename.
        By default, the file dialog opens in data/simulated.

        The simulation is paused before opening the file dialog.
        This prevents trajectory data from changing while the user is choosing
        the output filename or folder.
        """
        if not self.simulator.has_data():
            messagebox.showwarning(
                "No Data",
                "No trajectory data available. Please start the simulation first.",
            )
            return

        # Pause simulation before saving so the data stays unchanged
        # while the file dialog is open.
        self.pause_simulation()

        # Make sure the default data directory exists.
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        output_path = filedialog.asksaveasfilename(
            title="Save or append trajectory data",
            initialdir=DATA_DIR,
            initialfile=self.default_csv_filename,
            defaultextension=".csv",
            confirmoverwrite=False,
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )

        # If the user cancels the dialog, do nothing.
        if not output_path:
            return

        trajectory_df = create_simulation_dataframe(
            simulator=self.simulator,
            random_seed=42,
            start_time=self.simulation_start_time,
            reference_longitude=self.reference_longitude,
            reference_latitude=self.reference_latitude,
        )

        resolved_output_path = Path(output_path).resolve()
        saved_run_state = self.saved_run_states.get(resolved_output_path)
        if (
            not resolved_output_path.is_file()
            or resolved_output_path.stat().st_size == 0
        ):
            saved_run_state = None

        if saved_run_state is None:
            existing_run_id = None
            first_unsaved_sample = 0
        else:
            existing_run_id, first_unsaved_sample = saved_run_state

        if first_unsaved_sample >= len(trajectory_df):
            messagebox.showinfo(
                "No New Data",
                "No new trajectory samples are available.\n\n"
                f"Run ID: {existing_run_id}\n"
                f"File:\n{resolved_output_path}",
            )
            return

        try:
            save_result = save_trajectory_data(
                df=trajectory_df.iloc[first_unsaved_sample:],
                filename=output_path,
                existing_run_id=existing_run_id,
            )
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "Save Failed",
                f"Could not save trajectory data:\n\n{error}",
            )
            return

        self.saved_run_states[save_result.output_path.resolve()] = (
            save_result.run_id,
            len(trajectory_df),
        )

        if save_result.continued:
            message_title = "Trajectory Updated"
            message_text = (
                "New samples appended to the current trajectory run.\n\n"
                f"Run ID: {save_result.run_id}\n"
                f"File:\n{save_result.output_path}"
            )
        elif save_result.appended:
            message_title = "Trajectory Appended"
            message_text = (
                "New trajectory run appended to the existing CSV file.\n\n"
                f"Run ID: {save_result.run_id}\n"
                f"File:\n{save_result.output_path}"
            )
        else:
            message_title = "Trajectory Saved"
            message_text = (
                "New CSV file created.\n\n"
                f"Run ID: {save_result.run_id}\n"
                f"File:\n{save_result.output_path}"
            )

        messagebox.showinfo(
            message_title,
            message_text,
        )

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
        """
        Update the status display.
        """
        heading_deg = np.rad2deg(self.simulator.theta_current)
        omega = self.get_omega_from_steering()
        omega_deg = np.rad2deg(omega)
        speed = self.get_speed_from_slider()
        simulation_state = self.get_simulation_state()

        if abs(omega) < 1e-8:
            radius_text = "∞"
        else:
            radius_text = f"{speed / omega:.2f} m"

        if simulation_state == "running":
            simulation_text = "Running"
            simulation_color = "darkgreen"
        elif simulation_state == "paused":
            simulation_text = "Paused"
            simulation_color = "orange"
        else:
            simulation_text = "Stopped"
            simulation_color = "darkred"

        self.simulation_value_label.config(
            text=simulation_text,
            fg=simulation_color,
        )

        if self.coordinate_display_mode == "gps":
            longitude, latitude = local_to_gps_coordinates(
                [self.simulator.x_current],
                [self.simulator.y_current],
                reference_longitude=self.reference_longitude,
                reference_latitude=self.reference_latitude,
            )
            position_text = f"lon = {longitude[0]:.4f}°\nlat = {latitude[0]:.4f}°"
        elif self.coordinate_display_mode == "km":
            position_text = (
                f"x = {self.simulator.x_current / METERS_PER_KILOMETER:.3f} km\n"
                f"y = {self.simulator.y_current / METERS_PER_KILOMETER:.3f} km"
            )
        else:
            position_text = (
                f"x = {self.simulator.x_current:.2f} m\n"
                f"y = {self.simulator.y_current:.2f} m"
            )

        self.position_value_label.config(
            text=position_text,
        )
        self.heading_value_label.config(text=f"{heading_deg:.1f}°")
        self.omega_value_label.config(text=f"{omega_deg:.1f}°/s")
        self.speed_value_label.config(text=f"{speed:.2f} m/s")
        self.radius_value_label.config(text=radius_text)
        self.time_value_label.config(text=f"{self.simulator.current_time:.1f} s")

    def update_plot(self):
        """
        Update the trajectory plot.
        """
        draw_ship_plot(self)
