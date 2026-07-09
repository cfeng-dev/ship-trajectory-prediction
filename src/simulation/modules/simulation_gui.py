"""
@file simulation_gui.py
@description Provides a GUI to steer a simple 2D ship trajectory with simulation control and CSV export.
@date Created on: 06.07.2026
@author C.Feng
"""

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from modules.gui_config import GUIConfig
from modules.gui_help import show_help_window
from modules.gui_plot import update_plot as draw_ship_plot
from modules.simulation_core import ShipSimulator
from modules.simulation_io import (
    DATA_DIR,
    create_simulation_dataframe,
    save_trajectory_data,
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
        Create a simple desktop-style menu bar with File and Help menus.
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
        # Main content frame below the menu bar.
        content_frame = tk.Frame(
            self.root,
            bg=self.app_background_color,
        )
        content_frame.pack(fill=tk.BOTH, expand=True)

        # ==================================================
        # Left control panel
        # ==================================================
        control_frame = tk.Frame(
            content_frame,
            width=self.control_panel_width,
            bg=self.control_panel_color,
            padx=14,
            pady=14,
        )
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0), pady=10)
        control_frame.pack_propagate(False)

        tk.Label(
            control_frame,
            text="Simulation Control",
            font=("Arial", 12, "bold"),
            bg=self.control_panel_color,
        ).pack(pady=(0, 12))

        # ==================================================
        # Action buttons
        # ==================================================
        action_frame = tk.LabelFrame(
            control_frame,
            text="Actions",
            padx=10,
            pady=8,
            bg=self.control_panel_color,
        )
        action_frame.pack(fill=tk.X, pady=(0, self.section_spacing))

        self.simulation_button = tk.Button(
            action_frame,
            text="Start Simulation",
            width=18,
            command=self.toggle_simulation,
        )
        self.simulation_button.pack(pady=(2, 5))

        tk.Button(
            action_frame,
            text="Save CSV",
            width=18,
            command=self.save_csv,
        ).pack(pady=5)

        tk.Button(
            action_frame,
            text="Reset",
            width=18,
            command=self.reset,
        ).pack(pady=(5, 2))

        # ==================================================
        # Steering controls
        # ==================================================
        steering_frame = tk.LabelFrame(
            control_frame,
            text="Steering",
            padx=10,
            pady=8,
            bg=self.control_panel_color,
        )
        steering_frame.pack(fill=tk.X, pady=(0, self.section_spacing))

        tk.Label(
            steering_frame,
            text="Left  ←   0 °/s   →  Right",
            width=24,
            anchor="center",
            bg=self.control_panel_color,
        ).pack(pady=(0, 2))

        self.steering_slider = tk.Scale(
            steering_frame,
            from_=self.min_steering_deg_per_second,
            to=self.max_steering_deg_per_second,
            orient=tk.HORIZONTAL,
            length=180,
            resolution=self.steering_resolution,
            bg=self.control_panel_color,
            highlightbackground=self.control_panel_color,
        )
        self.steering_slider.set(0)
        self.steering_slider.pack(pady=(0, 5))

        tk.Button(
            steering_frame,
            text="Center Steering",
            width=18,
            command=self.center_steering,
        ).pack(pady=(2, 0))

        # ==================================================
        # Speed controls
        # ==================================================
        speed_frame = tk.LabelFrame(
            control_frame,
            text="Speed",
            padx=10,
            pady=8,
            bg=self.control_panel_color,
        )
        speed_frame.pack(fill=tk.X, pady=(0, self.section_spacing))

        tk.Label(
            speed_frame,
            text="Slow  ←   m/s   →  Fast",
            width=24,
            anchor="center",
            bg=self.control_panel_color,
        ).pack(pady=(0, 2))

        self.speed_slider = tk.Scale(
            speed_frame,
            from_=self.min_speed,
            to=self.max_speed,
            orient=tk.HORIZONTAL,
            length=180,
            resolution=self.speed_resolution,
            digits=3,
            bg=self.control_panel_color,
            highlightbackground=self.control_panel_color,
        )
        self.speed_slider.set(self.initial_speed)
        self.speed_slider.pack(pady=(0, 2))

        # ==================================================
        # Status area
        # ==================================================
        status_frame = tk.LabelFrame(
            control_frame,
            text="Status",
            padx=8,
            pady=6,
            bg=self.control_panel_color,
        )
        status_frame.pack(pady=(0, 0), fill=tk.X)

        # The value labels do not use a fixed character width.
        # This avoids cutting off long numbers when the position grows.
        self.simulation_value_label = tk.Label(
            status_frame,
            anchor="w",
            justify="left",
            font=self.status_value_font,
            bg=self.control_panel_color,
        )
        self.position_value_label = tk.Label(
            status_frame,
            anchor="w",
            justify="left",
            font=self.status_value_font,
            bg=self.control_panel_color,
        )
        self.heading_value_label = tk.Label(
            status_frame,
            anchor="w",
            justify="left",
            font=self.status_value_font,
            bg=self.control_panel_color,
        )
        self.omega_value_label = tk.Label(
            status_frame,
            anchor="w",
            justify="left",
            font=self.status_value_font,
            bg=self.control_panel_color,
        )
        self.speed_value_label = tk.Label(
            status_frame,
            anchor="w",
            justify="left",
            font=self.status_value_font,
            bg=self.control_panel_color,
        )
        self.radius_value_label = tk.Label(
            status_frame,
            anchor="w",
            justify="left",
            font=self.status_value_font,
            bg=self.control_panel_color,
        )
        self.time_value_label = tk.Label(
            status_frame,
            anchor="w",
            justify="left",
            font=self.status_value_font,
            bg=self.control_panel_color,
        )

        status_frame.columnconfigure(0, weight=0)
        status_frame.columnconfigure(1, weight=1)

        status_rows = [
            ("Simulation:", self.simulation_value_label),
            ("Position:", self.position_value_label),
            ("Heading:", self.heading_value_label),
            ("Omega:", self.omega_value_label),
            ("Speed:", self.speed_value_label),
            ("Radius:", self.radius_value_label),
            ("Time:", self.time_value_label),
        ]

        for row_index, (label_text, value_label) in enumerate(status_rows):
            tk.Label(
                status_frame,
                text=label_text,
                anchor="w",
                width=10,
                font=self.status_label_font,
                bg=self.control_panel_color,
            ).grid(
                row=row_index,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=self.status_row_padding_y,
            )

            value_label.grid(
                row=row_index,
                column=1,
                sticky="ew",
                pady=self.status_row_padding_y,
            )

        # ==================================================
        # Vertical separator between left and right area
        # ==================================================
        separator = tk.Frame(
            content_frame,
            bg=self.separator_color,
            width=2,
        )
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=10)

        # ==================================================
        # Right plot panel
        # ==================================================
        plot_frame = tk.Frame(
            content_frame,
            bg=self.plot_panel_color,
            padx=10,
            pady=10,
        )
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=10)

        self.figure = Figure(
            figsize=(self.figure_width, self.figure_height),
            facecolor=self.figure_background_color,
        )
        self.figure.patch.set_facecolor(self.figure_background_color)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(self.plot_background_color)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)

        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(
            bg=self.plot_panel_color,
            highlightbackground=self.plot_panel_color,
            highlightthickness=0,
        )
        canvas_widget.pack(fill=tk.BOTH, expand=True)

    def bind_keyboard_controls(self):
        """
        Bind keyboard keys to speed, steering, simulation controls, and CSV export.
        """
        self.root.bind("<Up>", self.increase_speed)
        self.root.bind("<Down>", self.decrease_speed)
        self.root.bind("<Left>", self.steer_left)
        self.root.bind("<Right>", self.steer_right)
        self.root.bind("<space>", self.toggle_simulation_with_keyboard)

        # Save CSV with Ctrl + S.
        self.root.bind("<Control-s>", self.save_csv_with_keyboard)
        self.root.bind("<Control-S>", self.save_csv_with_keyboard)

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
            title="Save trajectory data",
            initialdir=DATA_DIR,
            initialfile=self.default_csv_filename,
            defaultextension=".csv",
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
        )

        output_path = save_trajectory_data(
            df=trajectory_df,
            filename=output_path,
        )

        messagebox.showinfo(
            "Saved",
            f"Trajectory data saved to:\n{output_path}",
        )

    def reset(self):
        """
        Reset the simulation and clear the plot.
        """
        self.simulation_running = False
        self.simulation_started = False

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
        self.position_value_label.config(
            text=(
                f"x = {self.simulator.x_current:.2f} m\n"
                f"y = {self.simulator.y_current:.2f} m"
            )
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
