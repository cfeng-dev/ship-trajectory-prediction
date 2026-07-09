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
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.path import Path

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

        # ==================================================
        # Important adjustable parameters
        # ==================================================

        # Simulation timing
        self.simulation_dt = 0.05
        self.update_interval_ms = 100

        # Default CSV export settings
        self.default_csv_filename = "simulated_ship_trajectory.csv"

        # Initial simulation values
        self.initial_speed = 5.0
        self.observation_noise_sigma = 0.2

        # Steering slider settings in degrees per second [°/s]
        self.min_steering_deg_per_second = -45
        self.max_steering_deg_per_second = 45
        self.steering_resolution = 1

        # Speed slider settings in meters per second [m/s]
        self.min_speed = 1.0
        self.max_speed = 10.0
        self.speed_resolution = 0.1

        # Keyboard control step sizes
        self.keyboard_speed_step = 0.1
        self.keyboard_steering_step = 1

        # Ship visualization size in meters [m]
        self.ship_length = 10.0
        self.ship_width = 3.0

        # Ship visualization colors
        self.ship_facecolor = "lightsalmon"
        self.ship_edgecolor = "darkred"
        self.ship_alpha = 0.85
        self.ship_linewidth = 1.5

        # Ship center marker style
        self.show_ship_center = True
        self.ship_center_color = "darkred"
        self.ship_center_size = 8

        # ==================================================
        # GUI and plot style colors
        # ==================================================

        # Soft application background color.
        self.app_background_color = "#eef7fb"

        # Slightly stronger color for the left control panel.
        self.control_panel_color = "#e3eef4"

        # Soft color for the right plot panel.
        self.plot_panel_color = "#f4f9fc"

        # Thin separator between control panel and plot panel.
        self.separator_color = "#b7cbd6"

        # Slight blue background to create a water feeling.
        self.plot_background_color = "#eaf6fb"

        # Trajectory line color.
        self.trajectory_color = "#2a6f97"
        self.trajectory_linewidth = 2.0

        # Start point color.
        self.start_position_color = "black"

        # Grid style.
        self.grid_color = "#8fbcd4"
        self.grid_alpha = 0.75
        self.grid_linewidth = 1.0

        # Figure background.
        self.figure_background_color = self.plot_panel_color

        # ==================================================
        # Plot axis settings
        # ==================================================
        # Available modes:
        # "auto"   = show the full trajectory automatically
        # "fixed"  = use fixed axis limits
        # "follow" = smoothly follow the current ship position
        self.axis_mode = "auto"

        # Extra space around the trajectory when axis_mode = "auto"
        self.auto_axis_margin = 15

        # Visible plot size when axis_mode = "follow"
        self.view_width = 100
        self.view_height = 100

        # Smooth camera movement when axis_mode = "follow"
        # Smaller value = smoother but slower following.
        # Larger value = faster but less smooth.
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.camera_smoothness = 0.06

        # Fixed plot axis range when axis_mode = "fixed"
        self.axis_x_min = -10
        self.axis_x_max = 50
        self.axis_y_min = -30
        self.axis_y_max = 30

        # Tick distance on x-axis and y-axis [m]
        self.axis_tick_step = 10

        # Window and plot size
        self.window_width = 1200
        self.window_height = 720

        # Keep figure square
        self.figure_width = 6
        self.figure_height = 6

        # Width of the left control panel in pixels.
        # This gives the status values enough space to stay readable.
        self.control_panel_width = 280

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
        # This allows three GUI states:
        # Stopped, Running, Paused.
        self.simulation_running = False
        self.simulation_started = False

        # Initialize camera at the ship start position.
        # This is only used when axis_mode = "follow".
        self.camera_x = self.simulator.x_current
        self.camera_y = self.simulator.y_current

        # ==================================================
        # GUI layout
        # ==================================================
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.configure(bg=self.app_background_color)

        self.create_menu_bar()
        self.create_widgets()
        self.bind_keyboard_controls()
        self.update_simulation_button()
        self.update_status()
        self.update_plot()

        # Start continuous simulation loop.
        self.simulation_loop()

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
            text="Ship Controls",
            font=("Arial", 14, "bold"),
            bg=self.control_panel_color,
        ).pack(pady=(0, 10))

        self.simulation_button = tk.Button(
            control_frame,
            text="Start Simulation",
            width=18,
            command=self.toggle_simulation,
        )
        self.simulation_button.pack(pady=5)

        tk.Button(
            control_frame,
            text="Save CSV",
            width=18,
            command=self.save_csv,
        ).pack(pady=(20, 5))

        tk.Button(
            control_frame,
            text="Reset",
            width=18,
            command=self.reset,
        ).pack(pady=5)

        # ==================================================
        # Steering controls
        # ==================================================
        tk.Label(
            control_frame,
            text="Steering",
            font=("Arial", 11, "bold"),
            bg=self.control_panel_color,
        ).pack(pady=(25, 0))

        tk.Label(
            control_frame,
            text="Left  ←   0 °/s   →  Right",
            width=24,
            anchor="center",
            bg=self.control_panel_color,
        ).pack()

        self.steering_slider = tk.Scale(
            control_frame,
            from_=self.min_steering_deg_per_second,
            to=self.max_steering_deg_per_second,
            orient=tk.HORIZONTAL,
            length=180,
            resolution=self.steering_resolution,
            bg=self.control_panel_color,
            highlightbackground=self.control_panel_color,
        )
        self.steering_slider.set(0)
        self.steering_slider.pack(pady=5)

        tk.Button(
            control_frame,
            text="Center Steering",
            width=18,
            command=self.center_steering,
        ).pack(pady=5)

        # ==================================================
        # Speed controls
        # ==================================================
        tk.Label(
            control_frame,
            text="Speed",
            font=("Arial", 11, "bold"),
            bg=self.control_panel_color,
        ).pack(pady=(20, 0))

        tk.Label(
            control_frame,
            text="Slow  ←   m/s   →  Fast",
            width=24,
            anchor="center",
            bg=self.control_panel_color,
        ).pack()

        self.speed_slider = tk.Scale(
            control_frame,
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
        self.speed_slider.pack(pady=5)

        # ==================================================
        # Status area
        # ==================================================
        status_frame = tk.LabelFrame(
            control_frame,
            text="Status",
            padx=10,
            pady=8,
            bg=self.control_panel_color,
        )
        status_frame.pack(pady=(20, 0), fill=tk.X)

        # The value labels do not use a fixed character width.
        # This avoids cutting off long numbers when the position grows.
        self.simulation_value_label = tk.Label(
            status_frame,
            anchor="w",
            justify="left",
            bg=self.control_panel_color,
        )
        self.position_value_label = tk.Label(
            status_frame,
            anchor="w",
            justify="left",
            bg=self.control_panel_color,
        )
        self.heading_value_label = tk.Label(
            status_frame,
            anchor="w",
            justify="left",
            bg=self.control_panel_color,
        )
        self.omega_value_label = tk.Label(
            status_frame,
            anchor="w",
            justify="left",
            bg=self.control_panel_color,
        )
        self.speed_value_label = tk.Label(
            status_frame,
            anchor="w",
            justify="left",
            bg=self.control_panel_color,
        )
        self.radius_value_label = tk.Label(
            status_frame,
            anchor="w",
            justify="left",
            bg=self.control_panel_color,
        )
        self.time_value_label = tk.Label(
            status_frame,
            anchor="w",
            justify="left",
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
                font=("Arial", 10, "bold"),
                bg=self.control_panel_color,
            ).grid(row=row_index, column=0, sticky="w", padx=(0, 8), pady=2)

            value_label.grid(row=row_index, column=1, sticky="ew", pady=2)

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

    def center_window_on_screen(self, window):
        """
        Center a child window on the screen.

        Parameters
        ----------
        window : tk.Toplevel
            Child window to be centered.
        """
        window.update_idletasks()

        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        window_width = window.winfo_width()
        window_height = window.winfo_height()

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        window.geometry(f"+{x}+{y}")

    def show_help(self):
        """
        Show a help window with keyboard shortcuts and basic usage.

        The simulation is paused before opening the help window so the
        trajectory does not keep changing while the user is reading the
        instructions.
        """
        self.pause_simulation()

        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_window.resizable(False, False)
        help_window.configure(bg=self.app_background_color)

        # Keep the help window above the main window.
        help_window.transient(self.root)
        help_window.grab_set()

        main_frame = tk.Frame(
            help_window,
            padx=24,
            pady=18,
            bg=self.app_background_color,
        )
        main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            main_frame,
            text="Simulation Controls",
            font=("Arial", 13, "bold"),
            anchor="w",
            bg=self.app_background_color,
        ).pack(fill=tk.X, pady=(0, 12))

        # ==================================================
        # Keyboard controls
        # ==================================================
        keyboard_frame = tk.LabelFrame(
            main_frame,
            text="Keyboard shortcuts",
            padx=12,
            pady=10,
            bg=self.app_background_color,
        )
        keyboard_frame.pack(fill=tk.X, pady=(0, 12))

        keyboard_shortcuts = [
            ("↑ / ↓", "Increase / decrease speed"),
            ("← / →", "Steer left / right"),
            ("Space", "Start / pause / continue simulation"),
            ("Ctrl + S", "Save trajectory data as CSV"),
        ]

        for row, (key, description) in enumerate(keyboard_shortcuts):
            tk.Label(
                keyboard_frame,
                text=key,
                width=10,
                anchor="w",
                font=("Arial", 10, "bold"),
                bg=self.app_background_color,
            ).grid(row=row, column=0, sticky="w", pady=2)

            tk.Label(
                keyboard_frame,
                text=description,
                anchor="w",
                bg=self.app_background_color,
            ).grid(row=row, column=1, sticky="w", pady=2)

        # ==================================================
        # Menu controls
        # ==================================================
        menu_frame = tk.LabelFrame(
            main_frame,
            text="Menu",
            padx=12,
            pady=10,
            bg=self.app_background_color,
        )
        menu_frame.pack(fill=tk.X, pady=(0, 16))

        menu_descriptions = [
            ("File → Save CSV", "Pause simulation and save trajectory data"),
            ("File → Exit", "Close the application"),
            ("Help → Show Help", "Open this help window"),
        ]

        for row, (menu_item, description) in enumerate(menu_descriptions):
            tk.Label(
                menu_frame,
                text=menu_item,
                width=20,
                anchor="w",
                font=("Arial", 10, "bold"),
                bg=self.app_background_color,
            ).grid(row=row, column=0, sticky="w", pady=2)

            tk.Label(
                menu_frame,
                text=description,
                anchor="w",
                bg=self.app_background_color,
            ).grid(row=row, column=1, sticky="w", pady=2)

        # ==================================================
        # Button controls
        # ==================================================
        button_frame = tk.LabelFrame(
            main_frame,
            text="Buttons",
            padx=12,
            pady=10,
            bg=self.app_background_color,
        )
        button_frame.pack(fill=tk.X, pady=(0, 16))

        button_descriptions = [
            ("Start Simulation", "Start the interactive simulation"),
            ("Pause Simulation", "Pause the simulation without clearing data"),
            ("Continue Simulation", "Continue a paused simulation"),
            ("Save CSV", "Pause simulation and save trajectory data"),
            ("Reset", "Stop simulation and clear trajectory"),
            ("Center Steering", "Reset steering to 0 °/s"),
        ]

        for row, (button, description) in enumerate(button_descriptions):
            tk.Label(
                button_frame,
                text=button,
                width=19,
                anchor="w",
                font=("Arial", 10, "bold"),
                bg=self.app_background_color,
            ).grid(row=row, column=0, sticky="w", pady=2)

            tk.Label(
                button_frame,
                text=description,
                anchor="w",
                bg=self.app_background_color,
            ).grid(row=row, column=1, sticky="w", pady=2)

        tk.Button(
            main_frame,
            text="OK",
            width=12,
            command=lambda: (help_window.destroy(), self.root.focus_set()),
        ).pack(anchor="e")

        # Restore keyboard focus after closing the help window.
        help_window.protocol(
            "WM_DELETE_WINDOW",
            lambda: (help_window.destroy(), self.root.focus_set()),
        )

        # Show the help window in the center of the screen.
        self.center_window_on_screen(help_window)

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

    def create_ship_shape(self, theta):
        """
        Create a simple ship-like polygon in real-world meter size.

        The shape is not a regular triangle:
        - sharp bow at the front
        - wider middle section
        - narrower stern at the back

        The ship points in the direction of theta.
        """
        length = self.ship_length
        width = self.ship_width

        # Ship points for theta = 0.
        # The ship points to the right.
        ship_shape = np.array(
            [
                [0.50 * length, 0.00 * width],  # Bow / front tip
                [0.20 * length, 0.50 * width],  # Front upper side
                [-0.35 * length, 0.40 * width],  # Rear upper side
                [-0.50 * length, 0.20 * width],  # Stern upper corner
                [-0.50 * length, -0.20 * width],  # Stern lower corner
                [-0.35 * length, -0.40 * width],  # Rear lower side
                [0.20 * length, -0.50 * width],  # Front lower side
            ]
        )

        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        return ship_shape @ rotation_matrix.T

    def create_ship_legend_marker(self):
        """
        Create a small custom ship marker for the legend.

        This marker has the same ship-like shape as the real polygon,
        but it is normalized for display inside the legend.
        """
        half_width = 0.35

        vertices = [
            (1.00, 0.00),  # Bow / front tip
            (0.40, half_width),  # Front upper side
            (-0.70, 0.28),  # Rear upper side
            (-1.00, 0.14),  # Stern upper corner
            (-1.00, -0.14),  # Stern lower corner
            (-0.70, -0.28),  # Rear lower side
            (0.40, -half_width),  # Front lower side
            (1.00, 0.00),  # Close polygon
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        return Path(vertices, codes)

    def add_heading_marker(self):
        """
        Add a ship marker at the current ship position.

        The marker is drawn in real-world meter size.
        """
        x = self.simulator.x_current
        y = self.simulator.y_current
        theta = self.simulator.theta_current

        ship_shape = self.create_ship_shape(theta)

        # Move ship shape to current position.
        ship_shape[:, 0] += x
        ship_shape[:, 1] += y

        ship_patch = Polygon(
            ship_shape,
            closed=True,
            facecolor=self.ship_facecolor,
            edgecolor=self.ship_edgecolor,
            linewidth=self.ship_linewidth,
            alpha=self.ship_alpha,
            label="Ship",
            zorder=5,
        )

        self.ax.add_patch(ship_patch)

    def add_ship_center_marker(self):
        """
        Add a small marker at the ship center.

        This shows the actual simulated position of the ship.
        """
        if not self.show_ship_center:
            return

        self.ax.scatter(
            self.simulator.x_current,
            self.simulator.y_current,
            s=self.ship_center_size,
            color=self.ship_center_color,
            marker="o",
            zorder=6,
        )

    def update_legend(self):
        """
        Update the plot legend with a fixed heading marker.

        The legend icon does not rotate, because it is only a symbol.
        The real heading marker in the plot still rotates with the ship heading.
        """
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=self.trajectory_color,
                linewidth=self.trajectory_linewidth,
                label="Trajectory",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=self.start_position_color,
                linestyle="None",
                markersize=8,
                label="Start",
            ),
            Line2D(
                [0],
                [0],
                marker=self.create_ship_legend_marker(),
                markerfacecolor=self.ship_facecolor,
                markeredgecolor=self.ship_edgecolor,
                color=self.ship_edgecolor,
                linestyle="None",
                markersize=14,
                label="Ship",
            ),
        ]

        self.ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            handlelength=1.8,
            handletextpad=0.8,
            borderpad=0.5,
        )

    def update_axis_limits(self):
        """
        Update plot axis limits.

        axis_mode = "auto":
            Show the full trajectory automatically with a square axis range.

        axis_mode = "fixed":
            Use fixed axis limits.

        axis_mode = "follow":
            Smoothly follow the current ship position.
        """
        if self.axis_mode == "auto":
            if self.simulator.has_data():
                x_values = list(self.simulator.x_all) + [self.simulator.x_current]
                y_values = list(self.simulator.y_all) + [self.simulator.y_current]

                x_min_data = min(x_values)
                x_max_data = max(x_values)
                y_min_data = min(y_values)
                y_max_data = max(y_values)

                x_center = (x_min_data + x_max_data) / 2
                y_center = (y_min_data + y_max_data) / 2

                x_range = (x_max_data - x_min_data) + 2 * self.auto_axis_margin
                y_range = (y_max_data - y_min_data) + 2 * self.auto_axis_margin

                # Use the larger range for both axes.
                # This keeps the visible plot area square in meters.
                plot_range = max(x_range, y_range)

                x_min = x_center - plot_range / 2
                x_max = x_center + plot_range / 2
                y_min = y_center - plot_range / 2
                y_max = y_center + plot_range / 2
            else:
                x_center = (self.axis_x_min + self.axis_x_max) / 2
                y_center = (self.axis_y_min + self.axis_y_max) / 2

                x_range = self.axis_x_max - self.axis_x_min
                y_range = self.axis_y_max - self.axis_y_min

                plot_range = max(x_range, y_range)

                x_min = x_center - plot_range / 2
                x_max = x_center + plot_range / 2
                y_min = y_center - plot_range / 2
                y_max = y_center + plot_range / 2

        elif self.axis_mode == "follow":
            ship_x = self.simulator.x_current
            ship_y = self.simulator.y_current

            # Smoothly move the camera toward the current ship position.
            self.camera_x += self.camera_smoothness * (ship_x - self.camera_x)
            self.camera_y += self.camera_smoothness * (ship_y - self.camera_y)

            x_min = self.camera_x - self.view_width / 2
            x_max = self.camera_x + self.view_width / 2
            y_min = self.camera_y - self.view_height / 2
            y_max = self.camera_y + self.view_height / 2

        else:
            x_min = self.axis_x_min
            x_max = self.axis_x_max
            y_min = self.axis_y_min
            y_max = self.axis_y_max

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

        x_tick_min = np.floor(x_min / self.axis_tick_step) * self.axis_tick_step
        x_tick_max = np.ceil(x_max / self.axis_tick_step) * self.axis_tick_step

        y_tick_min = np.floor(y_min / self.axis_tick_step) * self.axis_tick_step
        y_tick_max = np.ceil(y_max / self.axis_tick_step) * self.axis_tick_step

        self.ax.set_xticks(
            np.arange(
                x_tick_min,
                x_tick_max + self.axis_tick_step,
                self.axis_tick_step,
            )
        )
        self.ax.set_yticks(
            np.arange(
                y_tick_min,
                y_tick_max + self.axis_tick_step,
                self.axis_tick_step,
            )
        )

    def update_plot(self):
        """
        Update the trajectory plot.
        """
        self.ax.clear()

        # Reapply plot background after clearing.
        self.ax.set_facecolor(self.plot_background_color)

        if self.simulator.has_data():
            # Add the current position to the displayed trajectory.
            # This makes the blue line connect to the current heading marker.
            x_plot = list(self.simulator.x_all) + [self.simulator.x_current]
            y_plot = list(self.simulator.y_all) + [self.simulator.y_current]

            self.ax.plot(
                x_plot,
                y_plot,
                color=self.trajectory_color,
                linewidth=self.trajectory_linewidth,
                label="Trajectory",
            )

            self.ax.scatter(
                self.simulator.x_all[0],
                self.simulator.y_all[0],
                color=self.start_position_color,
                label="Start",
                zorder=4,
            )

            # Show current ship shape and heading direction.
            self.add_heading_marker()

            # Show the simulated ship center.
            self.add_ship_center_marker()

        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]", rotation=0, labelpad=15, va="center")
        self.ax.set_title("Ship Trajectory Simulation")

        # Keep 1 meter on x-axis equal to 1 meter on y-axis.
        self.ax.set_aspect("equal", adjustable="box")

        self.ax.grid(
            True,
            color=self.grid_color,
            alpha=self.grid_alpha,
            linewidth=self.grid_linewidth,
        )

        # Update axis range and ticks depending on axis_mode.
        self.update_axis_limits()

        if self.simulator.has_data():
            self.update_legend()

        self.canvas.draw()
