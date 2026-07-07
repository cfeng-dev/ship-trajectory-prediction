"""
@file interactive_ship_gui.py
@description Provides a GUI to steer a simple 2D ship trajectory with motor control and CSV export.
@date Created on: 06.07.2026
@author C.Feng
"""

import tkinter as tk
from tkinter import messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.path import Path

from modules.ship_simulator import ShipSimulator
from modules.simulation_io import (
    create_simulation_dataframe,
    save_trajectory_data,
)


class ShipTrajectoryGUI:
    """
    GUI for interactively steering a 2D ship trajectory.

    The ship moves continuously when the motor is running.
    The steering slider controls the angular velocity.
    The speed slider controls the ship velocity.
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
        # Plot style colors
        # ==================================================

        # Slight blue background to create a water feeling
        self.plot_background_color = "#eaf6fb"

        # Trajectory line color
        self.trajectory_color = "#2a6f97"
        self.trajectory_linewidth = 2.0

        # Start point color
        self.start_position_color = "black"

        # Grid style
        self.grid_color = "#8fbcd4"
        self.grid_alpha = 0.75
        self.grid_linewidth = 1.0

        # Figure background
        self.figure_background_color = "white"

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
        self.window_width = 1100
        self.window_height = 700

        # Keep figure square
        self.figure_width = 6
        self.figure_height = 6

        # ==================================================
        # Simulation object
        # ==================================================
        self.simulator = ShipSimulator(
            v=self.initial_speed,
            sigma=self.observation_noise_sigma,
            dt=self.simulation_dt,
        )

        self.motor_running = False

        # Initialize camera at the ship start position.
        # This is only used when axis_mode = "follow".
        self.camera_x = self.simulator.x_current
        self.camera_y = self.simulator.y_current

        # ==================================================
        # GUI layout
        # ==================================================
        self.create_widgets()
        self.update_status()
        self.update_plot()

        # Start continuous simulation loop
        self.simulation_loop()

    def create_widgets(self):
        """
        Create buttons, sliders, labels, and plot area.
        """
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(
            control_frame,
            text="Ship Controls",
            font=("Arial", 14, "bold"),
        ).pack(pady=(0, 10))

        self.motor_button = tk.Button(
            control_frame,
            text="Start Motor",
            width=18,
            command=self.toggle_motor,
        )
        self.motor_button.pack(pady=5)

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
        ).pack(pady=(25, 0))

        tk.Label(
            control_frame,
            text="Left  ←   0 °/s   →  Right",
            width=24,
            anchor="center",
        ).pack()

        self.steering_slider = tk.Scale(
            control_frame,
            from_=self.min_steering_deg_per_second,
            to=self.max_steering_deg_per_second,
            orient=tk.HORIZONTAL,
            length=180,
            resolution=self.steering_resolution,
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
        ).pack(pady=(20, 0))

        tk.Label(
            control_frame,
            text="Slow  ←   m/s   →  Fast",
            width=24,
            anchor="center",
        ).pack()

        self.speed_slider = tk.Scale(
            control_frame,
            from_=self.min_speed,
            to=self.max_speed,
            orient=tk.HORIZONTAL,
            length=180,
            resolution=self.speed_resolution,
            digits=3,
        )
        self.speed_slider.set(self.initial_speed)
        self.speed_slider.pack(pady=5)

        self.status_label = tk.Label(
            control_frame,
            text="",
            justify=tk.LEFT,
        )
        self.status_label.pack(pady=(20, 0))

        # ==================================================
        # Plot area
        # ==================================================
        self.figure = Figure(figsize=(self.figure_width, self.figure_height))
        self.figure.patch.set_facecolor(self.figure_background_color)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(self.plot_background_color)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

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
            motor_running=self.motor_running,
        )

        self.update_status()
        self.update_plot()

        self.root.after(self.update_interval_ms, self.simulation_loop)

    def toggle_motor(self):
        """
        Start or stop the motor.
        """
        self.motor_running = not self.motor_running

        if self.motor_running:
            self.motor_button.config(text="Stop Motor")
        else:
            self.motor_button.config(text="Start Motor")

        self.update_status()

    def center_steering(self):
        """
        Reset steering slider to center.
        """
        self.steering_slider.set(0)

    def save_csv(self):
        """
        Save the simulated trajectory as CSV.
        """
        if not self.simulator.has_data():
            messagebox.showwarning(
                "No Data",
                "No trajectory data available. Please start the motor first.",
            )
            return

        trajectory_df = create_simulation_dataframe(
            simulator=self.simulator,
            random_seed=42,
        )

        output_path = save_trajectory_data(
            df=trajectory_df,
            filename="gui_steered_ship_trajectory.csv",
        )

        messagebox.showinfo(
            "Saved",
            f"Trajectory data saved to:\n{output_path}",
        )

    def reset(self):
        """
        Reset the simulation and clear the plot.
        """
        self.motor_running = False
        self.motor_button.config(text="Start Motor")
        self.steering_slider.set(0)
        self.speed_slider.set(self.initial_speed)

        self.simulator.reset()
        self.simulator.v = self.get_speed_from_slider()

        # Reset camera to the ship start position.
        # This is only used when axis_mode = "follow".
        self.camera_x = self.simulator.x_current
        self.camera_y = self.simulator.y_current

        self.update_status()
        self.update_plot()

    def update_status(self):
        """
        Update the status label.
        """
        heading_deg = np.rad2deg(self.simulator.theta_current)
        omega = self.get_omega_from_steering()
        omega_deg = np.rad2deg(omega)
        speed = self.get_speed_from_slider()

        if abs(omega) < 1e-8:
            radius_text = "∞"
        else:
            radius_text = f"{speed / omega:.2f} m"

        motor_text = "Running" if self.motor_running else "Stopped"

        self.status_label.config(
            text=(
                f"Motor: {motor_text}\n"
                f"Position: x={self.simulator.x_current:.2f} m, "
                f"y={self.simulator.y_current:.2f} m\n"
                f"Heading: {heading_deg:.1f}°\n"
                f"Omega: {omega_deg:.1f}°/s\n"
                f"Speed: {speed:.2f} m/s\n"
                f"Radius: {radius_text}\n"
                f"Time: {self.simulator.current_time:.1f} s"
            )
        )

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
            label="Current heading",
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
                label="True trajectory",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=self.start_position_color,
                linestyle="None",
                markersize=8,
                label="Start position",
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
                label="Current heading",
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

        # Reapply plot background after clearing
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
                label="True trajectory",
            )

            self.ax.scatter(
                self.simulator.x_all[0],
                self.simulator.y_all[0],
                color=self.start_position_color,
                label="Start position",
                zorder=4,
            )

            # Show current ship shape and heading direction.
            self.add_heading_marker()

            # Show the simulated ship center.
            self.add_ship_center_marker()

        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]", rotation=0, labelpad=15, va="center")
        self.ax.set_title("Interactive 2D Ship Trajectory")

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
