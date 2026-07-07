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

from ship_simulator import ShipSimulator
from simulation_io import (
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
        # Simulation parameters
        # ==================================================
        self.simulator = ShipSimulator(
            v=0.5,
            sigma=0.2,
            dt=0.05,
        )

        # GUI update interval in milliseconds
        self.update_interval_ms = 100

        self.motor_running = False

        # Ship visualization size in meters
        self.ship_length = 10.0
        self.ship_width = 3.0

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
            from_=-45,
            to=45,
            orient=tk.HORIZONTAL,
            length=180,
            resolution=1,
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
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            length=180,
            resolution=0.05,
            digits=3,
        )
        self.speed_slider.set(0.5)
        self.speed_slider.pack(pady=5)

        self.status_label = tk.Label(
            control_frame,
            text="",
            justify=tk.LEFT,
        )
        self.status_label.pack(pady=(20, 0))

        # Plot area
        self.figure = Figure(figsize=(7, 6))
        self.ax = self.figure.add_subplot(111)

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
        self.speed_slider.set(0.5)

        self.simulator.reset()
        self.simulator.v = self.get_speed_from_slider()

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
        Create a ship triangle in real-world meter size.

        The ship length and width are defined in meters.
        The triangle points in the direction of theta.
        """
        length = self.ship_length
        width = self.ship_width

        # Triangle pointing to the right when theta = 0.
        #
        #              front
        #                >
        # rear lower -------- rear upper
        #
        # The current ship position is approximately the center of the ship.
        ship_shape = np.array(
            [
                [length / 2, 0.0],  # Front tip
                [-length / 2, -width / 2],  # Rear lower corner
                [-length / 2, width / 2],  # Rear upper corner
            ]
        )

        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        return ship_shape @ rotation_matrix.T

    def add_heading_marker(self):
        """
        Add a red ship marker at the current ship position.

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
            color="red",
            label="Current heading",
            zorder=5,
        )

        self.ax.add_patch(ship_patch)

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
                color="tab:blue",
                label="True trajectory",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                linestyle="None",
                markersize=8,
                label="Start position",
            ),
            Line2D(
                [0],
                [0],
                marker=">",
                markerfacecolor="red",
                markeredgecolor="red",
                color="red",
                linestyle="None",
                markersize=10,
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

    def update_plot(self):
        """
        Update the trajectory plot.
        """
        self.ax.clear()

        if self.simulator.has_data():
            # Add the current position to the displayed trajectory.
            # This makes the blue line connect to the current heading marker.
            x_plot = list(self.simulator.x_all) + [self.simulator.x_current]
            y_plot = list(self.simulator.y_all) + [self.simulator.y_current]

            self.ax.plot(
                x_plot,
                y_plot,
                label="True trajectory",
            )

            self.ax.scatter(
                self.simulator.x_all[0],
                self.simulator.y_all[0],
                color="black",
                label="Start position",
                zorder=4,
            )

            # Show current ship position and heading direction.
            self.add_heading_marker()

        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]", rotation=0, labelpad=15, va="center")
        self.ax.set_title("Interactive 2D Ship Trajectory")

        # Keep 1 meter on x-axis equal to 1 meter on y-axis.
        self.ax.set_aspect("equal", adjustable="box")

        self.ax.grid(True)

        # Fixed axis range and ticks in meters.
        self.ax.set_xlim(-10, 50)
        self.ax.set_ylim(-30, 30)

        self.ax.set_xticks(np.arange(-10, 51, 10))
        self.ax.set_yticks(np.arange(-30, 31, 10))

        if self.simulator.has_data():
            self.update_legend()

        self.canvas.draw()


def main():
    root = tk.Tk()

    # Set initial window size: width x height.
    root.geometry("1100x700")

    ShipTrajectoryGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
