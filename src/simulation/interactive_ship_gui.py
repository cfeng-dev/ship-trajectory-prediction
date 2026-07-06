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

        # Maximum angular velocity in rad/s
        self.max_omega = np.deg2rad(15)

        self.motor_running = False

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
        Create buttons, slider, labels, and plot area.
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

        tk.Label(
            control_frame,
            text="Steering",
            font=("Arial", 11, "bold"),
        ).pack(pady=(25, 0))

        self.steering_slider = tk.Scale(
            control_frame,
            from_=-100,
            to=100,
            orient=tk.HORIZONTAL,
            length=180,
            label="Left  ←   0   →  Right",
        )
        self.steering_slider.set(0)
        self.steering_slider.pack(pady=5)

        tk.Button(
            control_frame,
            text="Center Steering",
            width=18,
            command=self.center_steering,
        ).pack(pady=5)

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
        Convert steering slider value to angular velocity.

        Returns
        -------
        omega : float
            Angular velocity in rad/s.
        """
        steering_value = self.steering_slider.get()

        # Positive slider value means steering to the right.
        # In mathematical coordinates, positive omega turns left,
        # therefore the sign is inverted here.
        omega = -(steering_value / 100.0) * self.max_omega

        return omega

    def simulation_loop(self):
        """
        Continuously run the simulation loop.
        """
        omega = self.get_omega_from_steering()

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

        self.simulator.reset()

        self.update_status()
        self.update_plot()

    def update_status(self):
        """
        Update the status label.
        """
        heading_deg = np.rad2deg(self.simulator.theta_current)
        omega = self.get_omega_from_steering()
        omega_deg = np.rad2deg(omega)

        if abs(omega) < 1e-8:
            radius_text = "∞"
        else:
            radius_text = f"{self.simulator.v / omega:.2f}"

        motor_text = "Running" if self.motor_running else "Stopped"

        self.status_label.config(
            text=(
                f"Motor: {motor_text}\n"
                f"Position: x={self.simulator.x_current:.2f}, "
                f"y={self.simulator.y_current:.2f}\n"
                f"Heading: {heading_deg:.1f}°\n"
                f"Omega: {omega_deg:.1f}°/s\n"
                f"Radius: {radius_text}\n"
                f"Time: {self.simulator.current_time:.1f} s"
            )
        )

    def add_heading_marker(self):
        """
        Add a fixed-size red triangle marker at the current ship position.

        The marker is rotated according to the current heading angle.
        """
        x = self.simulator.x_current
        y = self.simulator.y_current
        theta = self.simulator.theta_current

        # Convert heading to degrees.
        # Matplotlib triangle marker points upward by default,
        # therefore subtract 90 degrees so theta=0 points to the right.
        angle_deg = np.rad2deg(theta) - 90

        self.ax.scatter(
            x,
            y,
            s=180,
            color="red",
            marker=(3, 0, angle_deg),
            label="Current heading",
            zorder=5,
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
                marker="^",
                color="red",
                linestyle="None",
                markersize=12,
                label="Current heading",
            ),
        ]

        self.ax.legend(handles=legend_handles)

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
            )

            # Show current ship position and heading direction
            self.add_heading_marker()

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Interactive 2D Ship Trajectory")
        self.ax.axis("equal")
        self.ax.grid(True)

        if self.simulator.has_data():
            self.update_legend()

        self.canvas.draw()


def main():
    root = tk.Tk()
    ShipTrajectoryGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
