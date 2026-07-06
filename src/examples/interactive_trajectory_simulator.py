"""
@file interactive_trajectory_simulator.py
@description Provides a GUI to steer a simple 2D ship trajectory with motor control and CSV export.
@date Created on: 06.07.2026
@author C.Feng
"""

import tkinter as tk
from tkinter import messagebox

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from trajectory_models import (
    add_observation_noise,
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
        self.v = 1.0
        self.sigma = 0.2

        # Simulation time step in seconds
        self.dt = 0.1

        # GUI update interval in milliseconds
        self.update_interval_ms = 100

        # Maximum angular velocity in rad/s
        self.max_omega = np.deg2rad(20)

        self.motor_running = False

        self.reset_simulation_state()

        # ==================================================
        # GUI layout
        # ==================================================
        self.create_widgets()
        self.update_status()
        self.update_plot()

        # Start continuous simulation loop
        self.simulation_loop()

    def reset_simulation_state(self):
        """
        Reset the ship state and trajectory data.
        """
        self.x_current = 0.0
        self.y_current = 0.0
        self.theta_current = 0.0
        self.current_time = 0.0

        self.t_all = []
        self.x_all = []
        self.y_all = []
        self.theta_all = []
        self.omega_all = []
        self.radius_all = []
        self.motor_state_all = []

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

        # Positive steering should turn left, negative steering right
        omega = (steering_value / 100.0) * self.max_omega

        return omega

    def simulation_step(self):
        """
        Perform one simulation step if the motor is running.
        """
        if not self.motor_running:
            return

        omega = self.get_omega_from_steering()

        # Store current state before updating
        self.t_all.append(self.current_time)
        self.x_all.append(self.x_current)
        self.y_all.append(self.y_current)
        self.theta_all.append(self.theta_current)
        self.omega_all.append(omega)
        self.motor_state_all.append(self.motor_running)

        # Turning radius R = v / omega
        # For straight motion omega is almost zero, so radius is infinite
        if abs(omega) < 1e-8:
            radius = np.inf
        else:
            radius = self.v / omega

        self.radius_all.append(radius)

        # Update ship position and heading
        self.x_current += self.v * np.cos(self.theta_current) * self.dt
        self.y_current += self.v * np.sin(self.theta_current) * self.dt
        self.theta_current += omega * self.dt
        self.current_time += self.dt

    def simulation_loop(self):
        """
        Continuously run the simulation loop.
        """
        self.simulation_step()
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
        if len(self.x_all) == 0:
            messagebox.showwarning(
                "No Data",
                "No trajectory data available. Please start the motor first.",
            )
            return

        t = np.array(self.t_all)
        x_true = np.array(self.x_all)
        y_true = np.array(self.y_all)

        x_obs, y_obs = add_observation_noise(
            x=x_true,
            y=y_true,
            sigma=self.sigma,
            random_seed=42,
        )

        trajectory_df = pd.DataFrame(
            {
                "t": t,
                "x_true": x_true,
                "y_true": y_true,
                "x_obs": x_obs,
                "y_obs": y_obs,
                "theta": self.theta_all,
                "omega": self.omega_all,
                "radius": self.radius_all,
                "v": self.v,
                "sigma": self.sigma,
                "motor_running": self.motor_state_all,
            }
        )

        save_trajectory_data(
            df=trajectory_df,
            filename="gui_steered_ship_trajectory.csv",
        )

        messagebox.showinfo(
            "Saved",
            "Trajectory data saved as gui_steered_ship_trajectory.csv",
        )

    def reset(self):
        """
        Reset the simulation and clear the plot.
        """
        self.motor_running = False
        self.motor_button.config(text="Start Motor")
        self.steering_slider.set(0)

        self.reset_simulation_state()
        self.update_status()
        self.update_plot()

    def update_status(self):
        """
        Update the status label.
        """
        heading_deg = np.rad2deg(self.theta_current)
        omega = self.get_omega_from_steering()
        omega_deg = np.rad2deg(omega)

        if abs(omega) < 1e-8:
            radius_text = "∞"
        else:
            radius_text = f"{self.v / omega:.2f}"

        motor_text = "Running" if self.motor_running else "Stopped"

        self.status_label.config(
            text=(
                f"Motor: {motor_text}\n"
                f"Position: x={self.x_current:.2f}, y={self.y_current:.2f}\n"
                f"Heading: {heading_deg:.1f}°\n"
                f"Omega: {omega_deg:.1f}°/s\n"
                f"Radius: {radius_text}\n"
                f"Time: {self.current_time:.1f} s"
            )
        )

    def update_plot(self):
        """
        Update the trajectory plot.
        """
        self.ax.clear()

        if len(self.x_all) > 0:
            self.ax.plot(
                self.x_all,
                self.y_all,
                label="True trajectory",
            )

            self.ax.scatter(
                self.x_all[0],
                self.y_all[0],
                color="black",
                label="Start position",
            )

            self.ax.scatter(
                self.x_all[-1],
                self.y_all[-1],
                color="red",
                label="Current position",
            )

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Interactive 2D Ship Trajectory")
        self.ax.axis("equal")
        self.ax.grid(True)

        if len(self.x_all) > 0:
            self.ax.legend()

        self.canvas.draw()


def main():
    root = tk.Tk()
    app = ShipTrajectoryGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
