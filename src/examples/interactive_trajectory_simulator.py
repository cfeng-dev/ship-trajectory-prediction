"""
@file interactive_trajectory_simulator.py
@description Lets the user interactively steer a simple 2D ship trajectory and saves the simulated data.
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
    simulate_curved_trajectory,
    simulate_straight_trajectory,
)


class ShipTrajectoryGUI:
    """
    Simple GUI for interactively simulating a 2D ship trajectory.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("2D Ship Trajectory Simulator")

        # ==================================================
        # Simulation parameters
        # ==================================================
        self.v = 1.0
        self.radius = 10.0
        self.sigma = 0.2

        self.segment_duration = 5.0
        self.points_per_segment = 30

        self.reset_simulation_state()

        # ==================================================
        # GUI layout
        # ==================================================
        self.create_widgets()
        self.update_plot()

    def reset_simulation_state(self):
        """
        Reset the ship state and trajectory data.
        """
        self.x_current = 0.0
        self.y_current = 0.0
        self.theta_current = np.deg2rad(0)
        self.current_time = 0.0

        self.t_all = []
        self.x_all = []
        self.y_all = []
        self.segment_types = []

    def create_widgets(self):
        """
        Create buttons, labels, and plot area.
        """
        # Left control panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(
            control_frame,
            text="Ship Controls",
            font=("Arial", 14, "bold"),
        ).pack(pady=(0, 10))

        tk.Button(
            control_frame,
            text="Drive Straight",
            width=18,
            command=self.drive_straight,
        ).pack(pady=5)

        tk.Button(
            control_frame,
            text="Turn Left",
            width=18,
            command=self.turn_left,
        ).pack(pady=5)

        tk.Button(
            control_frame,
            text="Turn Right",
            width=18,
            command=self.turn_right,
        ).pack(pady=5)

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

        self.status_label = tk.Label(
            control_frame,
            text="Position: x=0.00, y=0.00\nHeading: 0.0°",
            justify=tk.LEFT,
        )
        self.status_label.pack(pady=(20, 0))

        # Right plot area
        self.figure = Figure(figsize=(7, 6))
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def append_segment(self, t_segment, x_segment, y_segment, segment_type):
        """
        Append a trajectory segment to the full trajectory.

        The first point of each new segment is removed to avoid duplicate
        points at the transition between two segments.
        """
        if len(self.t_all) > 0:
            t_segment = t_segment[1:]
            x_segment = x_segment[1:]
            y_segment = y_segment[1:]

        t_shifted = t_segment + self.current_time

        self.t_all.extend(t_shifted)
        self.x_all.extend(x_segment)
        self.y_all.extend(y_segment)
        self.segment_types.extend([segment_type] * len(t_segment))

        self.x_current = x_segment[-1]
        self.y_current = y_segment[-1]
        self.current_time = self.t_all[-1]

    def drive_straight(self):
        """
        Add a straight trajectory segment.
        """
        t_segment = np.linspace(
            0,
            self.segment_duration,
            self.points_per_segment,
        )

        x_segment, y_segment = simulate_straight_trajectory(
            t=t_segment,
            x0=self.x_current,
            y0=self.y_current,
            v=self.v,
            theta=self.theta_current,
        )

        self.append_segment(
            t_segment=t_segment,
            x_segment=x_segment,
            y_segment=y_segment,
            segment_type="straight",
        )

        self.update_status()
        self.update_plot()

    def turn_left(self):
        """
        Add a left-turn trajectory segment.
        """
        t_segment = np.linspace(
            0,
            self.segment_duration,
            self.points_per_segment,
        )

        x_segment, y_segment = simulate_curved_trajectory(
            t=t_segment,
            x0=self.x_current,
            y0=self.y_current,
            v=self.v,
            radius=self.radius,
            theta=self.theta_current,
        )

        omega = self.v / self.radius
        self.theta_current += omega * self.segment_duration

        self.append_segment(
            t_segment=t_segment,
            x_segment=x_segment,
            y_segment=y_segment,
            segment_type="left_curve",
        )

        self.update_status()
        self.update_plot()

    def turn_right(self):
        """
        Add a right-turn trajectory segment.
        """
        t_segment = np.linspace(
            0,
            self.segment_duration,
            self.points_per_segment,
        )

        x_segment, y_segment = simulate_curved_trajectory(
            t=t_segment,
            x0=self.x_current,
            y0=self.y_current,
            v=self.v,
            radius=-self.radius,
            theta=self.theta_current,
        )

        omega = self.v / (-self.radius)
        self.theta_current += omega * self.segment_duration

        self.append_segment(
            t_segment=t_segment,
            x_segment=x_segment,
            y_segment=y_segment,
            segment_type="right_curve",
        )

        self.update_status()
        self.update_plot()

    def save_csv(self):
        """
        Save the simulated trajectory as CSV.
        """
        if len(self.x_all) == 0:
            messagebox.showwarning(
                "No Data",
                "No trajectory data available. Please drive the ship first.",
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
                "trajectory_type": self.segment_types,
                "v": self.v,
                "sigma": self.sigma,
                "radius": self.radius,
            }
        )

        save_trajectory_data(
            df=trajectory_df,
            filename="gui_ship_trajectory.csv",
        )

        messagebox.showinfo(
            "Saved",
            "Trajectory data saved as gui_ship_trajectory.csv",
        )

    def reset(self):
        """
        Reset the simulation and clear the plot.
        """
        self.reset_simulation_state()
        self.update_status()
        self.update_plot()

    def update_status(self):
        """
        Update the status label.
        """
        heading_deg = np.rad2deg(self.theta_current)

        self.status_label.config(
            text=(
                f"Position: x={self.x_current:.2f}, y={self.y_current:.2f}\n"
                f"Heading: {heading_deg:.1f}°\n"
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
                marker="o",
                markersize=3,
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
        self.ax.legend()

        self.canvas.draw()


def main():
    root = tk.Tk()
    app = ShipTrajectoryGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
