"""Start the interactive 2D ship trajectory simulation GUI."""

import tkinter as tk

from ship_trajectory_prediction.simulation.gui import ShipTrajectoryGUI


def main():
    """
    Start the interactive ship trajectory GUI.
    """
    root = tk.Tk()

    app = ShipTrajectoryGUI(root)

    # Set initial window size: width x height.
    root.geometry(f"{app.window_width}x{app.window_height}")

    root.mainloop()


if __name__ == "__main__":
    main()
