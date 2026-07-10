"""
@file run_simulation_gui.py
@description Starts the interactive 2D ship trajectory simulation GUI.
@date Created on: 07.07.2026
@author C.Feng
"""

import tkinter as tk

from simulation.modules.simulation_gui import ShipTrajectoryGUI


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
