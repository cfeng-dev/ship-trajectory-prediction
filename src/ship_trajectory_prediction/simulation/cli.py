"""Start the interactive 2D ship trajectory simulation GUI."""

import tkinter as tk

from . import gui


def main():
    """
    Start the interactive ship trajectory GUI.
    """
    root = tk.Tk()

    app = gui.ShipTrajectoryGUI(root)

    # Set initial window size: width x height.
    root.geometry(f"{app.window_width}x{app.window_height}")

    root.mainloop()


if __name__ == "__main__":
    main()
