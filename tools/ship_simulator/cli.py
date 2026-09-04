"""Start the standalone 2D ship trajectory simulation GUI."""

from . import gui


def main():
    """
    Start the interactive ship trajectory GUI.
    """
    import tkinter as tk

    root = tk.Tk()

    app = gui.ShipTrajectoryGUI(root)

    # Set initial window size: width x height.
    root.geometry(f"{app.window_width}x{app.window_height}")

    root.mainloop()


if __name__ == "__main__":
    main()
