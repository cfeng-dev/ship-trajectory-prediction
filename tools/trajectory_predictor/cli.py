"""Launch the Bayesian CTRV trajectory predictor."""

import argparse


def main(argv=None):
    """Start the desktop application; help does not require a graphical display."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        import tkinter as tk
    except ImportError as error:
        parser.exit(
            2, f"Tkinter is not available in this Python installation: {error}\n"
        )

    from .gui import TrajectoryPredictor

    try:
        root = tk.Tk()
    except tk.TclError as error:
        parser.exit(2, f"Das Tk-Fenster konnte nicht gestartet werden: {error}\n")
    app = TrajectoryPredictor(root)
    try:
        root.mainloop()
    finally:
        app.worker.close()
        app.worker.join()
