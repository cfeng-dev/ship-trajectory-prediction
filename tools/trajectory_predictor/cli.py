"""Launch the Bayesian CTRV trajectory predictor."""

import argparse


def _import_trajectory_predictor():
    """Import the GUI only after command-line help and Tk checks complete."""
    from .gui import TrajectoryPredictor

    return TrajectoryPredictor


def _load_trajectory_predictor(parser):
    """Explain the shared-package requirement without masking other imports."""
    try:
        return _import_trajectory_predictor()
    except ModuleNotFoundError as error:
        if error.name != "bayestraj" and not error.name.startswith("bayestraj."):
            raise
        parser.exit(
            2,
            "Trajectory Predictor requires the shared bayestraj package. "
            "Run it from the complete project environment with "
            "`uv run trajectory-predictor`.\n",
        )


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

    TrajectoryPredictor = _load_trajectory_predictor(parser)

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
