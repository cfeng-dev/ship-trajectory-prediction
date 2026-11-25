"""Launch the CSV-only Evaluation Viewer."""

import argparse

from bayestraj.paths import project_path


def main(argv=None):
    """Keep help usable without Tk, a display, local CSVs or CmdStan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=project_path("results/evaluation"),
        help="Folder containing saved evaluation CSVs.",
    )
    arguments = parser.parse_args(argv)
    try:
        import tkinter as tk
    except ImportError as error:
        parser.exit(2, f"Tkinter is not available: {error}\n")
    from .gui import EvaluationViewer

    try:
        root = tk.Tk()
    except tk.TclError as error:
        parser.exit(2, f"Could not open the Tk window: {error}\n")
    app = None
    try:
        app = EvaluationViewer(root, input_dir=arguments.input_dir)
        root.mainloop()
    finally:
        if app is None:
            root.destroy()
        else:
            app.close()
