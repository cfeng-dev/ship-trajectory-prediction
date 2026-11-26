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
    default_observations = project_path(
        "data/processed/BR24_imm_tracked_trajectory.csv"
    )
    parser.add_argument(
        "--observations-csv",
        default=default_observations if default_observations.is_file() else None,
        help="Original time,x,y input CSV for the same experiment; uses source timestamps without normalization. Use --observations-csv= to load evaluation results without observations.",
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
        app = EvaluationViewer(
            root,
            input_dir=arguments.input_dir,
            observations_csv=arguments.observations_csv,
        )
        root.mainloop()
    finally:
        if app is None:
            root.destroy()
        else:
            app.close()
