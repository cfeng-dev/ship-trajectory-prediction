"""Launch the Bayesian CTRV posterior explorer."""

import argparse


def main(argv=None):
    """Start the desktop application; help does not require a graphical display."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        import tkinter as tk
    except ImportError as error:
        parser.exit(
            2, f"Tkinter ist in dieser Python-Installation nicht verfügbar: {error}\n"
        )

    from .gui import PosteriorExplorer

    try:
        root = tk.Tk()
    except tk.TclError as error:
        parser.exit(2, f"Das Tk-Fenster konnte nicht gestartet werden: {error}\n")
    app = PosteriorExplorer(root)
    try:
        root.mainloop()
    finally:
        app.worker.close()
        app.worker.join()
