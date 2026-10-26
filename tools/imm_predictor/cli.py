"""Launch the IMM ship trajectory predictor."""

import argparse
import logging


def _import_predictor_app():
    """Delay Tk and predictor imports until after command-line help has run."""
    from .gui import IMMPredictorApp

    return IMMPredictorApp


def main(argv=None):
    """Open the graphical predictor; command-line help needs no display."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        import tkinter as tk
    except ModuleNotFoundError as error:
        parser.exit(
            2,
            f"Tkinter is not available in this Python installation: {error}\n",
        )
    try:
        IMMPredictorApp = _import_predictor_app()
    except ModuleNotFoundError as error:
        if error.name != "bayestraj" and not error.name.startswith("bayestraj."):
            raise
        parser.exit(
            2,
            "IMM Predictor requires the shared bayestraj package. Run it from "
            "the complete project environment with `uv run imm-predictor`.\n",
        )
    try:
        root = tk.Tk()
    except tk.TclError as error:
        parser.exit(2, f"The Tk window could not start: {error}\n")
    app = None
    try:
        app = IMMPredictorApp(root)
        root.imm_predictor_app = app
        root.mainloop()
    except Exception:
        logging.getLogger("imm_predictor").exception("Desktop application failed")
        raise
    finally:
        if app is not None:
            app.close()
        else:
            try:
                root.destroy()
            except tk.TclError:
                pass
