"""Launch the local desktop BR24 IMM forecast player."""

import argparse
import logging


def _import_player_app():
    """Delay Tk and player imports until after command-line help has run."""
    from .gui import BR24IMMPlayerApp

    return BR24IMMPlayerApp


def main(argv=None):
    """Open the graphical player; command-line help needs no display."""
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
        BR24IMMPlayerApp = _import_player_app()
    except ModuleNotFoundError as error:
        if error.name != "bayestraj" and not error.name.startswith("bayestraj."):
            raise
        parser.exit(
            2,
            "BR24 IMM Player requires the shared bayestraj package. Run it from "
            "the complete project environment with `uv run br24-imm-player`.\n",
        )
    try:
        root = tk.Tk()
    except tk.TclError as error:
        parser.exit(2, f"The Tk window could not start: {error}\n")
    app = None
    try:
        app = BR24IMMPlayerApp(root)
        root.br24_imm_player_app = app
        root.mainloop()
    except Exception:
        logging.getLogger("br24_imm_player").exception("Desktop application failed")
        raise
    finally:
        if app is not None:
            app.close()
        else:
            try:
                root.destroy()
            except tk.TclError:
                pass
