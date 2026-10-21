"""Launch the local interactive BR24 IMM forecast player."""

import argparse


def main(argv=None):
    """Open the graphical player; command-line help needs no display."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        from .player import run_br24_imm_player
    except ModuleNotFoundError as error:
        if error.name != "bayestraj" and not error.name.startswith("bayestraj."):
            raise
        parser.exit(
            2,
            "BR24 IMM Player requires the shared bayestraj package. Run it from "
            "the complete project environment with `uv run br24-imm-player`.\n",
        )
    run_br24_imm_player()
