"""Non-modal help window and documentation content for Trajectory Predictor."""

import tkinter as tk

from .controls import ScrollableForm
from .view import CONTROL_BACKGROUND, FONT, TEXT_COLOR, create_styled_button

DIALOG_CONTENT_PADDING = 16
HELP_WINDOW_WIDTH = 660
HELP_DESCRIPTION_COLUMN_WIDTH = 24
HELP_DESCRIPTION_WRAP_LENGTH = 300
HELP_CONTENT_MAX_SCREEN_RATIO = 0.65

POSTERIOR_HELP_SECTIONS = (
    (
        "Keyboard shortcuts",
        (
            ("Space", "Start or pause plot playback"),
            ("← / →", "Move the display one observation backward or forward"),
            ("Toolbar", "Pan, zoom, or reset the plot view"),
        ),
    ),
    (
        "Menu",
        (
            ("File → Open CSV", "Select the CSV file for analysis"),
            ("File → Exit", "Exit Trajectory Predictor"),
            ("View", "Show or hide settings; open plot display options"),
            ("Settings", "Configure priors, inference, and analysis setup"),
            ("Help", "Open this help window"),
        ),
    ),
    (
        "Controls",
        (
            ("Start analysis", "Analyse the selected CSV with the current settings"),
            ("Cancel analysis", "Stop the active analysis and unlock settings"),
            ("Inference method", "Select RBPF, SMC, VI, or MCMC for the next analysis"),
            ("Follow ship", "Center the plot on the current ship position"),
            ("N slider", "Show the prior at N = 0 and posterior updates afterward"),
        ),
    ),
    (
        "Analysis setup",
        (
            ("Start index", "First CSV position to include; counting begins at 0"),
            (
                "Observation interval",
                "Use positions separated by at least this many seconds",
            ),
            ("Maximum observations", "Limit selected positions; leave empty for all"),
            (
                "Additional position noise",
                "Add reproducible local-position noise before inference; 0 keeps positions unchanged",
            ),
            ("Position-noise seed", "Seed used for the additional position noise"),
            (
                "Forecast steps",
                "Number of steps predicted after each position; 0 disables forecasts",
            ),
            (
                "Future trajectories",
                "Number of posterior forecast samples to draw; 0 hides them",
            ),
            ("Inference seed", "Seed used to make stochastic inference reproducible"),
            (
                "Playback interval",
                "Minimum wait before requesting the next N; computation can take longer",
            ),
        ),
    ),
    (
        "Display",
        (
            (
                "Plot display",
                "Select layers, forecast regions, and the follow-ship view span",
            ),
            ("Motion state", "Posterior for speed, heading, and turn rate"),
            ("Uncertainties", "Posterior for observation and process noise"),
        ),
    ),
)


def add_help_description_rows(parent, rows):
    """Add rows with a shared control column, aligned across help sections."""
    for row, (control, description) in enumerate(rows):
        tk.Label(
            parent,
            text=control,
            width=HELP_DESCRIPTION_COLUMN_WIDTH,
            font=("Arial", 10, "bold"),
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            anchor="nw",
            justify="left",
        ).grid(row=row, column=0, sticky="nw", padx=(0, 20), pady=3)
        tk.Label(
            parent,
            text=description,
            font=FONT,
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            anchor="nw",
            justify="left",
            wraplength=HELP_DESCRIPTION_WRAP_LENGTH,
        ).grid(row=row, column=1, sticky="nw", pady=3)


def help_content_height_for_screen(content_height, screen_height):
    """Match the ship simulator's readable, screen-bounded help area."""
    return min(content_height, int(screen_height * HELP_CONTENT_MAX_SCREEN_RATIO))


def help_window_position():
    """Place help in the top-left corner instead of centering it over the app."""
    return 0, 0


def help_window_height(content_height, screen_height):
    """Fit the help window to readable content within the available screen."""
    available_height = max(320, screen_height - 120)
    return min(max(320, content_height), available_height)


class PosteriorHelpWindow(tk.Toplevel):
    """Non-modal, scrollable instructions for the trajectory predictor."""

    def __init__(self, parent, *, on_close=None):
        super().__init__(parent)
        self._on_close = on_close
        self._closed = False
        self.withdraw()
        self.transient(parent)
        self.title("Help")
        self.configure(bg=CONTROL_BACKGROUND)
        self.resizable(False, False)

        main = tk.Frame(self, bg=CONTROL_BACKGROUND)
        main.pack(fill="both", expand=True)
        tk.Label(
            main,
            text="Trajectory Predictor",
            font=("Arial", 13, "bold"),
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            anchor="w",
        ).pack(fill="x", padx=24, pady=(18, 12))
        self._form = ScrollableForm(main, width=HELP_WINDOW_WIDTH - 60)
        self._form.pack(fill="both", expand=True, padx=24)
        for title, rows in POSTERIOR_HELP_SECTIONS:
            section = tk.LabelFrame(
                self._form.body,
                text=title,
                font=FONT,
                bg=CONTROL_BACKGROUND,
                fg=TEXT_COLOR,
                padx=12,
                pady=10,
            )
            section.pack(fill="x", pady=(0, 12))
            section.columnconfigure(1, weight=1)
            add_help_description_rows(section, rows)

        actions = tk.Frame(main, bg=CONTROL_BACKGROUND)
        actions.pack(fill="x", padx=24, pady=(12, 18))
        create_styled_button(actions, text="OK", width=12, command=self.close).pack(
            anchor="e"
        )
        self._form.bind_mouse_wheel()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.update_idletasks()
        self._form.canvas.configure(
            height=help_content_height_for_screen(
                self._form.body.winfo_reqheight(),
                self.winfo_screenheight(),
            )
        )
        self.update_idletasks()
        content_height = main.winfo_reqheight() + DIALOG_CONTENT_PADDING
        height = help_window_height(content_height, self.winfo_screenheight())
        left, top = help_window_position()
        self.geometry(f"{HELP_WINDOW_WIDTH}x{height}+{left}+{top}")
        self.deiconify()
        self.focus_set()

    def close(self):
        """Close the help window without changing the current analysis."""
        if self._closed:
            return
        self._closed = True
        if self._on_close is not None:
            self._on_close()
        self.destroy()
