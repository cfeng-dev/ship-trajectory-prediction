"""Non-modal help window and documentation content for Trajectory Predictor."""

import tkinter as tk

from bayestraj.inference.configuration import (
    DEFAULT_CTRV_ROLLING_OBSERVATION_COUNT,
)

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
            ("Space", "Start or pause plot playback after starting analysis"),
            (
                "← / →",
                "Move one observation backward or forward and pause playback",
            ),
            ("Toolbar", "Pan, zoom, or reset the plot view"),
        ),
    ),
    (
        "Menu",
        (
            ("File → Open CSV", "Select the CSV file for analysis"),
            ("File → Exit", "Exit Trajectory Predictor"),
            (
                "View",
                "Show or hide settings; select meters or kilometers; open plot display options",
            ),
            (
                "Settings",
                "Configure priors, inference, and analysis setup; locked during analysis",
            ),
            ("Help", "Open or focus this non-modal help window"),
            (
                "Main-window X",
                "Close child windows and stop analysis; a running calculation finishes before exit",
            ),
        ),
    ),
    (
        "Controls",
        (
            ("Start analysis", "Analyse the selected CSV with the current settings"),
            (
                "Cancel analysis",
                "Discard displayed results and unlock settings; an ongoing fit may finish in the background",
            ),
            (
                "Inference method",
                "Select SMC, Sequential VI, VI, or MCMC for the next analysis",
            ),
            (
                "VI / MCMC",
                "Refit the latest "
                f"{DEFAULT_CTRV_ROLLING_OBSERVATION_COUNT} observations at each update",
            ),
            (
                "Follow ship",
                "Center the plot on the current position while retaining manual zoom",
            ),
            (
                "Show settings",
                "Show or hide the left sidebar without restarting analysis",
            ),
            ("N slider", "Show the prior at N = 0 and posterior updates afterward"),
            (
                "Early updates",
                "With too few observations, forecasts may remain prior-predictive; the plot title identifies this phase",
            ),
            (
                "Analysis errors",
                "Playback stops and settings unlock after an error. Start a new analysis before continuing; failed filter states are not reused",
            ),
            (
                "Diagnostics",
                "Error details are printed to the launching terminal. No diagnostic log files are created",
            ),
        ),
    ),
    (
        "Analysis setup",
        (
            ("Start index", "First CSV position to include; counting begins at 0"),
            (
                "Observation interval",
                "Optional minimum position spacing in seconds; leave empty to use every original timestamp",
            ),
            ("Maximum observations", "Limit selected positions; leave empty for all"),
            (
                "Additional position noise",
                "Gaussian standard deviation per local position axis in meters; 0 keeps processed-trajectory positions unchanged",
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
        "Input and state displays",
        (
            (
                "CSV input",
                "Load a processed trajectory with time, x, y in seconds and local meters, or a BR24 radar tracking CSV",
            ),
            (
                "Reference state",
                "Unmodified reference position and time; motion values come from reference fields or position differences, not the posterior",
            ),
            (
                "Reference motion",
                "Supplied speed, yaw and turn rate take priority independently, including zero values. Missing quantities are marked (derived). Speed/course need two positions; course-change rate needs three. Course is undefined at rest and is not vessel orientation",
            ),
            (
                "Optional CSV columns",
                "BR24: ground_truth_velocity [m/s], ground_truth_yaw [rad], ground_truth_turn_rate [rad/s]. Processed time,x,y files also accept speed/velocity, theta/yaw and omega/turn_rate. Invalid supplied values are reported, not replaced",
            ),
            (
                "Reference turn rate",
                "Use supplied reference rates when available; otherwise derive backward wrapped yaw differences, or course differences if yaw is absent, using actual time spacing. Derived values are marked (derived) and displayed in degrees/s; unavailable values show a dash",
            ),
            (
                "Posterior medians",
                "Estimated speed, heading, turn rate and noise levels after conditioning on the available observations",
            ),
            (
                "Reference boundary",
                "BR24 ground truth is used only for display and evaluation, never for radar association or inference. Processed trajectories supply the position observations",
            ),
            (
                "Missing values",
                "A dash means unavailable or not yet identifiable; motion and noise estimates need sufficient observations",
            ),
            (
                "Units and time",
                "Position uses the selected m/km display, speed m/s, angles degrees and turn rate degrees/s; time is shown as h, min, s",
            ),
        ),
    ),
    (
        "Priors and inference",
        (
            (
                "Priors",
                "Set plausible thresholds and tail probabilities for speed, turn rate and noise; thresholds are not hard limits",
            ),
            (
                "Observation-noise floor",
                "Minimum effective position uncertainty; distinct from additional simulated position noise in Analysis setup",
            ),
            (
                "SMC",
                "Sequential Monte Carlo updates a weighted particle approximation as observations arrive",
            ),
            (
                "Sequential VI",
                "Online variational inference; distinct from the rolling batch VI option",
            ),
            (
                "Inference parameters",
                "Edit settings for the selected method, such as particle count, posterior draws or convergence controls",
            ),
            (
                "Apply / Cancel",
                "Apply saves validated settings for the next analysis; Cancel or the dialog X discards unapplied edits",
            ),
        ),
    ),
    (
        "Evaluation metrics",
        (
            (
                "Forecast ADE",
                "Mean 2D error of the current median forecast against available future reference positions, in meters; lower is better.",
            ),
            (
                "Forecast FDE",
                "Position error at the final forecast step in meters; lower is better.",
            ),
            (
                "Forecast RMSE",
                "Root mean squared 2D position error across all forecast steps in meters; lower is better.",
            ),
            (
                "Forecast Energy Score",
                "Joint 2D probabilistic forecast error; lower is better.",
            ),
            (
                "Predictive log density",
                "One-step-ahead log likelihood of the new observation; larger is better.",
            ),
            (
                "Joint 50% coverage",
                "Cumulative share of evaluated future positions inside per-step 2D 50% covariance ellipses, shown as percentage and count; not whole-path coverage.",
            ),
            (
                "Joint 90% coverage",
                "Cumulative share of evaluated future positions inside per-step 2D 90% covariance ellipses, shown as percentage and count; not whole-path coverage.",
            ),
            (
                "Inference time",
                "Wall-clock time for the current update, including forecast generation but excluding GUI drawing; not forecast accuracy.",
            ),
        ),
    ),
    (
        "BR24 radar association",
        (
            (
                "Radar input",
                "Raw BR24 tracking requires SMC and start index 0. Cycles keep their original timestamps; observation spacing and simulated position-noise settings do not modify this input",
            ),
            (
                "Status",
                "Initial measurement, Associated, or Prediction only when no detection is accepted",
            ),
            (
                "Cycle",
                "Original radar-cycle counter; N counts displayed updates, not detection IDs",
            ),
            ("Detection ID", "Detection selected within the current radar cycle"),
            (
                "NIS",
                "Normalized innovation squared: position discrepancy relative to predicted measurement uncertainty; unavailable at initialization",
            ),
            (
                "cov_xx / cov_xy / cov_yy",
                "Cartesian measurement covariance entries in square meters, not posterior or forecast covariance; cov_xy may be negative",
            ),
            (
                "No reference labels",
                "Tracking remains available; reference-state and reference-based forecast metrics show dashes",
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
            (
                "Reference trajectory",
                "Dashed gray reference path; visibility does not affect inference or evaluation",
            ),
            (
                "Forecast (median)",
                "Red coordinate-wise median of future latent positions, not a single sampled trajectory",
            ),
            (
                "Possible future trajectories",
                "Sampled future paths illustrating uncertainty; their display count does not change the full draws used for regions and metrics",
            ),
            (
                "Connect samples",
                "Off by default: paths start at their sampled current latent positions. On: replace the drawn start with the current displayed point for orientation only; forecasts and metrics stay unchanged",
            ),
            (
                "Apply / Cancel",
                "Apply commits plot changes without restarting inference; Cancel or X discards unapplied edits. Playback stays accessible",
            ),
            (
                "Follow-ship view span",
                "Viewport size in meters; manual pan/zoom stays available. Disable Follow ship to keep the view fixed",
            ),
        ),
    ),
    (
        "Prediction uncertainty",
        (
            (
                "50% / 90% regions",
                "Toggle each level independently; regions describe future latent positions, not added measurement noise",
            ),
            (
                "Horizon density region",
                "Filled density region pooling draws across future steps; not a simultaneous probability band for the whole path",
            ),
            (
                "Horizon density contours",
                "Contour outlines of the pooled horizon density instead of a filled region",
            ),
            (
                "Per-step density regions",
                "Separate filled density regions for each future timestamp, preserving step-specific uncertainty",
            ),
            (
                "Per-step density contours",
                "Separate density contour outlines for each future timestamp",
            ),
            (
                "Covariance ellipses",
                "Per-step 2D regions from empirical means and covariances; an elliptical approximation to the forecast draws",
            ),
            (
                "No uncertainty region",
                "Hide spatial regions without disabling forecasts, sampled paths or evaluation",
            ),
            (
                "Coverage versus display",
                "Coverage always uses per-step covariance ellipses, independently of the selected visual uncertainty style",
            ),
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
