"""Tk dialogs for IMM Predictor settings and help."""

from __future__ import annotations

import tkinter as tk
from dataclasses import replace
from tkinter import messagebox, ttk

from .player import (
    PLOT_DISPLAY_OPTIONS,
    PREDICTION_REGION_STYLE_LABELS,
    IMMPlayerConfig,
)
from .view import CONTROL_BACKGROUND, FONT, TEXT_COLOR, create_button


class PlaybackSettingsDialog:
    """Edit local drafts without blocking the main window."""

    def __init__(
        self, parent, config: IMMPlayerConfig, on_apply, *, plot_only=False
    ) -> None:
        self.window = tk.Toplevel(parent)
        self.window.withdraw()
        self._closed = False
        self._plot_only = plot_only
        self.window.title("Plot display" if plot_only else "Analysis setup")
        self.window.configure(bg=CONTROL_BACKGROUND)
        self.window.transient(parent)
        self._config = config
        self._on_apply = on_apply
        self._variables = {
            "prediction_sample_count": tk.StringVar(
                self.window, value=str(config.prediction_sample_count)
            ),
            "forecast_seed": tk.StringVar(self.window, value=str(config.forecast_seed)),
            **{
                name: tk.BooleanVar(self.window, value=getattr(config, name))
                for name in PLOT_DISPLAY_OPTIONS
            },
            "start_index": tk.StringVar(self.window, value=str(config.start_index)),
            "position_noise_std_m": tk.StringVar(
                self.window, value=f"{config.position_noise_std_m:g}"
            ),
            "position_noise_seed": tk.StringVar(
                self.window, value=str(config.position_noise_seed)
            ),
            "prediction_region_style": tk.StringVar(
                self.window,
                value=PREDICTION_REGION_STYLE_LABELS[config.prediction_region_style],
            ),
            "prediction_count": tk.StringVar(
                self.window, value=str(config.prediction_count)
            ),
            "playback_interval_seconds": tk.StringVar(
                self.window, value=f"{config.playback_interval_ms / 1000:g}"
            ),
            "follow_ship_view_span_meters": tk.StringVar(
                self.window, value=f"{config.follow_ship_view_span_meters:g}"
            ),
        }
        self._build()
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda _: self.close())
        self.window.update_idletasks()
        width = (
            max(430, self.window.winfo_reqwidth())
            if plot_only
            else min(560, max(400, parent.winfo_screenwidth() - 80))
        )
        height = (
            max(360, self.window.winfo_reqheight())
            if plot_only
            else min(
                max(320, self._content.winfo_reqheight() + 100),
                max(320, parent.winfo_screenheight() - 120),
            )
        )
        left = max(0, parent.winfo_rootx() + (parent.winfo_width() - width) // 2)
        top = max(0, parent.winfo_rooty() + (parent.winfo_height() - height) // 2)
        self.window.geometry(f"{width}x{height}+{left}+{top}")
        if plot_only:
            self.window.resizable(False, False)
        else:
            self.window.minsize(min(520, width), min(320, height))
        self.window.deiconify()

    def _build(self) -> None:
        if self._plot_only:
            self._build_plot_display()
            return
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        tk.Label(
            self.window,
            text="Apply updates these settings.\n"
            "A running analysis remains unchanged until \u201cStart analysis\u201d.",
            font=FONT,
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            justify="left",
            padx=14,
            pady=12,
        ).grid(row=0, column=0, sticky="ew")
        form = tk.Frame(self.window, bg=CONTROL_BACKGROUND)
        form.grid(row=1, column=0, sticky="nsew")
        canvas = tk.Canvas(form, bg=CONTROL_BACKGROUND, highlightthickness=0)
        scrollbar = tk.Scrollbar(form, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self._content = tk.Frame(canvas, bg=CONTROL_BACKGROUND, padx=12, pady=12)
        window = canvas.create_window((0, 0), window=self._content, anchor="nw")
        self._content.bind(
            "<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind(
            "<Configure>", lambda event: canvas.itemconfigure(window, width=event.width)
        )
        frame = tk.LabelFrame(
            self._content,
            text="Analysis setup",
            font=FONT,
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            padx=10,
            pady=8,
        )
        frame.pack(fill="x", pady=(0, 8))
        frame.columnconfigure(1, weight=1)
        labels = (
            ("Start index (from 0)", "start_index"),
            ("Additional position noise [m]", "position_noise_std_m"),
            ("Position-noise seed", "position_noise_seed"),
            ("Forecast steps (0 = off)", "prediction_count"),
            ("Future trajectories (0 = off)", "prediction_sample_count"),
            ("Trajectory-sampling seed", "forecast_seed"),
            ("Playback interval [s]", "playback_interval_seconds"),
        )
        for row, (label, name) in enumerate(labels):
            tk.Label(
                frame,
                text=label,
                bg=CONTROL_BACKGROUND,
                fg=TEXT_COLOR,
                font=FONT,
            ).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
            ttk.Entry(
                frame,
                textvariable=self._variables[name],
                width=16,
                style="IMMPredictor.TEntry",
            ).grid(row=row, column=1, sticky="ew", pady=(6, 0))

        def scroll(event):
            if canvas.yview() == (0.0, 1.0):
                return None
            direction = -1 if getattr(event, "num", None) == 4 or event.delta > 0 else 1
            canvas.yview_scroll(direction, "units")
            return "break"

        widgets = [canvas]
        while widgets:
            widget = widgets.pop()
            for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                widget.bind(event, scroll)
            widgets.extend(widget.winfo_children())
        buttons = tk.Frame(self.window, bg=CONTROL_BACKGROUND, padx=14, pady=12)
        buttons.grid(row=2, column=0, sticky="ew")
        create_button(buttons, "Apply", self.apply).pack(side="right", padx=(8, 0))
        create_button(buttons, "Cancel", self.close).pack(side="right")

    def _build_plot_display(self) -> None:
        tk.Label(
            self.window,
            text="Changes are applied only after selecting Apply.",
            font=FONT,
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            anchor="w",
            padx=14,
            pady=12,
        ).pack(fill="x")
        body = tk.Frame(self.window, bg=CONTROL_BACKGROUND, padx=14, pady=4)
        body.pack(fill="both", expand=True)
        for name, label in PLOT_DISPLAY_OPTIONS.items():
            tk.Checkbutton(
                body,
                text=label,
                variable=self._variables[name],
                bg=CONTROL_BACKGROUND,
                activebackground=CONTROL_BACKGROUND,
                fg=TEXT_COLOR,
                font=FONT,
                anchor="w",
            ).pack(fill="x", pady=3)
        row = tk.Frame(body, bg=CONTROL_BACKGROUND)
        row.pack(fill="x", pady=(8, 3))
        tk.Label(
            row,
            text="Prediction uncertainty",
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            font=FONT,
        ).pack(side="left")
        ttk.Combobox(
            row,
            textvariable=self._variables["prediction_region_style"],
            values=tuple(PREDICTION_REGION_STYLE_LABELS.values()),
            state="readonly",
            width=30,
        ).pack(side="right")
        row = tk.Frame(body, bg=CONTROL_BACKGROUND)
        row.pack(fill="x", pady=(8, 3))
        tk.Label(
            row,
            text="Follow-ship view span [m]",
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            font=FONT,
        ).pack(side="left")
        ttk.Entry(
            row,
            textvariable=self._variables["follow_ship_view_span_meters"],
            width=12,
            style="IMMPredictor.TEntry",
        ).pack(side="right")
        actions = tk.Frame(self.window, bg=CONTROL_BACKGROUND, padx=14, pady=12)
        actions.pack(fill="x")
        create_button(actions, "Apply", self.apply, width=12).pack(
            side="right", padx=(8, 0)
        )
        create_button(actions, "Cancel", self.close, width=12).pack(side="right")

    def apply(self) -> None:
        """Parse every field before handing one fresh configuration to the app."""
        if self._closed:
            return
        try:
            if self._plot_only:
                values = dict(
                    **{
                        name: self._variables[name].get()
                        for name in PLOT_DISPLAY_OPTIONS
                    },
                    prediction_region_style=next(
                        key
                        for key, label in PREDICTION_REGION_STYLE_LABELS.items()
                        if label == self._variables["prediction_region_style"].get()
                    ),
                    follow_ship_view_span_meters=float(
                        self._variables["follow_ship_view_span_meters"].get()
                    ),
                )
            else:
                values = dict(
                    start_index=int(self._variables["start_index"].get()),
                    position_noise_std_m=float(
                        self._variables["position_noise_std_m"].get()
                    ),
                    position_noise_seed=int(
                        self._variables["position_noise_seed"].get()
                    ),
                    prediction_count=int(self._variables["prediction_count"].get()),
                    prediction_sample_count=int(
                        self._variables["prediction_sample_count"].get()
                    ),
                    forecast_seed=int(self._variables["forecast_seed"].get()),
                    playback_interval_ms=round(
                        float(self._variables["playback_interval_seconds"].get()) * 1000
                    ),
                )
            config = replace(self._config, **values)
            applied = self._on_apply(config)
        except (TypeError, ValueError) as error:
            messagebox.showerror("Check settings", str(error), parent=self.window)
            return
        if applied:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.window.destroy()


IMM_HELP_SECTIONS = (
    (
        "Keyboard shortcuts",
        (
            ("Space", "Start or pause plot playback after starting analysis"),
            (
                "\u2190 / \u2192",
                "Move one radar cycle backward or forward; pause playback",
            ),
            ("Toolbar", "Pan, zoom, or reset the plot view"),
        ),
    ),
    (
        "Menu",
        (
            (
                "File \u2192 Open CSV",
                "Select a BR24 radar CSV before starting analysis",
            ),
            ("File \u2192 Exit", "Exit IMM Predictor"),
            (
                "View",
                "Show or hide settings; select meters or kilometers; open plot display options",
            ),
            (
                "Settings",
                "Configure analysis setup; locked until the active analysis is cancelled",
            ),
            ("Help", "Open this help window"),
        ),
    ),
    (
        "Controls",
        (
            (
                "Start analysis",
                "Load the selected CSV and show the N = 0 reference preview",
            ),
            (
                "Cancel analysis",
                "Stop the active analysis, clear displayed values, and unlock settings",
            ),
            (
                "Follow ship",
                "Center the plot on the current IMM position while retaining manual zoom",
            ),
            (
                "N slider",
                "N counts radar cycles from the selected start; each new cycle refreshes the forecast",
            ),
            (
                "N = 0",
                "Preview only; reference-state, track-state and evaluation values are not displayed",
            ),
            (
                "Analysis errors",
                "Playback stops and settings unlock after an error. Start a new analysis; a possibly partially updated filter is never retried",
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
            (
                "Start index",
                "First radar cycle to include, counting from 0; initializes a new track using its first detection",
            ),
            (
                "Additional position noise",
                "Extra Gaussian noise per position axis in meters; 0 keeps radar detections unchanged. Nominal sensor uncertainties stay unchanged",
            ),
            (
                "Position-noise seed",
                "Seed for reproducible additional radar-position noise",
            ),
            (
                "Forecast steps",
                "Number of future radar timestamps to predict; 0 disables forecasts and forecast metrics",
            ),
            ("Future trajectories", "Number of sampled IMM paths; 0 disables sampling"),
            (
                "Trajectory-sampling seed",
                "Seed for reproducible paths at each radar cycle; does not alter the analytic forecast",
            ),
            (
                "Playback interval",
                "Wait between playback steps in seconds; computation can take longer",
            ),
        ),
    ),
    (
        "State displays",
        (
            (
                "Reference state",
                "Ground-truth position, speed, heading and time at the current radar cycle, when available",
            ),
            (
                "Reference motion",
                "Position and time are required. Optional ground_truth_velocity [m/s], ground_truth_yaw [rad] and ground_truth_turn_rate [rad/s] take priority independently, including zero. Missing motion quantities are derived and marked (derived); invalid supplied values are reported",
            ),
            (
                "Derived direction",
                "Without yaw, Heading is the movement course from consecutive positions, not vessel orientation. It is unavailable at rest. Speed/course need two positions; course-change rate needs three",
            ),
            (
                "Reference turn rate",
                "Use supplied rates, otherwise wrapped yaw differences, or course differences if yaw is absent, over the actual preceding time interval. Shown in degrees/s with (derived); unavailable values show a dash",
            ),
            (
                "IMM track state",
                "Fused IMM position; speed and heading are derived from the latest two track positions, turn rate from three",
            ),
            (
                "Reference boundary",
                "Ground truth is used only for display and evaluation, never for radar association, filtering or forecasts",
            ),
            (
                "Missing values",
                "A dash means no value is available, including absent reference labels or disabled forecast metrics",
            ),
        ),
    ),
    (
        "Evaluation metrics",
        (
            (
                "Forecast ADE",
                "Mean 2D position error across the current forecast steps in meters; lower is better",
            ),
            (
                "Forecast FDE",
                "Position error at the final forecast step in meters; lower is better",
            ),
            (
                "Forecast RMSE",
                "Root mean squared 2D position error across the current forecast steps in meters; lower is better",
            ),
            (
                "Forecast Energy Score",
                "Probabilistic error of the Gaussian position forecasts, evaluated against reference positions; lower is better",
            ),
            (
                "Predictive log density",
                "Mean log density of future reference positions under the Gaussian forecasts; higher is better",
            ),
            (
                "Joint 50% coverage",
                "Cumulative share of forecast positions inside their 2D 50% covariance ellipses; shown as percentage and count",
            ),
            (
                "Joint 90% coverage",
                "Cumulative share of forecast positions inside their 2D 90% covariance ellipses; shown as percentage and count",
            ),
            (
                "Update time",
                "Time spent on the IMM initialization or filter update, excluding forecast generation, sampling and plotting",
            ),
        ),
    ),
    (
        "BR24 radar association",
        (
            (
                "Status",
                "Initial measurement, Associated, or Prediction only when no detection passes the association gate",
            ),
            (
                "Cycle",
                "Original radar-cycle counter; can differ from N when the start index is not 0",
            ),
            ("Detection ID", "Detection selected within the current radar cycle"),
            (
                "NIS",
                "Normalized innovation squared: discrepancy between predicted and detected position relative to innovation uncertainty",
            ),
            (
                "cov_xx / cov_xy / cov_yy",
                "Cartesian measurement covariance entries in square meters, not forecast covariance",
            ),
        ),
    ),
    (
        "Plot display and uncertainty",
        (
            (
                "Apply / Cancel",
                "Apply commits the local display changes; Cancel discards unapplied edits. The non-modal editor keeps playback accessible",
            ),
            (
                "Display layers",
                "Toggle the legend, reference trajectory, IMM track, current position, sampled paths and forecast mean",
            ),
            (
                "Prediction uncertainty",
                "Choose horizon or per-step density regions/contours, covariance ellipses, or no region; toggle 50% and 90% separately",
            ),
            (
                "Density regions",
                "Approximate Gaussian IMM position distributions. Horizon views pool future steps and are not simultaneous trajectory probability bands",
            ),
            (
                "Follow-ship view span",
                "Initial viewport span in meters; subsequent manual zoom is retained",
            ),
            (
                "Mode probabilities",
                "Forecast probabilities of the CV, CTRV and CA motion models",
            ),
            (
                "Forecast uncertainty",
                "Equivalent-area circle radii of the 50% and 90% position ellipses in meters",
            ),
            (
                "Sampled IMM trajectories",
                "Monte Carlo extension with uncertain posterior origins, model switches and process noise. Sample moments need not match the analytic IMM forecast exactly",
            ),
            (
                "Forecast mean / metrics",
                "The analytic mean, covariance regions and evaluation metrics are independent of the displayed sampled paths",
            ),
        ),
    ),
)


class IMMHelpWindow(tk.Toplevel):
    """Predictor-style non-modal help with aligned, scrollable sections."""

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
            text="IMM Predictor",
            font=("Arial", 13, "bold"),
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            anchor="w",
        ).pack(fill="x", padx=24, pady=(18, 12))
        form = tk.Frame(main, bg=CONTROL_BACKGROUND)
        form.pack(fill="both", expand=True, padx=24)
        canvas = self._canvas = tk.Canvas(
            form, bg=CONTROL_BACKGROUND, highlightthickness=0, width=600
        )
        scrollbar = tk.Scrollbar(form, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        body = self._body = tk.Frame(canvas, bg=CONTROL_BACKGROUND, padx=12, pady=12)
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind(
            "<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind(
            "<Configure>", lambda event: canvas.itemconfigure(window, width=event.width)
        )
        for title, rows in IMM_HELP_SECTIONS:
            section = tk.LabelFrame(
                body,
                text=title,
                font=FONT,
                bg=CONTROL_BACKGROUND,
                fg=TEXT_COLOR,
                padx=12,
                pady=10,
            )
            section.pack(fill="x", pady=(0, 12))
            section.columnconfigure(1, weight=1)
            for row, (label, description) in enumerate(rows):
                tk.Label(
                    section,
                    text=label,
                    width=24,
                    font=("Arial", 10, "bold"),
                    bg=CONTROL_BACKGROUND,
                    fg=TEXT_COLOR,
                    anchor="nw",
                    justify="left",
                ).grid(row=row, column=0, sticky="nw", padx=(0, 20), pady=3)
                tk.Label(
                    section,
                    text=description,
                    font=FONT,
                    bg=CONTROL_BACKGROUND,
                    fg=TEXT_COLOR,
                    anchor="nw",
                    justify="left",
                    wraplength=300,
                ).grid(row=row, column=1, sticky="nw", pady=3)
        widgets = [canvas]
        while widgets:
            widget = widgets.pop()
            for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                widget.bind(event, self._scroll)
            widgets.extend(widget.winfo_children())
        actions = tk.Frame(main, bg=CONTROL_BACKGROUND)
        actions.pack(fill="x", padx=24, pady=(12, 18))
        create_button(actions, "OK", self.close, width=12).pack(anchor="e")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", lambda _: self.close())
        self.update_idletasks()
        canvas.configure(
            height=min(body.winfo_reqheight(), int(self.winfo_screenheight() * 0.65))
        )
        self.update_idletasks()
        height = min(
            max(320, main.winfo_reqheight() + 16),
            max(320, self.winfo_screenheight() - 120),
        )
        self.geometry(f"660x{height}+0+0")
        self.deiconify()
        self.focus_set()

    def _scroll(self, event):
        if self._canvas.yview() == (0.0, 1.0):
            return None
        direction = -1 if getattr(event, "num", None) == 4 or event.delta > 0 else 1
        self._canvas.yview_scroll(direction, "units")
        return "break"

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._on_close is not None:
            self._on_close()
        self.destroy()
