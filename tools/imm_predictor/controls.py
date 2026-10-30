"""Scrollable left sidebar for the IMM Predictor."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np

from .player import IMMDisplayState
from .view import (
    CONTROL_BACKGROUND,
    FONT,
    HEADING_FONT,
    TEXT_COLOR,
    create_button,
    set_button_palette,
)


class IMMControls(tk.Frame):
    """Expose file, movement, evaluation, and association values in one sidebar."""

    def __init__(self, parent, *, on_open_file, on_cancel, on_start) -> None:
        super().__init__(parent, bg=CONTROL_BACKGROUND, width=340)
        self._on_open_file = on_open_file
        self._on_cancel = on_cancel
        self._on_start = on_start
        self._analysis_button: tk.Button | None = None
        self._data_file = tk.StringVar(self, value="")
        self._values = {
            name: tk.StringVar(self, value="-")
            for name in (
                "reference_position",
                "reference_speed",
                "reference_heading",
                "reference_turn_rate",
                "reference_time",
                "speed",
                "heading",
                "turn_rate",
                "ade",
                "fde",
                "rmse",
                "energy_score",
                "negative_log_predictive_density",
                "coverage_50",
                "coverage_90",
                "update_time",
                "association_status",
                "detection_id",
                "nis",
                "cov_xx",
                "cov_xy",
                "cov_yy",
                "cycle",
            )
        }
        self._build_scrollable_layout()

    def _build_scrollable_layout(self) -> None:
        canvas = tk.Canvas(self, bg=CONTROL_BACKGROUND, highlightthickness=0, width=340)
        self._canvas = canvas
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        body = tk.Frame(canvas, bg=CONTROL_BACKGROUND, padx=12, pady=12)
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        body.columnconfigure(0, weight=1)
        body.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window, width=event.width),
        )

        tk.Label(
            body,
            text="IMM online analysis",
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            font=HEADING_FONT,
        ).grid(row=0, column=0, sticky="ew", pady=(14, 10))
        self._add_analysis_section(body).grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self._add_data_section(body).grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self._add_value_section(
            body,
            "Reference state",
            (
                ("Position", "reference_position"),
                ("Speed", "reference_speed"),
                ("Heading", "reference_heading"),
                ("Turn rate", "reference_turn_rate"),
                ("Time", "reference_time"),
            ),
        ).grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self._add_value_section(
            body,
            "Posterior medians",
            (
                ("Speed", "speed"),
                ("Heading", "heading"),
                ("Turn rate", "turn_rate"),
            ),
        ).grid(row=4, column=0, sticky="ew", pady=(0, 8))
        self._add_value_section(
            body,
            "Evaluation metrics",
            (
                ("Forecast ADE (median)", "ade"),
                ("Forecast FDE (median)", "fde"),
                ("Forecast RMSE (median)", "rmse"),
                ("Forecast Energy Score", "energy_score"),
                ("NLPD", "negative_log_predictive_density"),
                ("Joint 50% coverage", "coverage_50"),
                ("Joint 90% coverage", "coverage_90"),
                ("Update time", "update_time"),
            ),
        ).grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self._add_value_section(
            body,
            "BR24 radar association",
            (
                ("Status", "association_status"),
                ("Cycle", "cycle"),
                ("Detection ID", "detection_id"),
                ("NIS", "nis"),
                ("cov_xx [m²]", "cov_xx"),
                ("cov_xy [m²]", "cov_xy"),
                ("cov_yy [m²]", "cov_yy"),
            ),
        ).grid(row=6, column=0, sticky="ew")
        self._bind_mouse_wheel()

    def _add_analysis_section(self, parent) -> tk.LabelFrame:
        section = self._section(parent, "Analysis")
        section.configure(pady=10)
        section.columnconfigure(0, weight=1)
        self._analysis_button = create_button(section, "Start analysis", self._on_start)
        self._analysis_button.grid(sticky="ew")
        return section

    def _add_data_section(self, parent) -> tk.LabelFrame:
        section = self._section(parent, "Data")
        section.columnconfigure(1, weight=1)
        tk.Label(
            section,
            text="CSV file",
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            font=FONT,
        ).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        file_frame = tk.Frame(section, bg=CONTROL_BACKGROUND)
        file_frame.grid(row=0, column=1, sticky="ew", pady=(6, 0))
        file_frame.columnconfigure(0, weight=1)
        entry = ttk.Entry(
            file_frame,
            textvariable=self._data_file,
            state="readonly",
            width=16,
            style="IMMPredictor.TEntry",
        )
        entry.grid(row=0, column=0, sticky="ew")
        self._file_entry = entry
        self._browse_button = ttk.Button(
            file_frame,
            text="…",
            command=self.choose_file,
            width=2,
            style="IMMPredictor.TButton",
        )
        self._browse_button.grid(row=0, column=1, sticky="ns", padx=(5, 0))
        return section

    def _add_value_section(self, parent, title: str, rows) -> tk.LabelFrame:
        section = self._section(parent, title)
        section.columnconfigure(1, weight=1)
        for row, (label, name) in enumerate(rows):
            tk.Label(
                section,
                text=f"{label}:",
                bg=CONTROL_BACKGROUND,
                fg=TEXT_COLOR,
                font=("Arial", 9, "bold"),
                anchor="w",
            ).grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=(2, 3))
            tk.Label(
                section,
                textvariable=self._values[name],
                bg=CONTROL_BACKGROUND,
                fg=TEXT_COLOR,
                font="TkFixedFont",
                justify="left",
                anchor="w",
                wraplength=185,
            ).grid(row=row, column=1, sticky="ew", pady=(2, 3))
        return section

    def _section(self, parent, title: str) -> tk.LabelFrame:
        return tk.LabelFrame(
            parent,
            text=title,
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            font=FONT,
            padx=10,
            pady=8,
        )

    def _on_mouse_wheel(self, event):
        if self._canvas.yview() == (0.0, 1.0):
            return None
        direction = -1 if getattr(event, "num", None) == 4 or event.delta > 0 else 1
        self._canvas.yview_scroll(direction, "units")
        return "break"

    def _on_touchpad_scroll(self, event):
        _delta_x, delta_y = self._canvas.tk.call(
            "tk::PreciseScrollDeltas",
            event.delta,
        )
        delta_y = float(delta_y)
        region = self._canvas.bbox("all")
        if delta_y == 0 or region is None:
            return "break"
        content_height = region[3] - region[1]
        first, last = self._canvas.yview()
        if content_height > 0 and last - first < 1:
            new_first = min(
                max(first - delta_y / content_height, 0), 1 - (last - first)
            )
            self._canvas.yview_moveto(new_first)
        return "break"

    def _bind_mouse_wheel(self) -> None:
        """Keep scrolling local to the sidebar, including all populated fields."""
        widgets = [self._canvas]
        while widgets:
            widget = widgets.pop()
            widget.bind("<MouseWheel>", self._on_mouse_wheel)
            widget.bind("<Button-4>", self._on_mouse_wheel)
            widget.bind("<Button-5>", self._on_mouse_wheel)
            try:
                widget.bind("<TouchpadScroll>", self._on_touchpad_scroll)
            except tk.TclError:
                pass  # TouchpadScroll requires Tk 9.
            widgets.extend(widget.winfo_children())

    def choose_file(self) -> None:
        """Ask for a tracking CSV and notify the application only if selected."""
        filename = filedialog.askopenfilename(
            parent=self.winfo_toplevel(),
            title="Open BR24 tracking CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if filename:
            self._on_open_file(Path(filename))

    def set_data_file(self, data_file: Path) -> None:
        """Show the successfully loaded data file in the readonly field."""
        self._data_file.set(str(data_file))

    def set_analysis_active(self, active: bool) -> None:
        """Offer the Predictor-style red cancel action for an active session."""
        self._file_entry.configure(state="disabled" if active else "readonly")
        self._browse_button.configure(state="disabled" if active else "normal")
        if self._analysis_button is not None:
            self._analysis_button.configure(
                text="Cancel analysis" if active else "Start analysis",
                command=self._on_cancel if active else self._on_start,
            )
            set_button_palette(
                self._analysis_button,
                background="#f7d9d9" if active else "#ffffff",
                hover_background="#efbcbc" if active else "#b8d8e8",
            )

    def clear_display_state(self) -> None:
        """Reset values until the user explicitly starts an analysis."""
        for value in self._values.values():
            value.set("-")

    def show_display_state(
        self, state: IMMDisplayState, *, coordinate_display_mode="m"
    ) -> None:
        """Render posterior medians, reference, metric, and association fields."""
        medians = state.posterior_medians
        metrics = state.metrics
        stage = state.stage
        for key, attribute, unit in (
            ("speed", "speed_mps", "m/s"),
            ("heading", "heading_degrees", "°"),
            ("turn_rate", "turn_rate_degrees_per_second", "°/s"),
        ):
            value = None if medians is None else getattr(medians, attribute)
            self._values[key].set(_format_optional(value, unit))
        reference = state.reference_state
        for name in ("position", "speed", "heading", "turn_rate", "time"):
            self._values[f"reference_{name}"].set("-")
        if reference is not None:
            self._values["reference_position"].set(
                _format_position(reference.position_meters, coordinate_display_mode)
            )
            self._values["reference_speed"].set(
                _format_optional(
                    reference.speed_mps, "m/s", derived=reference.speed_is_derived
                )
            )
            self._values["reference_heading"].set(
                _format_optional(
                    reference.heading_degrees, "°", derived=reference.heading_is_derived
                )
            )
            self._values["reference_turn_rate"].set(
                _format_optional(
                    reference.turn_rate_degrees_per_second,
                    "°/s",
                    derived=reference.turn_rate_is_derived,
                )
            )
            self._values["reference_time"].set(
                _format_elapsed_time(reference.time_seconds)
            )
        self._values["ade"].set(_format_optional(metrics.ade_m, "m"))
        self._values["fde"].set(_format_optional(metrics.fde_m, "m"))
        self._values["rmse"].set(_format_optional(metrics.rmse_m, "m"))
        self._values["energy_score"].set(_format_optional(metrics.energy_score_m, "m"))
        self._values["negative_log_predictive_density"].set(
            _format_optional(metrics.negative_log_predictive_density, "")
        )
        self._values["coverage_50"].set(
            _format_coverage(
                metrics.joint_coverage_50_count,
                metrics.joint_coverage_total,
            )
        )
        self._values["coverage_90"].set(
            _format_coverage(
                metrics.joint_coverage_90_count,
                metrics.joint_coverage_total,
            )
        )
        self._values["update_time"].set(
            _format_optional(stage.update_time_seconds, "s", decimals=3)
        )
        self._values["association_status"].set(
            "Initial measurement"
            if stage.observation_count == 1
            else "Associated"
            if stage.detection_id is not None
            else "Prediction only"
        )
        self._values["detection_id"].set(
            "-" if stage.detection_id is None else str(stage.detection_id)
        )
        self._values["nis"].set(_format_optional(stage.nis, ""))
        covariance = stage.measurement_covariance
        self._values["cov_xx"].set(
            _format_optional(
                None if covariance is None else covariance[0, 0],
                "",
                decimals=3,
            )
        )
        self._values["cov_xy"].set(
            _format_optional(
                None if covariance is None else covariance[0, 1],
                "",
                decimals=3,
            )
        )
        self._values["cov_yy"].set(
            _format_optional(
                None if covariance is None else covariance[1, 1],
                "",
                decimals=3,
            )
        )
        self._values["cycle"].set(str(stage.cycle_counter))


def _format_position(position, coordinate_display_mode: str) -> str:
    scale = 0.001 if coordinate_display_mode == "km" else 1.0
    return (
        f"x = {position[0] * scale:.2f} {coordinate_display_mode}\n"
        f"y = {position[1] * scale:.2f} {coordinate_display_mode}"
    )


def _format_optional(
    value: float | None, unit: str, *, decimals: int = 2, derived=False
) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    suffix = f" {unit}" if unit else ""
    return f"{value:.{decimals}f}{suffix}" + (" (derived)" if derived else "")


def _format_elapsed_time(seconds: float) -> str:
    """Match the Trajectory Predictor's concise elapsed-time display."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    whole_seconds = round(seconds)
    minutes, remaining_seconds = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min {remaining_seconds:02d} s"
    return f"{minutes} min {remaining_seconds:02d} s"


def _format_coverage(covered_count: int, total_count: int) -> str:
    if total_count < 1:
        return "-"
    return f"{100 * covered_count / total_count:.1f}% ({covered_count}/{total_count})"
