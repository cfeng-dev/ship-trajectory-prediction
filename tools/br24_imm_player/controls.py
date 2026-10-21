"""Scrollable left sidebar for the BR24 IMM desktop player."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np

from .player import IMMDisplayState
from .view import CONTROL_BACKGROUND, FONT, HEADING_FONT, TEXT_COLOR, create_button


class IMMControls(tk.Frame):
    """Expose file, movement, evaluation, and association values in one sidebar."""

    def __init__(self, parent, *, on_open_file, on_reload, on_start) -> None:
        super().__init__(parent, bg=CONTROL_BACKGROUND, width=340)
        self._on_open_file = on_open_file
        self._on_reload = on_reload
        self._on_start = on_start
        self._analysis_button: tk.Button | None = None
        self._data_file = tk.StringVar(self, value="")
        self._values = {
            name: tk.StringVar(self, value="-")
            for name in (
                "position",
                "speed",
                "heading",
                "turn_rate",
                "time",
                "ade",
                "fde",
                "rmse",
                "energy_score",
                "log_density",
                "coverage_50",
                "coverage_90",
                "association_status",
                "detection_id",
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
        canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        body.bind("<MouseWheel>", self._on_mouse_wheel)

        tk.Label(
            body,
            text="IMM online analysis",
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            font=HEADING_FONT,
        ).grid(row=0, column=0, sticky="ew", pady=(4, 12))
        self._add_analysis_section(body).grid(
            row=1, column=0, sticky="ew", pady=(0, 10)
        )
        self._add_data_section(body).grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self._add_value_section(
            body,
            "Current movement",
            (
                ("Position", "position"),
                ("Speed", "speed"),
                ("Heading", "heading"),
                ("Turn rate", "turn_rate"),
                ("Time", "time"),
            ),
        ).grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self._add_value_section(
            body,
            "Evaluation metrics",
            (
                ("Forecast ADE", "ade"),
                ("Forecast FDE", "fde"),
                ("Forecast RMSE", "rmse"),
                ("Forecast Energy Score", "energy_score"),
                ("Predictive log density", "log_density"),
                ("Joint 50% coverage", "coverage_50"),
                ("Joint 90% coverage", "coverage_90"),
            ),
        ).grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self._add_value_section(
            body,
            "BR24 radar association",
            (
                ("Status", "association_status"),
                ("Detection ID", "detection_id"),
                ("Cycle", "cycle"),
            ),
        ).grid(row=5, column=0, sticky="ew")

    def _add_analysis_section(self, parent) -> tk.LabelFrame:
        section = self._section(parent, "Analysis")
        self._analysis_button = create_button(section, "Start analysis", self._on_start)
        self._analysis_button.grid(sticky="ew", padx=8, pady=8)
        return section

    def _add_data_section(self, parent) -> tk.LabelFrame:
        section = self._section(parent, "Data")
        section.columnconfigure(0, weight=1)
        tk.Label(
            section,
            text="BR24 CSV file",
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            font=FONT,
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        entry = tk.Entry(
            section,
            textvariable=self._data_file,
            state="readonly",
            readonlybackground="#FFFFFF",
            fg="#3D617A",
            font=("Arial", 8),
        )
        entry.grid(row=1, column=0, sticky="ew", padx=(8, 4), pady=(0, 7))
        create_button(section, "Browse...", self.choose_file).grid(
            row=1, column=1, sticky="e", padx=(0, 8), pady=(0, 7)
        )
        create_button(section, "Reset analysis", self._on_reload).grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8)
        )
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
            ).grid(row=row, column=0, sticky="nw", padx=(8, 5), pady=3)
            tk.Label(
                section,
                textvariable=self._values[name],
                bg=CONTROL_BACKGROUND,
                fg=TEXT_COLOR,
                font=FONT,
                justify="left",
                anchor="w",
                wraplength=185,
            ).grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=3)
        return section

    def _section(self, parent, title: str) -> tk.LabelFrame:
        return tk.LabelFrame(
            parent,
            text=title,
            bg=CONTROL_BACKGROUND,
            fg="#123B59",
            font=FONT,
            relief="groove",
            borderwidth=1,
        )

    def _on_mouse_wheel(self, event):
        direction = -1 if event.delta > 0 else 1
        self._canvas.yview_scroll(direction, "units")
        return "break"

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
        """Distinguish an unstarted analysis from a displayed IMM session."""
        if self._analysis_button is not None:
            self._analysis_button.configure(
                text="Restart analysis" if active else "Start analysis"
            )

    def clear_display_state(self) -> None:
        """Reset values until the user explicitly starts an analysis."""
        for value in self._values.values():
            value.set("-")

    def show_display_state(self, state: IMMDisplayState) -> None:
        """Render one stage's movement, metric, and association fields."""
        movement = state.movement
        metrics = state.metrics
        stage = state.stage
        self._values["position"].set(
            f"x = {movement.position_meters[0]:.1f} m\ny = {movement.position_meters[1]:.1f} m"
        )
        self._values["speed"].set(_format_optional(movement.speed_mps, "m/s"))
        self._values["heading"].set(_format_optional(movement.heading_degrees, "deg"))
        self._values["turn_rate"].set(
            _format_optional(movement.turn_rate_degrees_per_second, "deg/s")
        )
        self._values["time"].set(f"{movement.time_seconds:.3f} s")
        self._values["ade"].set(_format_optional(metrics.ade_m, "m"))
        self._values["fde"].set(_format_optional(metrics.fde_m, "m"))
        self._values["rmse"].set(_format_optional(metrics.rmse_m, "m"))
        self._values["energy_score"].set(_format_optional(metrics.energy_score_m, "m"))
        self._values["log_density"].set(
            _format_optional(metrics.predictive_log_density, "")
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
        self._values["association_status"].set(
            "Associated"
            if stage.detection_id is not None
            else "No associated detection"
        )
        self._values["detection_id"].set(
            "-" if stage.detection_id is None else str(stage.detection_id)
        )
        self._values["cycle"].set(str(stage.cycle_counter))


def _format_optional(value: float | None, unit: str) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    suffix = f" {unit}" if unit else ""
    return f"{value:.2f}{suffix}"


def _format_coverage(covered_count: int, total_count: int) -> str:
    if total_count < 1:
        return "-"
    return f"{100 * covered_count / total_count:.1f}% ({covered_count}/{total_count})"
