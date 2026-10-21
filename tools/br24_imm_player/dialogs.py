"""Tk dialogs for BR24 IMM player settings and help."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from .player import PREDICTION_REGION_STYLE_LABELS, IMMPlayerConfig
from .view import FONT, TEXT_COLOR, WINDOW_BACKGROUND, create_button


class PlaybackSettingsDialog:
    """Edit a session configuration transactionally in a small modal dialog."""

    def __init__(self, parent, config: IMMPlayerConfig, on_apply) -> None:
        self.window = tk.Toplevel(parent)
        self.window.title("Playback, forecast and plot settings")
        self.window.configure(bg=WINDOW_BACKGROUND)
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.grab_set()
        self._config = config
        self._on_apply = on_apply
        self._variables = {
            "prediction_region_style": tk.StringVar(
                self.window,
                value=PREDICTION_REGION_STYLE_LABELS[config.prediction_region_style],
            ),
            "show_prediction_region_50": tk.BooleanVar(
                self.window, value=config.show_prediction_region_50
            ),
            "show_prediction_region_90": tk.BooleanVar(
                self.window, value=config.show_prediction_region_90
            ),
            "prediction_count": tk.StringVar(
                self.window, value=str(config.prediction_count)
            ),
            "playback_interval_ms": tk.StringVar(
                self.window, value=str(config.playback_interval_ms)
            ),
            "follow_ship_view_span_meters": tk.StringVar(
                self.window, value=f"{config.follow_ship_view_span_meters:g}"
            ),
        }
        self._build()
        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self) -> None:
        frame = tk.Frame(self.window, bg=WINDOW_BACKGROUND, padx=16, pady=14)
        frame.grid(sticky="nsew")
        labels = (
            ("Prediction horizon", "prediction_count"),
            ("Playback interval [ms]", "playback_interval_ms"),
            ("Follow-ship span [m]", "follow_ship_view_span_meters"),
        )
        for row, (label, name) in enumerate(labels):
            tk.Label(
                frame,
                text=label,
                bg=WINDOW_BACKGROUND,
                fg=TEXT_COLOR,
                font=FONT,
            ).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=5)
            tk.Entry(
                frame, textvariable=self._variables[name], width=16, font=FONT
            ).grid(row=row, column=1, sticky="ew", pady=5)
        row = len(labels)
        tk.Label(
            frame,
            text="Prediction uncertainty",
            bg=WINDOW_BACKGROUND,
            fg=TEXT_COLOR,
            font=FONT,
        ).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=5)
        ttk.Combobox(
            frame,
            textvariable=self._variables["prediction_region_style"],
            values=tuple(PREDICTION_REGION_STYLE_LABELS.values()),
            state="readonly",
            width=28,
        ).grid(row=row, column=1, sticky="ew", pady=5)
        for offset, probability in enumerate((50, 90), start=1):
            tk.Checkbutton(
                frame,
                text=f"Show {probability}% region",
                variable=self._variables[f"show_prediction_region_{probability}"],
                bg=WINDOW_BACKGROUND,
                fg=TEXT_COLOR,
                font=FONT,
            ).grid(row=row + offset, column=0, columnspan=2, sticky="w")
        tk.Label(
            frame,
            text="Density views approximate the Gaussian IMM position forecasts.\n"
            "Horizon views pool positions across forecast steps;\n"
            "they are not a simultaneous trajectory probability band.",
            bg=WINDOW_BACKGROUND,
            fg=TEXT_COLOR,
            font=FONT,
            justify="left",
        ).grid(row=row + 3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        buttons = tk.Frame(frame, bg=WINDOW_BACKGROUND)
        buttons.grid(row=row + 4, column=0, columnspan=2, sticky="e", pady=(12, 0))
        create_button(buttons, "Cancel", self.close).pack(side="right")
        create_button(buttons, "Apply", self.apply).pack(side="right", padx=(0, 7))

    def apply(self) -> None:
        """Parse every field before handing one fresh configuration to the app."""
        try:
            config = IMMPlayerConfig(
                prediction_region_style=next(
                    key
                    for key, label in PREDICTION_REGION_STYLE_LABELS.items()
                    if label == self._variables["prediction_region_style"].get()
                ),
                show_prediction_region_50=self._variables[
                    "show_prediction_region_50"
                ].get(),
                show_prediction_region_90=self._variables[
                    "show_prediction_region_90"
                ].get(),
                data_file=self._config.data_file,
                prediction_count=int(self._variables["prediction_count"].get()),
                playback_interval_ms=int(self._variables["playback_interval_ms"].get()),
                follow_ship_view_span_meters=float(
                    self._variables["follow_ship_view_span_meters"].get()
                ),
            )
            applied = self._on_apply(config)
        except (TypeError, ValueError) as error:
            messagebox.showerror("Check settings", str(error), parent=self.window)
            return
        if applied:
            self.close()

    def close(self) -> None:
        self.window.grab_release()
        self.window.destroy()


def show_help(parent) -> None:
    """Explain the online cycle and the reference-data boundary."""
    messagebox.showinfo(
        "BR24 IMM Player",
        "Each N step processes one new BR24 radar cycle and refreshes the "
        "multi-step IMM forecast.\n\n"
        "Reference positions, when present in the selected CSV, are displayed "
        "and evaluated only after the forecast. They are never tracker input.",
        parent=parent,
    )
