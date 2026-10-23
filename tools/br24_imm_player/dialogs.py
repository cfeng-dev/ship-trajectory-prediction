"""Tk dialogs for BR24 IMM player settings and help."""

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
    """Edit local drafts in modal analysis or non-modal plot settings."""

    def __init__(
        self, parent, config: IMMPlayerConfig, on_apply, *, plot_only=False
    ) -> None:
        self.window = tk.Toplevel(parent)
        self.window.withdraw()
        self._plot_only = plot_only
        self.window.title("Plot display" if plot_only else "Analysis setup")
        self.window.configure(bg=CONTROL_BACKGROUND)
        self.window.transient(parent)
        self._config = config
        self._on_apply = on_apply
        self._variables = {
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
        if not plot_only:
            self.window.grab_set()

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
                style="IMMPlayer.TEntry",
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
            style="IMMPlayer.TEntry",
        ).pack(side="right")
        actions = tk.Frame(self.window, bg=CONTROL_BACKGROUND, padx=14, pady=12)
        actions.pack(fill="x")
        create_button(actions, "Apply", self.apply, width=12).pack(
            side="right", padx=(8, 0)
        )
        create_button(actions, "Cancel", self.close, width=12).pack(side="right")

    def apply(self) -> None:
        """Parse every field before handing one fresh configuration to the app."""
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
        if not self._plot_only:
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
