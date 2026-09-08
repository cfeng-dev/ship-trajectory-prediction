"""Tk input controls; parsing and inference are kept outside the view."""

import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

from .settings import (
    COORDINATE_DISPLAY_MODES,
    LABELS,
    MAIN_DATA_FIELDS,
    METHOD_DISPLAY_LABELS,
    METHOD_DISPLAY_OPTIONS,
    default_form_values,
)
from .view import (
    CONTROL_BACKGROUND,
    FONT,
    TEXT_COLOR,
    create_styled_button,
    set_styled_button_palette,
)

CSV_STATE_ROW_KEYS = ("position", "speed", "heading", "turn_rate", "time")


def format_reference_position(x, y, *, coordinate_display_mode="m"):
    """Format local CSV coordinates like the simulator status display."""
    if not np.isfinite(x) or not np.isfinite(y):
        return "—"
    if coordinate_display_mode == "gps":
        return f"longitude = {x:.6f}°\nlatitude = {y:.6f}°"
    unit = "km" if coordinate_display_mode == "km" else "m"
    return f"x = {x:.2f} {unit}\ny = {y:.2f} {unit}"


class ScrollableForm(tk.Frame):
    """A form that remains accessible when the settings pane is small."""

    def __init__(self, parent, *, width=340):
        super().__init__(parent, bg=CONTROL_BACKGROUND)
        canvas = self.canvas = tk.Canvas(
            self, highlightthickness=0, width=width, bg=CONTROL_BACKGROUND
        )
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self.body = tk.Frame(canvas, bg=CONTROL_BACKGROUND, padx=12, pady=12)
        window = canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.columnconfigure(0, weight=1)
        self.body.bind(
            "<Configure>",
            lambda _: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>", lambda event: canvas.itemconfigure(window, width=event.width)
        )

    def bind_mouse_wheel(self):
        """Scroll only this form when the pointer is over its populated fields."""

        def scroll(event):
            if self.canvas.yview() == (0.0, 1.0):
                return None
            direction = -1 if getattr(event, "num", None) == 4 or event.delta > 0 else 1
            self.canvas.yview_scroll(direction, "units")
            return "break"

        widgets = [self.canvas, self.body]
        while widgets:
            widget = widgets.pop()
            widget.bind("<MouseWheel>", scroll)
            widget.bind("<Button-4>", scroll)
            widget.bind("<Button-5>", scroll)
            widgets.extend(widget.winfo_children())


def split_analysis_data_fields(fields):
    """Keep the inference choice with the action that consumes it."""
    analysis = {key: fields[key] for key in ("inference_method",) if key in fields}
    data = {key: value for key, value in fields.items() if key not in analysis}
    return analysis, data


class SettingsPanel(tk.Frame):
    """Editable data, prior and inference settings with an explicit apply action."""

    def __init__(self, parent, on_apply, on_cancel=None):
        super().__init__(parent, width=350, bg=CONTROL_BACKGROUND)
        self.pack_propagate(False)
        self._on_apply = on_apply
        self._on_cancel = on_cancel or (lambda: None)
        self.variables = {
            group: create_variables(self, fields)
            for group, fields in default_form_values().items()
        }
        self._analysis_input_widgets = []
        self.settings_form = ScrollableForm(self)
        self.data_form = self.settings_form
        self.settings_form.pack(fill="both", expand=True)
        body = self.settings_form.body
        tk.Label(
            body,
            text="Posterior analysis",
            font=("Arial", 12, "bold"),
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
        ).pack(pady=(14, 10))
        self.analysis_section = tk.LabelFrame(
            body,
            text="Analysis",
            font=FONT,
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            padx=10,
            pady=10,
        )
        self.analysis_section.pack(fill="x", pady=(0, 8))
        self.analysis_section.columnconfigure(0, weight=1)
        self.analysis_fields = tk.Frame(self.analysis_section, bg=CONTROL_BACKGROUND)
        analysis_values, data_values = split_analysis_data_fields(
            {key: self.variables["data"][key] for key in MAIN_DATA_FIELDS}
        )
        self.apply_button = create_styled_button(
            self.analysis_section, text="Start analysis", command=self._on_apply
        )
        self.apply_button.pack(fill="x")
        self.analysis_fields.columnconfigure(0, weight=1)
        populate_fields(
            self.analysis_fields,
            analysis_values,
            stacked=True,
            editable_widgets=self._analysis_input_widgets,
        )
        self.analysis_fields.pack(fill="x", pady=(8, 0))

        self.data_section = tk.LabelFrame(
            body,
            text="Data",
            font=FONT,
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            padx=10,
            pady=8,
        )
        self.data_section.pack(fill="x")
        self.data_section.columnconfigure(0, weight=1)
        populate_fields(
            self.data_section,
            data_values,
            choose_file=self.choose_file,
            stacked=True,
            editable_widgets=self._analysis_input_widgets,
        )
        self.posterior_state_section = tk.LabelFrame(
            body,
            text="Reference state",
            font=FONT,
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            padx=10,
            pady=8,
        )
        self.posterior_state_section.pack(fill="x", pady=(8, 0))
        self.posterior_state_values = {
            key: tk.StringVar(self, value="—") for key in CSV_STATE_ROW_KEYS
        }
        labels_by_key = {
            "position": "Position",
            "speed": "Speed",
            "heading": "Heading",
            "turn_rate": "Turn rate",
            "time": "Time",
        }
        for row, key in enumerate(CSV_STATE_ROW_KEYS):
            label = labels_by_key[key]
            tk.Label(
                self.posterior_state_section,
                text=f"{label}:",
                font=("Arial", 9, "bold"),
                bg=CONTROL_BACKGROUND,
                fg=TEXT_COLOR,
            ).grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=(2, 3))
            tk.Label(
                self.posterior_state_section,
                textvariable=self.posterior_state_values[key],
                font="TkFixedFont",
                bg=CONTROL_BACKGROUND,
                fg=TEXT_COLOR,
                justify="left",
            ).grid(row=row, column=1, sticky="nw", pady=(2, 3))
        self.settings_form.bind_mouse_wheel()

    def show_reference_state(self, state):
        """Show the unmodified reference state for the selected CSV row."""
        if state is None:
            for variable in self.posterior_state_values.values():
                variable.set("—")
            return
        x, y, heading, speed, turn_rate, coordinate_display_mode, time_seconds = state
        self.posterior_state_values["position"].set(
            format_reference_position(
                x, y, coordinate_display_mode=coordinate_display_mode
            )
        )
        self.posterior_state_values["heading"].set(
            f"{heading:.2f}°" if np.isfinite(heading) else "—"
        )
        self.posterior_state_values["speed"].set(
            f"{speed:.2f} m/s" if np.isfinite(speed) else "—"
        )
        self.posterior_state_values["turn_rate"].set(
            f"{turn_rate:.2f}°/s" if np.isfinite(turn_rate) else "—"
        )
        self.posterior_state_values["time"].set(
            f"{time_seconds:.1f} s" if np.isfinite(time_seconds) else "—"
        )

    def choose_file(self):
        """Choose a CSV without starting or replacing an analysis."""
        selected = filedialog.askopenfilename(
            parent=self,
            title="Select trajectory CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if selected:
            self.variables["data"]["data_file"].set(selected)

    def set_analysis_active(self, active):
        """Lock inputs whose changes would require a fresh inference run."""
        for widget in self._analysis_input_widgets:
            if isinstance(widget, ttk.Combobox):
                state = "disabled" if active else "readonly"
            else:
                state = "disabled" if active else "normal"
            widget.configure(state=state)
        self.apply_button.configure(state="disabled" if active else "normal")
        if active:
            self.apply_button.configure(
                text="Cancel analysis", command=self._on_cancel, state="normal"
            )
            set_styled_button_palette(
                self.apply_button,
                background="#f7d9d9",
                hover_background="#efbcbc",
            )
        else:
            self.apply_button.configure(
                text="Start analysis", command=self._on_apply, state="normal"
            )
            set_styled_button_palette(
                self.apply_button,
                background="#ffffff",
                hover_background="#b8d8e8",
            )

    def values(self):
        """Read a snapshot of the form on the Tk thread."""
        return {
            group: {key: variable.get() for key, variable in fields.items()}
            for group, fields in self.variables.items()
        }


def create_variables(parent, fields):
    """Create independent Tk variables, also used for uncommitted dialog drafts."""
    return {
        key: (
            tk.BooleanVar(parent, value=value)
            if isinstance(value, bool)
            else tk.StringVar(parent, value=value)
        )
        for key, value in fields.items()
    }


def populate_fields(
    parent, variables, *, choose_file=None, stacked=False, editable_widgets=None
):
    """Render common sidebar or dialog fields with their existing units and types."""
    for index, (key, variable) in enumerate(variables.items()):
        row = index * 2 if stacked else index
        column = 0 if stacked else 1
        field_row = row + 1 if stacked else row
        label_pady = (6, 2) if stacked else (6, 0)
        field_pady = (0, 6) if stacked else (6, 0)
        label = LABELS.get(key, key)
        if isinstance(variable, tk.BooleanVar):
            widget = tk.Checkbutton(
                parent,
                text=label,
                variable=variable,
                bg=CONTROL_BACKGROUND,
                activebackground=CONTROL_BACKGROUND,
                font=FONT,
                fg=TEXT_COLOR,
            )
            widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=6)
            if editable_widgets is not None:
                editable_widgets.append(widget)
            continue
        tk.Label(
            parent,
            text=label,
            wraplength=280,
            justify="left",
            font=FONT,
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
        ).grid(row=row, column=0, sticky="w", pady=label_pady, padx=(0, 8))
        if key == "data_file":
            file_frame = tk.Frame(parent, bg=CONTROL_BACKGROUND)
            file_frame.grid(row=field_row, column=column, sticky="ew", pady=field_pady)
            file_frame.columnconfigure(0, weight=1)
            entry = ttk.Entry(file_frame, textvariable=variable, width=16)
            entry.grid(row=0, column=0, sticky="ew")
            browse_button = create_styled_button(
                file_frame, text="…", width=2, command=choose_file
            )
            browse_button.grid(row=0, column=1, padx=(5, 0))
            if editable_widgets is not None:
                editable_widgets.extend((entry, browse_button))
            continue
        if key == "inference_method":
            display_value = METHOD_DISPLAY_LABELS.get(variable.get(), variable.get())
            variable.set(display_value)
        choices = {
            "inference_method": METHOD_DISPLAY_OPTIONS,
            "algorithm": ("meanfield", "fullrank"),
            "coordinate_display_mode": COORDINATE_DISPLAY_MODES,
        }.get(key)
        widget = (
            ttk.Combobox(
                parent,
                textvariable=variable,
                values=choices,
                state="readonly",
                width=14,
            )
            if choices
            else ttk.Entry(parent, textvariable=variable, width=14)
        )
        widget.grid(row=field_row, column=column, sticky="ew", pady=field_pady)
        if editable_widgets is not None:
            editable_widgets.append(widget)
