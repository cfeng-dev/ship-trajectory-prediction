"""Tk input controls; parsing and inference are kept outside the view."""

import tkinter as tk
from tkinter import filedialog, ttk

from .settings import (
    COORDINATE_DISPLAY_MODES,
    LABELS,
    MAIN_DATA_FIELDS,
    METHODS,
    default_form_values,
)
from .view import CONTROL_BACKGROUND, FONT, TEXT_COLOR, create_styled_button


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

    def __init__(self, parent, on_apply, on_reset=None):
        super().__init__(parent, width=350, bg=CONTROL_BACKGROUND)
        self.pack_propagate(False)
        on_reset = on_reset or (lambda: None)
        self.variables = {
            group: create_variables(self, fields)
            for group, fields in default_form_values().items()
        }
        self.settings_form = ScrollableForm(self)
        self.data_form = self.settings_form
        self.settings_form.pack(fill="both", expand=True)
        body = self.settings_form.body
        tk.Label(
            body,
            text="Posterior-Analyse",
            font=("Arial", 12, "bold"),
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
        ).pack(pady=(14, 10))
        self.analysis_section = tk.LabelFrame(
            body,
            text="Analyse",
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
            self.analysis_section, text="Analyse starten", command=on_apply
        )
        self.apply_button.pack(fill="x")
        self.reset_button = create_styled_button(
            self.analysis_section, text="Zurücksetzen", command=on_reset
        )
        self.reset_button.pack(fill="x", pady=(6, 0))
        self.analysis_fields.columnconfigure(1, weight=1)
        populate_fields(self.analysis_fields, analysis_values, stacked=False)
        self.analysis_fields.pack(fill="x", pady=(8, 0))

        self.data_section = tk.LabelFrame(
            body,
            text="Daten",
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
        )
        self.settings_form.bind_mouse_wheel()

    def choose_file(self):
        """Choose a CSV without starting or replacing an analysis."""
        selected = filedialog.askopenfilename(
            parent=self,
            title="Trajektorien-CSV auswählen",
            filetypes=(("CSV-Dateien", "*.csv"), ("Alle Dateien", "*.*")),
        )
        if selected:
            self.variables["data"]["data_file"].set(selected)

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


def populate_fields(parent, variables, *, choose_file=None, stacked=False):
    """Render common sidebar or dialog fields with their existing units and types."""
    for index, (key, variable) in enumerate(variables.items()):
        row = index * 2 if stacked else index
        column = 0 if stacked else 1
        field_row = row + 1 if stacked else row
        label_pady = (6, 2) if stacked else (6, 0)
        field_pady = (0, 6) if stacked else (6, 0)
        label = LABELS.get(key, key)
        if isinstance(variable, tk.BooleanVar):
            tk.Checkbutton(
                parent,
                text=label,
                variable=variable,
                bg=CONTROL_BACKGROUND,
                activebackground=CONTROL_BACKGROUND,
                font=FONT,
                fg=TEXT_COLOR,
            ).grid(row=row, column=0, columnspan=2, sticky="w", pady=6)
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
            ttk.Entry(file_frame, textvariable=variable, width=16).grid(
                row=0, column=0, sticky="ew"
            )
            create_styled_button(
                file_frame, text="…", width=2, command=choose_file
            ).grid(row=0, column=1, padx=(5, 0))
            continue
        choices = {
            "inference_method": METHODS,
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
