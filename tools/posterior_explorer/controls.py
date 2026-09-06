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


class SettingsPanel(tk.Frame):
    """Editable data, prior and inference settings with an explicit apply action."""

    def __init__(self, parent, on_apply):
        super().__init__(parent, width=350, bg=CONTROL_BACKGROUND)
        self.pack_propagate(False)
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
        actions = tk.LabelFrame(
            body,
            text="Analyse",
            font=FONT,
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            padx=10,
            pady=10,
        )
        actions.pack(fill="x", pady=(0, 8))
        self.apply_button = create_styled_button(
            actions, text="Neue Analyse", command=on_apply
        )
        self.apply_button.pack(fill="x")

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
            {key: self.variables["data"][key] for key in MAIN_DATA_FIELDS},
            choose_file=self.choose_file,
            stacked=True,
        )
        self.method_label = tk.Label(
            body,
            wraplength=285,
            justify="left",
            font=("Arial", 9),
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
        )
        self.method_label.pack(fill="x", pady=(12, 0))
        tk.Label(
            body,
            text="Priors, Inferenzparameter, Rauschen und Wiedergabeoptionen: Settings",
            wraplength=285,
            justify="left",
            font=("Arial", 9),
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
        ).pack(fill="x", pady=(12, 0))
        self.variables["data"]["inference_method"].trace_add("write", self._show_method)
        self._show_method()
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

    def _show_method(self, *_):
        method = self.variables["data"]["inference_method"].get()
        mode = (
            "Online: Beobachtungen werden nacheinander verarbeitet."
            if method in ("rbpf", "smc")
            else "Batch: neuer Fit je N, mit wachsendem Datenpräfix (ab N=3). "
            "Zunächst wenige Beobachtungen wählen."
        )
        self.method_label.configure(text=f"{method.upper()} — {mode}")

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
        ).grid(row=row, column=0, sticky="w", pady=(6, 2), padx=(0, 8))
        if key == "data_file":
            file_frame = tk.Frame(parent, bg=CONTROL_BACKGROUND)
            file_frame.grid(row=field_row, column=column, sticky="ew", pady=(0, 6))
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
        widget.grid(row=field_row, column=column, sticky="ew", pady=(0, 6))
