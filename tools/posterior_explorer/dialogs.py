"""Transactional editors for the explorer's advanced settings."""

import tkinter as tk
from tkinter import messagebox

from .controls import ScrollableForm, create_variables, populate_fields
from .settings import (
    DATA_OPTION_FIELDS,
    DISPLAY_OPTION_FIELDS,
    LABELS,
    validate_dialog_values,
)
from .view import CONTROL_BACKGROUND, FONT, TEXT_COLOR, create_styled_button

DIALOG_MINIMUM_HEIGHT = 320
DIALOG_SCREEN_MARGIN = 120
DIALOG_CONTENT_PADDING = 16
INFERENCE_DIALOG_WIDTH = 470
DATA_DIALOG_WIDTH = 560
WIDE_DIALOG_WIDTH = 650
PLOT_DISPLAY_WINDOW_WIDTH = 430
PLOT_DISPLAY_WINDOW_HEIGHT = 360


def dialog_height_for_content(content_height, screen_height):
    """Fit a short editor to its content within the available screen height."""
    available_height = max(DIALOG_MINIMUM_HEIGHT, screen_height - DIALOG_SCREEN_MARGIN)
    return min(max(DIALOG_MINIMUM_HEIGHT, content_height), available_height)


def dialog_width_for_group(group, screen_width):
    """Keep concise inference forms narrower than data and prior editors."""
    preferred_width = {
        "rbpf": INFERENCE_DIALOG_WIDTH,
        "smc": INFERENCE_DIALOG_WIDTH,
        "vi": INFERENCE_DIALOG_WIDTH,
        "mcmc": INFERENCE_DIALOG_WIDTH,
        "data": DATA_DIALOG_WIDTH,
        "priors": DATA_DIALOG_WIDTH,
    }.get(group, WIDE_DIALOG_WIDTH)
    return min(preferred_width, max(400, screen_width - 80))


class PlotDisplayWindow(tk.Toplevel):
    """Non-modal live controls for the frequently changed plot layers."""

    def __init__(self, parent, panel, *, on_close=None):
        super().__init__(parent)
        self.panel = panel
        self._on_close = on_close
        self._closed = False
        self.transient(parent)
        self.title("Plot-Anzeige")
        self.configure(bg=CONTROL_BACKGROUND)
        self.resizable(False, False)
        tk.Label(
            self,
            text="Änderungen werden sofort im Plot übernommen.",
            font=FONT,
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            anchor="w",
            padx=14,
            pady=12,
        ).pack(fill="x")
        body = tk.Frame(self, bg=CONTROL_BACKGROUND, padx=14, pady=4)
        body.pack(fill="both", expand=True)
        for key in DISPLAY_OPTION_FIELDS:
            tk.Checkbutton(
                body,
                text=LABELS[key],
                variable=panel.variables["data"][key],
                bg=CONTROL_BACKGROUND,
                activebackground=CONTROL_BACKGROUND,
                font=FONT,
                fg=TEXT_COLOR,
                anchor="w",
            ).pack(fill="x", pady=3)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.update_idletasks()
        left = max(
            0,
            parent.winfo_rootx()
            + (parent.winfo_width() - PLOT_DISPLAY_WINDOW_WIDTH) // 2,
        )
        top = max(
            0,
            parent.winfo_rooty()
            + (parent.winfo_height() - PLOT_DISPLAY_WINDOW_HEIGHT) // 2,
        )
        self.geometry(
            f"{PLOT_DISPLAY_WINDOW_WIDTH}x{PLOT_DISPLAY_WINDOW_HEIGHT}+{left}+{top}"
        )

    def close(self):
        """Close the live display panel without changing any selections."""
        if self._closed:
            return
        self._closed = True
        if self._on_close is not None:
            self._on_close()
        self.destroy()


class SettingsDialog(tk.Toplevel):
    """Edit a local draft; commit only validated values and never start inference."""

    def __init__(self, parent, panel, group):
        super().__init__(parent)
        self.panel = panel
        self.group = group
        self.transient(parent)
        self.configure(bg=CONTROL_BACKGROUND)
        title = {
            "priors": "Priors",
            "data": "Daten und Wiedergabe",
            "plot": "Plot-Anzeige",
        }.get(group, f"Inferenzparameter — {group.upper()}")
        self.title(title)
        width = dialog_width_for_group(group, parent.winfo_screenwidth())
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._header = tk.Label(
            self,
            text="Übernehmen aktualisiert die Einstellungen.\n"
            "Eine laufende Analyse bleibt unverändert bis „Analyse starten“.",
            font=FONT,
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            justify="left",
            wraplength=width - 50,
            padx=14,
            pady=12,
        )
        self._header.grid(row=0, column=0, sticky="ew")
        fields = panel.values()["data"] if group == "plot" else panel.values()[group]
        if group == "data":
            fields = {key: fields[key] for key in DATA_OPTION_FIELDS}
        elif group == "plot":
            fields = {key: fields[key] for key in DISPLAY_OPTION_FIELDS}
        self.variables = create_variables(self, fields)
        self._form = ScrollableForm(self, width=max(360, width - 70))
        self._form.grid(row=1, column=0, sticky="nsew")
        if group == "data":
            data_section = tk.LabelFrame(
                self._form.body,
                text="Daten und Wiedergabe",
                font=FONT,
                bg=CONTROL_BACKGROUND,
                fg=TEXT_COLOR,
                padx=10,
                pady=8,
            )
            data_section.pack(fill="x", pady=(0, 8))
            data_section.columnconfigure(1, weight=1)
            populate_fields(
                data_section,
                self.variables,
                choose_file=self.panel.choose_file,
            )
        else:
            populate_fields(self._form.body, self.variables)
        self._form.bind_mouse_wheel()
        self._actions = tk.Frame(self, bg=CONTROL_BACKGROUND, padx=14, pady=12)
        self._actions.grid(row=2, column=0, sticky="ew")
        create_styled_button(self._actions, text="Übernehmen", command=self.apply).pack(
            side="right", padx=(8, 0)
        )
        create_styled_button(self._actions, text="Abbrechen", command=self.cancel).pack(
            side="right"
        )
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.bind("<Escape>", lambda _: self.cancel())
        # Wait for mapping via an event, not a nested/blocking wait_visibility.
        self.bind("<Map>", self._on_map, add="+")
        self.update_idletasks()
        content_height = (
            self._header.winfo_reqheight()
            + self._form.body.winfo_reqheight()
            + self._actions.winfo_reqheight()
            + DIALOG_CONTENT_PADDING
        )
        height = dialog_height_for_content(content_height, self.winfo_screenheight())
        left = max(0, parent.winfo_rootx() + (parent.winfo_width() - width) // 2)
        top = max(0, parent.winfo_rooty() + (parent.winfo_height() - height) // 2)
        self.geometry(f"{width}x{height}+{left}+{top}")
        self.minsize(min(520, width), min(DIALOG_MINIMUM_HEIGHT, height))

    def _on_map(self, event):
        if event.widget is self:
            self.focus_set()

    def apply(self):
        """Validate this section before touching the shared form variables."""
        try:
            values = validate_dialog_values(
                self.group,
                {key: variable.get() for key, variable in self.variables.items()},
            )
        except (ValueError, TypeError) as error:
            messagebox.showerror("Einstellungen prüfen", str(error), parent=self)
            return
        target_group = "data" if self.group == "plot" else self.group
        for key, value in values.items():
            self.panel.variables[target_group][key].set(value)
        self.cancel()

    def cancel(self):
        """Discard the draft and close the non-modal editor."""
        if self.grab_current() is self:
            self.grab_release()
        self.destroy()
