"""Modal, transactional editors for the explorer's advanced settings."""

import tkinter as tk
from tkinter import messagebox

from .controls import ScrollableForm, create_variables, populate_fields
from .settings import DATA_OPTION_FIELDS, validate_dialog_values
from .view import CONTROL_BACKGROUND, FONT, TEXT_COLOR, create_styled_button

DIALOG_MINIMUM_HEIGHT = 320
DIALOG_SCREEN_MARGIN = 120
DIALOG_CONTENT_PADDING = 16


def dialog_height_for_content(content_height, screen_height):
    """Fit a short editor to its content within the available screen height."""
    available_height = max(DIALOG_MINIMUM_HEIGHT, screen_height - DIALOG_SCREEN_MARGIN)
    return min(max(DIALOG_MINIMUM_HEIGHT, content_height), available_height)


class SettingsDialog(tk.Toplevel):
    """Edit a local draft; commit only validated values and never start inference."""

    def __init__(self, parent, panel, group):
        super().__init__(parent)
        self.panel = panel
        self.group = group
        self.transient(parent)
        self.configure(bg=CONTROL_BACKGROUND)
        title = {"priors": "Priors", "data": "Daten und Darstellung"}.get(
            group, f"Inferenzparameter — {group.upper()}"
        )
        self.title(title)
        width = min(650, max(400, parent.winfo_screenwidth() - 80))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._header = tk.Label(
            self,
            text="Übernehmen aktualisiert die Einstellungen.\n"
            "Eine laufende Analyse bleibt unverändert bis „Neue Analyse“.",
            font=FONT,
            bg=CONTROL_BACKGROUND,
            fg=TEXT_COLOR,
            justify="left",
            wraplength=560,
            padx=14,
            pady=12,
        )
        self._header.grid(row=0, column=0, sticky="ew")
        fields = panel.values()[group]
        if group == "data":
            fields = {key: fields[key] for key in DATA_OPTION_FIELDS}
        self.variables = create_variables(self, fields)
        self._form = ScrollableForm(self, width=580)
        self._form.grid(row=1, column=0, sticky="nsew")
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
            self.grab_set()
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
        for key, value in values.items():
            self.panel.variables[self.group][key].set(value)
        self.cancel()

    def cancel(self):
        """Discard the draft and release the modal grab."""
        if self.grab_current() is self:
            self.grab_release()
        self.destroy()
