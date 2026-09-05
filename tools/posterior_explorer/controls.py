"""Tk input controls; parsing and inference are kept outside the view."""

import tkinter as tk
from tkinter import filedialog, ttk

from .settings import LABELS, METHODS, default_form_values


class ScrollableForm(ttk.Frame):
    """A form that remains accessible when the settings pane is small."""

    def __init__(self, parent):
        super().__init__(parent)
        canvas = tk.Canvas(self, highlightthickness=0, width=360)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self.body = ttk.Frame(canvas, padding=10)
        window = canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.columnconfigure(0, weight=1)
        self.body.bind(
            "<Configure>",
            lambda _: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>", lambda event: canvas.itemconfigure(window, width=event.width)
        )


class SettingsPanel(ttk.Frame):
    """Editable data, prior and inference settings with an explicit apply action."""

    def __init__(self, parent, on_apply):
        super().__init__(parent, padding=(8, 0, 8, 8))
        self.variables = {}
        self.method_forms = {}
        ttk.Label(
            self,
            text="Änderungen gelten erst nach „Neue Analyse“.",
            wraplength=355,
        ).pack(anchor="w", pady=(0, 8))
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        defaults = default_form_values()
        for group, title in (("data", "Daten"), ("priors", "Priors")):
            form = ScrollableForm(notebook)
            notebook.add(form, text=title)
            self._populate(form.body, group, defaults[group])
        method_tab = ttk.Frame(notebook)
        notebook.add(method_tab, text="Inferenz")
        self.method_label = ttk.Label(method_tab, padding=8, wraplength=355)
        self.method_label.pack(fill="x")
        for method in METHODS:
            form = ScrollableForm(method_tab)
            self._populate(form.body, method, defaults[method])
            self.method_forms[method] = form
        self.variables["data"]["inference_method"].trace_add("write", self._show_method)
        self._show_method()
        self.apply_button = ttk.Button(self, text="Neue Analyse", command=on_apply)
        self.apply_button.pack(fill="x", pady=(10, 0))

    def _populate(self, parent, group, fields):
        variables = self.variables[group] = {}
        for row, (key, value) in enumerate(fields.items()):
            variable = (
                tk.BooleanVar(parent, value=value)
                if isinstance(value, bool)
                else tk.StringVar(parent, value=value)
            )
            variables[key] = variable
            label = LABELS.get(key, key)
            if isinstance(value, bool):
                ttk.Checkbutton(parent, text=label, variable=variable).grid(
                    row=row, column=0, columnspan=2, sticky="w", pady=6
                )
                continue
            if key == "data_file":
                file_frame = ttk.Frame(parent)
                file_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
                file_frame.columnconfigure(0, weight=1)
                ttk.Label(file_frame, text=label).grid(row=0, column=0, sticky="w")
                ttk.Entry(file_frame, textvariable=variable).grid(
                    row=1, column=0, sticky="ew"
                )
                ttk.Button(
                    file_frame, text="…", width=3, command=self._choose_file
                ).grid(row=1, column=1, padx=(5, 0))
                continue
            ttk.Label(parent, text=label, wraplength=220).grid(
                row=row, column=0, sticky="w", pady=6, padx=(0, 8)
            )
            choices = {
                "inference_method": METHODS,
                "algorithm": ("meanfield", "fullrank"),
            }.get(key)
            if choices:
                widget = ttk.Combobox(
                    parent,
                    textvariable=variable,
                    values=choices,
                    state="readonly",
                    width=12,
                )
            else:
                widget = ttk.Entry(parent, textvariable=variable, width=14)
            widget.grid(row=row, column=1, sticky="ew", pady=6)

    def _choose_file(self):
        selected = filedialog.askopenfilename(
            parent=self,
            title="Trajektorien-CSV auswählen",
            filetypes=(("CSV-Dateien", "*.csv"), ("Alle Dateien", "*.*")),
        )
        if selected:
            self.variables["data"]["data_file"].set(selected)

    def _show_method(self, *_):
        method = self.variables["data"]["inference_method"].get()
        for form in self.method_forms.values():
            form.pack_forget()
        self.method_forms[method].pack(fill="both", expand=True)
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
