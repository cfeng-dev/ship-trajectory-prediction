"""CSV explorer using the Bayesian Predictor's native desktop presentation."""

import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from bayesian_predictor import view
from bayesian_predictor.controls import ScrollableForm
from bayesian_predictor.help import (
    DIALOG_CONTENT_PADDING,
    HELP_WINDOW_WIDTH,
    add_help_description_rows,
    help_content_height_for_screen,
    help_window_height,
    help_window_position,
)
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from .localization import help_sections, text
from .session import EvaluationLoader, Playback
from .settings import METRICS, ViewerOptions


class EvaluationViewer:
    def __init__(self, root, *, input_dir, observations_csv=None):
        self.root = root
        self.options = ViewerOptions()
        self.data = None
        self.playback = None
        self.plot = None
        self.canvas = None
        self.navigation = None
        self.timeline_canvas = None
        self.toolbar = None
        self._plot_settings_dialog = None
        self._playback_settings_dialog = None
        self._help_window = None
        self.selected_models = ()
        self._closed = False
        self._loading = False
        self._analysis_active = False
        self._load_worker = None
        self._load_poll_id = None
        self._source_inputs = []
        self._observation_inputs = []
        self.directory = tk.StringVar(root, value=str(input_dir))
        self.observations_csv = tk.StringVar(
            root, value=str(observations_csv) if observations_csv else ""
        )
        self.use_observations = tk.BooleanVar(
            root, value=bool(self.observations_csv.get().strip())
        )
        self.settings_visible = tk.BooleanVar(root, value=True)
        self.status = tk.StringVar(root)
        view.configure_window(root)
        root.title("Evaluation Viewer")
        self.sidebar = tk.Frame(root, bg=view.CONTROL_BACKGROUND)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        self.plot_host = tk.Frame(root, bg=view.PLOT_BACKGROUND)
        self.plot_host.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.plot_host.columnconfigure(0, weight=1)
        self.plot_host.rowconfigure(0, weight=1)
        self.placeholder = tk.Label(
            self.plot_host,
            text=self.tr("Choose the input paths, then click Start analysis."),
            bg=view.PLOT_BACKGROUND,
            fg=view.TEXT_COLOR,
            font=view.FONT,
        )
        self.placeholder.grid(row=0, column=0)
        self._build_sidebar()
        self._build_menu()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        for key in ("<Left>", "<Right>", "<space>"):
            root.bind(key, self._keyboard, add="+")

    def _create_plot_ui(self):
        from .plot_view import EvaluationPlot

        self.plot = EvaluationPlot(background=view.PLOT_BACKGROUND)
        self.canvas = view.PredictorFigureCanvas(
            self.plot.figure, master=self.plot_host
        )
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._build_timeline()
        self.toolbar = NavigationToolbar2Tk(
            self.canvas, self.plot_host, pack_toolbar=False
        )
        self.toolbar.grid(row=2, column=0, sticky="ew")
        view.configure_plot_toolbar(self.toolbar)
        self.canvas.mpl_connect("button_press_event", self._plot_click)
        self.canvas.mpl_connect("resize_event", lambda _: self.pause())
        self.navigation.set_language(self.options.language)
        self.navigation.set_follow_ship(self.options.follow_ship)
        self.navigation.set_settings_visible(self.settings_visible.get())

    def tr(self, source):
        return text(source, self.options.language)

    def _section(self, parent, source):
        section = tk.LabelFrame(
            parent,
            text=self.tr(source),
            bg=view.CONTROL_BACKGROUND,
            fg=view.TEXT_COLOR,
            font=view.FONT,
            padx=10,
            pady=8,
        )
        section.pack(fill="x", pady=(0, 10))
        return section

    def _label(self, parent, source):
        label = tk.Label(
            parent,
            text=self.tr(source),
            bg=view.CONTROL_BACKGROUND,
            fg=view.TEXT_COLOR,
            font=view.FONT,
            anchor="w",
            justify="left",
            wraplength=265,
        )
        label.pack(fill="x", pady=(2, 4))
        return label

    def _build_sidebar(self):
        for child in self.sidebar.winfo_children():
            child.destroy()
        form = ScrollableForm(self.sidebar, width=310)
        form.pack(fill="both", expand=True)
        self._form = form
        self._load_inputs = []
        form.on_scroll = self.pause
        body = form.body
        title = self._label(body, "Model evaluation")
        title.configure(font=("Arial", 12, "bold"), anchor="center", pady=10)
        group = self._section(body, "Analysis")
        self.load_button = view.create_styled_button(
            group, text=self.tr("Start analysis"), command=self.load_directory
        )
        self.load_button.pack(fill="x")
        group = self._section(body, "Data")
        self._label(group, "Evaluation directory")
        row = tk.Frame(group, bg=view.CONTROL_BACKGROUND)
        row.pack(fill="x")
        ttk.Entry(
            row, textvariable=self.directory, style="Predictor.TEntry", width=18
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            row,
            text="…",
            command=self.open_directory,
            style="Predictor.TButton",
            width=3,
        ).pack(side="right", padx=(5, 0))
        observation_toggle = tk.Checkbutton(
            group,
            text=self.tr("Use observation CSV"),
            variable=self.use_observations,
            command=self._sync_load_controls,
            bg=view.CONTROL_BACKGROUND,
            activebackground=view.CONTROL_BACKGROUND,
            fg=view.TEXT_COLOR,
            font=view.FONT,
            anchor="w",
        )
        observation_toggle.pack(fill="x", pady=(2, 4))
        self._load_inputs.append(observation_toggle)
        observation_row = tk.Frame(group, bg=view.CONTROL_BACKGROUND)
        observation_row.pack(fill="x")
        ttk.Entry(
            observation_row,
            textvariable=self.observations_csv,
            style="Predictor.TEntry",
            width=18,
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            observation_row,
            text="…",
            command=self.open_observations,
            style="Predictor.TButton",
            width=3,
        ).pack(side="right", padx=(5, 0))
        self._load_inputs.extend(row.winfo_children())
        self._observation_inputs = observation_row.winfo_children()
        self._load_inputs.extend(self._observation_inputs)
        self._source_inputs = self._load_inputs.copy()
        group = self._section(body, "Models")
        self.model_variables = {}
        if self.data is None:
            self._label(group, "No data loaded")
        else:
            for label in self.data.models:
                variable = tk.BooleanVar(self.root, value=label in self.selected_models)
                self.model_variables[label] = variable
                self._check(
                    group,
                    self.plot._model_label(label, self.options.language),
                    variable,
                    translated=True,
                )
        group = self._section(body, "Metric diagrams")
        available = self.data.available_metrics if self.data else METRICS
        self.metric_choices = {
            self.tr(value): key for key, value in METRICS.items() if key in available
        }
        self.metric_variables = {}
        for key, label in (
            ("top_metric", "Upper diagram"),
            ("bottom_metric", "Lower diagram"),
        ):
            variable = tk.StringVar(
                self.root, value=self.tr(METRICS[getattr(self.options, key)])
            )
            self.metric_variables[key] = variable
            if self.options.single_metric_layout and key == "bottom_metric":
                continue
            self._label(group, "Metric" if self.options.single_metric_layout else label)
            choice = ttk.Combobox(
                group,
                textvariable=variable,
                values=tuple(self.metric_choices),
                state="readonly",
                style="Predictor.TCombobox",
                width=26,
            )
            choice.bind(
                "<<ComboboxSelected>>",
                lambda _event, key=key: self._metric_selected(key),
            )
            choice.pack(fill="x", pady=(0, 7))
            self._load_inputs.append(choice)
        self._set_status()
        self._sync_load_controls()
        form.bind_mouse_wheel()

    def _check(self, group, source, variable, *, translated=False):
        checkbox = tk.Checkbutton(
            group,
            text=source if translated else self.tr(source),
            variable=variable,
            command=self._models_selected,
            bg=view.CONTROL_BACKGROUND,
            activebackground=view.CONTROL_BACKGROUND,
            fg=view.TEXT_COLOR,
            font=view.FONT,
            anchor="w",
            justify="left",
            wraplength=255,
        )
        checkbox.pack(fill="x", pady=2)
        self._load_inputs.append(checkbox)

    def _build_timeline(self):
        from .plot_view import PlaybackBar

        self.navigation = PlaybackBar(
            on_seek=self._slider,
            on_play=self.toggle_playback,
            on_follow=self.set_follow_ship,
            on_settings=self.set_settings_visible,
            background=view.PLOT_BACKGROUND,
        )
        self.timeline_canvas = view.PredictorFigureCanvas(
            self.navigation.figure, master=self.plot_host
        )
        self.timeline_canvas.get_tk_widget().configure(
            height=round(self.navigation.figure.bbox.height)
        )
        self.timeline_canvas.get_tk_widget().grid(row=1, column=0, sticky="ew")

    def _build_menu(self):
        if getattr(self, "_menu", None) is not None:
            # Keep the native menu alive while its language callback is running.
            for menu, index, source in self._menu_label_entries:
                menu.entryconfigure(index, label=self.tr(source))
            self.language_variable.set(self.options.language)
            return

        labels = []

        def add(parent, kind, source, **options):
            getattr(parent, f"add_{kind}")(label=self.tr(source), **options)
            labels.append((parent, parent.index("end"), source))

        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=0)
        add(file_menu, "command", "Open directory…", command=self.open_directory)
        add(file_menu, "command", "Reload", command=self.load_directory)
        file_menu.add_separator()
        add(file_menu, "command", "Exit", command=self.close)
        add(menu, "cascade", "File", menu=file_menu)
        view_menu = tk.Menu(menu, tearoff=0)
        add(
            view_menu,
            "checkbutton",
            "Show settings",
            variable=self.settings_visible,
            command=self.toggle_settings,
        )
        add(view_menu, "command", "Plot display…", command=self.open_plot_settings)
        add(menu, "cascade", "View", menu=view_menu)
        settings_menu = tk.Menu(menu, tearoff=0)
        add(settings_menu, "command", "Playback…", command=self.open_playback_settings)
        add(menu, "cascade", "Settings", menu=settings_menu)
        language = tk.Menu(menu, tearoff=0)
        self.language_variable = tk.StringVar(self.root, value=self.options.language)
        for label, value in (("English", "en"), ("Deutsch", "de")):
            add(
                language,
                "radiobutton",
                label,
                value=value,
                variable=self.language_variable,
                command=lambda selected=value: self.set_language(selected),
            )
        add(menu, "cascade", "Language", menu=language)
        help_menu = tk.Menu(menu, tearoff=0)
        add(
            help_menu,
            "command",
            "Show Help",
            command=self.show_help,
        )
        add(menu, "cascade", "Help", menu=help_menu)
        self._menu_label_entries = tuple(labels)
        self._menu = menu
        self.root.configure(menu=menu)

    def show_help(self):
        if self._closed:
            return
        if self._help_window is not None and self._help_window.winfo_exists():
            self._help_window.lift()
            self._help_window.focus_set()
            return self._help_window
        self._help_window = EvaluationHelpWindow(
            self.root, language=self.options.language, on_close=self._help_closed
        )
        return self._help_window

    def _help_closed(self):
        self._help_window = None

    def open_playback_settings(self):
        if self._closed or self._loading:
            return
        dialog = self._playback_settings_dialog
        if dialog is not None and dialog.winfo_exists():
            dialog.lift()
            dialog.focus_set()
            return dialog
        dialog = tk.Toplevel(self.root)
        self._playback_settings_dialog = dialog
        dialog.withdraw()
        dialog.transient(self.root)
        dialog.title(self.tr("Playback"))
        dialog.configure(bg=view.CONTROL_BACKGROUND)
        dialog.resizable(False, False)
        body = tk.Frame(dialog, bg=view.CONTROL_BACKGROUND, padx=14, pady=12)
        body.pack(fill="both", expand=True)
        self._label(body, "Changes are applied only after selecting Apply.")
        group = self._section(body, "Playback")
        self._label(group, "Playback interval [s]")
        variable = tk.StringVar(
            dialog, value=f"{self.options.playback_interval_seconds:g}"
        )
        ttk.Entry(
            group, textvariable=variable, style="Predictor.TEntry", width=12
        ).pack(anchor="w", pady=(0, 6))
        self._label(
            group, "Delay between saved forecast origins. Smaller values play faster."
        )

        def apply():
            if self._apply_playback_options(
                {"playback_interval_seconds": variable.get()}, parent=dialog
            ):
                self._close_playback_settings()

        actions = tk.Frame(body, bg=view.CONTROL_BACKGROUND)
        actions.pack(fill="x")
        view.create_styled_button(
            actions, text=self.tr("Apply"), width=12, command=apply
        ).pack(side="right", padx=(8, 0))
        view.create_styled_button(
            actions,
            text=self.tr("Cancel"),
            width=12,
            command=self._close_playback_settings,
        ).pack(side="right")
        dialog.protocol("WM_DELETE_WINDOW", self._close_playback_settings)
        self._show_dialog(dialog)
        return dialog

    def _close_playback_settings(self):
        dialog, self._playback_settings_dialog = self._playback_settings_dialog, None
        if dialog is not None and dialog.winfo_exists():
            dialog.destroy()

    def _apply_playback_options(self, values, *, parent=None):
        if self._closed or self._loading:
            return False
        try:
            options = replace(self.options, **values)
        except (ValueError, TypeError, tk.TclError) as error:
            messagebox.showerror(
                self.tr("Invalid settings"),
                self.tr(str(error)),
                parent=parent or self.root,
            )
            return False
        self.options = options
        if self.playback is not None:
            self.playback.set_interval(options.playback_interval_ms)
        return True

    def open_plot_settings(self):
        if self._closed or self._loading:
            return
        if (
            self._plot_settings_dialog is not None
            and self._plot_settings_dialog.winfo_exists()
        ):
            self._plot_settings_dialog.lift()
            self._plot_settings_dialog.focus_set()
            return self._plot_settings_dialog
        dialog = tk.Toplevel(self.root)
        self._plot_settings_dialog = dialog
        dialog.withdraw()
        dialog.title(self.tr("Plot display"))
        dialog.transient(self.root)
        dialog.configure(bg=view.CONTROL_BACKGROUND)
        dialog.resizable(False, False)
        body = tk.Frame(dialog, bg=view.CONTROL_BACKGROUND, padx=14, pady=12)
        body.pack(fill="both", expand=True)
        hint = self._label(body, "Changes are applied only after selecting Apply.")
        hint.configure(wraplength=0)
        columns = tk.Frame(body, bg=view.CONTROL_BACKGROUND)
        columns.pack(fill="both", expand=True)
        layers = tk.Frame(columns, bg=view.CONTROL_BACKGROUND)
        layers.grid(row=0, column=0, sticky="nw")
        controls = tk.Frame(columns, bg=view.CONTROL_BACKGROUND)
        controls.grid(row=0, column=1, sticky="nw", padx=(16, 0))
        variables = {}
        groups = (
            (
                layers,
                "General",
                (
                    ("show_trajectory_legend", "Show trajectory legend"),
                    ("show_metric_legend", "Show metric legend"),
                ),
            ),
            (
                layers,
                "Trajectory",
                (
                    ("show_reference_trajectory", "Show reference trajectory"),
                    ("show_reference_points", "Show reference points"),
                    (
                        "show_reference_forecast_segment",
                        "Show reference forecast segment",
                    ),
                    ("show_observed_trajectory", "Show observations through N"),
                    ("show_current_position", "Show current position"),
                    ("ship_view_span", "Follow-ship view span [m]"),
                ),
            ),
            (
                controls,
                "Plot layout",
                (("single_metric_layout", "One metric above trajectory"),),
            ),
            (
                controls,
                "Metric time axis",
                (
                    ("follow_metric_time", "Follow metric time"),
                    ("metric_window_width", "Window width [s]"),
                ),
            ),
        )
        for parent, title, fields in groups:
            group = self._section(parent, title)
            for key, label in fields:
                value = getattr(self.options, key)
                if isinstance(value, bool):
                    variable = tk.BooleanVar(dialog, value=value)
                    tk.Checkbutton(
                        group,
                        text=self.tr(label),
                        variable=variable,
                        bg=view.CONTROL_BACKGROUND,
                        activebackground=view.CONTROL_BACKGROUND,
                        fg=view.TEXT_COLOR,
                        font=view.FONT,
                        anchor="w",
                    ).pack(anchor="w", pady=2)
                else:
                    variable = tk.StringVar(dialog, value=str(value))
                    row = tk.Frame(group, bg=view.CONTROL_BACKGROUND)
                    row.pack(fill="x", pady=(8, 3))
                    tk.Label(
                        row,
                        text=self.tr(label),
                        bg=view.CONTROL_BACKGROUND,
                        fg=view.TEXT_COLOR,
                        font=view.FONT,
                    ).pack(side="left")
                    ttk.Entry(
                        row, textvariable=variable, style="Predictor.TEntry", width=12
                    ).pack(side="right", padx=(12, 0))
                variables[key] = variable

        def apply():
            values = {key: variable.get() for key, variable in variables.items()}
            if self._apply_plot_options(values, parent=dialog):
                self._close_plot_settings()

        actions = tk.Frame(body, bg=view.CONTROL_BACKGROUND)
        actions.pack(fill="x")
        view.create_styled_button(
            actions, text=self.tr("Apply"), width=12, command=apply
        ).pack(side="right", padx=(8, 0))
        view.create_styled_button(
            actions, text=self.tr("Cancel"), width=12, command=self._close_plot_settings
        ).pack(side="right")
        dialog.protocol("WM_DELETE_WINDOW", self._close_plot_settings)
        self._show_dialog(dialog)
        return dialog

    def _show_dialog(self, dialog):
        dialog.update_idletasks()
        left = max(
            0,
            self.root.winfo_rootx()
            + (self.root.winfo_width() - dialog.winfo_reqwidth()) // 2,
        )
        top = max(
            0,
            self.root.winfo_rooty()
            + (self.root.winfo_height() - dialog.winfo_reqheight()) // 2,
        )
        dialog.geometry(f"+{left}+{top}")
        dialog.deiconify()
        dialog.focus_set()

    def _close_plot_settings(self):
        dialog, self._plot_settings_dialog = self._plot_settings_dialog, None
        if dialog is not None and dialog.winfo_exists():
            dialog.destroy()

    def _apply_plot_options(self, values, *, parent=None):
        if self._loading:
            return False
        try:
            options = replace(self.options, **values)
        except (ValueError, TypeError, tk.TclError) as error:
            messagebox.showerror(
                self.tr("Invalid settings"), str(error), parent=parent or self.root
            )
            return False
        layout_changed = (
            options.single_metric_layout != self.options.single_metric_layout
        )
        self.options = options
        if layout_changed:
            self._build_sidebar()
        self._refresh_plots()
        return True

    def _metric_selected(self, key):
        if self._loading:
            return
        metric = self.metric_choices.get(self.metric_variables[key].get())
        if metric is not None:
            self.options = replace(self.options, **{key: metric})
            self._refresh_plots()

    def open_directory(self):
        if self._loading or self._analysis_active:
            return
        self.pause()
        selected = filedialog.askdirectory(
            parent=self.root,
            initialdir=self.directory.get(),
            title=self.tr("Open directory…"),
        )
        if selected:
            self.directory.set(selected)

    def open_observations(self):
        if self._loading or self._analysis_active:
            return
        self.pause()
        current_path = self.observations_csv.get().strip()
        selected = filedialog.askopenfilename(
            parent=self.root,
            initialdir=str(Path(current_path).expanduser().parent)
            if current_path
            else self.directory.get(),
            title=self.tr("Open observation CSV…"),
            filetypes=(("CSV", "*.csv"), ("All files", "*.*")),
        )
        if selected:
            self.observations_csv.set(selected)

    def load_directory(self, *, show_error=True):
        if self._closed or self._loading:
            return
        self.pause()
        observation_path = (
            self.observations_csv.get().strip() if self.use_observations.get() else ""
        )
        self._show_load_error = show_error
        self._load_message = "Preparing analysis …"
        self._analysis_active = True
        self._set_loading_ui(True)
        self._load_worker = EvaluationLoader(
            Path(self.directory.get()).expanduser(),
            observations_csv=observation_path or None,
        )
        self._load_poll_id = self.root.after(75, self._poll_load)

    def _sync_load_controls(self):
        active = self._loading or self._analysis_active
        self.load_button.configure(
            text=self.tr("Cancel analysis" if active else "Start analysis"),
            command=self.cancel_analysis if active else self.load_directory,
        )
        view.set_styled_button_palette(
            self.load_button,
            background="#f7d9d9" if active else "#ffffff",
            hover_background="#efbcbc" if active else "#b8d8e8",
        )
        for widget in self._load_inputs:
            disabled = (
                self._loading
                or (active and widget in self._source_inputs)
                or (
                    not self.use_observations.get()
                    and widget in self._observation_inputs
                )
            )
            widget.configure(
                state="disabled"
                if disabled
                else "readonly"
                if isinstance(widget, ttk.Combobox)
                else "normal"
            )
        for menu, index, source in getattr(self, "_menu_label_entries", ()):
            if source in {"Open directory…", "Reload"}:
                menu.entryconfigure(index, state="disabled" if active else "normal")

    def _set_loading_ui(self, active):
        self._loading = active
        self._sync_load_controls()
        if active:
            self.placeholder.configure(text=self.tr(self._load_message))
            self.placeholder.grid(row=0, column=0, rowspan=3, sticky="nsew")
            self.placeholder.tkraise()
        else:
            self.placeholder.grid_remove()
            if self.data is None:
                self.placeholder.configure(
                    text=self.tr("Choose the input paths, then click Start analysis.")
                )
                self.placeholder.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self._set_status()

    def _poll_load(self):
        self._load_poll_id = None
        if self._closed or self._load_worker is None:
            return
        for kind, value in self._load_worker.take_messages():
            if kind == "progress":
                self._load_message = value
                self.placeholder.configure(text=self.tr(value))
                self._set_status()
            elif kind == "ready":
                self._load_worker = None
                try:
                    self.root.update_idletasks()
                    self._apply_loaded_data(value)
                finally:
                    self._set_loading_ui(False)
                return
            elif kind == "error":
                self._load_worker = None
                self._analysis_active = False
                self._set_loading_ui(False)
                self.status.set(str(value))
                if self._show_load_error:
                    messagebox.showerror(
                        self.tr("CSV error"), str(value), parent=self.root
                    )
                return
        self._load_poll_id = self.root.after(75, self._poll_load)

    def cancel_analysis(self):
        if not (self._loading or self._analysis_active):
            return
        self._analysis_active = False
        self.pause()
        if self._load_worker is not None:
            self._load_worker.cancel()
            self._load_worker = None
        if self._load_poll_id is not None:
            self.root.after_cancel(self._load_poll_id)
            self._load_poll_id = None
        self._set_loading_ui(False)
        self.status.set(self.tr("Analysis cancelled. Settings unlocked."))

    def _apply_loaded_data(self, data):
        if self.plot is None:
            self._create_plot_ui()
        if self.playback is not None:
            self.playback.close()
        self.data = data
        self.selected_models = tuple(data.models)
        self.options = replace(
            self.options,
            top_metric=self.options.top_metric
            if self.options.top_metric in data.available_metrics
            else "ade_m",
            bottom_metric=self.options.bottom_metric
            if self.options.bottom_metric in data.available_metrics
            else "mean_negative_log_predictive_density",
        )
        self.playback = Playback(
            self.root,
            len(data.origins),
            self._frame,
            interval_ms=self.options.playback_interval_ms,
        )
        self._build_sidebar()
        self.navigation.set_count(len(data.origins))
        self.toolbar.update()
        self._frame(0)

    def _models_selected(self):
        if self._loading or self.data is None:
            return
        selected = tuple(
            label for label, variable in self.model_variables.items() if variable.get()
        )
        if not selected:
            for label, variable in self.model_variables.items():
                variable.set(label in self.selected_models)
            messagebox.showerror(
                self.tr("Invalid settings"),
                self.tr("Select at least one model."),
                parent=self.root,
            )
            return
        self.selected_models = selected
        self._refresh_plots()

    def _refresh_plots(self):
        if self.navigation is not None:
            self.navigation.set_follow_ship(self.options.follow_ship)
        if self.playback is not None:
            self._frame(self.playback.index)

    def _frame(self, index):
        self.plot.draw(self.data, index, self.options, self.selected_models)
        observation_count, maximum_observation_count = (
            [
                int((self.data.observations.time_seconds <= origin + 1e-6).sum())
                for origin in (self.data.origins[index], self.data.origins[-1])
            ]
            if self.data.observations is not None
            else (None, None)
        )
        self.navigation.show_index(
            index,
            observation_count=observation_count,
            maximum_observation_count=maximum_observation_count,
        )
        self._set_status()
        self._sync_play_button()
        self.canvas.draw_idle()

    def _set_status(self):
        if self._loading:
            self.status.set(self.tr(self._load_message))
        elif self.data is None:
            self.status.set(self.tr("Select a directory with saved evaluation CSVs."))
        else:
            index = self.playback.index if self.playback else 0
            self.status.set(
                f"{self.tr('Matched origins')}: {len(self.data.origins)}\n{self.tr('Forecast origin')}: {self.data.origins[index]:.3f} s\nH = {self.data.prediction_counts[index]} · +{self.data.horizon_seconds[index][-1]:.1f} s"
            )
            if self.data.observations is not None:
                count = int(
                    (
                        self.data.observations.time_seconds
                        <= self.data.origins[index] + 1e-6
                    ).sum()
                )
                self.status.set(self.status.get() + f"\nN = {count}")

    def _sync_play_button(self):
        if self.navigation is not None:
            playing = self.playback is not None and self.playback.playing
            self.navigation.set_playing(playing)

    def pause(self):
        if self.playback is not None:
            self.playback.pause()
        self._sync_play_button()

    def toggle_playback(self):
        if self._loading or self.playback is None:
            return
        if self.playback.playing:
            self.pause()
        else:
            self._analysis_active = True
            self._sync_load_controls()
            self.playback.start()

    def step(self, delta):
        if not self._loading and self.playback is not None:
            self.playback.seek(self.playback.index + delta)

    def _slider(self, value):
        if self._loading:
            return
        index = int(float(value))
        if self.playback is not None and index != self.playback.index:
            self.playback.seek(index)

    def _plot_click(self, event):
        if (
            not self._loading
            and self.data is not None
            and event.inaxes in self.plot.metric_axes
            and event.xdata is not None
            and event.button == 1
            and not self.toolbar.mode
        ):
            self.playback.seek(self.data.nearest_index(event.xdata))

    def _keyboard(self, event):
        if event.widget.winfo_class() in {
            "Entry",
            "TEntry",
            "TCombobox",
            "Scale",
            "TScale",
            "Button",
            "TButton",
            "Checkbutton",
            "TCheckbutton",
            "Radiobutton",
            "TRadiobutton",
        }:
            return None
        if event.keysym == "space":
            self.toggle_playback()
        else:
            self.step(-1 if event.keysym == "Left" else 1)
        return "break"

    def toggle_settings(self):
        self.pause()
        if self.navigation is not None:
            self.navigation.set_settings_visible(self.settings_visible.get())
        if self.settings_visible.get():
            self.sidebar.grid()
        else:
            self.sidebar.grid_remove()

    def set_settings_visible(self, visible):
        self.settings_visible.set(visible)
        self.toggle_settings()

    def set_follow_ship(self, follow):
        if self._loading:
            return
        self.options = replace(self.options, follow_ship=bool(follow))
        self._refresh_plots()

    def set_language(self, language):
        if self._loading:
            self.language_variable.set(self.options.language)
            return
        self.pause()
        model_draft = {
            label: variable.get() for label, variable in self.model_variables.items()
        }
        metric_draft = {
            key: self.metric_choices.get(variable.get())
            for key, variable in self.metric_variables.items()
        }
        self.options = replace(self.options, language=language)
        self._build_sidebar()
        for label, value in model_draft.items():
            self.model_variables[label].set(value)
        for key, value in metric_draft.items():
            if value is not None:
                self.metric_variables[key].set(self.tr(METRICS[value]))
        self._build_menu()
        if self._help_window is not None and self._help_window.winfo_exists():
            self._help_window.set_language(language)
        if self.navigation is not None:
            self.navigation.set_language(language)
        if self.playback is not None:
            self._frame(self.playback.index)
        elif self.plot is not None:
            self.plot.clear(language)
            self.canvas.draw_idle()
        else:
            self.placeholder.configure(
                text=self.tr("Choose the input paths, then click Start analysis.")
            )

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._close_plot_settings()
        self._close_playback_settings()
        if self._help_window is not None and self._help_window.winfo_exists():
            self._help_window.close()
        if self._load_worker is not None:
            self._load_worker.cancel()
            self._load_worker = None
        if self._load_poll_id is not None:
            self.root.after_cancel(self._load_poll_id)
            self._load_poll_id = None
        if self.playback is not None:
            self.playback.close()
        self.root.destroy()


class EvaluationHelpWindow(tk.Toplevel):
    """Use the Predictors' scrollable, non-modal help presentation."""

    def __init__(self, parent, *, language="en", on_close=None):
        super().__init__(parent)
        self._on_close = on_close
        self._closed = False
        self.withdraw()
        self.transient(parent)
        self.configure(bg=view.CONTROL_BACKGROUND)
        self.resizable(False, False)
        main = tk.Frame(self, bg=view.CONTROL_BACKGROUND)
        main.pack(fill="both", expand=True)
        tk.Label(
            main,
            text="Evaluation Viewer",
            font=("Arial", 13, "bold"),
            bg=view.CONTROL_BACKGROUND,
            fg=view.TEXT_COLOR,
            anchor="w",
        ).pack(fill="x", padx=24, pady=(18, 12))
        self._form = ScrollableForm(main, width=HELP_WINDOW_WIDTH - 60)
        self._form.pack(fill="both", expand=True, padx=24)
        actions = tk.Frame(main, bg=view.CONTROL_BACKGROUND)
        actions.pack(fill="x", padx=24, pady=(12, 18))
        view.create_styled_button(
            actions, text="OK", width=12, command=self.close
        ).pack(anchor="e")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", lambda _: self.close())
        self.set_language(language)
        self.update_idletasks()
        self._form.canvas.configure(
            height=help_content_height_for_screen(
                self._form.body.winfo_reqheight(), self.winfo_screenheight()
            )
        )
        self.update_idletasks()
        height = help_window_height(
            main.winfo_reqheight() + DIALOG_CONTENT_PADDING, self.winfo_screenheight()
        )
        left, top = help_window_position()
        self.geometry(f"{HELP_WINDOW_WIDTH}x{height}+{left}+{top}")
        self.deiconify()
        self.focus_set()

    def set_language(self, language):
        self.title(text("Help", language))
        for child in self._form.body.winfo_children():
            child.destroy()
        for heading, rows in help_sections(language):
            section = tk.LabelFrame(
                self._form.body,
                text=heading,
                font=view.FONT,
                bg=view.CONTROL_BACKGROUND,
                fg=view.TEXT_COLOR,
                padx=12,
                pady=10,
            )
            section.pack(fill="x", pady=(0, 12))
            section.columnconfigure(1, weight=1)
            add_help_description_rows(section, rows)
        self._form.bind_mouse_wheel()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._on_close is not None:
            self._on_close()
        self.destroy()
