"""CSV explorer using the Bayesian Predictor's native desktop presentation."""

import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from bayesian_predictor import view
from bayesian_predictor.controls import ScrollableForm
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from bayestraj.validation.evaluation_viewer import load_evaluation_directory

from .localization import text
from .plot_view import EvaluationPlot, PlaybackBar
from .session import Playback
from .settings import METRICS, ViewerOptions


class EvaluationViewer:
    def __init__(self, root, *, input_dir, observations_csv=None):
        self.root = root
        self.options = ViewerOptions()
        self.data = None
        self.playback = None
        self.selected_models = ()
        self._closed = False
        self.directory = tk.StringVar(root, value=str(input_dir))
        self.observations_csv = tk.StringVar(
            root, value=str(observations_csv) if observations_csv else ""
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
        self._build_sidebar()
        self._build_menu()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        for key in ("<Left>", "<Right>", "<space>"):
            root.bind(key, self._keyboard, add="+")
        self._initial_load = root.after(50, self._load_initial)

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
        form.on_scroll = self.pause
        body = form.body
        title = self._label(body, "Model evaluation")
        title.configure(font=("Arial", 12, "bold"), anchor="center", pady=10)
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
        self._label(group, "Observation CSV")
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
        view.create_styled_button(
            group, text=self.tr("Load CSVs"), command=self.load_directory
        ).pack(fill="x", pady=(8, 0))
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
            self._label(group, label)
            variable = tk.StringVar(
                self.root, value=self.tr(METRICS[getattr(self.options, key)])
            )
            self.metric_variables[key] = variable
            ttk.Combobox(
                group,
                textvariable=variable,
                values=tuple(self.metric_choices),
                state="readonly",
                style="Predictor.TCombobox",
                width=26,
            ).pack(fill="x", pady=(0, 7))
        self.display_variables = {}
        group = self._section(body, "Metric time axis")
        self._option_check(group, "follow_metric_time", "Follow metric time")
        self._option_entry(group, "metric_window_width", "Window width [s]")
        group = self._section(body, "Trajectory view")
        self._option_check(group, "follow_ship", "Follow ship")
        self._option_entry(group, "ship_view_span", "View span [m]")
        self._option_check(group, "show_legend", "Show legend")
        view.create_styled_button(
            body, text=self.tr("Apply"), command=self.apply_options
        ).pack(fill="x", pady=(0, 8))
        tk.Label(
            body,
            textvariable=self.status,
            bg=view.CONTROL_BACKGROUND,
            fg=view.TEXT_COLOR,
            justify="left",
            anchor="w",
            wraplength=275,
            font=view.FONT,
        ).pack(fill="x")
        self._set_status()
        form.bind_mouse_wheel()

    def _check(self, group, source, variable, *, translated=False):
        tk.Checkbutton(
            group,
            text=source if translated else self.tr(source),
            variable=variable,
            bg=view.CONTROL_BACKGROUND,
            activebackground=view.CONTROL_BACKGROUND,
            fg=view.TEXT_COLOR,
            font=view.FONT,
            anchor="w",
            justify="left",
            wraplength=255,
        ).pack(fill="x", pady=2)

    def _option_check(self, group, key, source):
        variable = tk.BooleanVar(self.root, value=getattr(self.options, key))
        self.display_variables[key] = variable
        self._check(group, source, variable)

    def _option_entry(self, group, key, source):
        self._label(group, source)
        variable = tk.StringVar(self.root, value=str(getattr(self.options, key)))
        self.display_variables[key] = variable
        ttk.Entry(
            group, textvariable=variable, style="Predictor.TEntry", width=12
        ).pack(anchor="w", pady=(0, 6))

    def _build_timeline(self):
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
        self.timeline_canvas.get_tk_widget().configure(height=90)
        self.timeline_canvas.get_tk_widget().grid(row=1, column=0, sticky="ew")

    def _build_menu(self):
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(
            label=self.tr("Open directory…"), command=self.open_directory
        )
        file_menu.add_command(label=self.tr("Reload"), command=self.load_directory)
        file_menu.add_separator()
        file_menu.add_command(label=self.tr("Exit"), command=self.close)
        menu.add_cascade(label=self.tr("File"), menu=file_menu)
        settings = tk.Menu(menu, tearoff=0)
        settings.add_checkbutton(
            label=self.tr("Show settings"),
            variable=self.settings_visible,
            command=self.toggle_settings,
        )
        menu.add_cascade(label=self.tr("Settings"), menu=settings)
        language = tk.Menu(menu, tearoff=0)
        self.language_variable = tk.StringVar(self.root, value=self.options.language)
        for label, value in (("English", "en"), ("Deutsch", "de")):
            language.add_radiobutton(
                label=label,
                value=value,
                variable=self.language_variable,
                command=lambda selected=value: self.set_language(selected),
            )
        menu.add_cascade(label=self.tr("Language"), menu=language)
        help_menu = tk.Menu(menu, tearoff=0)
        help_menu.add_command(
            label=self.tr("Show Help"),
            command=lambda: messagebox.showinfo(
                self.tr("Help"), self.tr("Help text"), parent=self.root
            ),
        )
        menu.add_cascade(label=self.tr("Help"), menu=help_menu)
        previous = getattr(self, "_menu", None)
        self.root.configure(menu=menu)
        self._menu = menu
        if previous is not None:
            self.root.after_idle(previous.destroy)

    def _load_initial(self):
        self._initial_load = None
        self.load_directory(show_error=False)

    def open_directory(self):
        self.pause()
        selected = filedialog.askdirectory(
            parent=self.root,
            initialdir=self.directory.get(),
            title=self.tr("Open directory…"),
        )
        if selected:
            self.directory.set(selected)
            self.load_directory()

    def open_observations(self):
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
            self.load_directory()

    def load_directory(self, *, show_error=True):
        self.pause()
        try:
            observation_path = self.observations_csv.get().strip()
            data = load_evaluation_directory(
                Path(self.directory.get()).expanduser(),
                observations_csv=observation_path or None,
            )
        except (OSError, ValueError) as error:
            self.status.set(str(error))
            if show_error:
                messagebox.showerror(self.tr("CSV error"), str(error), parent=self.root)
            return
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
        self.playback = Playback(self.root, len(data.origins), self._frame)
        self._build_sidebar()
        self.navigation.set_count(len(data.origins))
        self.toolbar.update()
        self._frame(0)

    def apply_options(self):
        self.pause()
        try:
            selected = tuple(
                label
                for label, variable in self.model_variables.items()
                if variable.get()
            )
            if self.data is not None and not selected:
                raise ValueError(self.tr("Select at least one model."))
            metrics = {
                key: self.metric_choices.get(variable.get())
                for key, variable in self.metric_variables.items()
            }
            if any(value is None for value in metrics.values()):
                raise ValueError(
                    self.tr("Select an available metric for each diagram.")
                )
            options = replace(
                self.options,
                **metrics,
                **{
                    key: variable.get()
                    for key, variable in self.display_variables.items()
                },
            )
        except (ValueError, tk.TclError) as error:
            messagebox.showerror(
                self.tr("Invalid settings"), str(error), parent=self.root
            )
            return
        self.options = options
        self.selected_models = selected
        self.navigation.set_follow_ship(options.follow_ship)
        self.toolbar.update()
        if self.playback is not None:
            self._frame(self.playback.index)

    def _frame(self, index):
        self.plot.draw(self.data, index, self.options, self.selected_models)
        self.navigation.show_index(index, elapsed_seconds=self.data.elapsed[index])
        self._set_status()
        self._sync_play_button()
        self.canvas.draw_idle()

    def _set_status(self):
        if self.data is None:
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
        playing = self.playback is not None and self.playback.playing
        self.navigation.set_playing(playing)

    def pause(self):
        if self.playback is not None:
            self.playback.pause()
        self._sync_play_button()

    def toggle_playback(self):
        if self.playback is None:
            return
        if self.playback.playing:
            self.pause()
        else:
            self.playback.start()

    def step(self, delta):
        if self.playback is not None:
            self.playback.seek(self.playback.index + delta)

    def _slider(self, value):
        index = int(float(value))
        if self.playback is not None and index != self.playback.index:
            self.playback.seek(index)

    def _plot_click(self, event):
        if (
            self.data is not None
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
        self.navigation.set_settings_visible(self.settings_visible.get())
        if self.settings_visible.get():
            self.sidebar.grid()
        else:
            self.sidebar.grid_remove()

    def set_settings_visible(self, visible):
        self.settings_visible.set(visible)
        self.toggle_settings()

    def set_follow_ship(self, follow):
        self.pause()
        self.options = replace(self.options, follow_ship=bool(follow))
        self.display_variables["follow_ship"].set(follow)
        if self.playback is not None:
            self._frame(self.playback.index)

    def set_language(self, language):
        self.pause()
        model_draft = {
            label: variable.get() for label, variable in self.model_variables.items()
        }
        metric_draft = {
            key: self.metric_choices.get(variable.get())
            for key, variable in self.metric_variables.items()
        }
        display_draft = {
            key: variable.get() for key, variable in self.display_variables.items()
        }
        self.options = replace(self.options, language=language)
        self._build_sidebar()
        for label, value in model_draft.items():
            self.model_variables[label].set(value)
        for key, value in metric_draft.items():
            if value is not None:
                self.metric_variables[key].set(self.tr(METRICS[value]))
        for key, value in display_draft.items():
            self.display_variables[key].set(value)
        self._build_menu()
        self.navigation.set_language(language)
        if self.playback is not None:
            self._frame(self.playback.index)
        else:
            self.plot.clear(language)
            self.canvas.draw_idle()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._initial_load is not None:
            self.root.after_cancel(self._initial_load)
            self._initial_load = None
        if self.playback is not None:
            self.playback.close()
        self.root.destroy()
