"""Dialog workflows for the standalone ship trajectory GUI."""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from . import controls
from . import io as simulation_io
from .config import MOTION_PARAMETER_FIELDS, parse_motion_parameters


def show_motion_parameters_dialog(gui):
    """Edit a local draft; only Apply changes the next simulation run."""
    if gui.simulation_started:
        messagebox.showwarning(
            "Movement Parameters Locked",
            "Reset the simulation before changing movement parameters.",
            parent=gui.root,
        )
        return None

    dialog = tk.Toplevel(gui.root)
    dialog.withdraw()
    dialog.title("Movement Parameters")
    dialog.resizable(False, False)
    dialog.configure(bg=gui.app_background_color)
    dialog.transient(gui.root)
    form = tk.Frame(dialog, bg=gui.app_background_color, padx=18, pady=16)
    form.pack(fill=tk.BOTH, expand=True)
    form.columnconfigure(1, weight=1)
    variables = {}
    for row, (name, label) in enumerate(MOTION_PARAMETER_FIELDS):
        variable = tk.StringVar(master=dialog, value=str(getattr(gui, name)))
        variables[name] = variable
        tk.Label(form, text=label, anchor="w", bg=gui.app_background_color).grid(
            row=row, column=0, sticky="w", padx=(0, 10), pady=4
        )
        tk.Entry(form, textvariable=variable, width=16, justify=tk.RIGHT).grid(
            row=row, column=1, sticky="ew", pady=4
        )
    tk.Label(
        form,
        text="Turn limits are symmetric. Arrow keys use the same step sizes.",
        bg=gui.app_background_color,
    ).grid(row=len(variables), column=0, columnspan=2, sticky="w", pady=(8, 0))
    buttons = tk.Frame(form, bg=gui.app_background_color)
    buttons.grid(
        row=len(variables) + 1, column=0, columnspan=2, sticky="e", pady=(14, 0)
    )

    def apply():
        values = {name: variable.get() for name, variable in variables.items()}
        if gui.apply_motion_parameters(values, parent=dialog):
            dialog.destroy()

    controls.create_styled_button(buttons, text="Apply", command=apply, width=10).pack(
        side=tk.LEFT, padx=(0, 8)
    )
    controls.create_styled_button(
        buttons, text="Cancel", command=dialog.destroy, width=10
    ).pack(side=tk.LEFT)
    dialog.bind("<Return>", lambda _event: apply())
    dialog.bind("<Escape>", lambda _event: dialog.destroy())
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    dialog.update_idletasks()
    width, height = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
    x = gui.root.winfo_rootx() + (gui.root.winfo_width() - width) // 2
    y = gui.root.winfo_rooty() + (gui.root.winfo_height() - height) // 2
    dialog.geometry(f"{width}x{height}+{max(0, x)}+{max(0, y)}")
    dialog.deiconify()
    dialog.grab_set()
    dialog.focus_set()
    return dialog


def apply_motion_parameters(gui, values, parent=None):
    """Validate all settings before changing sliders, configuration or motion."""
    if gui.simulation_started:
        messagebox.showwarning(
            "Movement Parameters Locked",
            "Reset the simulation before changing movement parameters.",
            parent=parent,
        )
        return False
    try:
        parameters = parse_motion_parameters(values)
    except ValueError as error:
        messagebox.showerror("Invalid Movement Parameters", str(error), parent=parent)
        return False
    for name, value in parameters.items():
        setattr(gui, name, value)
        setattr(gui.config, name, value)
    gui.speed_slider.configure(
        from_=gui.min_speed, to=gui.max_speed, resolution=gui.speed_resolution
    )
    gui.steering_slider.configure(
        from_=gui.max_steering_deg_per_second,
        to=gui.min_steering_deg_per_second,
        resolution=gui.steering_resolution,
    )
    gui.speed_slider.set(gui.initial_speed)
    gui.steering_slider.set(0.0)
    gui.simulator.v = gui.initial_speed
    gui.update_status()
    gui.update_plot()
    return True


def save_csv(gui):
    """Save the simulated trajectory to a user-selected CSV file."""
    if not gui.simulator.has_data():
        messagebox.showwarning(
            "No Data",
            "No trajectory data available. Please start the simulation first.",
        )
        return

    # Keep the trajectory unchanged while the file dialog is open.
    gui.pause_simulation()

    try:
        simulation_io.DATA_DIR.mkdir(parents=True, exist_ok=True)
        previous_path = getattr(gui, "last_saved_csv_path", None)
        if previous_path is not None and previous_path in gui.saved_run_states:
            initial_directory = previous_path.parent
            initial_filename = previous_path.name
        else:
            initial_directory = simulation_io.DATA_DIR
            initial_filename = simulation_io.next_trajectory_filename(
                initial_directory, default_filename=gui.default_csv_filename
            )
    except (OSError, ValueError) as error:
        messagebox.showerror(
            "Save Failed", f"Could not select a CSV filename:\n\n{error}"
        )
        return
    output_path = filedialog.asksaveasfilename(
        title="Save trajectory as time,x,y CSV",
        initialdir=initial_directory,
        initialfile=initial_filename,
        defaultextension=".csv",
        confirmoverwrite=False,
        filetypes=[
            ("CSV files", "*.csv"),
            ("All files", "*.*"),
        ],
    )

    if not output_path:
        return

    resolved_output_path = Path(output_path).resolve()
    first_unsaved_sample = gui.saved_run_states.get(resolved_output_path, 0)
    if not resolved_output_path.is_file() or resolved_output_path.stat().st_size == 0:
        first_unsaved_sample = 0

    try:
        trajectory_df = simulation_io.create_simulation_dataframe(
            gui.simulator, interval_seconds=gui.csv_export_interval_seconds
        )
        if first_unsaved_sample >= len(trajectory_df):
            messagebox.showinfo(
                "No New Data",
                "No new CSV sample is available yet.\n"
                f"Export interval: {gui.csv_export_interval_seconds:g} simulated seconds.\n\n"
                f"File:\n{resolved_output_path}",
            )
            return
        save_result = simulation_io.save_trajectory_data(
            df=trajectory_df.iloc[first_unsaved_sample:],
            filename=output_path,
            append=first_unsaved_sample > 0,
        )
    except (OSError, ValueError) as error:
        messagebox.showerror(
            "Save Failed",
            f"Could not save trajectory data:\n\n{error}",
        )
        return

    gui.saved_run_states[save_result.output_path.resolve()] = len(trajectory_df)
    gui.last_saved_csv_path = save_result.output_path.resolve()

    if save_result.appended:
        message_title = "Trajectory Updated"
        message_text = (
            "New samples and ground-truth states appended to the current trajectory.\n\n"
            f"File:\n{save_result.output_path}"
        )
    else:
        message_title = "Trajectory Saved"
        message_text = (
            "New trajectory CSV with ground-truth states created for the Bayesian Predictor.\n\n"
            f"File:\n{save_result.output_path}"
        )

    messagebox.showinfo(message_title, message_text)
