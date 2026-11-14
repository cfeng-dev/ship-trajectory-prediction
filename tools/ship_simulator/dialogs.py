"""Dialog workflows for the standalone ship trajectory GUI."""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np

from . import controls
from . import io as simulation_io


def _parse_gps_start_position(latitude_text, longitude_text):
    """Parse and validate a user-provided GPS start position."""
    try:
        latitude = float(latitude_text)
        longitude = float(longitude_text)
    except ValueError as error:
        raise ValueError("Latitude and longitude must be numeric values.") from error

    if not np.isfinite(latitude) or not np.isfinite(longitude):
        raise ValueError("Latitude and longitude must be finite values.")
    if not -90 < latitude < 90:
        raise ValueError("Latitude must be greater than -90° and less than 90°.")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180° and 180°.")

    return latitude, longitude


def show_gps_start_position_dialog(gui):
    """Open a dialog for configuring the GPS position of the local origin."""
    if gui.simulation_started:
        messagebox.showwarning(
            "GPS Position Locked",
            "Reset the simulation before changing the GPS start position.",
            parent=gui.root,
        )
        return

    dialog = tk.Toplevel(gui.root)
    dialog.title("GPS Start Position")
    dialog.resizable(False, False)
    dialog.configure(bg=gui.app_background_color)
    dialog.transient(gui.root)
    dialog.grab_set()

    form_frame = tk.Frame(
        dialog,
        bg=gui.app_background_color,
        padx=18,
        pady=16,
    )
    form_frame.pack(fill=tk.BOTH, expand=True)
    form_frame.columnconfigure(1, weight=1)

    latitude_var = tk.StringVar(value=f"{gui.reference_latitude:.8f}")
    longitude_var = tk.StringVar(value=f"{gui.reference_longitude:.8f}")

    tk.Label(
        form_frame,
        text="Latitude [°]:",
        anchor="w",
        bg=gui.app_background_color,
        fg="black",
    ).grid(row=0, column=0, sticky="w", padx=(0, 10), pady=4)
    latitude_entry = tk.Entry(
        form_frame,
        textvariable=latitude_var,
        width=20,
        justify=tk.RIGHT,
        bg="white",
        fg="black",
        insertbackground="black",
        selectbackground="#007aff",
        selectforeground="white",
        relief=tk.SOLID,
        borderwidth=1,
        highlightthickness=0,
    )
    latitude_entry.grid(row=0, column=1, sticky="ew", pady=4)

    tk.Label(
        form_frame,
        text="Longitude [°]:",
        anchor="w",
        bg=gui.app_background_color,
        fg="black",
    ).grid(row=1, column=0, sticky="w", padx=(0, 10), pady=4)
    longitude_entry = tk.Entry(
        form_frame,
        textvariable=longitude_var,
        width=20,
        justify=tk.RIGHT,
        bg="white",
        fg="black",
        insertbackground="black",
        selectbackground="#007aff",
        selectforeground="white",
        relief=tk.SOLID,
        borderwidth=1,
        highlightthickness=0,
    )
    longitude_entry.grid(row=1, column=1, sticky="ew", pady=4)

    button_frame = tk.Frame(form_frame, bg=gui.app_background_color)
    button_frame.grid(
        row=2,
        column=0,
        columnspan=2,
        sticky="e",
        pady=(14, 0),
    )

    def apply_position():
        if gui.apply_gps_start_position(
            latitude_var.get(),
            longitude_var.get(),
            parent=dialog,
        ):
            dialog.destroy()

    controls.create_styled_button(
        button_frame,
        text="Apply",
        width=10,
        command=apply_position,
    ).pack(side=tk.LEFT, padx=(0, 8))
    controls.create_styled_button(
        button_frame,
        text="Cancel",
        width=10,
        command=dialog.destroy,
    ).pack(side=tk.LEFT)

    dialog.bind("<Return>", lambda _event: apply_position())
    dialog.bind("<Escape>", lambda _event: dialog.destroy())

    dialog.update_idletasks()
    x = gui.root.winfo_rootx() + (gui.root.winfo_width() - dialog.winfo_width()) // 2
    y = gui.root.winfo_rooty() + (gui.root.winfo_height() - dialog.winfo_height()) // 2
    dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
    dialog.focus_set()


def apply_gps_start_position(gui, latitude_text, longitude_text, parent=None):
    """Validate and apply the GPS position of the local simulation origin."""
    if gui.simulation_started:
        messagebox.showwarning(
            "GPS Position Locked",
            "Reset the simulation before changing the GPS start position.",
            parent=parent,
        )
        return False

    try:
        latitude, longitude = _parse_gps_start_position(
            latitude_text,
            longitude_text,
        )
    except ValueError as error:
        messagebox.showerror(
            "Invalid GPS Position",
            str(error),
            parent=parent,
        )
        return False

    gui.reference_latitude = latitude
    gui.reference_longitude = longitude

    gui.update_status()
    gui.update_plot()

    messagebox.showinfo(
        "GPS Position Updated",
        "GPS start position updated successfully.\n\n"
        f"Latitude: {latitude:.8f}°\n"
        f"Longitude: {longitude:.8f}°",
        parent=parent,
    )

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

    simulation_io.DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = filedialog.asksaveasfilename(
        title="Save trajectory as time,x,y CSV",
        initialdir=simulation_io.DATA_DIR,
        initialfile=gui.default_csv_filename,
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

    if save_result.appended:
        message_title = "Trajectory Updated"
        message_text = (
            "New samples appended to the current time,x,y trajectory.\n\n"
            f"File:\n{save_result.output_path}"
        )
    else:
        message_title = "Trajectory Saved"
        message_text = (
            "New time,x,y CSV created for the Bayesian Predictor.\n\n"
            f"File:\n{save_result.output_path}"
        )

    messagebox.showinfo(message_title, message_text)
