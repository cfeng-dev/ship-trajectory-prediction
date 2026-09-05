"""Desktop presentation matching the ship simulator, without coupling the tools."""

import tkinter as tk

APP_BACKGROUND = "#eef7fb"
CONTROL_BACKGROUND = "#e3eef4"
PLOT_BACKGROUND = "#f4f9fc"
TEXT_COLOR = "#1f2933"
FONT = ("Arial", 10)


def create_styled_button(parent, *, text, command, width=18):
    """Use the simulator's white buttons with blue hover feedback."""
    button = tk.Button(
        parent,
        text=text,
        command=command,
        width=width,
        font=FONT,
        bg="#ffffff",
        fg=TEXT_COLOR,
        activebackground="#b8d8e8",
        activeforeground=TEXT_COLOR,
        disabledforeground="#7a8790",
        relief=tk.SOLID,
        borderwidth=1,
        highlightthickness=0,
        padx=10,
        pady=3,
    )
    button.bind("<Enter>", lambda _: button.configure(bg="#b8d8e8"))
    button.bind("<Leave>", lambda _: button.configure(bg="#ffffff"))
    return button


def configure_window(root):
    """Keep the window inside the screen and apply the simulator's light palette."""
    root.title("Bayesian CTRV — Posterior Explorer")
    width = min(1500, max(800, root.winfo_screenwidth() - 80))
    height = min(900, max(550, root.winfo_screenheight() - 100))
    left = max(0, (root.winfo_screenwidth() - width) // 2)
    top = max(0, (root.winfo_screenheight() - height) // 2 - 20)
    root.geometry(f"{width}x{height}+{left}+{top}")
    root.minsize(min(1000, width), min(620, height))
    root.configure(bg=APP_BACKGROUND)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)


def create_menu_bar(gui):
    """Expose application actions through the same four menus as the simulator."""
    menu_bar = tk.Menu(gui.root)
    file_menu = tk.Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="CSV öffnen…", command=gui.open_csv)
    file_menu.add_separator()
    file_menu.add_command(label="Schließen", command=gui.close)
    menu_bar.add_cascade(label="File", menu=file_menu)

    view_menu = tk.Menu(menu_bar, tearoff=0)
    view_menu.add_checkbutton(
        label="Einstellungen anzeigen",
        variable=gui.settings_visible_var,
        command=gui.toggle_settings,
    )
    view_menu.add_command(label="Plotansicht zurücksetzen", command=gui.reset_plot_view)
    menu_bar.add_cascade(label="View", menu=view_menu)

    settings_menu = tk.Menu(menu_bar, tearoff=0)
    settings_menu.add_command(
        label="Priors…", command=lambda: gui.show_settings_dialog("priors")
    )
    settings_menu.add_command(
        label="Inferenzparameter…", command=gui.show_inference_settings
    )
    settings_menu.add_command(
        label="Daten und Darstellung…", command=lambda: gui.show_settings_dialog("data")
    )
    menu_bar.add_cascade(label="Settings", menu=settings_menu)

    help_menu = tk.Menu(menu_bar, tearoff=0)
    help_menu.add_command(label="Bedienung und Tastenkürzel", command=gui.show_help)
    menu_bar.add_cascade(label="Help", menu=help_menu)
    gui.root.configure(menu=menu_bar)
