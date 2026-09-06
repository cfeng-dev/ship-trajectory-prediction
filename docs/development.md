# Development Guide

## Managing Dependencies

Dependencies are defined in `pyproject.toml`, while `uv.lock` stores the exact
resolved versions.

Install the existing locked environment without changing `uv.lock`:

```bash
uv sync --locked
```

After changing dependencies in `pyproject.toml`, update the lockfile and
environment with:

```bash
uv lock
uv sync
```

Commit `pyproject.toml` and `uv.lock` together.

## Command-Line Tools

Synchronize the project environment before running the tools:

```bash
uv sync --locked
```

### 1. Ship Simulator

Use the interactive simulator to generate synthetic ship-trajectory CSV files
when recorded trajectory data are unavailable. Run it from the project root:

```bash
uv run ship-simulator
```

### 2. Trajectory Data Checker

Run the checker from the project root. It verifies that a CSV follows the
project's trajectory-data format and basic quality requirements before it is
used by the models.

```bash
uv run trajectory_checker path/to/trajectory.csv
```

Only the first ten runs are shown by default. Show all runs with:

```bash
uv run trajectory_checker path/to/trajectory.csv --all-runs
```

### 3. Posterior Explorer

Inspect the ship's trajectory and Bayesian CTRV posterior updates in one desktop
window. Run from the project root after synchronizing the environment:

```bash
uv run posterior-explorer
```

Without refreshing the installed command, use the existing Windows environment:

```powershell
.venv\Scripts\python.exe -m tools.posterior_explorer
```

The window uses the same **File / View / Settings / Help** menu layout and light
blue control-panel style as the ship simulator. The **Neue Analyse** button stays
at the top of the sidebar, outside the scrollable data fields.

Choose a CSV, run, inference method and optional observation limit in **Daten**.
**File** also opens the CSV chooser. Under **Settings**, open **Priors**,
**Inferenzparameter** (for the currently selected method), or **Daten und
Darstellung** (forecast steps/sample paths, noise, seeds, playback interval and
legend). Dialogs edit a draft:
**Übernehmen** validates and keeps the edits; **Abbrechen** discards them. Neither
action restarts inference. Click **Neue Analyse** to apply all settings and begin
with an empty cache.

Use **View** to hide/show the settings pane or reset the plot view, and **Help**
for keyboard instructions. Closing through **File** has the same graceful
in-flight-fit behavior as closing the window.

Use Start/Pause or Space to play, the slider to select a stage, and Left/Right
with the plot focused to step through observations. Space in an input field keeps
its normal editing behavior. Zoom is retained as the posterior changes; the
Matplotlib toolbar resets the view or saves a figure. Switch between motion and
noise parameters using the posterior-group selector.

Zoom to the desired scale, then enable **Schiff folgen** below the trajectory,
above Start/Pause. The viewport centers on the current observed ship position
and follows playback, slider selection and backward steps without changing the
zoom scale. Disable it to pan freely again. The switch takes effect immediately
without a new analysis, is initially off, and also appears in the standalone
script. At N=0 there is no current position, so the viewport stays unchanged.

The trajectory panel also shows the latent forecast for the selected posterior:
a red coordinate-wise median and translucent example paths (not a probability
region). Forecasts use only observations through N; future recorded positions are
reference data, never fit inputs. The first line segment connects to the last
measured position for visual orientation, without changing the model's forecasts.
The default is 3 steps (30 seconds at 10-second intervals) and up to 20 sample
paths. Under **Settings > Daten und
Darstellung**, set **Vorhersageschritte** to 0 to disable forecasts, or
**Zukunftstrajektorien** to 0 to show only the median. Forecasts use the next
timestamps of the recording, so the horizon shortens near its end. At the last
recorded position there are no further forecast timestamps. At N=0 only priors
are shown; forecasts start at N=1 for RBPF/SMC and N=3 for VI/MCMC.

Under **Settings > Daten und Darstellung**, **Koordinatenanzeige** switches the
trajectory panel immediately between local metres, kilometres and GPS
(longitude/latitude). This affects only the displayed route, current position
and forecast; inference and posterior values remain in metres. GPS uses the
reference position of the selected route and adjusts the spatial aspect for its
latitude.

RBPF and SMC update an online filter. VI and MCMC refit an expanding data prefix
for each stage, starting at N=3; start with a small observation limit. They also
reserve the final route observation for the existing batch prediction interface.
All inference runs on one background worker. Waiting for a posterior does not
advance playback past that stage. Pause freezes the displayed stage; an in-flight
result may finish and enter the cache. Replacing an analysis discards its results,
and closing the window waits responsively for an in-flight fit to finish; fits
are not forcibly terminated.

The header-configured script remains available:
`experiments/posterior_analysis/plot_bayesian_ctrv_posterior_updates.py`.
Set `EXPERIMENT.prediction_count`, `EXPERIMENT.prediction_sample_count` and
`COORDINATE_DISPLAY_MODE` in its header to control the forecast and coordinate
display.
The GUI reuses its shared dashboard and inference loaders; it does not import
that script's editable header. GUI settings are local to the window, not saved
back into the script. Tkinter is part of standard Python installations, and no
additional Python dependency is required.

## Code Quality and Tests

GitHub Actions runs the tests on Python 3.10, 3.12, and 3.14 for every push and
pull request. Run the checks locally from the project root with:

```bash
uv run pytest
```

To apply Ruff formatting automatically:

```bash
uv run ruff format .
```
