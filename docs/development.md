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

Choose a CSV, run, inference method and optional observation limit in **Daten**.
The **Priors** and **Inferenz** tabs expose the prior and method-specific numerical
settings. Click **Neue Analyse** to apply edits and begin with an empty cache.
Hide the settings pane to give the plot more space.

Use Start/Pause or Space to play, the slider to select a stage, and Left/Right
with the plot focused to step through observations. Space in an input field keeps
its normal editing behavior. Zoom is retained as the posterior changes; the
Matplotlib toolbar resets the view or saves a figure. Switch between motion and
noise parameters using the posterior-group selector.

RBPF and SMC update an online filter. VI and MCMC refit an expanding data prefix
for each stage, starting at N=3; start with a small observation limit. They also
reserve the final route observation for the existing batch prediction interface.
All inference runs on one background worker. Waiting for a posterior does not
advance playback past that stage. Pause freezes the displayed stage; an in-flight
result may finish and enter the cache. Replacing an analysis discards its results,
and closing the window waits responsively for an in-flight fit to finish; fits
are not forcibly terminated.

The header-configured script remains available and unchanged:
`experiments/posterior_analysis/plot_bayesian_ctrv_posterior_updates.py`.
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
