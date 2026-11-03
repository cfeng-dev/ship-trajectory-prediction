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

### 2. Trajectory Predictor

Inspect the ship's trajectory and Bayesian CTRV posterior updates in one desktop
window. Run from the project root after synchronizing the environment:

```bash
uv run trajectory-predictor
```

Use **Language → English / Deutsch** to switch the interface and plot
labels without restarting analysis. English is selected at startup.

The built-in **Help** window describes the controls, settings, and keyboard
shortcuts. SMC and Sequential VI update online; VI and MCMC repeatedly fit the
latest 10 steps and are slower. For raw BR24 input, these windows count radar
cycles, including cycles without an accepted detection. Each method associates
raw detections using its own predicted state; ground truth is evaluation-only.

### 3. IMM Predictor

Inspect BR24 radar tracking and trajectory forecasts from the CV/CTRV/CA IMM:

```bash
uv run imm-predictor
```

The built-in **Help** window describes analysis setup, plot options, and
keyboard shortcuts. The optional local IMM implementation at
`external/IMM/br24_imm.py` and a BR24 tracking CSV are required for analysis.

## Project Paths

Resolve project-relative files with the shared helper, independently of the
working directory used to launch a script:

```python
from bayestraj.paths import project_path

DATA_FILE = project_path("data/processed/BR24Dataset20131107_with_reference.csv")
OUTPUT_CSV = project_path("results/bayesian_ctrv_smc_evaluation.csv")
```

Include the top-level directory, such as `data/`, `results/`, or `figures/`, in
the argument. Resolving a path does not create its file or directory.

## Model Evaluation

Run `experiments/model_evaluation/bayesian_ctrv.py` with your IDE's Run button or:

```bash
uv run python experiments/model_evaluation/bayesian_ctrv.py
```

Configure the inference method (`smc`, `vi`, `mcmc`, or `seq_vi`), priors,
and forecast settings in the script header. VI/MCMC use the shared default of
ten past points per fit: start with two points, grow to ten, then slide.
Evaluation starts at N=2; Sequential VI initializes once from three accepted
measurements by default, then updates one measurement at a time.

Results are saved to `results/bayesian_ctrv_<method>_evaluation.csv`
(`seq_vi` for Sequential VI).
Use `experiments/model_comparison/compare_bayesian_ctrv_imm.py` to compare saved
evaluation CSVs. Set its `IMM_CSV` and `BAYESIAN_*_CSV` paths; `None` excludes a
model. All selected models are compared on their common forecast origins.
`SHOW_POINTS` and `SHOW_MEAN_LINES` control curve markers and dashed mean lines.
`SHOW_SUMMARY_TABLE` displays all summary metrics in an additional table figure.
`METRIC_Y_SCALE` selects `linear` or `symlog` scaling for all metric curves.
`PLOT_LANGUAGE` selects German (`de`, default) or English (`en`) titles, time axes
and legends. Metric axes use short labels; titles include full English metric names.
Runtime labels follow the selected language.
Metric axes always include zero and preserve negative scores.
The symlog linear range adapts to each metric; coverage bars use a linear 0–100% axis.

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
