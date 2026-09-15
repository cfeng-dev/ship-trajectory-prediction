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

The built-in **Help** window describes the controls, settings, and keyboard
shortcuts. RBPF and SMC update online; VI and MCMC refit an expanding prefix and
are slower.

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

## Final Thesis Benchmark

The final benchmark evaluates the deterministic CTRV baseline, Bayesian VI with
expanding history, online RBPF/SMC, and optionally MCMC. It writes CSV files to
`results/thesis_benchmark/`, which is intentionally ignored because it can
contain results from the external recorded trajectory data.

Run a small development smoke benchmark with one short trajectory segment:

```bash
uv run python experiments/model_evaluation/final_benchmark.py --run-ids 102 --methods deterministic --prediction-counts 1 --noise-levels 0 --max-windows 2
```

Select multiple runs and the main inference methods:

```bash
uv run python experiments/model_evaluation/final_benchmark.py --run-ids 101,102,103 --methods deterministic,vi_expanding,rbpf,smc --prediction-counts 1,3,6 --noise-levels 0,2,5,10
```

Run the configured benchmark dimensions from the script's configuration section:

```bash
uv run python experiments/model_evaluation/final_benchmark.py
```

For an MCMC reference subset, include `mcmc_expanding` and restrict its runs,
horizons, and forecast origins explicitly:

```bash
uv run python experiments/model_evaluation/final_benchmark.py --run-ids 102 --methods vi_expanding,mcmc_expanding --prediction-counts 1,3 --noise-levels 5 --mcmc-run-ids 102 --mcmc-prediction-counts 3 --mcmc-window-indices 0,5,10
```

`predictions.csv` contains one row per held-out position and the detailed
workflow metrics. `runs.csv` contains one status record per run/method/horizon/
noise combination, including runtime and any captured error. `summary.csv`
aggregates successful combinations over trajectories, while `per_horizon.csv`
aggregates them separately for each actual timestamp-derived horizon. Coverage
columns describe joint 2D positional coverage at each horizon, not simultaneous
coverage of a complete future trajectory. Deterministic rows intentionally leave
probabilistic metrics empty rather than using substitute metrics.
