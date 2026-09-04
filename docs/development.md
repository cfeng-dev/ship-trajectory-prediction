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

The ship simulator is an auxiliary data-generation tool located under
`tools/ship_simulator`. Installing the project creates its command:

```bash
uv run ship-simulator
```

This command starts the interactive ship trajectory simulator.

Alternatively, start the simulator as a Python module from the project root:

```bash
uv run python -m ship_simulator
```

Validate and summarize a recorded or simulated trajectory CSV without changing
the file:

```bash
uv run trajectory-data-check data/simulated/example_simulated_trajectory.csv
```

The checker validates the shared schema, timestamps, coordinate and speed
values, per-run sampling intervals, and reports GPS-derived movement statistics.
Pass `--all-runs` to print every run summary for files containing many runs.

The project environment must be synchronized first with `uv sync --locked`.

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
