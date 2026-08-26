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

Installing the project creates the simulator command:

```bash
uv run ship-simulator
```

This command starts the interactive ship trajectory simulator.

Alternatively, the simulator can be started by running `cli.py` directly from
the project root:

```bash
uv run python src/bayestraj/simulation/cli.py
```

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
