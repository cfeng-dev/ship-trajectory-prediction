# Ship Trajectory Prediction

[![Tests](https://github.com/cfeng-dev/ship-trajectory-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/cfeng-dev/ship-trajectory-prediction/actions/workflows/tests.yml)

## Overview

This repository contains the source code, Bayesian models, and accompanying documentation for my Master's thesis.

The project investigates **probabilistic ship trajectory prediction using Bayesian methods**. The primary goal is to develop Bayesian models that predict future vessel trajectories while explicitly quantifying uncertainty in both observations and model parameters.

---

## Problem Description

Ship trajectory prediction plays an important role in maritime applications, including traffic monitoring, collision avoidance, and autonomous navigation. However, accurately predicting vessel movements remains challenging due to various sources of uncertainty.

Traditional deterministic prediction methods typically provide only a single estimated trajectory without expressing the associated uncertainty. In many real-world applications, this lack of uncertainty estimation can reduce the reliability of decision-making.

Bayesian modeling provides a principled probabilistic framework that enables uncertainty to be explicitly incorporated into trajectory prediction by estimating full posterior distributions rather than single point estimates.

---

## Objectives

The primary objective of this Master's thesis is to develop Bayesian models for probabilistic ship trajectory prediction.

The project aims to:

- Investigate Bayesian approaches for modeling vessel trajectories.
- Develop probabilistic trajectory prediction models using Stan and CmdStanPy.
- Quantify predictive uncertainty through Bayesian inference.
- Evaluate different Bayesian modeling strategies using real-world ship trajectory data.
- Visualize and analyze probabilistic trajectory predictions.

---

## Getting Started

### 1. Install Prerequisites

The following software is required:

- **[Git](https://git-scm.com/downloads)**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**

The following tools are optional:

- **[Visual Studio Code](https://code.visualstudio.com/)**
- **[Python extension for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=ms-python.python)**

A separate Python installation is not required. If necessary, `uv` downloads
and manages a compatible Python version automatically.

### 2. Clone the Repository

```bash
git clone https://github.com/cfeng-dev/ship-trajectory-prediction.git
cd ship-trajectory-prediction
```

### 3. Install the Project

```bash
uv sync --locked
```

This creates `.venv` and installs the project. No manual activation is needed
when using `uv run`.

### 4. Select the Python Interpreter in Visual Studio Code (Optional)

1. Press **Ctrl + Shift + P**.
2. Search for **Python: Select Interpreter**.
3. Select the interpreter inside `.venv`:

```text
# Windows
.venv\Scripts\python.exe

# macOS / Linux
.venv/bin/python
```

### 5. Install the C++ Toolchain

CmdStan requires a working C++17 toolchain to compile Stan models.

Windows users can install the required GNU C++ toolchain with CmdStanPy:

```bash
uv run python -m cmdstanpy.install_cxx_toolchain
```

This command installs the RTools/MinGW toolchain used by CmdStanPy to compile Stan models on Windows.

For macOS and Linux setup instructions, please refer to the [official Stan installation guide](https://mc-stan.org/install/#prerequisite-c17-toolchain).

### 6. Install CmdStan

Download and install the latest supported version of CmdStan:

```bash
uv run python -m cmdstanpy.install_cmdstan
```

### 7. Verify the Installation

Start the Python interpreter in the project environment:

```bash
uv run python
```

Then execute:

```python
import cmdstanpy

print(cmdstanpy.cmdstan_path())
```

A valid installation path, for example

```text
C:\Users\<username>\.cmdstan\cmdstan-2.39.0
```

confirms that CmdStan has been installed successfully.

Exit the Python interpreter:

```python
exit()
```

---

## Troubleshooting

### CmdStan installation fails on Windows

If the installation fails because commands such as `mingw32-make` or `cut` cannot be found, ensure that the following directories are included in the user `PATH` environment variable:

```text
%USERPROFILE%\.cmdstan\RTools40\mingw64\bin
%USERPROFILE%\.cmdstan\RTools40\usr\bin
```

Restart the terminal after updating the `PATH` and rerun:

```bash
uv run python -m cmdstanpy.install_cmdstan
```

For more detailed diagnostic output, run:

```bash
uv run python -m cmdstanpy.install_cmdstan --verbose --cores 1
```

---

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

---

## Command-Line Tools

Installing the project creates the simulator command:

```bash
uv run ship-simulator
```

This command starts the interactive ship trajectory simulator.

Alternatively, the simulator can be started by running `cli.py` directly from
the project root:

```bash
uv run python src/ship_trajectory_prediction/simulation/cli.py
```

The project environment must be synchronized first with `uv sync --locked`.

---

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
