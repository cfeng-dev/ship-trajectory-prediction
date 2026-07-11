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

The following software is required before setting up the project:

- **[Python 3.10 or later](https://www.python.org/downloads/)**
- **[Git](https://git-scm.com/downloads)**
- **[Visual Studio Code](https://code.visualstudio.com/)** (recommended)

### 2. Clone the Repository

```bash
git clone https://github.com/cfeng-dev/ship-trajectory-prediction.git
cd ship-trajectory-prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

After activation, the terminal should display:

```text
(.venv)
```

> **Important:** Before installing packages or running Python scripts, verify that `(.venv)` appears at the beginning of the terminal. Otherwise, packages may be installed into the global Python environment instead of the project's virtual environment.

### 5. Select the Python Interpreter in Visual Studio Code

1. Press **Ctrl + Shift + P**.
2. Search for **Python: Select Interpreter**.
3. Select:

```text
.venv\Scripts\python.exe
```

### 6. Install the Project

```bash
pip install -e .
```

### 7. Install the C++ Toolchain

CmdStan requires a working C++17 toolchain to compile Stan models.

Windows users can install the required GNU C++ toolchain with CmdStanPy:

```bash
python -m cmdstanpy.install_cxx_toolchain
```

This command installs the RTools/MinGW toolchain used by CmdStanPy to compile Stan models on Windows.

For macOS and Linux setup instructions, please refer to the [official Stan installation guide](https://mc-stan.org/install/#prerequisite-c17-toolchain).

### 8. Install CmdStan

Download and install the latest supported version of CmdStan:

```bash
python -m cmdstanpy.install_cmdstan
```

### 9. Verify the Installation

With the virtual environment activated, start the Python interpreter from the terminal:

```bash
python
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
python -m cmdstanpy.install_cmdstan
```

For more detailed diagnostic output, run:

```bash
python -m cmdstanpy.install_cmdstan --verbose --cores 1
```

---

## Managing Dependencies

Dependencies are declared in `pyproject.toml`, which is the single source of
truth for the project environment.

### Install the Project Dependencies

```bash
pip install -e .
```

### Install All Development Dependencies

```bash
pip install -e ".[dev]"
```

---

## Command-Line Tools

Installing the project creates the simulator command:

```bash
ship-simulator
```

This command starts the interactive ship trajectory simulator.

Alternatively, the simulator can be started by running `cli.py` directly from
the project root:

```bash
python src/ship_trajectory_prediction/simulation/cli.py
```

The project must be installed first (`pip install -e .`) so that the package
imports can be resolved.

---

## Code Quality and Tests

GitHub Actions runs the tests on Python 3.10 and 3.12 for every push and pull
request. Run the checks locally from the project root with:

```bash
pytest
```

To apply Ruff formatting automatically:

```bash
ruff format .
```
