# Examples

This directory contains small scripts for plotting simulated trajectory data,
verifying the project setup, and demonstrating basic CmdStanPy usage.

Run the scripts from the project root using `uv`. The CmdStanPy examples also
require CmdStan and a compatible C++ toolchain.

## Plot the simulated trajectory

Load the included simulated example dataset, resample it, and plot the selected
trajectory run:

```bash
uv run python examples/plot_simulated_trajectory.py
```

## Verify the CmdStan installation

Compile and sample from a minimal Stan model to check that CmdStan and the C++
toolchain are working correctly:

```bash
uv run python examples/cmdstan_check.py
```

## Run the Bernoulli example

Fit a simple Bayesian Bernoulli model to binary observations:

```bash
uv run python examples/bernoulli.py
```
