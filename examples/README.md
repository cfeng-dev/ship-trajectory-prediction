# Examples

This directory contains small scripts for verifying the project setup and
demonstrating basic CmdStanPy usage.

Run the scripts from the project root with the virtual environment activated
and CmdStan installed.

## Verify the CmdStan installation

Compile and sample from a minimal Stan model to check that CmdStan and the C++
toolchain are working correctly:

```bash
python examples/verify_cmdstan_installation.py
```

## Run the Bernoulli example

Fit a simple Bayesian Bernoulli model to binary observations:

```bash
python examples/bernoulli.py
```
