"""
@file bernoulli.py
@description Runs a Bayesian Bernoulli model using CmdStanPy to estimate the probability of success from binary observations.
@date Created on: 01.07.2026
@author C.Feng
"""

from pathlib import Path
from cmdstanpy import CmdStanModel

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to the Stan model
STAN_FILE = PROJECT_ROOT / "stan" / "examples" / "bernoulli.stan"


def main():
    # Compile the Stan model
    model = CmdStanModel(stan_file=str(STAN_FILE))

    # Example binary observations:
    # 1 = success
    # 0 = failure
    data = {
        "N": 10,
        "y": [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    }

    # Draw posterior samples
    fit = model.sample(
        data=data,
        chains=4,
        iter_warmup=500,
        iter_sampling=1000,
        seed=42,
    )

    # Print posterior summary
    print(fit.summary())

    # Extract and print posterior mean of theta
    print("\nPosterior mean of theta:")
    print(fit.stan_variable("theta").mean())


if __name__ == "__main__":
    main()
