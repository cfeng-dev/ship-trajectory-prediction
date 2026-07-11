"""
@file cmdstan_check.py
@description Verifies the CmdStan installation by compiling and running a minimal Stan model.
@date Created on: 01.07.2026
@author C.Feng
"""

from pathlib import Path

from cmdstanpy import CmdStanModel, cmdstan_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAN_FILE = PROJECT_ROOT / "stan" / "examples" / "test_cmdstan.stan"


def main():
    print("=" * 60)
    print("CmdStan Installation Test")
    print("=" * 60)

    print(f"CmdStan path : {cmdstan_path()}")
    print(f"Stan model   : {STAN_FILE}")

    if not STAN_FILE.exists():
        raise FileNotFoundError(f"Stan model not found:\n{STAN_FILE}")

    # Compile model
    model = CmdStanModel(stan_file=str(STAN_FILE))

    print("\n✓ Compilation successful")

    # Draw posterior samples
    fit = model.sample(
        chains=4,
        iter_warmup=500,
        iter_sampling=1000,
        seed=42,
        show_progress=True,
        refresh=100,
    )

    print("\n✓ Sampling successful")

    print("\nPosterior summary:")
    print(fit.summary())

    print("\nEnvironment is ready for Bayesian modeling.")


if __name__ == "__main__":
    main()
