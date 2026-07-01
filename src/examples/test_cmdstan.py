from pathlib import Path
from cmdstanpy import CmdStanModel, cmdstan_path


# Project root directory:
# src/examples/test_cmdstan.py -> parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Stan model to compile
MODEL_NAME = "bernoulli"


def main():
    print("=" * 60)
    print("CmdStan Installation Test")
    print("=" * 60)

    print(f"CmdStan path : {cmdstan_path()}")

    stan_file = PROJECT_ROOT / "stan" / "examples" / f"{MODEL_NAME}.stan"

    print(f"Stan model   : {stan_file}")

    if not stan_file.exists():
        raise FileNotFoundError(
            f"Stan model not found:\n{stan_file}\n\n"
            f"Please make sure the file exists at:\n"
            f"stan/examples/{MODEL_NAME}.stan"
        )

    model = CmdStanModel(stan_file=str(stan_file))

    print("\n✅ Compilation successful")
    print(f"Executable   : {model.exe_file}")

    print("\nEnvironment is ready for Bayesian modeling.")


if __name__ == "__main__":
    main()