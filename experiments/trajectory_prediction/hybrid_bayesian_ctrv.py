"""Run one hybrid Bayesian CTRV trajectory prediction."""

if __package__:
    from .bayesian_ctrv import run_cli
else:
    from bayesian_ctrv import run_cli

MODEL_VARIANT = "hybrid"


def main():
    """Run the shared trajectory-prediction pipeline with the hybrid model."""
    run_cli(model_variant=MODEL_VARIANT, description=__doc__)


if __name__ == "__main__":
    main()
