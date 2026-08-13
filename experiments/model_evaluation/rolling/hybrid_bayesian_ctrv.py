"""Evaluate hybrid Bayesian CTRV forecasts across rolling windows."""

if __package__:
    from .bayesian_ctrv import run_cli
else:
    from bayesian_ctrv import run_cli

MODEL_VARIANT = "hybrid"


def main():
    """Run the shared rolling evaluation with the hybrid Bayesian model."""
    run_cli(model_variant=MODEL_VARIANT, description=__doc__)


if __name__ == "__main__":
    main()
