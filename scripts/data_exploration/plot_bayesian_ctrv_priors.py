"""Plot and report the configured Bayesian CTRV prior distributions."""

from __future__ import annotations

import matplotlib.pyplot as plt

import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.validation.prior_plotting as prior_plotting
from bayestraj.validation.prior_reporting import print_prior_report

PRIORS = bayesian_model.BayesianCTRVPriors()


def main():
    """Print the configuration and show the configured prior figures."""
    print_prior_report(PRIORS)
    figures = {}
    for curve in prior_plotting.build_prior_curves(PRIORS):
        figure = prior_plotting.create_prior_figure(curve)
        figures[curve.filename_stem] = figure
        plt.show(block=True)
        plt.close(figure)
    return figures


if __name__ == "__main__":
    main()
