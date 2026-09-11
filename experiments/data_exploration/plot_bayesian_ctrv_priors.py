"""Plot and report the configured Bayesian CTRV prior distributions."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.validation.prior_plotting as prior_plotting
from bayestraj.validation.prior_reporting import print_prior_report

PRIORS = bayesian_model.BayesianCTRVPriors()


def main(argv=None):
    """Print the configuration and show the configured prior figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    arguments = parser.parse_args(argv)
    print_prior_report(PRIORS)
    curves = prior_plotting.build_prior_curves(PRIORS)
    if arguments.no_show:
        figures = prior_plotting.create_individual_figures(curves)
        for figure in figures.values():
            plt.close(figure)
        return figures
    figures = {}
    for curve in curves:
        figure = prior_plotting.create_prior_figure(curve)
        figures[curve.filename_stem] = figure
        plt.show(block=True)
        plt.close(figure)
    return figures


if __name__ == "__main__":
    main()
