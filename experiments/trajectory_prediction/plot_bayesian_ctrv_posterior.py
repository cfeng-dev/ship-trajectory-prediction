"""Fit Bayesian CTRV and display variational posterior plots sequentially."""

import argparse

import matplotlib.pyplot as plt

from ship_trajectory_prediction.evaluation.posterior import (
    plot_scalar_posterior,
    plot_state_credible_band,
    plot_state_posterior_at_time,
)
from ship_trajectory_prediction.evaluation.reporting import (
    posterior_parameter_summary,
    print_prediction_setup,
    print_variational_diagnostics,
)
from ship_trajectory_prediction.models.bayesian_ctrv import (
    NOISE_PARAMETER_NAMES,
    fit_bayesian_ctrv_model,
    variational_converged,
)
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import prepare_trajectory_window
from ship_trajectory_prediction.trajectory.io import read_ship_data

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

# Data and posterior selection
RUN_ID = 1
START_INDEX = 0
OBSERVATION_COUNT = 20
PREDICTION_COUNT = 5
POSTERIOR_TIME_INDICES = (0, OBSERVATION_COUNT - 1)
CREDIBLE_INTERVAL = 0.9

# Variational inference
VI_ITER = 20_000
VI_GRAD_SAMPLES = 1
VI_ELBO_SAMPLES = 100
VI_ETA = 1.0
VI_ADAPT_ITER = 50
VI_TOL_REL_OBJ = 0.01
VI_EVAL_ELBO = 100
VI_DRAWS = 1_000


def main(*, vi_algorithm="meanfield", seed=42, require_converged=False):
    """Fit one recorded window and display only posterior diagnostics."""
    trajectory_data = read_ship_data(DATA_FILE, run_id=RUN_ID)
    window = prepare_trajectory_window(
        trajectory_data,
        observation_count=OBSERVATION_COUNT,
        prediction_count=PREDICTION_COUNT,
        start_index=START_INDEX,
    )
    print_prediction_setup(
        "Bayesian CTRV Variational Posterior Visualization",
        data_file=DATA_FILE,
        run_id=RUN_ID,
        window=window,
        extra_rows=[
            ("Inference method", "VI"),
            ("VI algorithm", vi_algorithm),
            ("Seed", seed),
            (
                "Posterior time indices",
                ", ".join(str(index) for index in POSTERIOR_TIME_INDICES),
            ),
        ],
    )

    fit = fit_bayesian_ctrv_model(
        window,
        algorithm=vi_algorithm,
        iter=VI_ITER,
        grad_samples=VI_GRAD_SAMPLES,
        elbo_samples=VI_ELBO_SAMPLES,
        eta=VI_ETA,
        adapt_iter=VI_ADAPT_ITER,
        tol_rel_obj=VI_TOL_REL_OBJ,
        eval_elbo=VI_EVAL_ELBO,
        draws=VI_DRAWS,
        seed=seed,
        require_converged=require_converged,
    )
    converged = variational_converged(fit)
    print_variational_diagnostics(fit)
    print(f"CmdStan convergence criterion met: {converged}")
    if not converged:
        print(
            "WARNING: Treat these approximate posterior plots as preliminary; "
            "the VI convergence criterion was not met."
        )

    print("\nApproximate posterior parameter summary:")
    print(posterior_parameter_summary(fit, NOISE_PARAMETER_NAMES))
    _show_bayesian_ctrv_posterior_plots(
        fit,
        window,
        selected_time_indices=POSTERIOR_TIME_INDICES,
    )


def _show_bayesian_ctrv_posterior_plots(
    fit,
    window,
    *,
    selected_time_indices=POSTERIOR_TIME_INDICES,
):
    """Display posterior figures one at a time without saving them."""
    observed = window.observed_slice
    time_values = window.time_seconds[observed]

    for variable_name in NOISE_PARAMETER_NAMES:
        figure, _ = plot_scalar_posterior(
            fit,
            variable_name,
            credible_interval=CREDIBLE_INTERVAL,
        )
        _show_and_close(figure)

    figure, _ = plot_state_credible_band(
        fit,
        "speed_state",
        time_values,
        credible_interval=CREDIBLE_INTERVAL,
        observed_values=window.gps_speed_mps[observed],
    )
    _show_and_close(figure)

    figure, _ = plot_state_credible_band(
        fit,
        "turn_rate_state",
        time_values,
        credible_interval=CREDIBLE_INTERVAL,
    )
    _show_and_close(figure)

    for time_index in selected_time_indices:
        for variable_name in ("speed_state", "turn_rate_state"):
            figure, _ = plot_state_posterior_at_time(
                fit,
                variable_name,
                time_index,
                time_values=time_values,
                credible_interval=CREDIBLE_INTERVAL,
            )
            _show_and_close(figure)


def _show_and_close(figure):
    """Block until one posterior figure is closed, then release it."""
    plt.show(block=True)
    plt.close(figure)


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vi-algorithm",
        choices=("meanfield", "fullrank"),
        default="meanfield",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-converged",
        action="store_true",
        help="Abort instead of plotting if CmdStan reports non-converged VI.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(
        vi_algorithm=arguments.vi_algorithm,
        seed=arguments.seed,
        require_converged=arguments.require_converged,
    )
