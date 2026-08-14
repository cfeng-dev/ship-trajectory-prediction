"""Fit Bayesian CTRV and display variational posterior plots sequentially."""

import argparse

from ship_trajectory_prediction.models.bayesian_ctrv import (
    DEFAULT_VI_ADAPT_ITER,
    NOISE_PARAMETER_NAMES,
    BayesianCTRVPriors,
    fit_bayesian_ctrv_model,
    variational_converged,
)
from ship_trajectory_prediction.observations import prepare_trajectory_window
from ship_trajectory_prediction.observations.io import read_ship_data
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.validation.posterior_plotting import (
    show_bayesian_ctrv_posterior_plots,
    show_bayesian_ctrv_prior_update_plots,
)
from ship_trajectory_prediction.validation.reporting import (
    posterior_parameter_summary,
    print_prediction_setup,
    print_variational_diagnostics,
)

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

# Data and posterior selection
RUN_ID = 1
START_INDEX = 0
OBSERVATION_COUNT = 20
PREDICTION_COUNT = 5
PRIORS = BayesianCTRVPriors(
    speed_initial_prior_mean=3.524,
    speed_initial_prior_scale=0.365,
    turn_rate_initial_prior_mean=0.0,
    turn_rate_state_prior_scale=0.001698,
)
POSTERIOR_TIME_INDICES = (0, OBSERVATION_COUNT - 1)
CREDIBLE_INTERVAL = 0.9
PRIOR_UPDATE_OBSERVATION_COUNTS = (5, 10, 15, OBSERVATION_COUNT)
NOISE_PRIOR_SCALES = {
    name: getattr(PRIORS, f"{name}_prior_scale") for name in NOISE_PARAMETER_NAMES
}

# Variational inference
VI_ITER = 20_000
VI_GRAD_SAMPLES = 1
VI_ELBO_SAMPLES = 100
VI_ETA = 1.0
VI_ADAPT_ITER = DEFAULT_VI_ADAPT_ITER
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
            (
                "Prior update counts",
                ", ".join(str(count) for count in PRIOR_UPDATE_OBSERVATION_COUNTS),
            ),
            ("Noise priors", "Fixed half-normal distributions"),
            ("Initial speed prior", "Fixed across every prefix"),
            ("Heading/turn centers", "Position-informed for each prefix"),
        ],
    )

    fit = fit_bayesian_ctrv_model(
        window,
        priors=PRIORS,
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
    print(
        "GPS speed was not used for fitting and is plotted only as an "
        "external post-fit reference."
    )
    posterior_fits = _fit_prior_update_posteriors(
        trajectory_data,
        final_fit=fit,
        vi_algorithm=vi_algorithm,
        seed=seed,
        require_converged=require_converged,
    )
    show_bayesian_ctrv_prior_update_plots(
        posterior_fits,
        prior_scales=NOISE_PRIOR_SCALES,
        credible_interval=CREDIBLE_INTERVAL,
    )
    show_bayesian_ctrv_posterior_plots(
        fit,
        window,
        selected_time_indices=POSTERIOR_TIME_INDICES,
        credible_interval=CREDIBLE_INTERVAL,
        prior_scales=NOISE_PRIOR_SCALES,
        include_speed_gps_reference=True,
    )


def _fit_prior_update_posteriors(
    trajectory_data,
    *,
    final_fit,
    vi_algorithm,
    seed,
    require_converged,
):
    """Fit chronological prefixes while retaining one fixed noise prior."""
    observation_counts = tuple(sorted(set(PRIOR_UPDATE_OBSERVATION_COUNTS)))
    if not observation_counts or observation_counts[-1] != OBSERVATION_COUNT:
        raise ValueError(
            "PRIOR_UPDATE_OBSERVATION_COUNTS must end with OBSERVATION_COUNT."
        )
    if observation_counts[0] < 2:
        raise ValueError("Prior-update fits require at least two observations.")

    posterior_fits = {OBSERVATION_COUNT: final_fit}
    for observation_count in observation_counts[:-1]:
        print(
            "Fitting prior-update prefix: "
            f"n={observation_count} chronological observations"
        )
        prefix_window = prepare_trajectory_window(
            trajectory_data,
            observation_count=observation_count,
            prediction_count=PREDICTION_COUNT,
            start_index=START_INDEX,
        )
        prefix_fit = fit_bayesian_ctrv_model(
            prefix_window,
            priors=PRIORS,
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
        prefix_converged = variational_converged(prefix_fit)
        print(f"Prefix n={observation_count} converged: {prefix_converged}")
        posterior_fits[observation_count] = prefix_fit
    return posterior_fits


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
