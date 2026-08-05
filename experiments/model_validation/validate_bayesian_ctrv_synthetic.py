"""Validate Bayesian CTRV inference on synthetic trajectories."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
import pandas as pd

from ship_trajectory_prediction.evaluation.prediction_plotting import plot_prediction
from ship_trajectory_prediction.evaluation.reporting import (
    posterior_parameter_summary,
    posterior_variable_samples,
    print_prediction_setup,
    print_variational_diagnostics,
)
from ship_trajectory_prediction.models.bayesian_ctrv import (
    NOISE_PARAMETER_NAMES,
    BayesianCTRVPriors,
    VIRunResult,
    build_stan_data,
    compare_vi_runs,
    fit_bayesian_ctrv_model,
    summarize_predictions,
    variational_converged,
)
from ship_trajectory_prediction.simulation.synthetic_ctrv import (
    simulate_synthetic_ctrv_data,
)
from ship_trajectory_prediction.trajectory import prepare_trajectory_window

OBSERVATION_COUNT = 12
PREDICTION_COUNT = 4
PRIORS = BayesianCTRVPriors()
VI_ITER = 20_000
VI_GRAD_SAMPLES = 1
VI_ELBO_SAMPLES = 100
VI_ETA = 1.0
VI_ADAPT_ITER = 50
VI_TOL_REL_OBJ = 0.01
VI_EVAL_ELBO = 100
VI_DRAWS = 1_000
CREDIBLE_INTERVAL = 0.9
SYNTHETIC_SEED = 2026


def main(
    *,
    algorithms=("meanfield",),
    seeds=(11, 22, 33),
    observation_count=OBSERVATION_COUNT,
    prediction_count=PREDICTION_COUNT,
    vi_iter=VI_ITER,
    vi_draws=VI_DRAWS,
    require_converged=True,
    show_console=False,
    show_plot=False,
):
    """Run several VI initializations and report stability and recovery."""
    row_count = observation_count + prediction_count
    increments = np.resize(np.asarray([4.0, 5.0, 6.0, 5.0]), row_count - 1)
    time_seconds = np.concatenate(([0.0], np.cumsum(increments)))
    synthetic_data = simulate_synthetic_ctrv_data(
        count=row_count,
        time_seconds=time_seconds,
        seed=SYNTHETIC_SEED,
    )
    window = prepare_trajectory_window(
        synthetic_data,
        observation_count=observation_count,
        prediction_count=prediction_count,
    )
    stan_data = build_stan_data(window, priors=PRIORS)
    print_prediction_setup(
        "Synthetic Bayesian CTRV Validation",
        data_file=f"synthetic CTRV data (seed={SYNTHETIC_SEED})",
        run_id=0,
        window=window,
        extra_rows=[
            ("VI algorithms", ", ".join(algorithms)),
            ("Seeds", ", ".join(str(seed) for seed in seeds)),
            ("Observation model", "position only"),
            (
                "Initial speed center",
                f"{stan_data['speed_initial_prior_mean']:.3f} m/s (from positions)",
            ),
            (
                "Initial turn-rate center",
                f"{stan_data['turn_rate_initial_prior_mean']:.5f} rad/s",
            ),
            (
                "Turn-rate state prior scale",
                f"{stan_data['turn_rate_state_prior_scale']:.5f} rad/s",
            ),
            ("Turn-rate limit", f"{stan_data['turn_rate_limit']:.5f} rad/s"),
            ("VI tolerance", VI_TOL_REL_OBJ),
        ],
    )

    runs = []
    for algorithm in algorithms:
        for seed in seeds:
            started = perf_counter()
            fit = fit_bayesian_ctrv_model(
                window,
                priors=PRIORS,
                algorithm=algorithm,
                iter=vi_iter,
                grad_samples=VI_GRAD_SAMPLES,
                elbo_samples=VI_ELBO_SAMPLES,
                eta=VI_ETA,
                adapt_iter=VI_ADAPT_ITER,
                tol_rel_obj=VI_TOL_REL_OBJ,
                eval_elbo=VI_EVAL_ELBO,
                draws=vi_draws,
                seed=seed,
                require_converged=require_converged,
                show_console=show_console,
            )
            runtime_seconds = perf_counter() - started
            run = VIRunResult(
                seed=seed,
                algorithm=algorithm,
                fit=fit,
                runtime_seconds=runtime_seconds,
                converged=variational_converged(fit),
            )
            runs.append(run)
            print(f"\n{algorithm} seed {seed}:")
            print_variational_diagnostics(fit)

    comparison = compare_vi_runs(
        runs,
        window,
        credible_interval=CREDIBLE_INTERVAL,
    )
    display_columns = [
        "algorithm",
        "seed",
        "converged",
        "runtime_seconds",
        "final_iteration",
        "final_elbo",
        "endpoint_x_m",
        "endpoint_y_m",
        "ade_m",
        "fde_m",
        "mean_interval_width_m",
        "radial_coverage",
    ]
    print("\nVI run comparison:")
    print(comparison[display_columns].round(4).to_string(index=False))

    stability_columns = [
        "endpoint_x_m",
        "endpoint_y_m",
        "ade_m",
        "fde_m",
        "mean_interval_width_m",
        "radial_coverage",
    ]
    print("\nAcross-seed prediction stability:")
    print(
        comparison.groupby("algorithm")[stability_columns]
        .agg(["mean", "std"])
        .round(4)
        .to_string()
    )

    for algorithm in algorithms:
        run = next(result for result in runs if result.algorithm == algorithm)
        print(f"\n{algorithm} posterior noise summary (seed {run.seed}):")
        print(posterior_parameter_summary(run.fit, NOISE_PARAMETER_NAMES).round(6))
        _print_synthetic_recovery(
            run.fit,
            window,
            synthetic_data,
            credible_interval=CREDIBLE_INTERVAL,
        )

    if show_plot:
        plot_prediction(
            window,
            runs[0].fit,
            state_prediction_variable_names=(
                "x_state_prediction",
                "y_state_prediction",
            ),
        )
    return comparison


def _print_synthetic_recovery(
    fit,
    window,
    synthetic_data,
    *,
    credible_interval,
):
    """Print noise recovery plus latent-state and future interval coverage."""
    lower_probability = (1 - credible_interval) / 2
    upper_probability = 1 - lower_probability

    noise_rows = []
    for parameter_name in NOISE_PARAMETER_NAMES:
        samples = posterior_variable_samples(fit, parameter_name)
        noise_rows.append(
            {
                "parameter": parameter_name,
                "truth": synthetic_data[f"{parameter_name}_true"].iloc[0],
                "posterior_mean": np.mean(samples),
                "posterior_std": np.std(samples, ddof=1),
            }
        )
    print("\nSynthetic noise recovery:")
    print(pd.DataFrame(noise_rows).round(6).to_string(index=False))

    state_truth_columns = {
        "x_state": "x_true",
        "y_state": "y_true",
        "speed_state": "speed_true",
        "heading_state": "heading_true",
        "turn_rate_state": "turn_rate_true",
    }
    observed_truth = synthetic_data.iloc[: window.observation_count]
    state_rows = []
    for variable_name, truth_column in state_truth_columns.items():
        samples = posterior_variable_samples(fit, variable_name)
        truth = observed_truth[truth_column].to_numpy(dtype=float)
        median = np.median(samples, axis=0)
        lower = np.quantile(samples, lower_probability, axis=0)
        upper = np.quantile(samples, upper_probability, axis=0)
        state_rows.append(
            {
                "state": variable_name,
                "median_rmse": np.sqrt(np.mean((median - truth) ** 2)),
                "interval_coverage": np.mean((truth >= lower) & (truth <= upper)),
            }
        )
    print("\nSynthetic latent-state recovery:")
    print(pd.DataFrame(state_rows).round(4).to_string(index=False))

    future_truth = synthetic_data.iloc[window.observation_count :]
    prediction = summarize_predictions(
        fit,
        window,
        credible_interval=credible_interval,
    )
    future_rows = []
    for prefix, truth_column in (
        ("x_state", "x_true"),
        ("y_state", "y_true"),
        ("speed_state", "speed_true"),
        ("heading_state", "heading_true"),
        ("turn_rate_state", "turn_rate_true"),
    ):
        truth = future_truth[truth_column].to_numpy(dtype=float)
        covered = (truth >= prediction[f"{prefix}_lower"]) & (
            truth <= prediction[f"{prefix}_upper"]
        )
        future_rows.append(
            {
                "future_state": prefix,
                "interval_coverage": covered.mean(),
            }
        )
    print("\nSynthetic future-state coverage:")
    print(pd.DataFrame(future_rows).round(4).to_string(index=False))


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("meanfield", "fullrank"),
        default=["meanfield"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--observations", type=int, default=OBSERVATION_COUNT)
    parser.add_argument("--predictions", type=int, default=PREDICTION_COUNT)
    parser.add_argument("--vi-iter", type=int, default=VI_ITER)
    parser.add_argument("--vi-draws", type=int, default=VI_DRAWS)
    parser.add_argument(
        "--allow-nonconverged",
        action="store_true",
        help="Keep explicitly labelled VI runs that miss CmdStan convergence.",
    )
    parser.add_argument("--show-console", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(
        algorithms=tuple(arguments.algorithms),
        seeds=tuple(arguments.seeds),
        observation_count=arguments.observations,
        prediction_count=arguments.predictions,
        vi_iter=arguments.vi_iter,
        vi_draws=arguments.vi_draws,
        require_converged=not arguments.allow_nonconverged,
        show_console=arguments.show_console,
        show_plot=arguments.plot,
    )
