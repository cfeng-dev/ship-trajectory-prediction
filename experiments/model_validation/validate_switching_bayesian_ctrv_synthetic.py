"""Validate Switching Bayesian CTRV with a fixed four-phase synthetic track."""

import argparse
from time import perf_counter

import numpy as np
import pandas as pd

from ship_trajectory_prediction.evaluation.reporting import (
    posterior_variable_samples,
    print_variational_diagnostics,
)
from ship_trajectory_prediction.models.bayesian_ctrv import (
    BayesianCTRVPriors,
    VIRunResult,
    simulate_position_observations,
    variational_converged,
)
from ship_trajectory_prediction.models.switching_bayesian_ctrv import (
    MODE_NAMES,
    SwitchingCTRVConfig,
    compare_switching_vi_runs,
    compile_switching_bayesian_ctrv_model,
    fit_switching_bayesian_ctrv_model,
    summarize_mode_probabilities,
)
from ship_trajectory_prediction.simulation.synthetic_switching_ctrv import (
    simulate_synthetic_switching_ctrv_data,
)
from ship_trajectory_prediction.trajectory import prepare_trajectory_window

OBSERVATION_COUNT = 27
PREDICTION_COUNT = 4
DATA_SEED = 2026
VI_SEEDS = (11, 22, 33)
VI_ALGORITHM = "meanfield"
VI_ITER = 20_000
VI_DRAWS = 500
CREDIBLE_INTERVAL = 0.9


def main(
    *,
    vi_seeds=VI_SEEDS,
    vi_algorithm=VI_ALGORITHM,
    vi_iter=VI_ITER,
    vi_draws=VI_DRAWS,
):
    """Run multiple VI seeds without writing result artifacts."""
    switching_config = SwitchingCTRVConfig()
    data = simulate_synthetic_switching_ctrv_data(
        seed=DATA_SEED,
        stop_decay_time_seconds=switching_config.stop_speed_decay_time,
        stop_turn_decay_time_seconds=switching_config.stop_turn_decay_time,
    )
    model_data = data.loc[
        :, ["time", "run_id", "gps_latitude", "gps_longitude", "gps_speed"]
    ].copy()
    window = prepare_trajectory_window(
        model_data,
        observation_count=OBSERVATION_COUNT,
        prediction_count=PREDICTION_COUNT,
    )
    observations = simulate_position_observations(
        window,
        additional_noise_std_m=0.0,
        seed=DATA_SEED,
    )
    priors = BayesianCTRVPriors()
    runs = []
    mode_tables = []
    # Keep one-time compilation outside every per-seed runtime measurement.
    compile_switching_bayesian_ctrv_model()
    for seed in vi_seeds:
        print(f"\nSynthetic Switching CTRV VI seed {seed}")
        started = perf_counter()
        fit = fit_switching_bayesian_ctrv_model(
            window,
            priors=priors,
            config=switching_config,
            position_observations=observations,
            algorithm=vi_algorithm,
            iter=vi_iter,
            draws=vi_draws,
            seed=seed,
            require_converged=False,
        )
        runtime_seconds = perf_counter() - started
        converged = variational_converged(fit)
        print_variational_diagnostics(fit)
        print(f"CmdStan convergence criterion met: {converged}")
        runs.append(
            VIRunResult(
                seed=seed,
                algorithm=vi_algorithm,
                fit=fit,
                runtime_seconds=runtime_seconds,
                converged=converged,
            )
        )
        mode_table = summarize_mode_probabilities(
            fit,
            window,
            credible_interval=CREDIBLE_INTERVAL,
        )
        mode_table.insert(0, "seed", seed)
        mode_tables.append(mode_table)

    comparison = compare_switching_vi_runs(
        runs,
        window,
        credible_interval=CREDIBLE_INTERVAL,
    )
    modes = pd.concat(mode_tables, ignore_index=True)
    recovery = _mode_recovery_summary(modes, data)
    transition_stability = _transition_stability(runs)
    reconstruction = _trajectory_reconstruction_summary(runs, data)
    mode_stability = _mode_probability_stability(modes, data)
    predictive_stability, endpoint_stability = _posterior_predictive_stability(
        runs,
        window,
    )
    _print_results(
        comparison,
        recovery,
        transition_stability,
        reconstruction,
        mode_stability,
        predictive_stability,
        endpoint_stability,
    )

    return (
        comparison,
        recovery,
        transition_stability,
        reconstruction,
        mode_stability,
        predictive_stability,
        endpoint_stability,
    )


def _mode_recovery_summary(mode_tables, data):
    """Compare inferred probabilities with truth used only after fitting."""
    observed_truth = data["mode_true"].to_numpy(dtype=int)[1:OBSERVATION_COUNT]
    rows = []
    for seed, table in mode_tables.groupby("seed", sort=True):
        if len(table) != len(observed_truth):
            raise ValueError("Mode summary and synthetic truth are not aligned.")
        for mode_index, mode_name in MODE_NAMES.items():
            mask = observed_truth == mode_index
            rows.append(
                {
                    "seed": seed,
                    "true_mode": mode_name,
                    "transition_count": int(np.count_nonzero(mask)),
                    "mean_assigned_probability": float(
                        table.loc[mask, f"{mode_name}_probability"].mean()
                    ),
                    "most_likely_fraction": float(
                        np.mean(
                            table.loc[mask, "most_likely_mode"].to_numpy() == mode_index
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _transition_stability(runs):
    """Summarize posterior transition means and their spread across VI seeds."""
    matrices = []
    for run in runs:
        samples = posterior_variable_samples(run.fit, "transition_probability")
        matrices.append(np.mean(samples, axis=0))
    matrices = np.asarray(matrices)
    rows = []
    for source_index, source_name in MODE_NAMES.items():
        for target_index, target_name in MODE_NAMES.items():
            values = matrices[:, source_index - 1, target_index - 1]
            rows.append(
                {
                    "source_mode": source_name,
                    "target_mode": target_name,
                    "across_seed_mean": float(np.mean(values)),
                    "across_seed_std": float(
                        np.std(values, ddof=1) if len(values) > 1 else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _trajectory_reconstruction_summary(runs, data):
    """Evaluate observed latent x/y reconstruction against held-back truth."""
    x_true = data["x_true"].to_numpy(dtype=float)[:OBSERVATION_COUNT]
    y_true = data["y_true"].to_numpy(dtype=float)[:OBSERVATION_COUNT]
    lower_probability = (1 - CREDIBLE_INTERVAL) / 2
    upper_probability = 1 - lower_probability
    rows = []
    for run in runs:
        x_samples = _finite_draw_matrix(
            run.fit,
            "x_state",
            expected_width=OBSERVATION_COUNT,
        )
        y_samples = _finite_draw_matrix(
            run.fit,
            "y_state",
            expected_width=OBSERVATION_COUNT,
        )
        if x_samples.shape != y_samples.shape:
            raise ValueError("Synthetic x/y state draws must have matching shapes.")

        x_median = np.median(x_samples, axis=0)
        y_median = np.median(y_samples, axis=0)
        position_error = np.hypot(x_median - x_true, y_median - y_true)
        draw_distance = np.hypot(
            x_samples - x_median,
            y_samples - y_median,
        )
        radial_radius = np.quantile(
            draw_distance,
            CREDIBLE_INTERVAL,
            axis=0,
        )
        x_width = np.quantile(x_samples, upper_probability, axis=0) - np.quantile(
            x_samples,
            lower_probability,
            axis=0,
        )
        y_width = np.quantile(y_samples, upper_probability, axis=0) - np.quantile(
            y_samples,
            lower_probability,
            axis=0,
        )
        rows.append(
            {
                "seed": run.seed,
                "algorithm": run.algorithm,
                "observed_state_ade_m": float(np.mean(position_error)),
                "observed_state_rmse_m": float(np.sqrt(np.mean(position_error**2))),
                "observed_state_final_error_m": float(position_error[-1]),
                "observed_state_radial_coverage": float(
                    np.mean(position_error <= radial_radius)
                ),
                "observed_state_mean_credible_radius_m": float(np.mean(radial_radius)),
                "observed_state_mean_marginal_interval_width_m": float(
                    np.mean(0.5 * (x_width + y_width))
                ),
            }
        )
    return pd.DataFrame(rows)


def _mode_probability_stability(mode_tables, data):
    """Measure cross-seed agreement of observed transition probabilities."""
    rows = []
    for transition_index, table in mode_tables.groupby(
        "transition_index",
        sort=True,
    ):
        destination_index = int(transition_index)
        true_mode = int(data["mode_true"].iloc[destination_index])
        most_likely_counts = table["most_likely_mode"].value_counts()
        consensus_mode = int(most_likely_counts.index[0])
        row = {
            "transition_index": destination_index,
            "time": table["time"].iloc[0],
            "t": float(table["t"].iloc[0]),
            "true_mode": true_mode,
            "true_mode_name": MODE_NAMES[true_mode],
            "seed_count": len(table),
            "consensus_most_likely_mode": consensus_mode,
            "consensus_most_likely_mode_name": MODE_NAMES[consensus_mode],
            "most_likely_mode_agreement": float(
                most_likely_counts.iloc[0] / len(table)
            ),
        }
        for _, mode_name in MODE_NAMES.items():
            values = table[f"{mode_name}_probability"].to_numpy(dtype=float)
            row[f"{mode_name}_probability_across_seed_mean"] = float(np.mean(values))
            row[f"{mode_name}_probability_across_seed_std"] = (
                _sample_standard_deviation(values)
            )
            row[f"{mode_name}_probability_across_seed_range"] = float(np.ptp(values))
        rows.append(row)
    return pd.DataFrame(rows)


def _posterior_predictive_stability(runs, window):
    """Summarize cross-seed future distributions and endpoint variation."""
    lower_probability = (1 - CREDIBLE_INTERVAL) / 2
    upper_probability = 1 - lower_probability
    x_medians = []
    y_medians = []
    prediction_radii = []
    interval_widths = []
    for run in runs:
        x_samples = _finite_draw_matrix(
            run.fit,
            "x_observation_prediction",
            expected_width=window.prediction_count,
        )
        y_samples = _finite_draw_matrix(
            run.fit,
            "y_observation_prediction",
            expected_width=window.prediction_count,
        )
        if x_samples.shape != y_samples.shape:
            raise ValueError(
                "Synthetic posterior-predictive x/y draws must have matching shapes."
            )
        x_median = np.median(x_samples, axis=0)
        y_median = np.median(y_samples, axis=0)
        x_medians.append(x_median)
        y_medians.append(y_median)
        prediction_radii.append(
            np.quantile(
                np.hypot(
                    x_samples - x_median,
                    y_samples - y_median,
                ),
                CREDIBLE_INTERVAL,
                axis=0,
            )
        )
        x_width = np.quantile(x_samples, upper_probability, axis=0) - np.quantile(
            x_samples,
            lower_probability,
            axis=0,
        )
        y_width = np.quantile(y_samples, upper_probability, axis=0) - np.quantile(
            y_samples,
            lower_probability,
            axis=0,
        )
        interval_widths.append(0.5 * (x_width + y_width))

    x_medians = np.asarray(x_medians)
    y_medians = np.asarray(y_medians)
    prediction_radii = np.asarray(prediction_radii)
    interval_widths = np.asarray(interval_widths)
    if x_medians.shape[0] == 0:
        raise ValueError("runs must contain at least one synthetic VI fit.")
    x_center = np.mean(x_medians, axis=0)
    y_center = np.mean(y_medians, axis=0)
    median_position_deviation = np.hypot(
        x_medians - x_center,
        y_medians - y_center,
    )
    prediction = window.prediction_slice
    prediction_times = window.time_seconds[prediction]
    origin_time = window.time_seconds[window.observation_count - 1]
    predictive = pd.DataFrame(
        {
            "time": window.timestamps[prediction],
            "t": prediction_times,
            "horizon_seconds": prediction_times - origin_time,
            "seed_count": x_medians.shape[0],
            "x_median_across_seed_mean_m": x_center,
            "x_median_across_seed_std_m": _column_sample_standard_deviation(x_medians),
            "y_median_across_seed_mean_m": y_center,
            "y_median_across_seed_std_m": _column_sample_standard_deviation(y_medians),
            "median_position_deviation_mean_m": np.mean(
                median_position_deviation,
                axis=0,
            ),
            "median_position_deviation_max_m": np.max(
                median_position_deviation,
                axis=0,
            ),
            "prediction_radius_across_seed_mean_m": np.mean(
                prediction_radii,
                axis=0,
            ),
            "prediction_radius_across_seed_std_m": (
                _column_sample_standard_deviation(prediction_radii)
            ),
            "marginal_interval_width_across_seed_mean_m": np.mean(
                interval_widths,
                axis=0,
            ),
            "marginal_interval_width_across_seed_std_m": (
                _column_sample_standard_deviation(interval_widths)
            ),
        }
    )
    endpoint = pd.DataFrame(
        [
            {
                "seed_count": x_medians.shape[0],
                "horizon_seconds": float(predictive["horizon_seconds"].iloc[-1]),
                "endpoint_x_m_mean": float(x_center[-1]),
                "endpoint_x_m_std": _sample_standard_deviation(x_medians[:, -1]),
                "endpoint_y_m_mean": float(y_center[-1]),
                "endpoint_y_m_std": _sample_standard_deviation(y_medians[:, -1]),
                "endpoint_position_deviation_mean_m": float(
                    np.mean(median_position_deviation[:, -1])
                ),
                "endpoint_position_deviation_max_m": float(
                    np.max(median_position_deviation[:, -1])
                ),
                "endpoint_prediction_radius_m_mean": float(
                    np.mean(prediction_radii[:, -1])
                ),
                "endpoint_prediction_radius_m_std": _sample_standard_deviation(
                    prediction_radii[:, -1]
                ),
                "endpoint_marginal_interval_width_m_mean": float(
                    np.mean(interval_widths[:, -1])
                ),
                "endpoint_marginal_interval_width_m_std": (
                    _sample_standard_deviation(interval_widths[:, -1])
                ),
            }
        ]
    )
    return predictive, endpoint


def _finite_draw_matrix(fit, variable_name, *, expected_width):
    """Return one finite posterior draw-by-time matrix."""
    samples = posterior_variable_samples(fit, variable_name)
    if (
        samples.ndim != 2
        or samples.shape[0] < 1
        or samples.shape[1] != expected_width
        or not np.all(np.isfinite(samples))
    ):
        raise ValueError(
            f"{variable_name} must contain finite draw-by-time samples with "
            f"width {expected_width}."
        )
    return samples


def _sample_standard_deviation(values):
    """Return sample standard deviation, or zero for one value."""
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1) if values.size > 1 else 0.0)


def _column_sample_standard_deviation(values):
    """Return column-wise sample deviations, or zeros for one row."""
    values = np.asarray(values, dtype=float)
    if values.shape[0] == 1:
        return np.zeros(values.shape[1], dtype=float)
    return np.std(values, axis=0, ddof=1)


def _print_results(
    comparison,
    recovery,
    transition_stability,
    reconstruction,
    mode_stability,
    predictive_stability,
    endpoint_stability,
):
    """Print accuracy, truth recovery, and cross-seed stability tables."""
    print("\nSynthetic prediction and VI stability:")
    print(comparison.round(3).to_string(index=False))
    print("\nObserved latent trajectory reconstruction:")
    print(reconstruction.round(3).to_string(index=False))
    print("\nQualitative mode recovery (truth used only for evaluation):")
    print(recovery.round(3).to_string(index=False))
    probability_std_columns = [
        f"{mode_name}_probability_across_seed_std" for mode_name in MODE_NAMES.values()
    ]
    print("\nObserved mode-probability stability:")
    print(
        mode_stability[
            [
                "transition_index",
                "true_mode_name",
                "most_likely_mode_agreement",
                *probability_std_columns,
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("\nTransition-matrix stability:")
    print(transition_stability.round(3).to_string(index=False))
    print("\nPosterior-predictive stability by horizon:")
    print(predictive_stability.round(3).to_string(index=False))
    print("\nPosterior-predictive endpoint stability:")
    print(endpoint_stability.round(3).to_string(index=False))


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=VI_SEEDS)
    parser.add_argument(
        "--vi-algorithm",
        choices=("meanfield", "fullrank"),
        default=VI_ALGORITHM,
    )
    parser.add_argument("--vi-iter", type=int, default=VI_ITER)
    parser.add_argument("--vi-draws", type=int, default=VI_DRAWS)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(
        vi_seeds=arguments.seeds,
        vi_algorithm=arguments.vi_algorithm,
        vi_iter=arguments.vi_iter,
        vi_draws=arguments.vi_draws,
    )
