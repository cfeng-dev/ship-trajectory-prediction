"""Fit the Switching Bayesian CTRV model to one recorded trajectory window."""

import argparse
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .config import ExperimentConfig
else:
    from config import ExperimentConfig

from ship_trajectory_prediction.evaluation.metrics import (
    evaluate_position_predictions,
    print_position_evaluation,
)
from ship_trajectory_prediction.evaluation.posterior import (
    plot_state_credible_band,
)
from ship_trajectory_prediction.evaluation.reporting import (
    posterior_parameter_summary,
    posterior_variable_samples,
    print_prediction_setup,
    print_variational_diagnostics,
)
from ship_trajectory_prediction.models.bayesian_ctrv import (
    NOISE_PARAMETER_NAMES,
    BayesianCTRVPriors,
    simulate_position_observations,
    variational_converged,
)
from ship_trajectory_prediction.models.switching_bayesian_ctrv import (
    MODE_NAMES,
    SwitchingCTRVConfig,
    fit_switching_bayesian_ctrv_model,
    summarize_mode_probabilities,
    summarize_switching_predictions,
    summarize_transition_probabilities,
)
from ship_trajectory_prediction.paths import project_path
from ship_trajectory_prediction.trajectory import prepare_trajectory_window
from ship_trajectory_prediction.trajectory.io import read_ship_data

DATA_FILE = project_path(
    "data/raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

EXPERIMENT = ExperimentConfig(
    run_id=1,  # Trajectory run to fit.
    start_index=0,  # First point of the selected window.
    observation_count=20,  # Position points used for fitting.
    prediction_count=5,  # Held-out future position points.
    additional_position_noise_std_m=2.0,  # Per x/y axis [m]; 0 disables.
    position_noise_seed=2026,  # Reproduces the added position noise.
    inference_method="vi",  # Switching inference currently supports VI only.
    inference_seed=42,  # Reproduces variational inference.
)
PRIORS = BayesianCTRVPriors(
    position_initial_prior_scale=5.0,  # Initial x/y uncertainty [m].
    speed_initial_prior_scale=0.75,  # Initial speed uncertainty [m/s].
    heading_initial_prior_scale=0.35,  # Initial heading uncertainty [rad].
    turn_rate_state_prior_scale=None,  # Derive from observed positions.
    sigma_position_gps_prior_scale=5.0,  # Measurement-noise scale [m].
    sigma_position_process_prior_scale=0.5,  # Position drift [m/sqrt(s)].
    sigma_speed_process_prior_scale=0.05,  # Speed drift [(m/s)/sqrt(s)].
    sigma_turn_rate_process_prior_scale=0.001,  # Turn drift [(rad/s)/sqrt(s)].
)
SWITCHING_CONFIG = SwitchingCTRVConfig(
    initial_mode_probability=(1 / 3, 1 / 3, 1 / 3),  # Stop, cruise, maneuver.
    alpha_transition=(  # Dirichlet row priors favor staying in one mode.
        (20.0, 1.0, 1.0),
        (1.0, 20.0, 1.0),
        (1.0, 1.0, 20.0),
    ),
    position_process_multiplier=(0.5, 1.0, 3.0),  # Per-mode position noise.
    speed_process_multiplier=(0.25, 1.0, 4.0),  # Per-mode speed noise.
    turn_rate_process_multiplier=(0.25, 1.0, 4.0),  # Per-mode turn noise.
    stop_speed_decay_time=20.0,  # Stop-mode speed decay time [s].
    stop_turn_decay_time=10.0,  # Stop-mode turn-rate decay time [s].
)
VI_CONFIG = {
    "algorithm": "meanfield",  # "meanfield" or "fullrank".
    "iter": 20_000,  # Maximum optimization iterations.
    "grad_samples": 1,  # Samples per gradient estimate.
    "elbo_samples": 100,  # Samples per ELBO estimate.
    "eta": 1.0,  # Initial step size.
    "adapt_iter": 50,  # Step-size adaptation iterations.
    "tol_rel_obj": 0.01,  # Relative ELBO stopping tolerance.
    "eval_elbo": 100,  # ELBO evaluation interval.
    "draws": 1_000,  # Posterior draws retained in memory.
    "require_converged": False,  # Allow preliminary non-converged VI.
}
CREDIBLE_INTERVAL = 0.9  # Central 90% posterior interval.


def main(
    *,
    inference_method=EXPERIMENT.inference_method,
    vi_algorithm=VI_CONFIG["algorithm"],
    seed=EXPERIMENT.inference_seed,
    position_noise_std_m=EXPERIMENT.additional_position_noise_std_m,
    position_noise_seed=EXPERIMENT.position_noise_seed,
    require_converged=VI_CONFIG["require_converged"],
    show_plots=True,
):
    """Fit one switching model and report diagnostics without writing files."""
    if (
        not isinstance(inference_method, str)
        or inference_method.strip().lower() != "vi"
    ):
        raise ValueError("Switching Bayesian CTRV currently supports only VI.")
    inference_method = inference_method.strip().lower()
    trajectory_data = read_ship_data(DATA_FILE, run_id=EXPERIMENT.run_id)
    window = prepare_trajectory_window(
        trajectory_data,
        observation_count=EXPERIMENT.observation_count,
        prediction_count=EXPERIMENT.prediction_count,
        start_index=EXPERIMENT.start_index,
    )
    position_observations = simulate_position_observations(
        window,
        additional_noise_std_m=position_noise_std_m,
        seed=position_noise_seed,
    )
    vi_config = {
        **VI_CONFIG,
        "algorithm": vi_algorithm,
        "require_converged": require_converged,
    }
    forecast_horizon_seconds = float(
        window.time_seconds[-1] - window.time_seconds[window.observation_count - 1]
    )
    _print_setup(
        window=window,
        inference_method=inference_method,
        vi_algorithm=vi_algorithm,
        inference_seed=seed,
        position_observations=position_observations,
        forecast_horizon_seconds=forecast_horizon_seconds,
    )

    fit_started = perf_counter()
    fit = fit_switching_bayesian_ctrv_model(
        window,
        priors=PRIORS,
        config=SWITCHING_CONFIG,
        position_observations=position_observations,
        seed=seed,
        **vi_config,
    )
    runtime_seconds = perf_counter() - fit_started
    converged = variational_converged(fit)
    print(f"\nModel fit and forecast runtime: {runtime_seconds:.2f} s")
    print_variational_diagnostics(fit)
    print(f"CmdStan convergence criterion met: {converged}")
    if not converged:
        print(
            "WARNING: Treat mode probabilities and forecasts as preliminary; "
            "the VI convergence criterion was not met."
        )

    print("\nPosterior noise-parameter summary:")
    print(posterior_parameter_summary(fit, NOISE_PARAMETER_NAMES))
    evaluation = evaluate_position_predictions(
        fit,
        window,
        credible_interval=CREDIBLE_INTERVAL,
        position_variable_names=(
            "x_observation_prediction",
            "y_observation_prediction",
        ),
    )
    print_position_evaluation(evaluation)
    print(
        "GPS speed was not used for fitting, prior construction, "
        "initialization, mode inference, or prediction."
    )

    gps_speed_reference_available = bool(np.all(np.isfinite(window.gps_speed_mps)))
    prediction_summary = summarize_switching_predictions(
        fit,
        window,
        credible_interval=CREDIBLE_INTERVAL,
        include_speed_gps_reference=gps_speed_reference_available,
    )
    mode_summary = summarize_mode_probabilities(
        fit,
        window,
        credible_interval=CREDIBLE_INTERVAL,
    )
    transition_summary = summarize_transition_probabilities(
        fit,
        credible_interval=CREDIBLE_INTERVAL,
    )
    if show_plots:
        _show_figures_sequentially(
            fit,
            window,
            position_observations,
            mode_summary,
            transition_summary,
            gps_speed_reference_available=gps_speed_reference_available,
        )
    return fit, prediction_summary, mode_summary, transition_summary, evaluation


def _print_setup(
    *,
    window,
    inference_method,
    vi_algorithm,
    inference_seed,
    position_observations,
    forecast_horizon_seconds,
):
    """Print concise reproducibility information for one switching fit."""
    noise_std_m = position_observations.additional_noise_std_m
    noise_description = (
        f"{noise_std_m:g} m (seed={position_observations.noise_seed})"
        if noise_std_m > 0
        else "disabled"
    )
    print_prediction_setup(
        "Switching Bayesian CTRV State-Space Prediction",
        data_file=DATA_FILE,
        run_id=EXPERIMENT.run_id,
        window=window,
        extra_rows=[
            ("Inference method", inference_method.upper()),
            ("VI algorithm", vi_algorithm),
            ("Inference seed", inference_seed),
            ("Additional position noise", noise_description),
            ("Forecast horizon", f"{forecast_horizon_seconds:g} s"),
            ("Modes", ", ".join(MODE_NAMES.values())),
        ],
    )


def _show_figures_sequentially(
    fit,
    window,
    position_observations,
    mode_summary,
    transition_summary,
    *,
    gps_speed_reference_available,
):
    """Show each diagnostic only after the previous figure was closed."""
    _show_and_close(
        _plot_trajectory(
            fit,
            window,
            position_observations,
        )
    )
    _show_and_close(_plot_mode_probabilities(mode_summary))
    _show_and_close(_plot_transition_matrix(transition_summary))

    observed_times = window.timestamps[window.observed_slice]
    speed_reference_options = {}
    speed_title = "Latent speed"
    if gps_speed_reference_available:
        speed_reference_options = {
            "reference_values": window.gps_speed_mps[window.observed_slice],
            "reference_label": "GPS speed reference (not used for fitting)",
        }
        speed_title = "Latent speed; GPS speed was not used for fitting"
    speed_figure, _ = plot_state_credible_band(
        fit,
        "speed_state",
        observed_times,
        credible_interval=CREDIBLE_INTERVAL,
        title=speed_title,
        **speed_reference_options,
    )
    _show_and_close(speed_figure)

    turn_rate_figure, _ = plot_state_credible_band(
        fit,
        "turn_rate_state",
        observed_times,
        credible_interval=CREDIBLE_INTERVAL,
        title="Latent turn rate",
    )
    _show_and_close(turn_rate_figure)


def _show_and_close(figure):
    """Display one figure as a blocking window and always release it."""
    try:
        plt.show(block=True)
    finally:
        plt.close(figure)


def _plot_trajectory(fit, window, position_observations):
    """Plot future draws connected to each draw's final latent state."""
    x_state = posterior_variable_samples(fit, "x_state")
    y_state = posterior_variable_samples(fit, "y_state")
    x_prediction = posterior_variable_samples(fit, "x_state_prediction")
    y_prediction = posterior_variable_samples(fit, "y_state_prediction")
    if x_state.shape != y_state.shape or x_prediction.shape != y_prediction.shape:
        raise ValueError("Posterior x and y state draws must have matching shapes.")
    connected_x = np.column_stack((x_state[:, -1], x_prediction))
    connected_y = np.column_stack((y_state[:, -1], y_prediction))

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.plot(
        position_observations.x_meters,
        position_observations.y_meters,
        color="tab:blue",
        linewidth=2,
        label=(
            "Noise-augmented observations"
            if position_observations.additional_noise_std_m > 0
            else "Position observations"
        ),
    )
    prediction = window.prediction_slice
    axis.plot(
        window.x_meters[prediction],
        window.y_meters[prediction],
        color="black",
        linestyle="--",
        linewidth=2,
        label="Held-out trajectory",
    )
    sample_indices = np.linspace(
        0,
        len(connected_x) - 1,
        num=min(100, len(connected_x)),
        dtype=int,
    )
    for sample_index in sample_indices:
        axis.plot(
            connected_x[sample_index],
            connected_y[sample_index],
            color="tab:red",
            alpha=0.05,
            linewidth=1,
        )
    axis.plot(
        np.median(connected_x, axis=0),
        np.median(connected_y, axis=0),
        color="tab:red",
        linewidth=2,
        label="Posterior median",
    )
    axis.set_title("Switching Bayesian CTRV Prediction")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def _plot_mode_probabilities(mode_summary):
    """Plot posterior mean probabilities for all observed transition modes."""
    figure, axis = plt.subplots(figsize=(11, 5))
    for mode_name, color in zip(
        MODE_NAMES.values(),
        ("tab:gray", "tab:blue", "tab:orange"),
        strict=True,
    ):
        axis.plot(
            mode_summary["time"],
            mode_summary[f"{mode_name}_probability"],
            color=color,
            linewidth=2,
            label=mode_name.title(),
        )
    axis.set_title("Posterior motion-mode probabilities")
    axis.set_xlabel("Time")
    axis.set_ylabel("Probability")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def _plot_transition_matrix(transition_summary):
    """Plot the posterior mean mode-transition matrix."""
    mode_names = list(MODE_NAMES.values())
    matrix = transition_summary.pivot(
        index="from_mode_name",
        columns="to_mode_name",
        values="probability_mean",
    ).loc[mode_names, mode_names]
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="Blues")
    for row_index in range(3):
        for column_index in range(3):
            axis.text(
                column_index,
                row_index,
                f"{matrix.iloc[row_index, column_index]:.2f}",
                ha="center",
                va="center",
            )
    axis.set_xticks(range(3), [name.title() for name in mode_names])
    axis.set_yticks(range(3), [name.title() for name in mode_names])
    axis.set_xlabel("Next mode")
    axis.set_ylabel("Previous mode")
    axis.set_title("Posterior mean transition probability")
    figure.colorbar(image, ax=axis, label="Probability")
    figure.tight_layout()
    return figure


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vi-algorithm",
        choices=("meanfield", "fullrank"),
        default=VI_CONFIG["algorithm"],
    )
    parser.add_argument("--seed", type=int, default=EXPERIMENT.inference_seed)
    parser.add_argument(
        "--position-noise-std-m",
        type=float,
        default=EXPERIMENT.additional_position_noise_std_m,
    )
    parser.add_argument(
        "--position-noise-seed",
        type=int,
        default=EXPERIMENT.position_noise_seed,
    )
    parser.add_argument(
        "--require-converged",
        action="store_true",
        default=VI_CONFIG["require_converged"],
    )
    parser.add_argument(
        "--no-plots",
        action="store_false",
        dest="show_plots",
        help="Run the fit without opening diagnostic figures.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(
        vi_algorithm=arguments.vi_algorithm,
        seed=arguments.seed,
        position_noise_std_m=arguments.position_noise_std_m,
        position_noise_seed=arguments.position_noise_seed,
        require_converged=arguments.require_converged,
        show_plots=arguments.show_plots,
    )
