"""Plot deterministic and Bayesian rolling trajectory evaluations."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import ship_trajectory_prediction.validation.prediction_plotting as prediction_plotting

ROLLING_FIGURE_SIZE = (11, 8)
DEFAULT_SAMPLE_TRAJECTORIES_PER_FORECAST = 15


@dataclass(frozen=True, slots=True)
class RollingPosteriorPlotData:
    """Posterior paths for one rolling window in a shared coordinate frame."""

    forecast_origin_x: float
    forecast_origin_y: float
    x_samples: np.ndarray
    y_samples: np.ndarray
    forecast_time_seconds: np.ndarray


def plot_bayesian_rolling_predictions(
    route_x,
    route_y,
    posterior_plot_groups,
    *,
    initial_observation_count,
    window_mode,
    sample_trajectories_per_forecast=(DEFAULT_SAMPLE_TRAJECTORIES_PER_FORECAST),
    sample_seed=42,
    observed_route_x=None,
    observed_route_y=None,
    history_position_count=None,
    observed_trajectory_label="Anfängliche Beobachtungen",
    title_prefix="Rollierende bayessche CTRV-Prognose",
    forecast_label="Rollierende Posterior-Mediane",
    sample_label="Posterior-prädiktive Trajektorien aller Prognosen",
):
    """Plot rolling Bayesian paths and posterior-predictive uncertainty."""
    window_mode_label = _window_mode_label(window_mode)
    posterior_plot_groups = tuple(posterior_plot_groups)
    if not posterior_plot_groups:
        raise ValueError("posterior_plot_groups must not be empty.")
    if observed_route_x is None:
        observed_route_x = route_x
    if observed_route_y is None:
        observed_route_y = route_y

    forecast_paths = tuple(
        (
            np.concatenate(
                ([group.forecast_origin_x], np.median(group.x_samples, axis=0))
            ),
            np.concatenate(
                ([group.forecast_origin_y], np.median(group.y_samples, axis=0))
            ),
        )
        for group in posterior_plot_groups
    )
    forecast_origin_x = [group.forecast_origin_x for group in posterior_plot_groups]
    forecast_origin_y = [group.forecast_origin_y for group in posterior_plot_groups]
    sample_paths = _select_rolling_sample_paths(
        posterior_plot_groups,
        sample_trajectories_per_forecast=sample_trajectories_per_forecast,
        sample_seed=sample_seed,
    )

    history_suffix = (
        f", K={int(history_position_count)}"
        if history_position_count is not None
        else ""
    )
    figure, axis = prediction_plotting.plot_trajectory_paths(
        observed_path=(
            observed_route_x[:initial_observation_count],
            observed_route_y[:initial_observation_count],
        ),
        reference_path=(route_x, route_y),
        forecast_paths=forecast_paths,
        sample_paths=sample_paths,
        prediction_origins=(forecast_origin_x, forecast_origin_y),
        posterior_draw_groups=tuple(
            (group.x_samples, group.y_samples) for group in posterior_plot_groups
        ),
        forecast_time_groups=tuple(
            group.forecast_time_seconds for group in posterior_plot_groups
        ),
        annotate_prediction_regions=False,
        title=f"{title_prefix} ({window_mode_label}{history_suffix})",
        observed_label=observed_trajectory_label,
        reference_label="Aufgezeichnete Trajektorie",
        forecast_label=forecast_label,
        sample_label=sample_label,
        prediction_origin_label="Startpunkte der Prognosen",
        figsize=ROLLING_FIGURE_SIZE,
        forecast_alpha=0.35,
        forecast_linewidth=1.5,
    )
    plt.show()
    return figure, axis


def plot_deterministic_rolling_predictions(
    route_x,
    route_y,
    predictions,
    *,
    initial_observation_count,
    window_mode,
    observed_route_x,
    observed_route_y,
    position_noise_std_m,
):
    """Plot deterministic rolling forecasts with shared scientific styling."""
    window_mode_label = _window_mode_label(window_mode)
    forecast_paths = []
    forecast_origin_x = []
    forecast_origin_y = []
    for _, group in predictions.groupby("window_index", sort=True):
        forecast_paths.append(
            (
                np.concatenate(
                    (
                        [group["forecast_origin_x_route"].iloc[0]],
                        group["x_predicted_route"],
                    )
                ),
                np.concatenate(
                    (
                        [group["forecast_origin_y_route"].iloc[0]],
                        group["y_predicted_route"],
                    )
                ),
            )
        )
        forecast_origin_x.append(group["forecast_origin_x_route"].iloc[0])
        forecast_origin_y.append(group["forecast_origin_y_route"].iloc[0])

    figure, axis = prediction_plotting.plot_trajectory_paths(
        observed_path=(
            observed_route_x[:initial_observation_count],
            observed_route_y[:initial_observation_count],
        ),
        reference_path=(route_x, route_y),
        forecast_paths=forecast_paths,
        prediction_origins=(forecast_origin_x, forecast_origin_y),
        title=(f"Rollierende deterministische CTRV-Prognose ({window_mode_label})"),
        observed_label=(
            "Verrauschte Anfangsbeobachtungen"
            if position_noise_std_m > 0
            else "Anfängliche Beobachtungen"
        ),
        reference_label="Aufgezeichnete Trajektorie",
        forecast_label="Rollierende deterministische CTRV-Prognosen",
        prediction_origin_label="Startpunkte der Prognosen",
        figsize=ROLLING_FIGURE_SIZE,
        forecast_alpha=0.65,
        forecast_linewidth=1.3,
    )
    plt.show()
    return figure, axis


def _select_rolling_sample_paths(
    posterior_plot_groups,
    *,
    sample_trajectories_per_forecast,
    sample_seed,
):
    """Select reproducible posterior paths for every rolling forecast."""
    if (
        isinstance(sample_trajectories_per_forecast, bool)
        or not isinstance(sample_trajectories_per_forecast, int)
        or sample_trajectories_per_forecast < 0
    ):
        raise ValueError(
            "sample_trajectories_per_forecast must be a non-negative integer."
        )
    if isinstance(sample_seed, bool) or not isinstance(sample_seed, int):
        raise ValueError("sample_seed must be an integer.")

    if sample_trajectories_per_forecast == 0:
        return ()
    random_generator = np.random.default_rng(sample_seed)
    sample_paths = []
    for group in posterior_plot_groups:
        sample_count = min(
            sample_trajectories_per_forecast,
            group.x_samples.shape[0],
        )
        draw_indices = np.sort(
            random_generator.choice(
                group.x_samples.shape[0],
                sample_count,
                replace=False,
            )
        )
        for draw_index in draw_indices:
            sample_paths.append(
                (
                    np.concatenate(
                        ([group.forecast_origin_x], group.x_samples[draw_index])
                    ),
                    np.concatenate(
                        ([group.forecast_origin_y], group.y_samples[draw_index])
                    ),
                )
            )
    return tuple(sample_paths)


def _window_mode_label(window_mode):
    """Return the German label for one rolling-window mode."""
    try:
        return {
            "sliding": "gleitendes Fenster",
            "expanding": "wachsendes Fenster",
        }[window_mode]
    except (KeyError, TypeError) as error:
        raise ValueError("window_mode must be 'sliding' or 'expanding'.") from error
