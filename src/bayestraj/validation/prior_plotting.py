"""Build and draw the configured Bayesian CTRV prior distributions."""

from dataclasses import dataclass
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np

import bayestraj.models.bayesian_ctrv as bayesian_model

DENSITY_POINT_COUNT = 1_000
PLOT_TAIL_PROBABILITY = 1e-3
INDIVIDUAL_FIGURE_SIZE = (8.0, 5.0)
TITLE_PAD_POINTS = 12
CURVE_COLOR = "#24557A"
CENTRAL_COLOR = "#4C956C"
TAIL_COLOR = "#D17A22"
SHOW_LEGEND = True


@dataclass(frozen=True, slots=True)
class PriorCurve:
    """Plot-ready density and its interpretable configured region."""

    filename_stem: str
    title: str
    x_label: str
    x_values: np.ndarray
    density: np.ndarray
    central_lower: float
    central_upper: float
    central_probability: float
    thresholds: tuple[float, ...]
    x_ticks: tuple[float, ...] = ()
    x_limits: tuple[float, float] | None = None
    central_legend_label: str | None = None
    threshold_legend_label: str = "Konfigurierter Grenzwert"


def build_prior_curves(priors: bayesian_model.BayesianCTRVPriors):
    """Return the six configured priors in presentation units."""
    if not isinstance(priors, bayesian_model.BayesianCTRVPriors):
        raise TypeError("priors must be a BayesianCTRVPriors instance.")
    speed_x = np.linspace(0.0, _symmetric_quantile(priors.speed_prior_scale), DENSITY_POINT_COUNT)
    speed_density = np.sqrt(2.0 / np.pi) / priors.speed_prior_scale * np.exp(
        -0.5 * (speed_x / priors.speed_prior_scale) ** 2
    )
    heading_x = np.linspace(-180.0, 180.0, DENSITY_POINT_COUNT)
    turn_threshold = priors.turn_rate_prior_abs_rate_deg_s
    turn_scale = float(np.rad2deg(priors.turn_rate_prior_scale))
    turn_x = np.linspace(-_symmetric_quantile(turn_scale), _symmetric_quantile(turn_scale), DENSITY_POINT_COUNT)
    turn_process_mean = float(np.rad2deg(1.0 / priors.sigma_turn_rate_process_prior_rate))
    return (
        PriorCurve("prior_initial_speed", "Prior: Anfangsgeschwindigkeit", r"$v_1$ [m/s]", speed_x, speed_density, 0.0, priors.speed_prior_upper_mps, 1.0 - priors.speed_prior_tail_probability, (priors.speed_prior_upper_mps,)),
        PriorCurve("prior_initial_heading", "Prior: Anfangskurswinkel", r"$\theta_1$ [$^\circ$]", heading_x, np.full_like(heading_x, 1.0 / 360.0), -180.0, 180.0, 1.0, (-180.0, 180.0), (-270.0, -180.0, -90.0, 0.0, 90.0, 180.0, 270.0), (-270.0, 270.0), "Gleichverteilte Anfangsrichtungen", "Konfigurierter Winkelbereich"),
        PriorCurve("prior_initial_turn_rate", "Prior: initiale Drehrate", r"$\omega_1$ [$^\circ$/s]", turn_x, _normal_density(turn_x, turn_scale), -turn_threshold, turn_threshold, 1.0 - priors.turn_rate_prior_tail_probability, (-turn_threshold, turn_threshold)),
        _exponential_curve("prior_position_observation_noise", "Prior: Positionsmessrauschen", r"$\sigma_{\mathrm{obs}}$ [m]", priors.sigma_position_observation_prior_rate, priors.sigma_position_observation_prior_upper_m, priors.sigma_position_observation_prior_tail_probability),
        _exponential_curve("prior_speed_process_noise", "Prior: Geschwindigkeits-Prozessrauschen", r"$\sigma_v$ [m/s]", priors.sigma_speed_process_prior_rate, priors.sigma_speed_process_prior_upper_mps, priors.sigma_speed_process_prior_tail_probability),
        _exponential_curve("prior_turn_rate_process_noise", "Prior: Drehraten-Prozessrauschen", r"$\sigma_\omega$ [$^\circ$/s]", 1.0 / turn_process_mean, priors.sigma_turn_rate_process_prior_upper_deg_s, priors.sigma_turn_rate_process_prior_tail_probability),
    )


def create_individual_figures(curves, *, show_legend=SHOW_LEGEND):
    """Create one thesis-ready figure for every prior."""
    return {curve.filename_stem: create_prior_figure(curve, show_legend=show_legend) for curve in curves}


def create_prior_figure(curve, *, show_legend=SHOW_LEGEND):
    """Create one thesis-ready figure for a prior."""
    figure, axis = plt.subplots(figsize=INDIVIDUAL_FIGURE_SIZE)
    _draw_prior(axis, curve, show_legend=show_legend)
    figure.tight_layout()
    return figure


def _draw_prior(axis, curve: PriorCurve, *, show_legend=True) -> None:
    central = (curve.x_values >= curve.central_lower) & (curve.x_values <= curve.central_upper)
    axis.plot(curve.x_values, curve.density, color=CURVE_COLOR, linewidth=2.2, label="Prior-Dichte")
    axis.fill_between(curve.x_values, curve.density, where=central, color=CENTRAL_COLOR, alpha=0.20, label=curve.central_legend_label or f"{_format_percentage(curve.central_probability)} innerhalb der Grenze")
    if np.any(~central):
        axis.fill_between(curve.x_values, curve.density, where=~central, color=TAIL_COLOR, alpha=0.22, label=f"{_format_percentage(1.0 - curve.central_probability)} au?erhalb der Grenze")
    for index, threshold in enumerate(curve.thresholds):
        axis.axvline(threshold, color=TAIL_COLOR, linestyle="--", linewidth=1.6, label=curve.threshold_legend_label if index == 0 else "_nolegend_", zorder=3)
    axis.set_title(curve.title, fontsize=16, pad=TITLE_PAD_POINTS)
    axis.set_xlabel(curve.x_label, fontsize=13)
    axis.set_ylabel("Dichte", fontsize=13)
    axis.grid(alpha=0.25, linewidth=0.8)
    x_lower, x_upper = curve.x_limits or (curve.x_values[0], curve.x_values[-1])
    axis.set_xlim(x_lower, x_upper)
    axis.set_ylim(bottom=0.0)
    base_ticks = curve.x_ticks or tuple(axis.get_xticks())
    axis.set_xticks(sorted({*(tick for tick in base_ticks if x_lower <= tick <= x_upper), *curve.thresholds}))
    axis.tick_params(labelsize=11)
    if show_legend:
        axis.legend(loc="upper right", fontsize=10, framealpha=0.9)


def _exponential_curve(filename_stem, title, x_label, rate, configured_upper, tail_probability):
    x_values = np.linspace(0.0, -np.log(PLOT_TAIL_PROBABILITY) / rate, DENSITY_POINT_COUNT)
    return PriorCurve(filename_stem, title, x_label, x_values, rate * np.exp(-rate * x_values), 0.0, configured_upper, 1.0 - tail_probability, (configured_upper,))


def _symmetric_quantile(scale):
    return scale * NormalDist().inv_cdf(1.0 - PLOT_TAIL_PROBABILITY / 2.0)


def _normal_density(values, scale):
    return np.exp(-0.5 * (values / scale) ** 2) / (scale * np.sqrt(2.0 * np.pi))


def _format_percentage(probability):
    return f"{100.0 * probability:.0f} %"
