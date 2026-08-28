"""Plot and report the configured Bayesian CTRV prior distributions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np

import bayestraj.models.bayesian_ctrv as bayesian_model

PRIORS = bayesian_model.BayesianCTRVPriors()

DENSITY_POINT_COUNT = 1_000
PLOT_TAIL_PROBABILITY = 1e-3
INDIVIDUAL_FIGURE_SIZE = (8.0, 5.0)
CURVE_COLOR = "#24557A"
CENTRAL_COLOR = "#4C956C"
TAIL_COLOR = "#D17A22"
SHOW_LEGEND = True  # False hides the legend in every prior plot.


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


def main(argv=None):
    """Print the configuration and show the configured prior figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Create and close the figures without opening interactive windows.",
    )
    arguments = parser.parse_args(argv)

    print_prior_report(PRIORS)
    curves = build_prior_curves(PRIORS)
    if arguments.no_show:
        figures = create_individual_figures(curves)
        for figure in figures.values():
            plt.close(figure)
    else:
        figures = {}
        for curve in curves:
            figure = create_prior_figure(curve)
            figures[curve.filename_stem] = figure
            plt.show(block=True)
            plt.close(figure)
    return figures


def build_prior_curves(priors: bayesian_model.BayesianCTRVPriors):
    """Return the six configured priors in presentation units."""
    if not isinstance(priors, bayesian_model.BayesianCTRVPriors):
        raise TypeError("priors must be a BayesianCTRVPriors instance.")

    speed_probability = 1.0 - priors.speed_prior_tail_probability
    speed_x = np.linspace(
        0.0,
        _symmetric_normal_absolute_upper_quantile(
            priors.speed_prior_scale,
            PLOT_TAIL_PROBABILITY,
        ),
        DENSITY_POINT_COUNT,
    )
    speed_density = (
        np.sqrt(2.0 / np.pi)
        / priors.speed_prior_scale
        * np.exp(-0.5 * (speed_x / priors.speed_prior_scale) ** 2)
    )

    heading_x = np.linspace(-180.0, 180.0, DENSITY_POINT_COUNT)
    heading_density = np.full_like(heading_x, 1.0 / 360.0)

    turn_rate_scale_deg_s = float(np.rad2deg(priors.turn_rate_prior_scale))
    turn_rate_threshold_deg_s = (
        priors.turn_rate_prior_abs_heading_change_deg
        / priors.turn_rate_prior_reference_interval_seconds
    )
    turn_rate_limit = _symmetric_normal_absolute_upper_quantile(
        turn_rate_scale_deg_s,
        PLOT_TAIL_PROBABILITY,
    )
    turn_rate_x = np.linspace(
        -turn_rate_limit,
        turn_rate_limit,
        DENSITY_POINT_COUNT,
    )
    turn_rate_density = _normal_density(turn_rate_x, turn_rate_scale_deg_s)

    observation_curve = _exponential_curve(
        filename_stem="prior_position_observation_noise",
        title="Prior: Positionsmessrauschen",
        x_label=r"$\sigma_{\mathrm{obs}}$ [m]",
        rate=priors.sigma_position_observation_prior_rate,
        configured_upper=priors.sigma_position_observation_prior_upper_m,
        tail_probability=priors.sigma_position_observation_prior_tail_probability,
    )
    speed_process_curve = _exponential_curve(
        filename_stem="prior_speed_process_noise",
        title="Prior: Geschwindigkeits-Prozessrauschen",
        x_label=r"$\sigma_v$ [m/s]",
        rate=priors.sigma_speed_process_prior_rate,
        configured_upper=priors.sigma_speed_process_prior_upper_mps,
        tail_probability=priors.sigma_speed_process_prior_tail_probability,
    )
    turn_process_mean_deg_s = float(
        np.rad2deg(1.0 / priors.sigma_turn_rate_process_prior_rate)
    )
    turn_process_rate_per_deg_s = 1.0 / turn_process_mean_deg_s
    turn_process_curve = _exponential_curve(
        filename_stem="prior_turn_rate_process_noise",
        title="Prior: Drehraten-Prozessrauschen",
        x_label=r"$\sigma_\omega$ [$^\circ$/s]",
        rate=turn_process_rate_per_deg_s,
        configured_upper=priors.sigma_turn_rate_process_prior_upper_deg_s,
        tail_probability=priors.sigma_turn_rate_process_prior_tail_probability,
    )

    return (
        PriorCurve(
            filename_stem="prior_initial_speed",
            title="Prior: Anfangsgeschwindigkeit",
            x_label=r"$v_1$ [m/s]",
            x_values=speed_x,
            density=speed_density,
            central_lower=0.0,
            central_upper=priors.speed_prior_upper_mps,
            central_probability=speed_probability,
            thresholds=(priors.speed_prior_upper_mps,),
        ),
        PriorCurve(
            filename_stem="prior_initial_heading",
            title="Prior: Anfangskurswinkel",
            x_label=r"$\theta_1$ [$^\circ$]",
            x_values=heading_x,
            density=heading_density,
            central_lower=-180.0,
            central_upper=180.0,
            central_probability=1.0,
            thresholds=(-180.0, 180.0),
            x_ticks=(-270.0, -180.0, -90.0, 0.0, 90.0, 180.0, 270.0),
            x_limits=(-270.0, 270.0),
            central_legend_label="Gleichverteilte Anfangsrichtungen",
            threshold_legend_label="Konfigurierter Winkelbereich",
        ),
        PriorCurve(
            filename_stem="prior_initial_turn_rate",
            title="Prior: initiale Drehrate",
            x_label=r"$\omega_1$ [$^\circ$/s]",
            x_values=turn_rate_x,
            density=turn_rate_density,
            central_lower=-turn_rate_threshold_deg_s,
            central_upper=turn_rate_threshold_deg_s,
            central_probability=1.0 - priors.turn_rate_prior_tail_probability,
            thresholds=(-turn_rate_threshold_deg_s, turn_rate_threshold_deg_s),
        ),
        observation_curve,
        speed_process_curve,
        turn_process_curve,
    )


def create_individual_figures(curves, *, show_legend=SHOW_LEGEND):
    """Create one thesis-ready figure for every prior."""
    return {
        curve.filename_stem: create_prior_figure(
            curve,
            show_legend=show_legend,
        )
        for curve in curves
    }


def create_prior_figure(curve, *, show_legend=SHOW_LEGEND):
    """Create one thesis-ready figure for a prior."""
    figure, axis = plt.subplots(figsize=INDIVIDUAL_FIGURE_SIZE)
    _draw_prior(axis, curve, show_legend=show_legend)
    figure.tight_layout()
    return figure


def print_prior_report(priors: bayesian_model.BayesianCTRVPriors) -> None:
    """Print configured assumptions separately from derived parameters."""
    turn_threshold_deg_s = (
        priors.turn_rate_prior_abs_heading_change_deg
        / priors.turn_rate_prior_reference_interval_seconds
    )
    turn_scale_deg_s = float(np.rad2deg(priors.turn_rate_prior_scale))
    turn_process_mean_rad_s = 1.0 / priors.sigma_turn_rate_process_prior_rate
    turn_process_mean_deg_s = float(np.rad2deg(turn_process_mean_rad_s))

    print("Bayesian CTRV prior configuration")
    print("---------------------------------")
    print("Initial speed v_1:")
    print("  distribution: Half-Normal")
    print(
        "  configured statement: "
        f"P(v_1 < {priors.speed_prior_upper_mps:g} m/s) = "
        f"{1.0 - priors.speed_prior_tail_probability:.1%}"
    )
    print(f"  derived scale: {priors.speed_prior_scale:.6g} m/s")
    print("Initial heading theta_1:")
    print("  distribution: Uniform")
    print("  configured range: [-pi, pi] rad = [-180, 180] deg")
    print("  configured statement: all initial directions are equally plausible")
    print("Initial turn rate omega_1:")
    print("  distribution: Normal")
    print(
        "  configured statement: "
        f"P(|omega_1| < {turn_threshold_deg_s:g} deg/s) = "
        f"{1.0 - priors.turn_rate_prior_tail_probability:.1%}"
    )
    print(
        f"  derived scale: {priors.turn_rate_prior_scale:.6g} rad/s "
        f"({turn_scale_deg_s:.6g} deg/s)"
    )
    _print_exponential_report(
        label="Position observation noise sigma_obs",
        upper=priors.sigma_position_observation_prior_upper_m,
        unit="m",
        tail_probability=priors.sigma_position_observation_prior_tail_probability,
        rate=priors.sigma_position_observation_prior_rate,
        rate_unit="1/m",
    )
    _print_exponential_report(
        label="Speed process noise sigma_v",
        upper=priors.sigma_speed_process_prior_upper_mps,
        unit="m/s",
        tail_probability=priors.sigma_speed_process_prior_tail_probability,
        rate=priors.sigma_speed_process_prior_rate,
        rate_unit="s/m",
    )
    print("Turn-rate process noise sigma_omega:")
    print("  distribution: Exponential")
    print(
        "  configured statement: "
        f"P(sigma_omega < {priors.sigma_turn_rate_process_prior_upper_deg_s:g} "
        f"deg/s) = {1.0 - priors.sigma_turn_rate_process_prior_tail_probability:.1%}"
    )
    print(f"  derived rate: {priors.sigma_turn_rate_process_prior_rate:.6g} s/rad")
    print(
        f"  derived mean: {turn_process_mean_rad_s:.6g} rad/s "
        f"({turn_process_mean_deg_s:.6g} deg/s)"
    )


def _draw_prior(
    axis,
    curve: PriorCurve,
    *,
    show_legend=True,
) -> None:
    """Draw one density with configured central and tail regions."""
    central = (curve.x_values >= curve.central_lower) & (
        curve.x_values <= curve.central_upper
    )
    axis.plot(
        curve.x_values,
        curve.density,
        color=CURVE_COLOR,
        linewidth=2.2,
        label="Prior-Dichte",
    )
    central_label = curve.central_legend_label or (
        f"{_format_percentage(curve.central_probability)} innerhalb der Grenze"
    )
    axis.fill_between(
        curve.x_values,
        curve.density,
        where=central,
        color=CENTRAL_COLOR,
        alpha=0.20,
        label=central_label,
    )
    if np.any(~central):
        axis.fill_between(
            curve.x_values,
            curve.density,
            where=~central,
            color=TAIL_COLOR,
            alpha=0.22,
            label=(
                f"{_format_percentage(1.0 - curve.central_probability)} "
                "außerhalb der Grenze"
            ),
        )
    for index, threshold in enumerate(curve.thresholds):
        axis.axvline(
            threshold,
            color=TAIL_COLOR,
            linestyle="--",
            linewidth=1.6,
            label=curve.threshold_legend_label if index == 0 else "_nolegend_",
            zorder=3,
        )
    axis.set_title(curve.title, fontsize=16)
    axis.set_xlabel(curve.x_label, fontsize=13)
    axis.set_ylabel("Dichte", fontsize=13)
    axis.grid(alpha=0.25, linewidth=0.8)
    x_lower, x_upper = curve.x_limits or (
        curve.x_values[0],
        curve.x_values[-1],
    )
    axis.set_xlim(x_lower, x_upper)
    axis.set_ylim(bottom=0.0)
    base_ticks = curve.x_ticks or tuple(axis.get_xticks())
    visible_ticks = (tick for tick in base_ticks if x_lower <= tick <= x_upper)
    axis.set_xticks(sorted({*visible_ticks, *curve.thresholds}))
    axis.set_xlim(x_lower, x_upper)
    axis.tick_params(labelsize=11)
    if show_legend:
        axis.legend(loc="upper right", fontsize=10, framealpha=0.9)


def _exponential_curve(
    *,
    filename_stem,
    title,
    x_label,
    rate,
    configured_upper,
    tail_probability,
):
    """Return one exponential density in its displayed units."""
    x_values = np.linspace(
        0.0,
        -np.log(PLOT_TAIL_PROBABILITY) / rate,
        DENSITY_POINT_COUNT,
    )
    return PriorCurve(
        filename_stem=filename_stem,
        title=title,
        x_label=x_label,
        x_values=x_values,
        density=rate * np.exp(-rate * x_values),
        central_lower=0.0,
        central_upper=configured_upper,
        central_probability=1.0 - tail_probability,
        thresholds=(configured_upper,),
    )


def _print_exponential_report(
    *,
    label,
    upper,
    unit,
    tail_probability,
    rate,
    rate_unit,
):
    """Print one configured exponential statement and its derived parameters."""
    print(f"{label}:")
    print("  distribution: Exponential")
    print(
        f"  configured statement: P({label.split()[-1]} < {upper:g} {unit}) = "
        f"{1.0 - tail_probability:.1%}"
    )
    print(f"  derived rate: {rate:.6g} {rate_unit}")
    print(f"  derived mean: {1.0 / rate:.6g} {unit}")


def _symmetric_normal_absolute_upper_quantile(scale, tail_probability):
    """Return an absolute Normal quantile from a two-sided tail probability."""
    return scale * NormalDist().inv_cdf(1.0 - tail_probability / 2.0)


def _normal_density(values, scale):
    """Return the zero-centered Normal density for an array."""
    return np.exp(-0.5 * (values / scale) ** 2) / (scale * np.sqrt(2.0 * np.pi))


def _format_percentage(probability):
    """Return a whole percentage using German typographic spacing."""
    return f"{100.0 * probability:.0f} %"


if __name__ == "__main__":
    main()
