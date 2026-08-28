"""Interactive prior-posterior analysis for the Bayesian CTRV model."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from scipy.stats import gaussian_kde

import bayestraj.forecasting.inference as inference
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.models.bayesian_observations as observation_support
import bayestraj.observations.io as observations_io
import bayestraj.observations.window as observation_window
import bayestraj.validation.reporting as reporting

PARAMETER_NAMES = (
    "initial_speed",
    "initial_heading",
    "initial_turn_rate",
    "position_observation_noise",
    "speed_process_noise",
    "turn_rate_process_noise",
)

DENSITY_POINT_COUNT = 1_000
PLOT_TAIL_PROBABILITY = 1e-3
TITLE_PAD_POINTS = 12
PRIOR_COLOR = "#6B7280"
POSTERIOR_COLOR = "#24557A"
POSTERIOR_FILL_COLOR = "#4C956C"
ACTIVE_BUTTON_COLOR = "#E5E7EB"
INACTIVE_BUTTON_COLOR = "#F3F4F6"


@dataclass(frozen=True, slots=True)
class PriorPosteriorParameterSpec:
    """Describe one prior and its matching Stan posterior variable."""

    parameter_name: str
    posterior_variable: str
    state_index: int | None
    display_unit: str
    title: str
    x_label: str
    support: str


@dataclass(frozen=True, slots=True)
class PosteriorUpdate:
    """Posterior draws from one separate batch fit of an observation prefix."""

    observation_count: int
    samples: np.ndarray

    def __post_init__(self) -> None:
        if (
            isinstance(self.observation_count, bool)
            or not isinstance(self.observation_count, int)
            or self.observation_count < 3
        ):
            raise ValueError("observation_count must be an integer of at least 3.")
        samples = np.asarray(self.samples, dtype=float).copy()
        if samples.ndim != 1 or samples.size < 2 or not np.all(np.isfinite(samples)):
            raise ValueError("samples must be a finite vector with at least two draws.")
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)


@dataclass(frozen=True, slots=True)
class PosteriorSummary:
    """Central posterior estimate and interval in presentation units."""

    center: float
    lower: float
    upper: float


class PriorPosteriorNavigator:
    """Update one prior-posterior axis with previous and next buttons."""

    def __init__(
        self,
        figure,
        axis,
        spec,
        priors,
        updates,
        x_values,
        *,
        show_legend,
    ):
        self.figure = figure
        self.axis = axis
        self.spec = spec
        self.priors = priors
        self.updates = tuple(updates)
        self.x_values = x_values
        self.show_legend = show_legend
        self._stage_index = 0

        previous_axis = figure.add_axes((0.31, 0.025, 0.16, 0.065))
        next_axis = figure.add_axes((0.53, 0.025, 0.16, 0.065))
        self.previous_button = Button(previous_axis, "Zurück")
        self.next_button = Button(next_axis, "Weiter")
        self.previous_button.on_clicked(self.show_previous)
        self.next_button.on_clicked(self.show_next)
        self._draw()

    @property
    def observation_count(self) -> int:
        """Return zero for the prior stage or the current fitted prefix size."""
        if self._stage_index == 0:
            return 0
        return self.updates[self._stage_index - 1].observation_count

    def show_previous(self, _event) -> None:
        """Show the preceding observation prefix if one exists."""
        if self._stage_index == 0:
            return
        self._stage_index -= 1
        self._draw()

    def show_next(self, _event) -> None:
        """Show the following observation prefix if one exists."""
        if self._stage_index == len(self.updates):
            return
        self._stage_index += 1
        self._draw()

    def _draw(self) -> None:
        self.axis.clear()
        prior_density = evaluate_prior_density(
            self.spec,
            self.priors,
            self.x_values,
        )
        self.axis.plot(
            self.x_values,
            prior_density,
            color=PRIOR_COLOR,
            linestyle="--",
            linewidth=2.0,
            label="Prior-Dichte",
        )

        if self._stage_index == 0:
            status = "N = 0 – Prior ohne Beobachtungen"
        else:
            update = self.updates[self._stage_index - 1]
            posterior_density = evaluate_posterior_density(
                self.spec,
                update.samples,
                self.x_values,
            )
            self.axis.plot(
                self.x_values,
                posterior_density,
                color=POSTERIOR_COLOR,
                linewidth=2.2,
                label="Posterior-Dichte",
            )
            self.axis.fill_between(
                self.x_values,
                posterior_density,
                color=POSTERIOR_FILL_COLOR,
                alpha=0.18,
            )
            status = f"N = {update.observation_count} – separate Batch-Anpassung"

        self.axis.set_title(
            f"Prior-Posterior-Aktualisierung: {self.spec.title}\n{status}",
            fontsize=15,
            pad=TITLE_PAD_POINTS,
        )
        self.axis.set_xlabel(self.spec.x_label, fontsize=13)
        self.axis.set_ylabel("Dichte", fontsize=13)
        self.axis.set_xlim(float(self.x_values[0]), float(self.x_values[-1]))
        if self.spec.support == "circular":
            self.axis.set_xticks([-270.0, -180.0, -90.0, 0.0, 90.0, 180.0, 270.0])
        self.axis.set_ylim(bottom=0.0)
        self.axis.grid(alpha=0.25, linewidth=0.8)
        self.axis.tick_params(labelsize=11)
        if self.show_legend:
            self.axis.legend(loc="upper right", fontsize=10, framealpha=0.9)
        self._update_button(self.previous_button, self._stage_index > 0)
        self._update_button(
            self.next_button,
            self._stage_index < len(self.updates),
        )
        self.figure.canvas.draw_idle()

    @staticmethod
    def _update_button(button: Button, active: bool) -> None:
        button.set_active(active)
        button.ax.set_facecolor(
            ACTIVE_BUTTON_COLOR if active else INACTIVE_BUTTON_COLOR
        )
        button.label.set_color("black" if active else "#9CA3AF")


_PARAMETER_METADATA = {
    "initial_speed": (
        "speed_state",
        0,
        "m/s",
        "Anfangsgeschwindigkeit",
        r"$v_1$ [m/s]",
        "positive",
    ),
    "initial_heading": (
        "heading_initial",
        None,
        "°",
        "Anfangskurswinkel",
        r"$\theta_1$ [$^\circ$]",
        "circular",
    ),
    "initial_turn_rate": (
        "turn_rate_state",
        0,
        "°/s",
        "Initiale Drehrate",
        r"$\omega_1$ [$^\circ$/s]",
        "real",
    ),
    "position_observation_noise": (
        "sigma_position_observation",
        None,
        "m",
        "Positionsmessrauschen",
        r"$\sigma_{\mathrm{obs}}$ [m]",
        "positive",
    ),
    "speed_process_noise": (
        "sigma_speed_process",
        None,
        "m/s",
        "Geschwindigkeits-Prozessrauschen",
        r"$\sigma_v$ [m/s]",
        "positive",
    ),
    "turn_rate_process_noise": (
        "sigma_turn_rate_process",
        None,
        "°/s",
        "Drehraten-Prozessrauschen",
        r"$\sigma_\omega$ [$^\circ$/s]",
        "positive",
    ),
}


def build_parameter_spec(
    parameter_name: str,
    priors: bayesian_model.BayesianCTRVPriors,
) -> PriorPosteriorParameterSpec:
    """Return plot and posterior metadata for one configured prior."""
    if not isinstance(priors, bayesian_model.BayesianCTRVPriors):
        raise TypeError("priors must be a BayesianCTRVPriors instance.")
    if parameter_name not in _PARAMETER_METADATA:
        allowed = ", ".join(PARAMETER_NAMES)
        raise ValueError(f"parameter_name must be one of: {allowed}.")

    metadata = _PARAMETER_METADATA[parameter_name]
    return PriorPosteriorParameterSpec(parameter_name, *metadata)


def extract_posterior_samples(
    fit,
    spec: PriorPosteriorParameterSpec,
) -> np.ndarray:
    """Extract one parameter from a VI/MCMC fit in presentation units."""
    values = reporting.posterior_variable_samples(fit, spec.posterior_variable)
    if spec.state_index is not None:
        if values.ndim != 2 or values.shape[1] <= spec.state_index:
            raise ValueError(
                f"Posterior variable {spec.posterior_variable!r} must contain "
                "one state vector per draw."
            )
        values = values[:, spec.state_index]
    elif values.ndim != 1:
        raise ValueError(
            f"Posterior variable {spec.posterior_variable!r} must contain "
            "one scalar per draw."
        )

    if spec.parameter_name in {
        "initial_heading",
        "initial_turn_rate",
        "turn_rate_process_noise",
    }:
        values = np.rad2deg(values)
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(
            f"Posterior variable {spec.posterior_variable!r} must contain finite draws."
        )
    return values


def evaluate_prior_density(
    spec: PriorPosteriorParameterSpec,
    priors: bayesian_model.BayesianCTRVPriors,
    x_values,
) -> np.ndarray:
    """Evaluate one configured prior density in presentation units."""
    x_values = np.asarray(x_values, dtype=float)
    if spec.parameter_name == "initial_speed":
        scale = priors.speed_prior_scale
        density = np.sqrt(2.0 / np.pi) / scale * np.exp(-0.5 * (x_values / scale) ** 2)
        return np.where(x_values >= 0.0, density, 0.0)
    if spec.parameter_name == "initial_heading":
        return np.where(np.abs(x_values) <= 180.0, 1.0 / 360.0, 0.0)
    if spec.parameter_name == "initial_turn_rate":
        scale = float(np.rad2deg(priors.turn_rate_prior_scale))
        return np.exp(-0.5 * (x_values / scale) ** 2) / (scale * np.sqrt(2.0 * np.pi))

    rates = {
        "position_observation_noise": (priors.sigma_position_observation_prior_rate),
        "speed_process_noise": priors.sigma_speed_process_prior_rate,
        "turn_rate_process_noise": (
            priors.sigma_turn_rate_process_prior_rate * np.pi / 180.0
        ),
    }
    rate = rates[spec.parameter_name]
    density = rate * np.exp(-rate * x_values)
    return np.where(x_values >= 0.0, density, 0.0)


def evaluate_posterior_density(
    spec: PriorPosteriorParameterSpec,
    samples,
    x_values,
) -> np.ndarray:
    """Estimate one posterior density while respecting its support."""
    samples = np.asarray(samples, dtype=float)
    x_values = np.asarray(x_values, dtype=float)
    if spec.support == "positive":
        density = _kernel_density(samples, x_values) + _kernel_density(
            samples,
            -x_values,
        )
        return np.where(x_values >= 0.0, density, 0.0)
    if spec.support == "circular":
        angles = _wrap_degrees(samples)
        circular_mean = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(angles)))))
        centered = _wrap_degrees(angles - circular_mean)
        offsets = _wrap_degrees(x_values - circular_mean)
        density = (
            _kernel_density(centered, offsets)
            + _kernel_density(centered, offsets - 360.0)
            + _kernel_density(centered, offsets + 360.0)
        )
        return np.where(np.abs(x_values) <= 180.0, density, 0.0)
    return _kernel_density(samples, x_values)


def create_prior_posterior_figure(
    spec: PriorPosteriorParameterSpec,
    priors: bayesian_model.BayesianCTRVPriors,
    updates,
    *,
    show_legend=True,
):
    """Create one interactive figure for a single configured parameter."""
    updates = tuple(updates)
    if not updates:
        raise ValueError("updates must contain at least one posterior fit.")
    counts = tuple(update.observation_count for update in updates)
    if counts != tuple(sorted(set(counts))):
        raise ValueError("Posterior observation counts must be strictly increasing.")

    x_values = _build_density_grid(spec, priors, updates)
    figure, axis = plt.subplots(figsize=(8.0, 5.4))
    figure.subplots_adjust(bottom=0.24, top=0.82)
    navigator = PriorPosteriorNavigator(
        figure,
        axis,
        spec,
        priors,
        updates,
        x_values,
        show_legend=show_legend,
    )
    return figure, navigator


def fit_prior_posterior_updates(
    trajectory_data,
    *,
    parameter_name,
    observation_counts,
    start_index,
    position_noise_std_m,
    position_noise_seed,
    priors,
    inference_method,
    inference_config,
    inference_seed,
    fit_model=None,
):
    """Fit one Bayesian CTRV parameter to consistent observation prefixes."""
    counts = _validate_observation_counts(observation_counts)
    _, inference_method = inference.normalize_inference_configuration(
        "batch",
        inference_method,
    )
    spec = build_parameter_spec(parameter_name, priors)
    if fit_model is None:
        fit_model = bayesian_model.fit_bayesian_ctrv_model

    maximum_window = observation_window.prepare_trajectory_window(
        trajectory_data,
        observation_count=counts[-1],
        prediction_count=1,
        start_index=start_index,
    )
    maximum_observations = observation_support.simulate_position_observations(
        maximum_window,
        position_noise_std_m=position_noise_std_m,
        seed=position_noise_seed,
    )

    updates = []
    for count in counts:
        print(f"Berechne separaten Batch-Posterior fuer N = {count} ...")
        window = observation_window.prepare_trajectory_window(
            trajectory_data,
            observation_count=count,
            prediction_count=1,
            start_index=start_index,
        )
        observations = observation_support.PositionObservations(
            time_seconds=maximum_observations.time_seconds[:count],
            x_meters=maximum_observations.x_meters[:count],
            y_meters=maximum_observations.y_meters[:count],
            position_noise_std_m=position_noise_std_m,
            noise_seed=position_noise_seed,
        )
        fit = fit_model(
            window,
            priors=priors,
            position_observations=observations,
            inference_method=inference_method,
            seed=inference_seed,
            **dict(inference_config),
        )
        updates.append(
            PosteriorUpdate(
                count,
                extract_posterior_samples(fit, spec),
            )
        )
    return spec, tuple(updates)


def run_bayesian_ctrv_prior_posterior_analysis(
    *,
    data_file,
    run_id,
    parameter_name,
    observation_counts,
    start_index,
    position_noise_std_m,
    position_noise_seed,
    priors,
    inference_method,
    vi_config,
    mcmc_config,
    inference_seed,
    credible_interval,
    show_legend,
    show=True,
):
    """Run and display one standalone Bayesian CTRV prior update analysis."""
    _, normalized_method = inference.normalize_inference_configuration(
        "batch",
        inference_method,
    )
    inference_config = dict(vi_config if normalized_method == "vi" else mcmc_config)
    observation_counts = _validate_observation_counts(observation_counts)
    credible_interval = _validate_credible_interval(credible_interval)
    trajectory_data = (
        observations_io.read_ship_data(data_file, run_id=run_id)
        .sort_values("time")
        .reset_index(drop=True)
    )
    if trajectory_data.empty:
        raise ValueError(f"No trajectory rows found for run_id={run_id}.")

    parameter_spec = build_parameter_spec(parameter_name, priors)
    print("=" * 72)
    print("Bayessche CTRV Prior-Posterior-Aktualisierung")
    print("=" * 72)
    print(f"Parameter             : {parameter_spec.title}")
    print(f"Run ID                : {run_id}")
    print(f"Startindex            : {start_index}")
    print(f"Inferenzmethode       : {normalized_method.upper()}")
    print(
        "Beobachtungsstaende  : Prior, "
        + ", ".join(f"N={count}" for count in observation_counts)
    )
    print(
        "Interpretation        : Jeder Posterior stammt aus einer separaten "
        "Batch-Anpassung."
    )

    spec, updates = fit_prior_posterior_updates(
        trajectory_data,
        parameter_name=parameter_name,
        observation_counts=observation_counts,
        start_index=start_index,
        position_noise_std_m=position_noise_std_m,
        position_noise_seed=position_noise_seed,
        priors=priors,
        inference_method=normalized_method,
        inference_config=inference_config,
        inference_seed=inference_seed,
    )
    print(f"Prior                 : {_describe_prior(spec, priors)}")
    for update in updates:
        summary = summarize_posterior_update(
            spec,
            update,
            credible_interval=credible_interval,
        )
        percentage = 100.0 * credible_interval
        center_label = "Kreismittel" if spec.support == "circular" else "Median"
        print(f"\nPosterior nach N={update.observation_count}:")
        print(f"  {center_label}: {summary.center:.3f} {spec.display_unit}")
        print(
            f"  {percentage:g} %-Intervall: "
            f"[{summary.lower:.3f}, {summary.upper:.3f}] {spec.display_unit}"
        )

    figure, navigator = create_prior_posterior_figure(
        spec,
        priors,
        updates,
        show_legend=show_legend,
    )
    if show:
        plt.show(block=True)
    plt.close(figure)
    return figure, navigator


def summarize_posterior_update(
    spec: PriorPosteriorParameterSpec,
    update: PosteriorUpdate,
    *,
    credible_interval=0.9,
) -> PosteriorSummary:
    """Summarize one posterior with circular handling for heading."""
    credible_interval = _validate_credible_interval(credible_interval)
    lower_probability = (1.0 - credible_interval) / 2.0
    upper_probability = 1.0 - lower_probability
    if spec.support == "circular":
        angles = _wrap_degrees(update.samples)
        center = float(np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(angles))))))
        centered = _wrap_degrees(angles - center)
        lower, upper = np.quantile(
            centered,
            [lower_probability, upper_probability],
        )
        return PosteriorSummary(center, center + float(lower), center + float(upper))

    lower, center, upper = np.quantile(
        update.samples,
        [lower_probability, 0.5, upper_probability],
    )
    return PosteriorSummary(float(center), float(lower), float(upper))


def _build_density_grid(spec, priors, updates):
    if spec.support == "circular":
        return np.linspace(-270.0, 270.0, DENSITY_POINT_COUNT)

    posterior_values = np.concatenate([update.samples for update in updates])
    if spec.parameter_name == "initial_speed":
        prior_upper = priors.speed_prior_scale * NormalDist().inv_cdf(
            1.0 - PLOT_TAIL_PROBABILITY / 2.0
        )
    elif spec.parameter_name == "initial_turn_rate":
        scale = float(np.rad2deg(priors.turn_rate_prior_scale))
        prior_upper = scale * NormalDist().inv_cdf(1.0 - PLOT_TAIL_PROBABILITY / 2.0)
    else:
        rates = {
            "position_observation_noise": (
                priors.sigma_position_observation_prior_rate
            ),
            "speed_process_noise": priors.sigma_speed_process_prior_rate,
            "turn_rate_process_noise": (
                priors.sigma_turn_rate_process_prior_rate * np.pi / 180.0
            ),
        }
        prior_upper = -np.log(PLOT_TAIL_PROBABILITY) / rates[spec.parameter_name]

    posterior_upper = float(np.quantile(np.abs(posterior_values), 0.995)) * 1.1
    upper = max(float(prior_upper), posterior_upper, np.finfo(float).eps)
    lower = -upper if spec.support == "real" else 0.0
    return np.linspace(lower, upper, DENSITY_POINT_COUNT)


def _validate_observation_counts(observation_counts):
    counts = tuple(observation_counts)
    if not counts or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 3
        for count in counts
    ):
        raise ValueError(
            "observation_counts must contain integers greater than or equal to 3."
        )
    if counts != tuple(sorted(set(counts))):
        raise ValueError("observation_counts must be strictly increasing.")
    return counts


def _validate_credible_interval(credible_interval):
    try:
        credible_interval = float(credible_interval)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("credible_interval must be between 0 and 1.") from error
    if not 0.0 < credible_interval < 1.0:
        raise ValueError("credible_interval must be between 0 and 1.")
    return credible_interval


def _describe_prior(spec, priors):
    descriptions = {
        "initial_speed": (f"Halbnormal(Skala={priors.speed_prior_scale:.4g} m/s)"),
        "initial_heading": "Gleichverteilung(-180°, 180°)",
        "initial_turn_rate": (
            f"Normal(0, {np.rad2deg(priors.turn_rate_prior_scale):.4g} °/s)"
        ),
        "position_observation_noise": (
            f"Exponential(Rate={priors.sigma_position_observation_prior_rate:.4g} 1/m)"
        ),
        "speed_process_noise": (
            f"Exponential(Rate={priors.sigma_speed_process_prior_rate:.4g} s/m)"
        ),
        "turn_rate_process_noise": (
            f"Exponential(Rate={priors.sigma_turn_rate_process_prior_rate:.4g} s/rad)"
        ),
    }
    return descriptions[spec.parameter_name]


def _kernel_density(samples, x_values):
    try:
        return gaussian_kde(samples)(x_values)
    except (ValueError, np.linalg.LinAlgError):
        bandwidth = max(float(np.ptp(x_values)) / 200.0, 1e-6)
        standardized = (x_values[:, np.newaxis] - samples[np.newaxis, :]) / bandwidth
        return np.mean(np.exp(-0.5 * standardized**2), axis=1) / (
            bandwidth * np.sqrt(2.0 * np.pi)
        )


def _wrap_degrees(values):
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0
