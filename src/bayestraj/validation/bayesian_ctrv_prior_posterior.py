"""Interactive prior-posterior analysis for the Bayesian CTRV model."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.stats import gaussian_kde

import bayestraj.inference.ctrv_rbpf as rbpf
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.observations.io as observations_io
import bayestraj.observations.position as observation_support
import bayestraj.observations.window as observation_window
import bayestraj.validation.reporting as reporting

PARAMETER_NAMES = (
    "current_speed",
    "current_heading",
    "current_turn_rate",
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


@dataclass(frozen=True, slots=True)
class PriorPosteriorParameterSpec:
    """Describe one prior and its matching posterior variable."""

    parameter_name: str
    posterior_variable: str
    state_index: int | None
    display_unit: str
    title: str
    x_label: str
    support: str


@dataclass(frozen=True, slots=True)
class PosteriorUpdate:
    """Posterior draws and diagnostics after one sequential update."""

    observation_count: int
    samples: np.ndarray
    effective_sample_size: float | None = None
    particle_count: int | None = None
    resample_count: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.observation_count, bool)
            or not isinstance(self.observation_count, int)
            or self.observation_count < 1
        ):
            raise ValueError("observation_count must be a positive integer.")
        samples = np.asarray(self.samples, dtype=float).copy()
        if samples.ndim != 1 or samples.size < 2 or not np.all(np.isfinite(samples)):
            raise ValueError("samples must be a finite vector with at least two draws.")
        if self.particle_count is not None and (
            isinstance(self.particle_count, bool)
            or not isinstance(self.particle_count, int)
            or self.particle_count < 1
        ):
            raise ValueError("particle_count must be a positive integer or None.")
        if self.effective_sample_size is not None:
            effective_sample_size = float(self.effective_sample_size)
            if (
                not np.isfinite(effective_sample_size)
                or effective_sample_size <= 0.0
                or (
                    self.particle_count is not None
                    and effective_sample_size > self.particle_count
                )
            ):
                raise ValueError(
                    "effective_sample_size must be positive and no larger than "
                    "particle_count."
                )
            object.__setattr__(
                self,
                "effective_sample_size",
                effective_sample_size,
            )
        if self.resample_count is not None and (
            isinstance(self.resample_count, bool)
            or not isinstance(self.resample_count, int)
            or self.resample_count < 0
        ):
            raise ValueError("resample_count must be a non-negative integer or None.")
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)


@dataclass(frozen=True, slots=True)
class PosteriorSummary:
    """Central posterior estimate and interval in presentation units."""

    center: float
    lower: float
    upper: float


class PriorPosteriorNavigator:
    """Select and display prior-posterior stages with one slider."""

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
        observation_counts=None,
        update_loader=None,
    ):
        self.figure = figure
        self.axis = axis
        self.spec = spec
        self.priors = priors
        self.updates = tuple(updates)
        self.observation_counts = (
            tuple(update.observation_count for update in self.updates)
            if observation_counts is None
            else tuple(observation_counts)
        )
        self._updates_by_count = {
            update.observation_count: update for update in self.updates
        }
        self._update_loader = update_loader
        self.x_values = x_values
        self.show_legend = show_legend
        self._stage_index = 0

        slider_axis = figure.add_axes((0.18, 0.055, 0.64, 0.045))
        slider_steps = np.asarray((0, *self.observation_counts), dtype=float)
        self.slider = Slider(
            slider_axis,
            "N",
            0,
            self.observation_counts[-1],
            valinit=0,
            valstep=slider_steps,
            valfmt="%0.0f",
        )
        self._slider_release_connection = figure.canvas.mpl_connect(
            "button_release_event",
            self.show_selected_observation_count,
        )
        self._key_press_connection = figure.canvas.mpl_connect(
            "key_press_event",
            self.handle_key_press,
        )
        self._draw()

    @property
    def observation_count(self) -> int:
        """Return zero for the prior stage or the current fitted prefix size."""
        if self._stage_index == 0:
            return 0
        return self.observation_counts[self._stage_index - 1]

    def show_selected_observation_count(self, _event) -> None:
        """Load and show the slider stage after the slider is released."""
        observation_count = int(round(self.slider.val))
        if observation_count == self.observation_count:
            return
        if observation_count == 0:
            self._stage_index = 0
            self._draw()
            return

        try:
            selected_index = self.observation_counts.index(observation_count)
        except ValueError as error:
            raise ValueError(
                "The slider selected an unavailable observation count."
            ) from error

        loaded_update = False
        for missing_count in self.observation_counts[: selected_index + 1]:
            if missing_count in self._updates_by_count:
                continue
            if self._update_loader is None:
                raise RuntimeError("No update loader is available for this stage.")
            update = self._update_loader(missing_count)
            if not isinstance(update, PosteriorUpdate):
                raise TypeError("update_loader must return a PosteriorUpdate.")
            if update.observation_count != missing_count:
                raise ValueError(
                    "update_loader returned an unexpected observation count."
                )
            self._updates_by_count[missing_count] = update
            loaded_update = True

        if loaded_update:
            self.x_values = _build_density_grid(
                self.spec,
                self.priors,
                tuple(self._updates_by_count.values()),
            )
        self._stage_index = selected_index + 1
        self._draw()

    def handle_key_press(self, event) -> None:
        """Move the selected posterior stage with the left or right arrow key."""
        if event.key not in {"left", "right"}:
            return
        step = -1 if event.key == "left" else 1
        stage_index = min(
            max(self._stage_index + step, 0),
            len(self.observation_counts),
        )
        if stage_index == self._stage_index:
            return
        observation_count = (
            0 if stage_index == 0 else self.observation_counts[stage_index - 1]
        )
        self.slider.set_val(observation_count)
        self.show_selected_observation_count(event)

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
            label="Ausgangs-Prior",
        )

        if self._stage_index == 0:
            status = "N = 0 – Ausgangs-Prior ohne Beobachtungen"
        else:
            update = self._updates_by_count[self.observation_count]
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
            status = f"N = {update.observation_count}"

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
        self.figure.canvas.draw_idle()


_PARAMETER_METADATA = {
    "current_speed": (
        "speed_at_origin",
        None,
        "m/s",
        "Aktuelle Geschwindigkeit",
        r"$v_N$ [m/s]",
        "positive",
    ),
    "current_heading": (
        "heading_at_origin",
        None,
        "°",
        "Aktueller Kurswinkel",
        r"$\theta_N$ [$^\circ$]",
        "circular",
    ),
    "current_turn_rate": (
        "turn_rate_at_origin",
        None,
        "°/s",
        "Aktuelle Drehrate",
        r"$\omega_N$ [$^\circ$/s]",
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
    """Extract one parameter through the shared fit interface in display units."""
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
        "current_heading",
        "current_turn_rate",
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
    if spec.parameter_name == "current_speed":
        scale = priors.speed_prior_scale
        density = np.sqrt(2.0 / np.pi) / scale * np.exp(-0.5 * (x_values / scale) ** 2)
        return np.where(x_values >= 0.0, density, 0.0)
    if spec.parameter_name == "current_heading":
        return np.where(np.abs(x_values) <= 180.0, 1.0 / 360.0, 0.0)
    if spec.parameter_name == "current_turn_rate":
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


def create_sequential_prior_posterior_figure(
    spec: PriorPosteriorParameterSpec,
    priors: bayesian_model.BayesianCTRVPriors,
    update_loader,
    *,
    maximum_observation_count,
    show_legend=True,
):
    """Create a navigator that loads consecutive posterior prefixes on demand."""
    if (
        isinstance(maximum_observation_count, bool)
        or not isinstance(maximum_observation_count, int)
        or maximum_observation_count < 1
    ):
        raise ValueError("maximum_observation_count must be a positive integer.")
    if not callable(update_loader):
        raise TypeError("update_loader must be callable.")

    observation_counts = tuple(range(1, maximum_observation_count + 1))
    x_values = _build_density_grid(spec, priors, ())
    figure, axis = plt.subplots(figsize=(8.0, 5.4))
    figure.subplots_adjust(bottom=0.24, top=0.82)
    navigator = PriorPosteriorNavigator(
        figure,
        axis,
        spec,
        priors,
        (),
        x_values,
        show_legend=show_legend,
        observation_counts=observation_counts,
        update_loader=update_loader,
    )
    return figure, navigator


def create_rbpf_posterior_update_loader(
    trajectory_data,
    *,
    parameter_name,
    start_index,
    position_noise_std_m,
    position_noise_seed,
    priors,
    rbpf_config,
    rbpf_seed,
    initialize_filter=None,
):
    """Create one persistent RBPF and expose its consecutive posterior states."""
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or start_index < 0
    ):
        raise ValueError("start_index must be a non-negative integer.")
    if not isinstance(priors, bayesian_model.BayesianCTRVPriors):
        raise TypeError("priors must be a BayesianCTRVPriors instance.")
    if not isinstance(rbpf_config, rbpf.SequentialCTRVFilterConfig):
        raise TypeError("rbpf_config must be a SequentialCTRVFilterConfig instance.")
    position_noise_std_m = observation_support.validate_non_negative_finite(
        "position_noise_std_m",
        position_noise_std_m,
    )
    position_noise_seed = observation_support.validate_non_negative_integer(
        "position_noise_seed",
        position_noise_seed,
    )
    rbpf_seed = observation_support.validate_non_negative_integer(
        "rbpf_seed",
        rbpf_seed,
    )
    maximum_observation_count = len(trajectory_data) - start_index
    if maximum_observation_count < bayesian_model.MIN_OBSERVATION_COUNT + 1:
        raise ValueError(
            "The selected trajectory must provide at least four consecutive positions."
        )

    spec = build_parameter_spec(parameter_name, priors)
    complete_window = observation_window.prepare_trajectory_window(
        trajectory_data,
        observation_count=maximum_observation_count - 1,
        prediction_count=1,
        start_index=start_index,
    )
    time_seconds = np.asarray(complete_window.time_seconds, dtype=float)
    x_observed = np.asarray(complete_window.x_meters, dtype=float).copy()
    y_observed = np.asarray(complete_window.y_meters, dtype=float).copy()
    if position_noise_std_m > 0.0:
        noise_generator = np.random.default_rng(position_noise_seed)
        x_observed += noise_generator.normal(
            0.0,
            position_noise_std_m,
            maximum_observation_count,
        )
        y_observed += noise_generator.normal(
            0.0,
            position_noise_std_m,
            maximum_observation_count,
        )

    if initialize_filter is None:
        initialize_filter = rbpf.SequentialBayesianCTRVFilter.initialize
    if not callable(initialize_filter):
        raise TypeError("initialize_filter must be callable.")
    print("Initialisiere RBPF mit N = 1 ...")
    online_filter = initialize_filter(
        time_seconds[:1],
        x_observed[:1],
        y_observed[:1],
        priors=priors,
        config=rbpf_config,
        seed=rbpf_seed,
    )

    def load_update(observation_count):
        if (
            isinstance(observation_count, bool)
            or not isinstance(observation_count, int)
            or not 1 <= observation_count <= maximum_observation_count
        ):
            raise ValueError(
                "observation_count must be within the available trajectory prefix."
            )
        if observation_count < online_filter.processed_observation_count:
            raise ValueError("The RBPF update loader cannot move backward.")
        while online_filter.processed_observation_count < observation_count:
            index = online_filter.processed_observation_count
            print(f"Aktualisiere RBPF mit Messpunkt N = {index + 1} ...")
            online_filter.update(
                time_seconds[index],
                x_observed[index],
                y_observed[index],
            )
        fit = online_filter.sample_current_posterior(seed=rbpf_seed)
        return PosteriorUpdate(
            observation_count=observation_count,
            samples=extract_posterior_samples(fit, spec),
            effective_sample_size=online_filter.effective_sample_size,
            particle_count=rbpf_config.particle_count,
            resample_count=online_filter.resample_count,
        )

    return spec, maximum_observation_count, load_update


def run_bayesian_ctrv_prior_posterior_analysis(
    *,
    data_file,
    run_id,
    parameter_name,
    start_index,
    position_noise_std_m,
    position_noise_seed,
    priors,
    rbpf_config,
    rbpf_seed,
    credible_interval,
    show_legend,
    show=True,
):
    """Run one interactive Bayesian CTRV RBPF update analysis."""
    credible_interval = _validate_credible_interval(credible_interval)
    trajectory_data = (
        observations_io.read_ship_data(data_file, run_id=run_id)
        .sort_values("time")
        .reset_index(drop=True)
    )
    if trajectory_data.empty:
        raise ValueError(f"No trajectory rows found for run_id={run_id}.")

    spec, maximum_observation_count, load_update = create_rbpf_posterior_update_loader(
        trajectory_data,
        parameter_name=parameter_name,
        start_index=start_index,
        position_noise_std_m=position_noise_std_m,
        position_noise_seed=position_noise_seed,
        priors=priors,
        rbpf_config=rbpf_config,
        rbpf_seed=rbpf_seed,
    )
    print("=" * 72)
    print("Bayessche CTRV Prior-Posterior-Aktualisierung")
    print("=" * 72)
    print(f"Parameter             : {spec.title}")
    print(f"Run ID                : {run_id}")
    print(f"Startindex            : {start_index}")
    print("Inferenzmethode       : RBPF")
    print(
        "Beobachtungsstaende  : Prior, "
        f"N=1 bis N={maximum_observation_count} (Schrittweite 1)"
    )
    print(
        "Interpretation        : Jeder Schritt aktualisiert denselben "
        "sequentiellen Filter."
    )
    print(
        "Berechnung            : Zielwert am Schieberegler; fehlende Punkte "
        "werden sequenziell verarbeitet."
    )
    print(f"Partikel              : {rbpf_config.particle_count}")
    print(f"Posteriorziehungen    : {rbpf_config.posterior_draw_count}")
    print(f"Prior                 : {_describe_prior(spec, priors)}")

    def load_and_report_update(observation_count):
        update = load_update(observation_count)
        summary = summarize_posterior_update(
            spec,
            update,
            credible_interval=credible_interval,
        )
        percentage = 100.0 * credible_interval
        center_label = "Kreismittel" if spec.support == "circular" else "Median"
        display_unit = _terminal_display_unit(spec)
        print(f"\nPosterior nach N={update.observation_count}:")
        print(f"  {center_label}: {summary.center:.3f} {display_unit}")
        print(
            f"  {percentage:g} %-Intervall: "
            f"[{summary.lower:.3f}, {summary.upper:.3f}] {display_unit}"
        )
        if (
            update.effective_sample_size is not None
            and update.particle_count is not None
        ):
            print(
                f"  ESS: {update.effective_sample_size:.1f} / {update.particle_count}"
            )
        if update.resample_count is not None:
            print(f"  Resamplings: {update.resample_count}")
        return update

    figure, navigator = create_sequential_prior_posterior_figure(
        spec,
        priors,
        load_and_report_update,
        maximum_observation_count=maximum_observation_count,
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

    if spec.parameter_name == "current_speed":
        prior_upper = priors.speed_prior_scale * NormalDist().inv_cdf(
            1.0 - PLOT_TAIL_PROBABILITY / 2.0
        )
    elif spec.parameter_name == "current_turn_rate":
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

    posterior_upper = 0.0
    if updates:
        posterior_values = np.concatenate([update.samples for update in updates])
        posterior_upper = float(np.quantile(np.abs(posterior_values), 0.995)) * 1.1
    upper = max(float(prior_upper), posterior_upper, np.finfo(float).eps)
    lower = -upper if spec.support == "real" else 0.0
    return np.linspace(lower, upper, DENSITY_POINT_COUNT)


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
        "current_speed": (f"Halbnormal(Skala={priors.speed_prior_scale:.4g} m/s)"),
        "current_heading": "Gleichverteilung(-180 Grad, 180 Grad)",
        "current_turn_rate": (
            f"Normal(0, {np.rad2deg(priors.turn_rate_prior_scale):.4g} Grad/s)"
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


def _terminal_display_unit(spec):
    return {
        "°": "Grad",
        "°/s": "Grad/s",
    }.get(spec.display_unit, spec.display_unit)


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
