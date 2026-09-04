"""Interactive trajectory and posterior dashboard for Bayesian CTRV updates."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

import bayestraj.inference.ctrv_rbpf as rbpf
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.numeric_validation as numeric_validation
import bayestraj.observations.io as observations_io
import bayestraj.observations.window as observation_window
import bayestraj.validation.bayesian_ctrv_prior_posterior as prior_posterior

PARAMETER_NAMES = prior_posterior.PARAMETER_NAMES
PARAMETER_GROUPS = {
    "motion": (
        "current_speed",
        "current_heading",
        "current_turn_rate",
    ),
    "noise": (
        "position_observation_noise",
        "speed_process_noise",
        "turn_rate_process_noise",
    ),
}
PARAMETER_GROUP_LABELS = {
    "motion": "Bewegungszustand",
    "noise": "Unsicherheiten",
}
FIGURE_SIZE = (15.0, 8.5)
DEFAULT_PLAYBACK_INTERVAL_MS = 1_000


@dataclass(frozen=True, slots=True)
class PosteriorDashboardTrajectory:
    """Reference route and the positions supplied to the online filter."""

    reference_x: np.ndarray
    reference_y: np.ndarray
    observed_x: np.ndarray
    observed_y: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            name: np.asarray(getattr(self, name), dtype=float).copy()
            for name in (
                "reference_x",
                "reference_y",
                "observed_x",
                "observed_y",
            )
        }
        shapes = {values.shape for values in arrays.values()}
        if len(shapes) != 1 or any(values.ndim != 1 for values in arrays.values()):
            raise ValueError("Trajectory coordinates must be matching vectors.")
        if not arrays["reference_x"].size or any(
            not np.all(np.isfinite(values)) for values in arrays.values()
        ):
            raise ValueError("Trajectory coordinates must be finite and non-empty.")
        for name, values in arrays.items():
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class PosteriorDashboardUpdate:
    """All parameter draws and diagnostics after one sequential update."""

    observation_count: int
    samples_by_parameter: dict[str, np.ndarray]
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
        if tuple(self.samples_by_parameter) != PARAMETER_NAMES:
            raise ValueError(
                "samples_by_parameter must contain all dashboard parameters in order."
            )
        samples_by_parameter = {}
        for name, samples in self.samples_by_parameter.items():
            samples = np.asarray(samples, dtype=float).copy()
            if (
                samples.ndim != 1
                or samples.size < 2
                or not np.all(np.isfinite(samples))
            ):
                raise ValueError(
                    "Each dashboard parameter must contain at least two finite draws."
                )
            samples.setflags(write=False)
            samples_by_parameter[name] = samples
        object.__setattr__(
            self,
            "samples_by_parameter",
            MappingProxyType(samples_by_parameter),
        )


class PosteriorDashboardNavigator:
    """Synchronize one route and three switchable posterior panels."""

    def __init__(
        self,
        figure,
        trajectory_axis,
        posterior_axes,
        trajectory,
        priors,
        update_loader,
        *,
        maximum_observation_count,
        show_legend,
        playback_interval_ms,
    ):
        self.figure = figure
        self.trajectory_axis = trajectory_axis
        self.posterior_axes = tuple(posterior_axes)
        self.trajectory = trajectory
        self.priors = priors
        self._update_loader = update_loader
        self.maximum_observation_count = maximum_observation_count
        self.show_legend = show_legend
        self._updates_by_count = {}
        self._observation_count = 0
        self._parameter_group = "motion"
        self._trajectory_has_been_drawn = False
        self._is_playing = False
        self._slider_interaction_active = False
        self._specs_by_name = {
            name: prior_posterior.build_parameter_spec(name, priors)
            for name in PARAMETER_NAMES
        }
        self._x_values_by_name = {
            name: prior_posterior._build_density_grid(spec, priors, ())
            for name, spec in self._specs_by_name.items()
        }

        playback_axis = figure.add_axes((0.025, 0.05, 0.085, 0.055))
        self.playback_button = Button(playback_axis, "Start")
        self._playback_timer = figure.canvas.new_timer(interval=playback_interval_ms)
        self._playback_timer.add_callback(self.advance_playback)

        slider_axis = figure.add_axes((0.15, 0.065, 0.55, 0.035))
        slider_steps = np.arange(maximum_observation_count + 1, dtype=float)
        self.slider = Slider(
            slider_axis,
            "N",
            0,
            maximum_observation_count,
            valinit=0,
            valstep=slider_steps,
            valfmt="%0.0f",
        )
        selector_axis = figure.add_axes((0.77, 0.035, 0.2, 0.1))
        selector_axis.set_title("Posterior-Gruppe", fontsize=10)
        self.group_selector = RadioButtons(
            selector_axis,
            tuple(PARAMETER_GROUP_LABELS.values()),
            active=0,
        )
        self._slider_press_connection = figure.canvas.mpl_connect(
            "button_press_event",
            self._handle_mouse_press,
        )
        self._slider_release_connection = figure.canvas.mpl_connect(
            "button_release_event",
            self._handle_mouse_release,
        )
        self._key_press_connection = figure.canvas.mpl_connect(
            "key_press_event",
            self.handle_key_press,
        )
        self.playback_button.on_clicked(self.toggle_playback)
        self.group_selector.on_clicked(self._select_parameter_group)
        self._draw()

    @property
    def observation_count(self) -> int:
        """Return zero for the prior stage or the selected trajectory prefix."""
        return self._observation_count

    @property
    def parameter_group(self) -> str:
        """Return the key of the currently displayed parameter group."""
        return self._parameter_group

    @property
    def parameter_names(self) -> tuple[str, str, str]:
        """Return the three parameters visible in the posterior column."""
        return PARAMETER_GROUPS[self.parameter_group]

    @property
    def is_playing(self) -> bool:
        """Return whether automatic posterior playback is active."""
        return self._is_playing

    def toggle_playback(self, _event) -> None:
        """Start, pause, or restart automatic posterior playback."""
        if self.is_playing:
            self._stop_playback()
            return
        if self.observation_count == self.maximum_observation_count:
            self.slider.set_val(0)
            self.show_selected_observation_count(None)
        self._is_playing = True
        self._update_playback_button_label()
        self._playback_timer.start()

    def advance_playback(self) -> None:
        """Advance automatic playback by exactly one observation stage."""
        if not self.is_playing:
            return
        observation_count = self.observation_count + 1
        self.slider.set_val(observation_count)
        self._show_observation_count(observation_count)
        if observation_count == self.maximum_observation_count:
            self._stop_playback()

    def show_selected_observation_count(self, _event) -> None:
        """Load missing stages and display the slider-selected observation count."""
        if self.is_playing:
            self._stop_playback()
        observation_count = int(round(self.slider.val))
        self._show_observation_count(observation_count)

    def _handle_mouse_press(self, event) -> None:
        self._slider_interaction_active = event.inaxes is self.slider.ax

    def _handle_mouse_release(self, event) -> None:
        if not self._slider_interaction_active:
            return
        self._slider_interaction_active = False
        self.show_selected_observation_count(event)

    def _show_observation_count(self, observation_count) -> None:
        if observation_count == self.observation_count:
            return
        if observation_count == 0:
            self._observation_count = 0
            self._draw()
            self._update_playback_button_label()
            return

        loaded_update = False
        for missing_count in range(1, observation_count + 1):
            if missing_count in self._updates_by_count:
                continue
            update = self._update_loader(missing_count)
            if not isinstance(update, PosteriorDashboardUpdate):
                raise TypeError("update_loader must return a PosteriorDashboardUpdate.")
            if update.observation_count != missing_count:
                raise ValueError(
                    "update_loader returned an unexpected observation count."
                )
            self._updates_by_count[missing_count] = update
            loaded_update = True

        if loaded_update:
            self._update_density_grids()
        self._observation_count = observation_count
        self._draw()
        self._update_playback_button_label()

    def handle_key_press(self, event) -> None:
        """Control playback or move one stage with the supported keys."""
        if event.key in {" ", "space"}:
            self.toggle_playback(event)
            return
        if event.key not in {"left", "right"}:
            return
        if self.is_playing:
            self._stop_playback()
        step = -1 if event.key == "left" else 1
        observation_count = min(
            max(self.observation_count + step, 0),
            self.maximum_observation_count,
        )
        if observation_count == self.observation_count:
            return
        self.slider.set_val(observation_count)
        self._show_observation_count(observation_count)

    def _select_parameter_group(self, selected_label) -> None:
        for group, label in PARAMETER_GROUP_LABELS.items():
            if selected_label == label:
                self._parameter_group = group
                self._draw()
                return
        raise ValueError("The selected posterior group is unavailable.")

    def _stop_playback(self) -> None:
        self._playback_timer.stop()
        self._is_playing = False
        self._update_playback_button_label()

    def _update_playback_button_label(self) -> None:
        if self.is_playing:
            label = "Pause"
        elif self.observation_count == self.maximum_observation_count:
            label = "Neu starten"
        else:
            label = "Start"
        self.playback_button.label.set_text(label)
        self.figure.canvas.draw_idle()

    def _update_density_grids(self) -> None:
        for parameter_name, spec in self._specs_by_name.items():
            updates = tuple(
                prior_posterior.PosteriorUpdate(
                    update.observation_count,
                    update.samples_by_parameter[parameter_name],
                )
                for update in self._updates_by_count.values()
            )
            self._x_values_by_name[parameter_name] = (
                prior_posterior._build_density_grid(spec, self.priors, updates)
            )

    def _draw(self) -> None:
        self._draw_trajectory()
        for axis, parameter_name in zip(
            self.posterior_axes,
            self.parameter_names,
            strict=True,
        ):
            self._draw_posterior(axis, parameter_name)
        self.figure.suptitle(
            f"Bayessche CTRV Posterior-Aktualisierung — N = {self.observation_count}",
            fontsize=15,
            fontweight="bold",
        )
        self.figure.canvas.draw_idle()

    def _draw_trajectory(self) -> None:
        axis = self.trajectory_axis
        view_limits = None
        if self._trajectory_has_been_drawn:
            view_limits = (axis.get_xlim(), axis.get_ylim())
        axis.clear()
        axis.plot(
            self.trajectory.reference_x,
            self.trajectory.reference_y,
            color="#9CA3AF",
            linewidth=1.5,
            label="Aufgezeichnete Trajektorie",
            zorder=1,
        )
        if self.observation_count > 0:
            observed_slice = slice(0, self.observation_count)
            axis.plot(
                self.trajectory.observed_x[observed_slice],
                self.trajectory.observed_y[observed_slice],
                color="#24557A",
                linewidth=2.2,
                label="Beobachtungen bis N",
                zorder=2,
            )
            current_index = self.observation_count - 1
            axis.plot(
                self.trajectory.observed_x[current_index],
                self.trajectory.observed_y[current_index],
                marker="o",
                markersize=7,
                linestyle="none",
                color="#D97706",
                label="Aktuelle Position",
                zorder=3,
            )
        axis.set_title("Schiffsbewegung", fontsize=13, pad=10)
        axis.set_xlabel("Ostposition x [m]", fontsize=11)
        axis.set_ylabel("Nordposition y [m]", fontsize=11)
        axis.grid(alpha=0.25, linewidth=0.8)
        axis.tick_params(labelsize=10)
        axis.set_aspect("equal", adjustable="datalim")
        if view_limits is not None:
            axis.set_xlim(view_limits[0])
            axis.set_ylim(view_limits[1])
        self._trajectory_has_been_drawn = True
        if self.show_legend:
            axis.legend(loc="best", fontsize=9, framealpha=0.9)

    def _draw_posterior(self, axis, parameter_name) -> None:
        axis.clear()
        spec = self._specs_by_name[parameter_name]
        x_values = self._x_values_by_name[parameter_name]
        prior_density = prior_posterior.evaluate_prior_density(
            spec,
            self.priors,
            x_values,
        )
        axis.plot(
            x_values,
            prior_density,
            color=prior_posterior.PRIOR_COLOR,
            linestyle="--",
            linewidth=1.6,
            label="Ausgangs-Prior",
        )
        if self.observation_count > 0:
            samples = self._updates_by_count[
                self.observation_count
            ].samples_by_parameter[parameter_name]
            posterior_density = prior_posterior.evaluate_posterior_density(
                spec,
                samples,
                x_values,
            )
            axis.plot(
                x_values,
                posterior_density,
                color=prior_posterior.POSTERIOR_COLOR,
                linewidth=1.9,
                label="Posterior-Dichte",
            )
            axis.fill_between(
                x_values,
                posterior_density,
                color=prior_posterior.POSTERIOR_FILL_COLOR,
                alpha=0.18,
            )
        axis.set_title(spec.title, fontsize=11, pad=6)
        axis.set_xlabel(spec.x_label, fontsize=10)
        axis.set_ylabel("Dichte", fontsize=10)
        axis.set_xlim(float(x_values[0]), float(x_values[-1]))
        if spec.support == "circular":
            axis.set_xticks([-180.0, -90.0, 0.0, 90.0, 180.0])
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25, linewidth=0.8)
        axis.tick_params(labelsize=9)
        if self.show_legend and axis is self.posterior_axes[0]:
            axis.legend(loc="upper right", fontsize=8, framealpha=0.9)


def create_sequential_posterior_dashboard_figure(
    trajectory,
    priors,
    update_loader,
    *,
    maximum_observation_count,
    show_legend=True,
    playback_interval_ms=DEFAULT_PLAYBACK_INTERVAL_MS,
):
    """Create one interactive route and multi-parameter posterior figure."""
    if not isinstance(trajectory, PosteriorDashboardTrajectory):
        raise TypeError("trajectory must be a PosteriorDashboardTrajectory.")
    if not isinstance(priors, bayesian_model.BayesianCTRVPriors):
        raise TypeError("priors must be a BayesianCTRVPriors instance.")
    if not callable(update_loader):
        raise TypeError("update_loader must be callable.")
    if (
        isinstance(maximum_observation_count, bool)
        or not isinstance(maximum_observation_count, int)
        or not 1 <= maximum_observation_count <= trajectory.reference_x.size
    ):
        raise ValueError("maximum_observation_count must fit within the trajectory.")
    if (
        isinstance(playback_interval_ms, bool)
        or not isinstance(playback_interval_ms, (int, np.integer))
        or playback_interval_ms < 1
    ):
        raise ValueError("playback_interval_ms must be a positive integer.")
    playback_interval_ms = int(playback_interval_ms)

    figure = plt.figure(figsize=FIGURE_SIZE)
    grid = figure.add_gridspec(
        3,
        2,
        left=0.06,
        right=0.98,
        bottom=0.2,
        top=0.9,
        width_ratios=(1.25, 1.0),
        hspace=0.68,
        wspace=0.28,
    )
    trajectory_axis = figure.add_subplot(grid[:, 0])
    posterior_axes = tuple(figure.add_subplot(grid[row, 1]) for row in range(3))
    navigator = PosteriorDashboardNavigator(
        figure,
        trajectory_axis,
        posterior_axes,
        trajectory,
        priors,
        update_loader,
        maximum_observation_count=maximum_observation_count,
        show_legend=show_legend,
        playback_interval_ms=playback_interval_ms,
    )
    return figure, navigator


def create_rbpf_posterior_dashboard_loader(
    trajectory_data,
    *,
    start_index,
    position_noise_std_m,
    position_noise_seed,
    priors,
    rbpf_config,
    rbpf_seed,
    initialize_filter=None,
):
    """Create one RBPF stream that exposes all dashboard parameters per stage."""
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
    position_noise_std_m = numeric_validation.validate_non_negative_finite(
        "position_noise_std_m",
        position_noise_std_m,
    )
    position_noise_seed = numeric_validation.validate_non_negative_integer(
        "position_noise_seed",
        position_noise_seed,
    )
    rbpf_seed = numeric_validation.validate_non_negative_integer(
        "rbpf_seed",
        rbpf_seed,
    )
    maximum_observation_count = len(trajectory_data) - start_index
    if maximum_observation_count < bayesian_model.MIN_OBSERVATION_COUNT + 1:
        raise ValueError(
            "The selected trajectory must provide at least four consecutive positions."
        )

    complete_window = observation_window.prepare_trajectory_window(
        trajectory_data,
        observation_count=maximum_observation_count - 1,
        prediction_count=1,
        start_index=start_index,
    )
    time_seconds = np.asarray(complete_window.time_seconds, dtype=float)
    reference_x = np.asarray(complete_window.x_meters, dtype=float)
    reference_y = np.asarray(complete_window.y_meters, dtype=float)
    observed_x = reference_x.copy()
    observed_y = reference_y.copy()
    if position_noise_std_m > 0.0:
        noise_generator = np.random.default_rng(position_noise_seed)
        observed_x += noise_generator.normal(
            0.0,
            position_noise_std_m,
            maximum_observation_count,
        )
        observed_y += noise_generator.normal(
            0.0,
            position_noise_std_m,
            maximum_observation_count,
        )
    trajectory = PosteriorDashboardTrajectory(
        reference_x,
        reference_y,
        observed_x,
        observed_y,
    )

    if initialize_filter is None:
        initialize_filter = rbpf.SequentialBayesianCTRVFilter.initialize
    if not callable(initialize_filter):
        raise TypeError("initialize_filter must be callable.")
    print("Initialisiere RBPF mit N = 1 ...")
    online_filter = initialize_filter(
        time_seconds[:1],
        observed_x[:1],
        observed_y[:1],
        priors=priors,
        config=rbpf_config,
        seed=rbpf_seed,
    )
    specs = tuple(
        prior_posterior.build_parameter_spec(parameter_name, priors)
        for parameter_name in PARAMETER_NAMES
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
                observed_x[index],
                observed_y[index],
            )
        fit = online_filter.sample_current_posterior(seed=rbpf_seed)
        samples_by_parameter = {
            spec.parameter_name: prior_posterior.extract_posterior_samples(fit, spec)
            for spec in specs
        }
        return PosteriorDashboardUpdate(
            observation_count=observation_count,
            samples_by_parameter=samples_by_parameter,
            effective_sample_size=online_filter.effective_sample_size,
            particle_count=rbpf_config.particle_count,
            resample_count=online_filter.resample_count,
        )

    return trajectory, maximum_observation_count, load_update


def run_bayesian_ctrv_posterior_dashboard(
    *,
    data_file,
    run_id,
    start_index,
    position_noise_std_m,
    position_noise_seed,
    priors,
    rbpf_config,
    rbpf_seed,
    playback_interval_ms,
    show_legend,
    show=True,
):
    """Run the interactive trajectory and posterior dashboard."""
    trajectory_data = (
        observations_io.read_ship_data(data_file, run_id=run_id)
        .sort_values("time")
        .reset_index(drop=True)
    )
    if trajectory_data.empty:
        raise ValueError(f"No trajectory rows found for run_id={run_id}.")

    trajectory, maximum_observation_count, load_update = (
        create_rbpf_posterior_dashboard_loader(
            trajectory_data,
            start_index=start_index,
            position_noise_std_m=position_noise_std_m,
            position_noise_seed=position_noise_seed,
            priors=priors,
            rbpf_config=rbpf_config,
            rbpf_seed=rbpf_seed,
        )
    )
    print("=" * 72)
    print("Bayessche CTRV Posterior-Aktualisierung mit Schiffsbewegung")
    print("=" * 72)
    print(f"Run ID                : {run_id}")
    print(f"Startindex            : {start_index}")
    print("Inferenzmethode       : RBPF")
    print(
        "Beobachtungsstaende  : Prior, "
        f"N=1 bis N={maximum_observation_count} (Schrittweite 1)"
    )
    print("Posterior-Gruppen     : Bewegungszustand, Unsicherheiten")
    print("Navigation            : Schieberegler oder Pfeiltasten links/rechts")
    print(f"Partikel              : {rbpf_config.particle_count}")
    print(f"Posteriorziehungen    : {rbpf_config.posterior_draw_count}")

    figure, navigator = create_sequential_posterior_dashboard_figure(
        trajectory,
        priors,
        load_update,
        maximum_observation_count=maximum_observation_count,
        show_legend=show_legend,
        playback_interval_ms=playback_interval_ms,
    )
    if show:
        plt.show(block=True)
    plt.close(figure)
    return figure, navigator
