"""Interactive trajectory and posterior dashboard for Bayesian CTRV updates."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.layout_engine import LayoutEngine
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

import bayestraj.inference.configuration as inference
import bayestraj.inference.ctrv_cmdstan as batch_inference
import bayestraj.inference.ctrv_rbpf as rbpf
import bayestraj.inference.ctrv_smc as smc
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.numeric_validation as numeric_validation
import bayestraj.observations.io as observations_io
import bayestraj.observations.window as observation_window
import bayestraj.validation.bayesian_ctrv_prior_posterior as prior_posterior
import bayestraj.validation.metrics as validation_metrics
import bayestraj.validation.prediction_plotting as prediction_plotting
import bayestraj.validation.reporting as reporting

from . import coordinates

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
NOISE_PARAMETER_NAMES = frozenset(PARAMETER_GROUPS["noise"])
PARAMETER_POSTERIOR_MINIMUM_OBSERVATION_COUNTS = {
    "current_speed": 2,
    "current_heading": 2,
    "current_turn_rate": 3,
    "position_observation_noise": 2,
    "speed_process_noise": 2,
    "turn_rate_process_noise": 2,
}
PARAMETER_PRIOR_TITLES = {
    "current_speed": "Initial speed prior",
    "current_heading": "Initial heading prior",
    "current_turn_rate": "Initial turn-rate prior",
    "position_observation_noise": "Initial position-observation-noise prior",
    "speed_process_noise": "Initial speed-process-noise prior",
    "turn_rate_process_noise": "Initial turn-rate-process-noise prior",
}
PARAMETER_POSTERIOR_TITLES = {
    "current_speed": "Speed posterior",
    "current_heading": "Heading posterior",
    "current_turn_rate": "Turn-rate posterior",
    "position_observation_noise": "Position-observation-noise posterior",
    "speed_process_noise": "Speed-process-noise posterior",
    "turn_rate_process_noise": "Turn-rate-process-noise posterior",
}
PARAMETER_GROUP_LABELS = {
    "motion": "Motion state",
    "noise": "Uncertainties",
}
FIGURE_SIZE = (15.0, 8.5)
DEFAULT_PLAYBACK_INTERVAL_MS = 1_000
DEFAULT_PREDICTION_COUNT = 3
DEFAULT_PREDICTION_SAMPLE_COUNT = 20
ANALYSIS_METRICS_MINIMUM_OBSERVATION_COUNT = 2
FOLLOW_SHIP_VIEW_SPAN_METERS = 600.0
COORDINATE_DISPLAY_MODES = prediction_plotting.PLOT_COORDINATE_MODES


def normalize_coordinate_display_mode(coordinate_display_mode):
    """Return the supported display mode, falling back to metres with a warning."""
    return prediction_plotting.normalize_plot_coordinate_mode(coordinate_display_mode)


def _validate_follow_ship_view_span_m(value):
    """Return a finite positive follow-ship viewport span in metres."""
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "follow_ship_view_span_m must be a finite positive number."
        ) from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError("follow_ship_view_span_m must be a finite positive number.")
    return value


@dataclass(frozen=True, slots=True)
class PosteriorDashboardConfig:
    """Configuration of one Bayesian CTRV posterior-update dashboard."""

    run_id: int
    start_index: int
    maximum_observation_count: int | None
    position_noise_std_m: float
    position_noise_seed: int
    inference_method: str
    inference_seed: int
    prediction_count: int = DEFAULT_PREDICTION_COUNT
    prediction_sample_count: int = DEFAULT_PREDICTION_SAMPLE_COUNT
    observation_interval_seconds: float = 10.0

    def __post_init__(self) -> None:
        """Normalize the selected batch or online inference method."""
        _, inference_method = inference.normalize_inference_method(
            self.inference_method,
            online_inference_methods=inference.CTRV_ONLINE_INFERENCE_METHODS,
        )
        object.__setattr__(self, "inference_method", inference_method)
        for name in ("prediction_count", "prediction_sample_count"):
            object.__setattr__(
                self,
                name,
                numeric_validation.validate_non_negative_integer(
                    name, getattr(self, name)
                ),
            )
        object.__setattr__(
            self,
            "observation_interval_seconds",
            numeric_validation.validate_positive_finite(
                "observation_interval_seconds",
                self.observation_interval_seconds,
            ),
        )


@dataclass(frozen=True, slots=True)
class PosteriorDashboardTrajectory:
    """Reference route, inference positions and their optional GPS origin."""

    reference_x: np.ndarray
    reference_y: np.ndarray
    observed_x: np.ndarray
    observed_y: np.ndarray
    reference_speed_mps: np.ndarray | None = None
    reference_heading_degrees: np.ndarray | None = None
    reference_turn_rate_degrees_per_second: np.ndarray | None = None
    reference_time_seconds: np.ndarray | None = None
    reference_longitude: float | None = None
    reference_latitude: float | None = None

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
        for name in (
            "reference_speed_mps",
            "reference_heading_degrees",
            "reference_turn_rate_degrees_per_second",
            "reference_time_seconds",
        ):
            values = getattr(self, name)
            if values is None:
                values = np.full(arrays["reference_x"].shape, np.nan)
            else:
                values = np.asarray(values, dtype=float).copy()
                if values.shape != arrays["reference_x"].shape or values.ndim != 1:
                    raise ValueError(f"{name} must match the trajectory coordinates.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        longitude = self.reference_longitude
        latitude = self.reference_latitude
        if (longitude is None) != (latitude is None):
            raise ValueError(
                "reference_longitude and reference_latitude must be provided together."
            )
        if longitude is not None:
            try:
                longitude = float(longitude)
                latitude = float(latitude)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "reference_longitude and reference_latitude must be numeric."
                ) from error
            if not np.isfinite(longitude) or not np.isfinite(latitude):
                raise ValueError(
                    "reference_longitude and reference_latitude must be finite."
                )
            if not -90.0 < latitude < 90.0:
                raise ValueError(
                    "reference_latitude must be between -90 and 90 degrees."
                )
            object.__setattr__(self, "reference_longitude", longitude)
            object.__setattr__(self, "reference_latitude", latitude)

    def reference_state_at(self, observation_count, *, coordinate_display_mode="m"):
        """Return reference values that are identifiable at one display stage."""
        observation_count = int(observation_count)
        coordinate_display_mode = normalize_coordinate_display_mode(
            coordinate_display_mode
        )
        if observation_count <= 0:
            return (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                coordinate_display_mode,
                np.nan,
            )
        index = min(max(observation_count - 1, 0), len(self.reference_x) - 1)
        x, y = self.reference_x[index], self.reference_y[index]
        if coordinate_display_mode == "km":
            x /= coordinates.METERS_PER_KILOMETER
            y /= coordinates.METERS_PER_KILOMETER
        elif coordinate_display_mode == "gps":
            if self.reference_longitude is None or self.reference_latitude is None:
                raise ValueError("GPS display requires a reference GPS position.")
            longitude, latitude = coordinates.local_to_gps_coordinates(
                np.asarray([x]),
                np.asarray([y]),
                self.reference_longitude,
                self.reference_latitude,
            )
            x, y = longitude[0], latitude[0]
        return (
            x,
            y,
            self.reference_heading_degrees[index] if observation_count >= 2 else np.nan,
            self.reference_speed_mps[index] if observation_count >= 2 else np.nan,
            (
                self.reference_turn_rate_degrees_per_second[index]
                if observation_count >= 3
                else np.nan
            ),
            coordinate_display_mode,
            self.reference_time_seconds[index],
        )


@dataclass(frozen=True, slots=True)
class PosteriorDashboardForecast:
    """Compact, immutable latent forecast for one cached posterior stage.

    Positions are local metres; offsets are seconds after the last observation.
    Only the median and a limited set of paired draws are retained for playback.
    """

    time_offsets_seconds: np.ndarray
    median_positions: np.ndarray
    sample_positions: np.ndarray

    def __post_init__(self) -> None:
        offsets = np.asarray(self.time_offsets_seconds, dtype=float).copy()
        median = np.asarray(self.median_positions, dtype=float).copy()
        samples = np.asarray(self.sample_positions, dtype=float).copy()
        numeric_validation.validate_finite_vector("time_offsets_seconds", offsets)
        if offsets[0] <= 0 or np.any(np.diff(offsets) <= 0):
            raise ValueError(
                "Forecast offsets must be positive and strictly increasing."
            )
        if (
            median.shape != (offsets.size, 2)
            or samples.ndim != 3
            or samples.shape[1:] != median.shape
            or not np.all(np.isfinite(median))
            or not np.all(np.isfinite(samples))
        ):
            raise ValueError(
                "Forecast positions must be finite, matching (time, xy) arrays."
            )
        for name, values in (
            ("time_offsets_seconds", offsets),
            ("median_positions", median),
            ("sample_positions", samples),
        ):
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class DashboardAnalysisMetrics:
    """Forecast accuracy and computation diagnostics for one displayed stage."""

    ade_m: float | None
    fde_m: float | None
    joint_coverage_count: int
    joint_coverage_total: int
    inference_time_seconds: float | None


@dataclass(frozen=True, slots=True)
class PosteriorDashboardUpdate:
    """All parameter draws and diagnostics after one inference stage."""

    observation_count: int
    samples_by_parameter: dict[str, np.ndarray]
    effective_sample_size: float | None = None
    particle_count: int | None = None
    resample_count: int | None = None
    forecast: PosteriorDashboardForecast | None = None
    inference_time_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.forecast is not None and not isinstance(
            self.forecast, PosteriorDashboardForecast
        ):
            raise TypeError("forecast must be a PosteriorDashboardForecast or None.")
        if self.inference_time_seconds is not None and (
            not np.isfinite(self.inference_time_seconds)
            or self.inference_time_seconds < 0
        ):
            raise ValueError("inference_time_seconds must be finite and non-negative.")
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


def posterior_medians_at(updates_by_count, observation_count):
    """Return only posterior medians identifiable at one display stage."""
    update = updates_by_count.get(observation_count)
    if update is None:
        return None
    return {
        parameter_name: (
            float(np.median(update.samples_by_parameter[parameter_name]))
            if observation_count
            >= PARAMETER_POSTERIOR_MINIMUM_OBSERVATION_COUNTS[parameter_name]
            else None
        )
        for parameter_name in PARAMETER_NAMES
    }


def analysis_metrics_at(trajectory, updates_by_count, observation_count):
    """Evaluate the displayed forecast and all available 90% 2D regions to date."""
    if observation_count < ANALYSIS_METRICS_MINIMUM_OBSERVATION_COUNT:
        return DashboardAnalysisMetrics(
            ade_m=None,
            fde_m=None,
            joint_coverage_count=0,
            joint_coverage_total=0,
            inference_time_seconds=None,
        )
    update = updates_by_count.get(observation_count)
    forecast = None if update is None else update.forecast
    ade_m, fde_m = None, None
    if forecast is not None:
        actual_positions = _forecast_reference_positions(trajectory, update)
        if actual_positions.size:
            errors_m = np.linalg.norm(
                forecast.median_positions[: len(actual_positions)] - actual_positions,
                axis=1,
            )
            ade_m = float(np.mean(errors_m))
            fde_m = float(errors_m[-1])

    covered_count, total_count = 0, 0
    for count, cached_update in updates_by_count.items():
        if (
            count < ANALYSIS_METRICS_MINIMUM_OBSERVATION_COUNT
            or count > observation_count
            or cached_update.forecast is None
        ):
            continue
        forecast = cached_update.forecast
        actual_positions = _forecast_reference_positions(trajectory, cached_update)
        sample_count = min(
            len(actual_positions),
            forecast.sample_positions.shape[1],
        )
        if forecast.sample_positions.shape[0] < 2:
            continue
        for index in range(sample_count):
            region = validation_metrics.empirical_covariance_regions(
                forecast.sample_positions[:, index, 0],
                forecast.sample_positions[:, index, 1],
                probabilities=(0.9,),
            )[0.9]
            covered_count += region.contains(*actual_positions[index])
            total_count += 1
    return DashboardAnalysisMetrics(
        ade_m=ade_m,
        fde_m=fde_m,
        joint_coverage_count=covered_count,
        joint_coverage_total=total_count,
        inference_time_seconds=(
            None if update is None else update.inference_time_seconds
        ),
    )


def _forecast_reference_positions(trajectory, update):
    """Return held-out reference positions aligned with one forecast update."""
    forecast = update.forecast
    if forecast is None:
        return np.empty((0, 2))
    start_index = update.observation_count
    stop_index = min(
        start_index + len(forecast.median_positions), len(trajectory.reference_x)
    )
    return np.column_stack(
        (
            trajectory.reference_x[start_index:stop_index],
            trajectory.reference_y[start_index:stop_index],
        )
    )


class _PosteriorDashboardLayout(LayoutEngine):
    """Keep text-sized clearances during resize and toolbar view restoration."""

    _adjust_compatible = True
    _colorbar_gridspec = False

    def __init__(
        self,
        trajectory_axis,
        posterior_axes_by_group,
        playback_axis,
        slider_axis,
        selector_axis,
        follow_axis,
    ):
        super().__init__()
        self.trajectory_axis = trajectory_axis
        self.posterior_axes_by_group = posterior_axes_by_group
        self.playback_axis = playback_axis
        self.slider_axis = slider_axis
        self.selector_axis = selector_axis
        self.follow_axis = follow_axis
        self.display_mode = "compact"
        self.active_group = "motion"

    def execute(self, figure) -> None:
        """Reserve fixed-point gaps and a separate footer before every draw.

        Fonts have fixed point sizes, so percentage-only gaps become too small
        in an embedded window. The lower bounds only guard transient, near-zero
        canvas sizes during packing.
        """
        width_points = max(400.0, figure.get_figwidth() * 72.0)
        height_points = max(320.0, figure.get_figheight() * 72.0)
        bottom_margin = 116.0  # Footer, its title, x labels and clear separation.
        top_margin = 52.0  # Figure title and top-panel title.
        panel_gap = 58.0  # Upper x label, lower title and breathing room.
        panel_height = max(
            1.0, (height_points - bottom_margin - top_margin - 2 * panel_gap) / 3
        )
        grid = (
            self.posterior_axes_by_group["motion"][0].get_subplotspec().get_gridspec()
        )
        grid.update(
            left=max(0.06, 76.0 / width_points),
            right=0.98,
            bottom=bottom_margin / height_points,
            top=1.0 - top_margin / height_points,
            hspace=panel_gap / panel_height,
            wspace=0.28,
        )
        # GridSpec.update only moves pyplot-managed axes automatically. Embedded
        # Figures have no pyplot manager, so apply their subplot positions too.
        self.trajectory_axis.set_position(
            self.trajectory_axis.get_subplotspec().get_position(figure)
        )
        motion_axes = self.posterior_axes_by_group["motion"]
        noise_axes = self.posterior_axes_by_group["noise"]
        if self.display_mode == "expanded":
            for axis in (*motion_axes, *noise_axes):
                axis.set_visible(True)
                axis.set_position(axis.get_subplotspec().get_position(figure))
        else:
            for group, axes in self.posterior_axes_by_group.items():
                visible = group == self.active_group
                for index, axis in enumerate(axes):
                    axis.set_visible(visible)
                    if visible:
                        motion_position = (
                            motion_axes[index].get_subplotspec().get_position(figure)
                        )
                        noise_position = (
                            noise_axes[index].get_subplotspec().get_position(figure)
                        )
                        axis.set_position(
                            (
                                motion_position.x0,
                                motion_position.y0,
                                noise_position.x1 - motion_position.x0,
                                motion_position.height,
                            )
                        )
        playback_width = max(0.085, 64.0 / width_points)
        if self.display_mode == "expanded":
            self.selector_axis.set_visible(False)
            selector_left = 0.97
        else:
            self.selector_axis.set_visible(True)
            selector_width = max(0.2, 165.0 / width_points)
            selector_left = 0.97 - selector_width
            self.selector_axis.set_position(
                (
                    selector_left,
                    12.0 / height_points,
                    selector_width,
                    44.0 / height_points,
                )
            )
        slider_left = 0.07 + playback_width
        self.playback_axis.set_position(
            (0.025, 18.0 / height_points, playback_width, 28.0 / height_points)
        )
        self.follow_axis.set_position(
            (0.025, 54.0 / height_points, 165.0 / width_points, 24.0 / height_points)
        )
        self.slider_axis.set_position(
            (
                slider_left,
                29.0 / height_points,
                selector_left - slider_left - 0.06,
                10.0 / height_points,
            )
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
        minimum_posterior_observation_count,
        show_legend,
        playback_interval_ms,
        request_update=None,
        prediction_count=0,
        coordinate_display_mode="m",
        posterior_axes_by_group=None,
        show_reference_trajectory=True,
        show_observed_trajectory=True,
        show_current_position=True,
        show_sample_trajectories=True,
        show_median_forecast=True,
        show_prediction_region_50=True,
        show_prediction_region_90=True,
        follow_ship_view_span_m=FOLLOW_SHIP_VIEW_SPAN_METERS,
        on_state_change=None,
        on_metrics_change=None,
        on_medians_change=None,
    ):
        self.figure = figure
        self.trajectory_axis = trajectory_axis
        if posterior_axes_by_group is None:
            posterior_axes_by_group = {"motion": tuple(posterior_axes)}
        self.posterior_axes_by_group = {
            group: tuple(axes) for group, axes in posterior_axes_by_group.items()
        }
        self.posterior_axes = self.posterior_axes_by_group["motion"]
        self.trajectory = trajectory
        self.priors = priors
        self._update_loader = update_loader
        self._request_update = request_update
        self._requested_observation_count = 0
        self._density_grids_dirty = False
        self.maximum_observation_count = maximum_observation_count
        self.minimum_posterior_observation_count = minimum_posterior_observation_count
        self.show_legend = show_legend
        self.show_reference_trajectory = show_reference_trajectory
        self.show_observed_trajectory = show_observed_trajectory
        self.show_current_position = show_current_position
        self.show_sample_trajectories = show_sample_trajectories
        self.show_median_forecast = show_median_forecast
        self.show_prediction_region_50 = show_prediction_region_50
        self.show_prediction_region_90 = show_prediction_region_90
        self.follow_ship_view_span_m = _validate_follow_ship_view_span_m(
            follow_ship_view_span_m
        )
        self.prediction_count = prediction_count
        self.coordinate_display_mode = normalize_coordinate_display_mode(
            coordinate_display_mode
        )
        self._reset_trajectory_view_for_coordinate_change = False
        self._validate_coordinate_display_mode()
        self._on_state_change = on_state_change
        self._on_metrics_change = on_metrics_change
        self._on_medians_change = on_medians_change
        self._updates_by_count = {}
        self._observation_count = 0
        self._parameter_group = "motion"
        self._posterior_display_mode = "compact"
        self._trajectory_has_been_drawn = False
        self._follow_ship_view_needs_focus = False
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
        selector_axis.set_title("Posterior group", fontsize=10)
        self.group_selector = RadioButtons(
            selector_axis,
            tuple(PARAMETER_GROUP_LABELS.values()),
            active=0,
        )
        follow_axis = figure.add_axes((0.025, 0.08, 0.2, 0.04))
        follow_axis.set_frame_on(False)
        self.follow_checkbox = CheckButtons(
            follow_axis,
            ("Follow ship",),
            (False,),
            useblit=False,
            label_props={"fontsize": [10]},
            frame_props={"s": [196], "linewidth": [1.2]},
            check_props={"s": [196], "linewidth": [1.5]},
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
        self.follow_checkbox.on_clicked(self._follow_ship_changed)
        layout = _PosteriorDashboardLayout(
            trajectory_axis,
            self.posterior_axes_by_group,
            playback_axis,
            slider_axis,
            selector_axis,
            follow_axis,
        )
        self._layout = layout
        figure.set_layout_engine(layout)
        layout.execute(figure)
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
    def posterior_display_mode(self) -> str:
        """Return whether one or both posterior groups are currently visible."""
        return self._posterior_display_mode

    def set_posterior_display_mode(self, mode) -> None:
        """Show one selected group or both posterior groups side by side."""
        if mode not in {"compact", "expanded"}:
            raise ValueError("mode must be 'compact' or 'expanded'.")
        if self._posterior_display_mode == mode:
            return
        self._posterior_display_mode = mode
        self._layout.display_mode = mode
        self._layout.active_group = self._parameter_group
        self._layout.execute(self.figure)
        self._set_group_selector_visible(mode == "compact")
        self._draw()
        self.figure.canvas.draw_idle()

    def _set_group_selector_visible(self, visible) -> None:
        """Hide every artist of the compact-only posterior-group selector."""
        self.group_selector.ax.set_visible(visible)
        self.group_selector._buttons.set_visible(visible)
        self.group_selector.ax.title.set_visible(visible)
        self.group_selector.active = visible
        for label in self.group_selector.labels:
            label.set_visible(visible)

    @property
    def parameter_names(self) -> tuple[str, str, str]:
        """Return the three parameters visible in the posterior column."""
        return PARAMETER_GROUPS[self.parameter_group]

    @property
    def is_playing(self) -> bool:
        """Return whether automatic posterior playback is active."""
        return self._is_playing

    @property
    def is_waiting(self) -> bool:
        """Return whether the requested stage is still being computed."""
        return self._requested_observation_count != self.observation_count

    @property
    def requested_observation_count(self) -> int:
        """Return the navigation target, including a pending background update."""
        return self._requested_observation_count

    def accept_update(self, update) -> None:
        """Cache a background result on the UI thread and show it when requested."""
        if not isinstance(update, PosteriorDashboardUpdate):
            raise TypeError("update must be a PosteriorDashboardUpdate.")
        _validate_update_observation_count(
            update.observation_count,
            minimum=self.minimum_posterior_observation_count,
            maximum=self.maximum_observation_count,
        )
        self._updates_by_count[update.observation_count] = update
        self._density_grids_dirty = True
        if self.is_waiting:
            self._show_observation_count(self._requested_observation_count)

    def pause_playback(self) -> None:
        """Pause immediately, retaining any in-flight result only in the cache."""
        self._stop_playback()
        if self._request_update is not None:
            self._requested_observation_count = self.observation_count
            self._request_update(self.observation_count)
            self.slider.set_val(self.observation_count)

    def disconnect(self) -> None:
        """Stop playback and release widget callbacks when an embedded view closes."""
        self._playback_timer.stop()
        self._is_playing = False
        for connection in (
            self._slider_press_connection,
            self._slider_release_connection,
            self._key_press_connection,
        ):
            self.figure.canvas.mpl_disconnect(connection)
        for widget in (
            self.playback_button,
            self.slider,
            self.group_selector,
            self.follow_checkbox,
        ):
            widget.disconnect_events()

    def toggle_playback(self, _event) -> None:
        """Start, pause, or restart automatic posterior playback."""
        if self.is_playing:
            self.pause_playback()
            return
        if self.observation_count == self.maximum_observation_count:
            self.slider.set_val(0)
            self.show_selected_observation_count(None)
        self._is_playing = True
        self._update_playback_button_label()
        self._playback_timer.start()

    def advance_playback(self) -> None:
        """Advance automatic playback by exactly one observation stage."""
        if not self.is_playing or self.is_waiting:
            return
        observation_count = self.observation_count + 1
        self.slider.set_val(observation_count)
        self._show_observation_count(observation_count)

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
        self._requested_observation_count = observation_count
        if self._request_update is not None:
            self._request_update(observation_count)
        if observation_count == self.observation_count:
            return
        if observation_count == 0:
            self._observation_count = 0
            self._draw()
            self._update_playback_button_label()
            return

        loaded_update = self._density_grids_dirty
        for missing_count in range(
            self.minimum_posterior_observation_count,
            observation_count + 1,
        ):
            if missing_count in self._updates_by_count:
                continue
            if self._request_update is not None:
                return
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
            self._density_grids_dirty = False
        self._observation_count = observation_count
        self._draw()
        self._update_playback_button_label()
        if self.is_playing and self.observation_count == self.maximum_observation_count:
            self._stop_playback()

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
        if observation_count == self.observation_count and not self.is_waiting:
            return
        self.slider.set_val(observation_count)
        self._show_observation_count(observation_count)

    def _select_parameter_group(self, selected_label) -> None:
        for group, label in PARAMETER_GROUP_LABELS.items():
            if selected_label == label:
                self._parameter_group = group
                self.posterior_axes = self.posterior_axes_by_group[group]
                self._layout.active_group = group
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
            label = "Restart"
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
        if self._posterior_display_mode == "expanded":
            for group, axes in self.posterior_axes_by_group.items():
                for index, (axis, parameter_name) in enumerate(
                    zip(axes, PARAMETER_GROUPS[group], strict=True)
                ):
                    self._draw_posterior(
                        axis,
                        parameter_name,
                        show_legend_axis=(group == "motion" and index == 0),
                    )
        else:
            for axis, parameter_name in zip(
                self.posterior_axes,
                self.parameter_names,
                strict=True,
            ):
                self._draw_posterior(axis, parameter_name)
        self.figure.suptitle(
            f"Bayesian CTRV posterior update — N = {self.observation_count}",
            fontsize=15,
            fontweight="bold",
        )
        self.figure.canvas.draw_idle()
        if self._on_state_change is not None:
            self._on_state_change(
                self.trajectory.reference_state_at(
                    self.observation_count,
                    coordinate_display_mode=self.coordinate_display_mode,
                )
            )
        if self._on_metrics_change is not None:
            self._on_metrics_change(
                analysis_metrics_at(
                    self.trajectory,
                    self._updates_by_count,
                    self.observation_count,
                )
            )
        if self._on_medians_change is not None:
            self._on_medians_change(
                posterior_medians_at(self._updates_by_count, self.observation_count)
            )

    def _draw_trajectory(self) -> None:
        axis = self.trajectory_axis
        view_limits = None
        if (
            self._trajectory_has_been_drawn
            and not self._reset_trajectory_view_for_coordinate_change
        ):
            view_limits = (axis.get_xlim(), axis.get_ylim())
        axis.clear()
        reference_x, reference_y = self._display_coordinates(
            self.trajectory.reference_x,
            self.trajectory.reference_y,
        )
        observed_x, observed_y = self._display_coordinates(
            self.trajectory.observed_x,
            self.trajectory.observed_y,
        )
        if self.show_reference_trajectory:
            axis.plot(
                reference_x,
                reference_y,
                color="#9CA3AF",
                linewidth=1.5,
                label="Recorded trajectory",
                zorder=1,
            )
        if self.observation_count > 0:
            observed_slice = slice(0, self.observation_count)
            if self.show_observed_trajectory:
                axis.plot(
                    observed_x[observed_slice],
                    observed_y[observed_slice],
                    color="#24557A",
                    linewidth=2.2,
                    label="Observations through N",
                    zorder=2,
                )
            current_index = self.observation_count - 1
            if self.show_current_position:
                axis.plot(
                    observed_x[current_index],
                    observed_y[current_index],
                    marker="o",
                    markersize=7,
                    linestyle="none",
                    color="#D97706",
                    label="Current position",
                    zorder=3,
                )
        forecast_legend_handles = self._draw_forecast(axis)
        x_label, y_label, spatial_aspect = self._coordinate_display_spec()
        axis.set_xlabel(x_label, fontsize=11)
        axis.set_ylabel(y_label, fontsize=11)
        axis.grid(alpha=0.25, linewidth=0.8)
        axis.tick_params(labelsize=10)
        axis.set_aspect(spatial_aspect, adjustable="datalim")
        if self.coordinate_display_mode == "gps":
            axis.ticklabel_format(style="plain", useOffset=False)
        if view_limits is not None:
            axis.set_xlim(view_limits[0])
            axis.set_ylim(view_limits[1])
        self._center_trajectory_on_ship()
        self._trajectory_has_been_drawn = True
        self._reset_trajectory_view_for_coordinate_change = False
        if self.show_legend:
            handles, labels = axis.get_legend_handles_labels()
            handles.extend(forecast_legend_handles)
            labels.extend(handle.get_label() for handle in forecast_legend_handles)
            axis.legend(
                handles,
                labels,
                loc="upper right",
                fontsize=9,
                framealpha=0.9,
            )

    def _follow_ship_changed(self, _label) -> None:
        """Focus the ship view or restore the complete recorded trajectory."""
        if self.follow_checkbox.get_status()[0]:
            self._follow_ship_view_needs_focus = True
            self._center_trajectory_on_ship()
        else:
            self._follow_ship_view_needs_focus = False
            self._trajectory_has_been_drawn = False
            self._draw_trajectory()
        self.figure.canvas.draw_idle()

    def _center_trajectory_on_ship(self) -> None:
        """Translate the current viewport; preserve its spans and axis directions."""
        if not self.follow_checkbox.get_status()[0] or self.observation_count == 0:
            return
        axis = self.trajectory_axis
        index = self.observation_count - 1
        position_x, position_y = self._display_coordinates(
            np.asarray([self.trajectory.observed_x[index]]),
            np.asarray([self.trajectory.observed_y[index]]),
        )
        if self._follow_ship_view_needs_focus:
            self._set_focused_ship_view(index)
            self._follow_ship_view_needs_focus = False
            return
        for get_limits, set_limits, position in (
            (axis.get_xlim, axis.set_xlim, position_x[0]),
            (axis.get_ylim, axis.set_ylim, position_y[0]),
        ):
            lower, upper = get_limits()
            half_span = (upper - lower) / 2.0
            set_limits(position - half_span, position + half_span)

    def _set_focused_ship_view(self, index) -> None:
        """Show a fixed real-world window around the current ship position."""
        half_span_meters = self.follow_ship_view_span_m / 2.0
        current_x = self.trajectory.observed_x[index]
        current_y = self.trajectory.observed_y[index]
        x_values, _ = self._display_coordinates(
            np.asarray([current_x - half_span_meters, current_x + half_span_meters]),
            np.asarray([current_y, current_y]),
        )
        _, y_values = self._display_coordinates(
            np.asarray([current_x, current_x]),
            np.asarray([current_y - half_span_meters, current_y + half_span_meters]),
        )
        for get_limits, set_limits, values in (
            (self.trajectory_axis.get_xlim, self.trajectory_axis.set_xlim, x_values),
            (self.trajectory_axis.get_ylim, self.trajectory_axis.set_ylim, y_values),
        ):
            lower, upper = sorted(map(float, values))
            current_lower, current_upper = get_limits()
            if current_lower > current_upper:
                set_limits(upper, lower)
            else:
                set_limits(lower, upper)

    def _draw_forecast(self, axis) -> list:
        """Draw the selected forecast and its posterior-predictive regions."""
        update = self._updates_by_count.get(self.observation_count)
        forecast = None if update is None else update.forecast
        title = "Ship movement"
        legend_handles = []
        if forecast is not None:
            origin_x, origin_y = self._display_coordinates(
                np.asarray([self.trajectory.observed_x[self.observation_count - 1]]),
                np.asarray([self.trajectory.observed_y[self.observation_count - 1]]),
            )
            origin = np.array([origin_x[0], origin_y[0]])
            region_probabilities = tuple(
                probability
                for probability, enabled in (
                    (0.5, self.show_prediction_region_50),
                    (0.9, self.show_prediction_region_90),
                )
                if enabled
            )
            display_sample_positions = None
            if forecast.sample_positions.shape[0] >= 2 and (
                self.show_sample_trajectories or region_probabilities
            ):
                display_sample_positions = np.empty_like(forecast.sample_positions)
                for time_index in range(forecast.sample_positions.shape[1]):
                    sample_x, sample_y = self._display_coordinates(
                        forecast.sample_positions[:, time_index, 0],
                        forecast.sample_positions[:, time_index, 1],
                    )
                    display_sample_positions[:, time_index] = np.column_stack(
                        (sample_x, sample_y)
                    )
                legend_handles = prediction_plotting._draw_prediction_regions(
                    axis,
                    (
                        display_sample_positions[:, :, 0],
                        display_sample_positions[:, :, 1],
                    ),
                    np.concatenate(([0.0], forecast.time_offsets_seconds)),
                    annotate_time=False,
                    region_probabilities=region_probabilities,
                )
            # Connect to the last measured point for orientation, as in the
            # standalone prediction plot; this does not re-anchor model draws.
            for index, positions in enumerate(
                forecast.sample_positions if self.show_sample_trajectories else ()
            ):
                if display_sample_positions is None:
                    sample_x, sample_y = self._display_coordinates(
                        positions[:, 0], positions[:, 1]
                    )
                else:
                    sample_x = display_sample_positions[index, :, 0]
                    sample_y = display_sample_positions[index, :, 1]
                path = np.vstack((origin, np.column_stack((sample_x, sample_y))))
                axis.plot(
                    path[:, 0],
                    path[:, 1],
                    color="#DC2626",
                    alpha=0.14,
                    linewidth=0.9,
                    zorder=2,
                    label="Possible future trajectories"
                    if index == 0
                    else "_nolegend_",
                )
            median_x, median_y = self._display_coordinates(
                forecast.median_positions[:, 0], forecast.median_positions[:, 1]
            )
            path = np.vstack((origin, np.column_stack((median_x, median_y))))
            if self.show_median_forecast:
                axis.plot(
                    path[:, 0],
                    path[:, 1],
                    color="#DC2626",
                    linewidth=2.2,
                    marker=".",
                    markersize=4,
                    zorder=4,
                    label="Forecast (median)",
                )
            title += f" · Forecast +{forecast.time_offsets_seconds[-1]:g} s"
        elif self.prediction_count:
            message = (
                "End of recorded trajectory: no further forecast times."
                if self.observation_count == self.trajectory.reference_x.size
                else f"Forecast from N = {self.minimum_posterior_observation_count}"
            )
            axis.text(
                0.02,
                0.02,
                message,
                transform=axis.transAxes,
                fontsize=8,
                va="bottom",
                wrap=True,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
        axis.set_title(title, fontsize=13, pad=10)
        return legend_handles

    def set_display_options(
        self,
        *,
        show_legend,
        show_reference_trajectory,
        show_observed_trajectory,
        show_current_position,
        show_sample_trajectories,
        show_median_forecast,
        show_prediction_region_50,
        show_prediction_region_90,
        follow_ship_view_span_m=None,
    ) -> None:
        """Apply presentation-only toggles without loading another update."""
        options = {
            "show_legend": show_legend,
            "show_reference_trajectory": show_reference_trajectory,
            "show_observed_trajectory": show_observed_trajectory,
            "show_current_position": show_current_position,
            "show_sample_trajectories": show_sample_trajectories,
            "show_median_forecast": show_median_forecast,
            "show_prediction_region_50": show_prediction_region_50,
            "show_prediction_region_90": show_prediction_region_90,
        }
        if any(not isinstance(value, (bool, np.bool_)) for value in options.values()):
            raise ValueError("Display options must be boolean values.")
        for name, value in options.items():
            setattr(self, name, bool(value))
        if follow_ship_view_span_m is not None:
            self.follow_ship_view_span_m = _validate_follow_ship_view_span_m(
                follow_ship_view_span_m
            )
            if self.follow_checkbox.get_status()[0]:
                self._follow_ship_view_needs_focus = True
        self._draw()

    def set_coordinate_display_mode(self, coordinate_display_mode) -> None:
        """Redraw cached spatial results in a new unit without another inference run."""
        coordinate_display_mode = normalize_coordinate_display_mode(
            coordinate_display_mode
        )
        if coordinate_display_mode == self.coordinate_display_mode:
            return
        previous_mode = self.coordinate_display_mode
        self.coordinate_display_mode = coordinate_display_mode
        try:
            self._validate_coordinate_display_mode()
        except Exception:
            self.coordinate_display_mode = previous_mode
            raise
        self._reset_trajectory_view_for_coordinate_change = True
        self._follow_ship_view_needs_focus = self.follow_checkbox.get_status()[0]
        self._draw_trajectory()
        self.figure.canvas.draw_idle()
        if self._on_state_change is not None:
            self._on_state_change(
                self.trajectory.reference_state_at(
                    self.observation_count,
                    coordinate_display_mode=self.coordinate_display_mode,
                )
            )

    def _validate_coordinate_display_mode(self) -> None:
        if self.coordinate_display_mode == "gps" and (
            self.trajectory.reference_longitude is None
            or self.trajectory.reference_latitude is None
        ):
            raise ValueError(
                "GPS coordinate display requires reference_longitude and "
                "reference_latitude."
            )

    def _display_coordinates(self, x_meters, y_meters):
        """Convert local metre coordinates only at the presentation boundary."""
        if self.coordinate_display_mode == "m":
            return x_meters, y_meters
        if self.coordinate_display_mode == "km":
            return (
                x_meters / coordinates.METERS_PER_KILOMETER,
                y_meters / coordinates.METERS_PER_KILOMETER,
            )
        return coordinates.local_to_gps_coordinates(
            x_meters,
            y_meters,
            self.trajectory.reference_longitude,
            self.trajectory.reference_latitude,
        )

    def _coordinate_display_spec(self):
        """Return readable labels and spatial scaling for the active display mode."""
        if self.coordinate_display_mode == "m":
            return "Easting x [m]", "Northing y [m]", 1.0
        if self.coordinate_display_mode == "km":
            return "Easting x [km]", "Northing y [km]", 1.0
        return (
            "Longitude [°]",
            "Latitude [°]",
            float(1.0 / np.cos(np.radians(self.trajectory.reference_latitude))),
        )

    def _draw_posterior(
        self,
        axis,
        parameter_name,
        *,
        show_legend_axis=None,
    ) -> None:
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
            label="Initial prior",
        )
        minimum_posterior_observation_count = max(
            self.minimum_posterior_observation_count,
            PARAMETER_POSTERIOR_MINIMUM_OBSERVATION_COUNTS.get(parameter_name, 0),
        )
        posterior_available = (
            self.observation_count >= minimum_posterior_observation_count
        )
        if posterior_available:
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
                label="Posterior density",
            )
            axis.fill_between(
                x_values,
                posterior_density,
                color=prior_posterior.POSTERIOR_FILL_COLOR,
                alpha=0.18,
            )
        elif self.observation_count > 0:
            axis.text(
                0.5,
                0.88,
                (f"Posterior available from N = {minimum_posterior_observation_count}"),
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                color="0.35",
            )
        title = spec.title
        if parameter_name in PARAMETER_PRIOR_TITLES:
            title = (
                PARAMETER_POSTERIOR_TITLES[parameter_name]
                if posterior_available
                else PARAMETER_PRIOR_TITLES[parameter_name]
            )
        axis.set_title(title, fontsize=11, pad=6)
        axis.set_xlabel(spec.x_label, fontsize=10)
        axis.set_ylabel("Density", fontsize=10)
        if parameter_name in NOISE_PARAMETER_NAMES:
            axis.set_xscale("log")
        axis.set_xlim(float(x_values[0]), float(x_values[-1]))
        if spec.support == "circular":
            axis.set_xticks([-180.0, -90.0, 0.0, 90.0, 180.0])
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25, linewidth=0.8)
        axis.tick_params(labelsize=9)
        if show_legend_axis is None:
            show_legend_axis = axis is self.posterior_axes[0]
        if self.show_legend and show_legend_axis:
            axis.legend(loc="upper right", fontsize=8, framealpha=0.9)


def create_sequential_posterior_dashboard_figure(
    trajectory,
    priors,
    update_loader=None,
    *,
    maximum_observation_count,
    minimum_posterior_observation_count=1,
    show_legend=True,
    playback_interval_ms=DEFAULT_PLAYBACK_INTERVAL_MS,
    figure=None,
    request_update=None,
    prediction_count=0,
    coordinate_display_mode="m",
    show_reference_trajectory=True,
    show_observed_trajectory=True,
    show_current_position=True,
    show_sample_trajectories=True,
    show_median_forecast=True,
    show_prediction_region_50=True,
    show_prediction_region_90=True,
    follow_ship_view_span_m=FOLLOW_SHIP_VIEW_SPAN_METERS,
    on_state_change=None,
    on_metrics_change=None,
    on_medians_change=None,
):
    """Create a dashboard, optionally using an embedded canvas and async requests.

    With ``request_update``, deliver results via ``navigator.accept_update`` on
    the UI thread. Otherwise ``update_loader`` is called synchronously as before.
    Attach an embedded canvas to ``figure`` before calling this factory so its
    playback timer uses the host GUI's event loop.
    """
    if not isinstance(trajectory, PosteriorDashboardTrajectory):
        raise TypeError("trajectory must be a PosteriorDashboardTrajectory.")
    if not isinstance(priors, bayesian_model.BayesianCTRVPriors):
        raise TypeError("priors must be a BayesianCTRVPriors instance.")
    if request_update is None and not callable(update_loader):
        raise TypeError("update_loader must be callable for synchronous navigation.")
    if request_update is not None and not callable(request_update):
        raise TypeError("request_update must be callable.")
    if (
        isinstance(maximum_observation_count, bool)
        or not isinstance(maximum_observation_count, int)
        or not 1 <= maximum_observation_count <= trajectory.reference_x.size
    ):
        raise ValueError("maximum_observation_count must fit within the trajectory.")
    if (
        isinstance(minimum_posterior_observation_count, bool)
        or not isinstance(minimum_posterior_observation_count, (int, np.integer))
        or not 1 <= minimum_posterior_observation_count <= maximum_observation_count
    ):
        raise ValueError(
            "minimum_posterior_observation_count must fit within the dashboard."
        )
    minimum_posterior_observation_count = int(minimum_posterior_observation_count)
    if (
        isinstance(playback_interval_ms, bool)
        or not isinstance(playback_interval_ms, (int, np.integer))
        or playback_interval_ms < 1
    ):
        raise ValueError("playback_interval_ms must be a positive integer.")
    playback_interval_ms = int(playback_interval_ms)
    prediction_count = numeric_validation.validate_non_negative_integer(
        "prediction_count",
        prediction_count,
    )

    if figure is None:
        figure = plt.figure(figsize=FIGURE_SIZE)
    grid = figure.add_gridspec(
        3,
        3,
        left=0.06,
        right=0.98,
        width_ratios=(1.25, 0.5, 0.5),
        wspace=0.28,
    )
    trajectory_axis = figure.add_subplot(grid[:, 0])
    posterior_axes_by_group = {
        "motion": tuple(figure.add_subplot(grid[row, 1]) for row in range(3)),
        "noise": tuple(figure.add_subplot(grid[row, 2]) for row in range(3)),
    }
    posterior_axes = posterior_axes_by_group["motion"]
    navigator = PosteriorDashboardNavigator(
        figure,
        trajectory_axis,
        posterior_axes,
        trajectory,
        priors,
        update_loader,
        maximum_observation_count=maximum_observation_count,
        minimum_posterior_observation_count=minimum_posterior_observation_count,
        show_legend=show_legend,
        playback_interval_ms=playback_interval_ms,
        request_update=request_update,
        prediction_count=prediction_count,
        coordinate_display_mode=coordinate_display_mode,
        posterior_axes_by_group=posterior_axes_by_group,
        show_reference_trajectory=show_reference_trajectory,
        show_observed_trajectory=show_observed_trajectory,
        show_current_position=show_current_position,
        show_sample_trajectories=show_sample_trajectories,
        show_median_forecast=show_median_forecast,
        show_prediction_region_50=show_prediction_region_50,
        show_prediction_region_90=show_prediction_region_90,
        follow_ship_view_span_m=follow_ship_view_span_m,
        on_state_change=on_state_change,
        on_metrics_change=on_metrics_change,
        on_medians_change=on_medians_change,
    )
    return figure, navigator


def create_posterior_dashboard_loader(
    trajectory_data,
    *,
    experiment,
    priors,
    vi_config,
    mcmc_config,
    rbpf_config,
    smc_config,
    initialize_online_filter=None,
    fit_batch_model=None,
):
    """Create the selected batch or online posterior-update loader."""
    if not isinstance(experiment, PosteriorDashboardConfig):
        raise TypeError("experiment must be a PosteriorDashboardConfig instance.")
    if not isinstance(priors, bayesian_model.BayesianCTRVPriors):
        raise TypeError("priors must be a BayesianCTRVPriors instance.")
    inference_mode, inference_method = inference.normalize_inference_method(
        experiment.inference_method,
        online_inference_methods=inference.CTRV_ONLINE_INFERENCE_METHODS,
    )
    online_mode = inference_mode == "online"
    trajectory, time_seconds, maximum_observation_count = (
        _prepare_posterior_dashboard_trajectory(
            trajectory_data,
            experiment=experiment,
            reserve_prediction=not online_mode,
        )
    )
    specs = tuple(
        prior_posterior.build_parameter_spec(parameter_name, priors)
        for parameter_name in PARAMETER_NAMES
    )

    if online_mode:
        if inference_method == "rbpf":
            if not isinstance(rbpf_config, rbpf.SequentialCTRVFilterConfig):
                raise TypeError(
                    "rbpf_config must be a SequentialCTRVFilterConfig instance."
                )
            filter_type = rbpf.SequentialBayesianCTRVFilter
            particle_filter_config = rbpf_config
        else:
            if not isinstance(smc_config, smc.SequentialMonteCarloCTRVConfig):
                raise TypeError(
                    "smc_config must be a SequentialMonteCarloCTRVConfig instance."
                )
            filter_type = smc.SequentialMonteCarloCTRVFilter
            particle_filter_config = smc_config
        if initialize_online_filter is None:
            initialize_online_filter = filter_type.initialize
        if not callable(initialize_online_filter):
            raise TypeError("initialize_online_filter must be callable.")
        online_filter = initialize_online_filter(
            time_seconds[:1],
            trajectory.observed_x[:1],
            trajectory.observed_y[:1],
            priors=priors,
            config=particle_filter_config,
            seed=experiment.inference_seed,
        )

        def load_online_update(observation_count):
            _validate_update_observation_count(
                observation_count,
                minimum=1,
                maximum=maximum_observation_count,
            )
            if observation_count < online_filter.processed_observation_count:
                raise ValueError("The online update loader cannot move backward.")
            while online_filter.processed_observation_count < observation_count:
                index = online_filter.processed_observation_count
                online_filter.update(
                    time_seconds[index],
                    trajectory.observed_x[index],
                    trajectory.observed_y[index],
                )
            fit = online_filter.sample_current_posterior(seed=experiment.inference_seed)
            future_times = time_seconds[
                observation_count : observation_count + experiment.prediction_count
            ]
            forecast = None
            if future_times.size:
                forecast_fit = online_filter.forecast(
                    future_times,
                    seed=1_000_000 + experiment.inference_seed,
                )
                forecast = _extract_dashboard_forecast(
                    forecast_fit,
                    future_times - time_seconds[observation_count - 1],
                    experiment.prediction_sample_count,
                )
            return PosteriorDashboardUpdate(
                observation_count=observation_count,
                samples_by_parameter=_extract_dashboard_samples(fit, specs),
                effective_sample_size=online_filter.effective_sample_size,
                particle_count=particle_filter_config.particle_count,
                resample_count=online_filter.resample_count,
                forecast=forecast,
            )

        return trajectory, maximum_observation_count, 1, load_online_update

    selected_config = dict(vi_config if inference_method == "vi" else mcmc_config)
    if fit_batch_model is None:
        fit_batch_model = batch_inference.fit_bayesian_ctrv_model
    if not callable(fit_batch_model):
        raise TypeError("fit_batch_model must be callable.")

    def load_batch_update(observation_count):
        _validate_update_observation_count(
            observation_count,
            minimum=bayesian_model.MIN_OBSERVATION_COUNT,
            maximum=maximum_observation_count,
        )
        window = observation_window.prepare_trajectory_window(
            trajectory_data,
            observation_count=observation_count,
            prediction_count=max(
                1,
                min(experiment.prediction_count, len(time_seconds) - observation_count),
            ),
            start_index=experiment.start_index,
        )
        position_observations = bayesian_model.PositionObservations(
            time_seconds=window.time_seconds[window.observed_slice],
            x_meters=trajectory.observed_x[:observation_count],
            y_meters=trajectory.observed_y[:observation_count],
            position_noise_std_m=experiment.position_noise_std_m,
            noise_seed=experiment.position_noise_seed,
        )
        fit = fit_batch_model(
            window,
            priors=priors,
            position_observations=position_observations,
            inference_method=inference_method,
            seed=experiment.inference_seed,
            **selected_config,
        )
        return PosteriorDashboardUpdate(
            observation_count=observation_count,
            samples_by_parameter=_extract_dashboard_samples(fit, specs),
            forecast=(
                _extract_dashboard_forecast(
                    fit,
                    window.time_seconds[window.prediction_slice]
                    - window.time_seconds[observation_count - 1],
                    experiment.prediction_sample_count,
                )
                if experiment.prediction_count
                else None
            ),
        )

    return (
        trajectory,
        maximum_observation_count,
        bayesian_model.MIN_OBSERVATION_COUNT,
        load_batch_update,
    )


def _prepare_posterior_dashboard_trajectory(
    trajectory_data,
    *,
    experiment,
    reserve_prediction,
):
    start_index = experiment.start_index
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or start_index < 0
    ):
        raise ValueError("start_index must be a non-negative integer.")
    selected_trajectory_data = trajectory_data.iloc[start_index:]
    if len(selected_trajectory_data) < bayesian_model.MIN_OBSERVATION_COUNT + 1:
        raise ValueError(
            f"Start index {start_index} is too large: the selected trajectory has "
            f"{len(trajectory_data)} positions, but at least four consecutive "
            "positions are required."
        )
    selected_trajectory_data = _select_trajectory_rows_at_interval(
        selected_trajectory_data,
        experiment.observation_interval_seconds,
    )
    available_observation_count = len(selected_trajectory_data)
    if available_observation_count < bayesian_model.MIN_OBSERVATION_COUNT + 1:
        raise ValueError(
            f"Observation interval {experiment.observation_interval_seconds:g} s "
            f"leaves only {available_observation_count} positions; at least four "
            "consecutive positions are required."
        )
    maximum_supported_count = available_observation_count - int(reserve_prediction)
    maximum_observation_count = experiment.maximum_observation_count
    if maximum_observation_count is None:
        maximum_observation_count = maximum_supported_count
    if (
        isinstance(maximum_observation_count, bool)
        or not isinstance(maximum_observation_count, int)
        or not bayesian_model.MIN_OBSERVATION_COUNT
        <= maximum_observation_count
        <= maximum_supported_count
    ):
        raise ValueError(
            "maximum_observation_count must fit within the selected trajectory."
        )

    complete_window = observation_window.prepare_trajectory_window(
        selected_trajectory_data,
        observation_count=available_observation_count - 1,
        prediction_count=1,
        start_index=0,
    )
    time_seconds = np.asarray(complete_window.time_seconds, dtype=float)
    reference_x = np.asarray(complete_window.x_meters, dtype=float)
    reference_y = np.asarray(complete_window.y_meters, dtype=float)
    reference_heading_degrees, reference_turn_rate_degrees_per_second = (
        _reference_heading_and_turn_rate(
            selected_trajectory_data,
            start_index=0,
            count=available_observation_count,
            time_seconds=time_seconds,
            reference_x=reference_x,
            reference_y=reference_y,
        )
    )
    observed_x = reference_x.copy()
    observed_y = reference_y.copy()
    position_noise_std_m = numeric_validation.validate_non_negative_finite(
        "position_noise_std_m",
        experiment.position_noise_std_m,
    )
    position_noise_seed = numeric_validation.validate_non_negative_integer(
        "position_noise_seed",
        experiment.position_noise_seed,
    )
    if position_noise_std_m > 0.0:
        noise_generator = np.random.default_rng(position_noise_seed)
        observed_x += noise_generator.normal(
            0.0,
            position_noise_std_m,
            available_observation_count,
        )
        observed_y += noise_generator.normal(
            0.0,
            position_noise_std_m,
            available_observation_count,
        )
    trajectory = PosteriorDashboardTrajectory(
        reference_x,
        reference_y,
        observed_x,
        observed_y,
        reference_speed_mps=complete_window.gps_speed_mps,
        reference_heading_degrees=reference_heading_degrees,
        reference_turn_rate_degrees_per_second=reference_turn_rate_degrees_per_second,
        reference_time_seconds=time_seconds,
        reference_longitude=complete_window.reference_longitude,
        reference_latitude=complete_window.reference_latitude,
    )
    return trajectory, time_seconds, maximum_observation_count


def _select_trajectory_rows_at_interval(trajectory_data, interval_seconds):
    """Keep the first row and later rows at least one interval apart in time."""
    interval_seconds = numeric_validation.validate_positive_finite(
        "observation_interval_seconds",
        interval_seconds,
    )
    if trajectory_data.empty:
        return trajectory_data.copy()
    timestamps = pd.to_datetime(trajectory_data["time"], utc=True)
    interval = pd.Timedelta(seconds=interval_seconds)
    selected_indices = [0]
    last_timestamp = timestamps.iloc[0]
    for index, timestamp in enumerate(timestamps.iloc[1:], start=1):
        if timestamp - last_timestamp >= interval:
            selected_indices.append(index)
            last_timestamp = timestamp
    return trajectory_data.iloc[selected_indices].copy()


def _validate_update_observation_count(observation_count, *, minimum, maximum):
    if (
        isinstance(observation_count, bool)
        or not isinstance(observation_count, int)
        or not minimum <= observation_count <= maximum
    ):
        raise ValueError(f"observation_count must be between {minimum} and {maximum}.")


def _extract_dashboard_forecast(fit, time_offsets_seconds, sample_count):
    """Summarize all latent draws, keeping only a bounded selection of full paths."""
    x = reporting.posterior_variable_samples(fit, "x_prediction")
    y = reporting.posterior_variable_samples(fit, "y_prediction")
    if (
        x.ndim != 2
        or x.shape != y.shape
        or x.shape[0] < 2
        or x.shape[1] != len(time_offsets_seconds)
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("Forecast draws must be finite matching (draw, time) arrays.")
    positions = np.stack((x, y), axis=-1)
    indices = np.linspace(
        0, len(positions) - 1, min(sample_count, len(positions)), dtype=int
    )
    return PosteriorDashboardForecast(
        time_offsets_seconds=time_offsets_seconds,
        median_positions=np.median(positions, axis=0),
        sample_positions=positions[indices],
    )


def _extract_dashboard_samples(fit, specs):
    return {
        spec.parameter_name: prior_posterior.extract_posterior_samples(fit, spec)
        for spec in specs
    }


def _reference_heading_and_turn_rate(
    trajectory_data, *, start_index, count, time_seconds, reference_x, reference_y
):
    """Use simulation truth columns when present, otherwise raw-route derivatives."""
    ordered_data = (
        trajectory_data.sort_values("time")
        .reset_index(drop=True)
        .iloc[start_index : start_index + count]
    )
    if {"theta", "omega"}.issubset(ordered_data.columns):
        heading = np.rad2deg(np.asarray(ordered_data["theta"], dtype=float))
        turn_rate = np.rad2deg(np.asarray(ordered_data["omega"], dtype=float))
        if np.all(np.isfinite(heading)) and np.all(np.isfinite(turn_rate)):
            return heading, turn_rate
    delta_x = np.diff(reference_x)
    delta_y = np.diff(reference_y)
    heading_radians = np.arctan2(delta_y, delta_x)
    heading = np.empty(count)
    heading[0] = heading_radians[0]
    heading[1:] = heading_radians
    turn_rate = np.empty(count)
    turn_rate[0] = 0.0
    turn_rate[1:] = np.diff(np.unwrap(heading)) / np.diff(time_seconds)
    return np.rad2deg(heading), np.rad2deg(turn_rate)


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
    experiment = PosteriorDashboardConfig(
        run_id=0,
        start_index=start_index,
        maximum_observation_count=None,
        position_noise_std_m=position_noise_std_m,
        position_noise_seed=position_noise_seed,
        inference_method="rbpf",
        inference_seed=rbpf_seed,
        prediction_count=0,  # Preserve this legacy posterior-only convenience API.
    )
    trajectory, maximum_observation_count, _, load_update = (
        create_posterior_dashboard_loader(
            trajectory_data,
            experiment=experiment,
            priors=priors,
            vi_config={},
            mcmc_config={},
            rbpf_config=rbpf_config,
            smc_config=None,
            initialize_online_filter=initialize_filter,
        )
    )
    return trajectory, maximum_observation_count, load_update


def run_bayesian_ctrv_posterior_dashboard(
    *,
    data_file,
    experiment,
    priors,
    vi_config,
    mcmc_config,
    rbpf_config,
    smc_config,
    playback_interval_ms,
    show_legend,
    coordinate_display_mode="m",
    show=True,
):
    """Run the interactive trajectory and posterior dashboard."""
    if not isinstance(experiment, PosteriorDashboardConfig):
        raise TypeError("experiment must be a PosteriorDashboardConfig instance.")
    trajectory_data = (
        observations_io.read_ship_data(data_file, run_id=experiment.run_id)
        .sort_values("time")
        .reset_index(drop=True)
    )
    if trajectory_data.empty:
        raise ValueError(f"No trajectory rows found for run_id={experiment.run_id}.")

    (
        trajectory,
        maximum_observation_count,
        minimum_posterior_observation_count,
        load_update,
    ) = create_posterior_dashboard_loader(
        trajectory_data,
        experiment=experiment,
        priors=priors,
        vi_config=vi_config,
        mcmc_config=mcmc_config,
        rbpf_config=rbpf_config,
        smc_config=smc_config,
    )
    figure, navigator = create_sequential_posterior_dashboard_figure(
        trajectory,
        priors,
        load_update,
        maximum_observation_count=maximum_observation_count,
        minimum_posterior_observation_count=(minimum_posterior_observation_count),
        show_legend=show_legend,
        playback_interval_ms=playback_interval_ms,
        prediction_count=experiment.prediction_count,
        coordinate_display_mode=coordinate_display_mode,
    )
    if show:
        plt.show(block=True)
    plt.close(figure)
    return figure, navigator
